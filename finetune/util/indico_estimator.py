import numpy as np
import tensorflow as tf
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import training
from tensorflow.estimator import ModeKeys

def placeholder_like(tensor):
    return tf.placeholder(tensor.dtype, shape=tensor.shape)


def parse_input_fn_result(result):
    iterator = tf.data.make_initializable_iterator(result)
    init = iterator.initializer
    result = iterator.get_next()
    return result, init


class IndicoEstimator(tf.estimator.Estimator):
    def __init__(self, *args, **kwargs):
        self.estimator_spec = None
        self.g = None
        self.features_real = None
        self.placeholder_feats = None
        self.predictions = None
        self.ds_init = None
        self.mon_sess = None
        self._cached_predict = False
        self.available_data = []
        super().__init__(*args, **kwargs)

    def get_features_from_fn(self, input_fn, predict=True):
        with tf.Graph().as_default() as g:
            result = self._call_input_fn(input_fn, ModeKeys.PREDICT)
            features, initializer = parse_input_fn_result(result)
            if type(features) == tuple and predict:
                features = features[0]
            with tf.Session() as sess:
                sess.run(initializer)
                while True:
                    try:
                        self.available_data.append(sess.run(features))
                    except tf.errors.OutOfRangeError:
                        break
            return features

    def data_generator(self):
        yield from self.available_data
        self.available_data = []

    def close_predict(self):
        #tf.reset_default_graph()
        self.estimator_spec = None
        self.features_real = None
        self.ds_init = None
        self.predictions = None
        self.g = None
        if self.mon_sess is not None:
            self.mon_sess.close()
        self.mon_sess = None

    def cached_predict(self,
                input_fn,
                predict_keys=None,
                hooks=None,
                checkpoint_path=None,
                yield_single_examples=True):
        # Check that model has been trained.
        self.g = self.g or tf.Graph()
        random_seed.set_random_seed(self._config.tf_random_seed)
        features = self.get_features_from_fn(input_fn)
        with self.g.as_default():
            if self.estimator_spec is None:
                self._create_and_assert_global_step(self.g)
                if not checkpoint_path:
                    checkpoint_path = checkpoint_management.latest_checkpoint(
                        self._model_dir)
                if not checkpoint_path:
                    logging.info('Could not find trained model in model_dir: {}, running '
                                 'initialization to predict.'.format(self._model_dir))
                with tf.device('/cpu:0'):
                    iterator = tf.data.make_initializable_iterator(
                        tf.data.Dataset.from_generator(
                            self.data_generator,
                            tf.contrib.framework.nest.map_structure(lambda f: f.dtype, features),
                            tf.contrib.framework.nest.map_structure(lambda f: f.shape, features),
                        )
                    )
                self.ds_init = iterator.initializer
                model_fn_input = iterator.get_next()
                self.estimator_spec = self._call_model_fn(model_fn_input, None, ModeKeys.PREDICT, self.config)
                # Call to warm_start has to be after model_fn is called.
                self._maybe_warm_start(checkpoint_path)

                self.predictions = self._extract_keys(self.estimator_spec.predictions, predict_keys)
                all_hooks = hooks or []
                all_hooks.extend(list(self.estimator_spec.prediction_hooks or []))

                self.mon_sess = training.MonitoredSession(
                    session_creator=training.ChiefSessionCreator(
                        checkpoint_filename_with_path=checkpoint_path,
                        master=self._config.master,
                        scaffold=self.estimator_spec.scaffold,
                        config=self._session_config),
                    hooks=all_hooks)

            self.mon_sess.run(self.ds_init)
            while True:
                try:
                    preds_evaluated = self.mon_sess.run(self.predictions)
                    if not yield_single_examples:
                        yield preds_evaluated
                    elif not isinstance(self.predictions, dict):
                        for pred in preds_evaluated:
                            yield pred
                    else:
                        for i in range(self._extract_batch_length(preds_evaluated)):
                            yield {
                                key: value[i]
                                for key, value in preds_evaluated.items()
                            }
                except tf.errors.OutOfRangeError:
                    self.available_data = []

class Scheduler:
    def __init__(self, models, max_models_per_gpu, num_gpus):
        assert num_gpus == 1 # not yet
        self.models = models
        self.loaded_models = list()
        self.max_models_per_gpu = max_models_per_gpu

    def _rotate_in_model(self, model_id):
        print(self.loaded_models)
        model = self.models[model_id]
        if model not in self.loaded_models:
            if len(self.loaded_models) + 1 > self.max_models_per_gpu:
                self.loaded_models.pop(0).close_predict()
        else:
            self.loaded_models.remove(model) # put it back at the end of the queue
        self.loaded_models.append(model)
        return model

    def predict(self, model_id, input_fn, predict_keys=None, hooks=None, checkpoint_path=None, yield_single_examples=True):
        model = self._rotate_in_model(model_id)
        return model.indico_predict(
            input_fn, predict_keys=predict_keys, hooks=hooks, checkpoint_path=checkpoint_path, yield_single_examples=yield_single_examples
        )
        
