import warnings

import numpy as np

from finetune.encoding.input_encoder import get_spacy


def assign_associations(associations, none_value, idx_lookup):
    candidates = dict()
    for association in associations:
        for bpe_idx, candidate_idx, candidate_label, candidate_prob in association:
            if candidate_label == none_value:
                continue
            if idx_lookup[bpe_idx] not in candidates:
                candidates[idx_lookup[bpe_idx]] = []
            candidates[idx_lookup[bpe_idx]].append(
                (idx_lookup[candidate_idx], candidate_label, candidate_prob)
            )

    # TODO some how sample these candidates eg maximum probabilities, to fit some schema
    candidates = {
        k: max(v, key=lambda x: x[2]) for k, v in candidates.items()
    }  # for now just pick maximum prob
    return candidates


def _merge_confidences(annotation):
    """
    Collapse list of confidences down to a single mean confidence.
    """
    if not "confidence" in annotation or not len(annotation["confidence"]):
        return

    labels = annotation["confidence"][0].keys()
    annotation["confidence"] = {
        label: np.mean(
            [confidences[label] for confidences in annotation["confidence"]]
        ).tolist()
        for label in labels
    }


def round_to_nearest_start_and_end(label, token_starts, token_ends, text):
    # Update label start / end / text to align with nearest token start_token and end
    # Applies in-place modification to `label` obj.
    end_distances = np.abs(token_ends - label["end"])
    label["end"] = token_ends[np.argmin(end_distances)]

    token_starts = token_starts[token_starts < label["end"]]
    start_distances = np.abs(np.asarray(token_starts) - label["start"])
    label["start"] = token_starts[np.argmin(start_distances)]

    label["text"] = text[label["start"] : label["end"]]


def finetune_to_indico_sequence(
    raw_texts,
    subseqs,
    labels,
    probs=None,
    none_value=None,
    subtoken_predictions=False,
    associations=None,
    bio_tagging=False,
):
    """
    Maps from the labeled substring format into the 'indico' format. This is the exact inverse operation to
    :meth indico_to_finetune_sequence:.

    The indico format is as follows:
        Raw text for X,
        Labels as a list of dicts, with each dict in the form:
        {
            'start': <Character index of the start_token of the labeled sequence>,
            'end': <Character index of the end of the labeled sequence>,
            'label': <A categorical label (int or string) that represents the category of the subsequence,
            'text': <Optionally, a field with the subsequence contained between the start and end.
        }

    The Labeled substring, or finetune internal, format is as follows.
    Each item of the data is a list strings of the form:
        ["The quick brown", "fox", "jumped over the lazy", ...]
    With the corresponding labels:
        ["PAD", "animal", "PAD", ...]

    It is the :param none_value: that is used to populate the PAD labels.
    :param data: A list of segmented text of the form list(list(str))
    :param labels: Categorical labels for each sub-string in data.
    :param none_value: The none value used to encode the input format.
    :return: Texts, annoatations both in the 'indico' format.
    """
    annotations = []
    spacy_docs = get_spacy().pipe(raw_texts)
    loop_vals = zip(raw_texts, spacy_docs, subseqs, labels, probs or [None] * len(raw_texts))
    for doc_idx, (raw_text, spacy_tokens, doc_seq, label_seq, prob_seq) in enumerate(loop_vals):
        spacy_token_starts = np.asarray([token.idx for token in spacy_tokens])
        spacy_token_ends = np.asarray([token.idx + len(token.text) for token in spacy_tokens])
        doc_annotations = []
        annotation_ranges = set()
        raw_annotation_end = 0
        raw_annotation_start = 0
        subtoken_to_label_idx = []
        for i, (sub_str, raw_label, confidences) in enumerate(
            zip(doc_seq, label_seq, prob_seq or [None] * len(doc_seq))
        ):
            subtoken_to_label_idx.append(len(doc_annotations))
            if not isinstance(raw_label, tuple):
                multilabel = False
                label_list = [raw_label]
            else:
                multilabel = True
                label_list = raw_label

            for label_idx, label in enumerate(label_list):
                stripped_text = sub_str.strip()
                start_of_search = raw_annotation_start if label_idx != 0 else raw_annotation_end 
                if subtoken_predictions:
                    raw_annotation_start = raw_text.find(sub_str, start_of_search)
                    raw_annotation_end = raw_annotation_start + len(sub_str)
                else:
                    raw_annotation_start = raw_text.find(
                        stripped_text, start_of_search
                    )
                    raw_annotation_end = raw_annotation_start + len(stripped_text)

                if raw_annotation_start == -1:
                    warnings.warn(
                        "Failed to find predicted sequence: {} in text".format(
                            sub_str
                        )
                    )
                    continue


                extended_existing_label = False
                for item in (doc_annotations if multilabel else doc_annotations[-1:]):
                    # handle case where we extend existing annotation
                    if (
                        # same label
                        item["label"] == label
                        # and only separated by whitespace
                        and item["end"] <= raw_annotation_end
                        and not raw_text[item["end"]: raw_annotation_start].strip()
                        # and not BIO tagging
                        and not bio_tagging
                    ):
                        item["end"] = raw_annotation_end
                        item["text"] = raw_text[item["start"]: raw_annotation_end]
                        if "confidence" in item and confidences is not None:
                            item["confidence"].append(confidences)
                        extended_existing_label = True
                        break

                if extended_existing_label or label == none_value:
                    continue

                annotation_start, annotation_end = (
                    int(raw_annotation_start),
                    int(raw_annotation_end),
                )

                annotation = {
                    "start": annotation_start,
                    "end": annotation_end,
                    "label": label,
                    "text": raw_text[annotation_start:annotation_end],
                }

                # if we don't want to allow subtoken predictions, adjust start and end to match
                # the start and ends of the nearest full tokens
                if not subtoken_predictions:
                    round_to_nearest_start_and_end(
                        annotation, spacy_token_starts, spacy_token_ends, raw_text
                    )

                if confidences is not None:
                    annotation["confidence"] = [confidences]

                if annotation["start"] >= annotation["end"]:
                    continue

                # prevent duplicate annotation edge case
                annotation_tuple = (annotation["start"], annotation["end"], label)
                if annotation_tuple not in annotation_ranges:
                    annotation_ranges.add(annotation_tuple)
                    doc_annotations.append(annotation)

        doc_annotations = sorted(
            [dict(items) for items in doc_annotations], key=lambda x: span(x)
        )

        for annotation in doc_annotations:
            _merge_confidences(annotation)

        annotations.append(doc_annotations)
    return raw_texts, annotations


def span(annotation):
    # represents annotation by its char start and end idx
    return (annotation["start"], annotation["end"])


def sorted_insert(annotations, annotation):
    # iterate in reverse order until you find the proper location to insert the new annotation
    idx = 0
    n = len(annotations)
    while idx < len(annotations) and span(annotation) < span(annotations[n - idx - 1]):
        idx += 1

    # mutate annotations list in place
    annotations.insert(n - idx, annotation)


def overlap(current_annotation, annotation):
    return (
        current_annotation["start"] < annotation["end"] <= current_annotation["end"]
    ) or (annotation["start"] < current_annotation["end"] <= annotation["end"])


def overlap_handler(current_annotation, annotation, text, multi_label):
    """
    Scenarios:
        <> --> current_annotation
        [] --> annotation
        
    1) < [ > ]
    2) [ < > ]
    3) < [ ] >
    """
    if current_annotation["start"] <= annotation["start"]:
        first, second = current_annotation, annotation
    else:
        first, second = annotation, current_annotation

    final_delimiter = min(first["end"], second["end"])
    final_label = second["label"] if (second["end"] > first["end"]) else first["label"]
    overlapping_text = text[second["start"] : final_delimiter]
    end = max(first["end"], second["end"])

    first_chunk = {
        "start": first["start"],
        "end": second["start"],
        "label": first["label"],
        "text": text[first["start"] : second["start"]],
    }

    if multi_label:
        second_label = first["label"] | second["label"]
    else:
        if first["label"] != second["label"] and (len(overlapping_text.strip()) > 1):
            warnings.warn(
                "Found overlapping annotations: {} and {}. \n"
                "Consider setting `multi_label_sequences` to `True` in your config.".format(
                    annotation, current_annotation
                )
            )
        spacy_tokens = get_spacy()(text)
        spacy_token_starts = [token.idx for token in spacy_tokens]
        if second["label"] in spacy_token_starts:
            second_label = second["label"]
        elif final_delimiter in spacy_token_starts:
            second_label = first["label"]
        else:
            second_label = first["label"]

    second_chunk = {
        "start": second["start"],
        "end": final_delimiter,
        "label": second_label,
        "text": overlapping_text,
    }

    third_chunk = {
        "start": final_delimiter,
        "end": end,
        "label": final_label,
        "text": text[final_delimiter:end],
    }
    chunks = [first_chunk, second_chunk, third_chunk]
    chunks = [c for c in chunks if c["start"] != c["end"]]
    return chunks
