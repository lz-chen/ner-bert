import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from .utils import tokens2spans, bert_labels2tokens, voting_choicer, first_choicer
from sklearn_crfsuite.metrics import flat_classification_report
from pathlib import Path


def plot_by_class_curve(history, metric_, sup_labels):
    by_class = get_by_class_metric(history, metric_, sup_labels)
    vals = list(by_class.values())
    x = np.arange(len(vals[0]))
    args = []
    for val in vals:
        args.append(x)
        args.append(val)
    plt.figure(figsize=(15, 10))
    plt.grid(True)
    plt.plot(*args)
    plt.legend(list(by_class.keys()))
    _, _ = plt.yticks(np.arange(0, 1, step=0.1))
    plt.show()


def get_metrics_by_class(text_res, sup_labels):
    # text_res = flat_classification_report(y_true, y_pred, labels=labels, digits=3)
    res = {}
    for line in text_res.split("\n"):
        line = line.split()
        if len(line) and line[0] in sup_labels:
            res[line[0]] = {key: val for key, val in zip(["prec", "rec", "f1"], line[1:-1])}
    return res


def get_by_class_metric(history, metric_, sup_labels):
    res = defaultdict(list)
    for h in history:
        h = get_metrics_by_class(h, sup_labels)
        for class_, metrics_ in h.items():
            res[class_].append(float(metrics_[metric_]))
    return res


def get_max_metric(history, metric_, sup_labels, return_idx=False):
    by_class = get_by_class_metric(history, metric_, sup_labels)
    by_class_arr = np.array(list(by_class.values()))
    idx = np.array(by_class_arr.sum(0)).argmax()
    if return_idx:
        return list(zip(by_class.keys(), by_class_arr[:, idx])), idx
    return list(zip(by_class.keys(), by_class_arr[:, idx]))


def get_mean_max_metric(history, metric_="f1", return_idx=False):
    m_idx = 0
    if metric_ == "f1":
        m_idx = 2
    elif m_idx == "rec":
        m_idx = 1
    metrics = [float(h.split("\n")[-2].split()[3 + m_idx]) for h in history]
    idx = np.argmax(metrics)
    res = metrics[idx]
    if return_idx:
        return idx, res
    return res


def bert_preds_to_y(dl, preds, ignore_labels=["O"], fn=first_choicer):  # original
    tokens, labels = bert_labels2tokens(dl, preds, fn)
    spans_pred = tokens2spans(tokens, labels)  # list of (span, pred_tag), span can be more than one token
    tokens, labels = bert_labels2tokens(dl, [x.labels for x in dl.dataset], fn)
    spans_true = tokens2spans(tokens, labels)
    set_labels = set()
    for idx in range(len(spans_pred)):
        while len(spans_pred[idx]) < len(spans_true[idx]):
            spans_pred[idx].append(("", "O"))
        while len(spans_pred[idx]) > len(spans_true[idx]):
            spans_true[idx].append(("O", "O"))
        set_labels.update([y for x, y in spans_true[idx]])
    set_labels -= set(ignore_labels)
    y_true, y_pred = [[y[1] for y in x] for x in spans_true], [[y[1] for y in x] for x in spans_pred]
    return tokens, y_true, y_pred, list(set_labels)


def bert_preds_to_ys(dl, preds, ignore_labels=["O"], fn=first_choicer):
    pred_tokens, pred_labels = bert_labels2tokens(dl, preds, fn)
    true_tokens, true_labels = bert_labels2tokens(dl, [x.labels for x in dl.dataset], fn)
    pred_labels, set_labels = _clean_tags(pred_labels)
    true_labels, set_labels = _clean_tags(true_labels)
    set_labels -= set(ignore_labels)
    assert all(len(pred_tokens[i]) == len(pred_labels[i]) for i in range(len(pred_tokens))) is True
    assert all(len(pred_tokens[i]) == len(true_tokens[i]) for i in range(len(pred_tokens))) is True
    assert all(len(true_tokens[i]) == len(true_labels[i]) for i in range(len(true_tokens))) is True

    return true_tokens, true_labels, pred_labels, list(set_labels)


def _clean_tags(tag_sequences):
    set_labels = set()
    clean_tag_sequences = []
    for tag_sequence in tag_sequences:
        clean_tag_sequence = []
        for tag in tag_sequence:
            if tag == 'B_O':
                clean_tag_sequence.append("O")
            else:
                tag = tag.split('_')[-1]
                clean_tag_sequence.append(tag)
                set_labels.add(tag)
        clean_tag_sequences.append(clean_tag_sequence)
    return clean_tag_sequences, set_labels


def get_bert_span_report(dl, preds, ignore_labels=["O"], fn=first_choicer):
    # tokens, labels = bert_labels2tokens(dl, preds, fn)
    # spans_pred = tokens2spans(tokens, labels)
    # tokens, labels = bert_labels2tokens(dl, [x.labels for x in dl.dataset], fn)
    # spans_true = tokens2spans(tokens, labels)
    # set_labels = set()
    # for idx in range(len(spans_pred)):
    #     while len(spans_pred[idx]) < len(spans_true[idx]):
    #         spans_pred[idx].append(("", "O"))
    #     while len(spans_pred[idx]) > len(spans_true[idx]):
    #         spans_true[idx].append(("O", "O"))
    #     set_labels.update([y for x, y in spans_true[idx]])
    # set_labels -= set(ignore_labels)
    _, y_true, y_pred, set_labels = bert_preds_to_y(dl, preds, ignore_labels, fn)
    return flat_classification_report(y_true, y_pred, set_labels, digits=3)


def get_elmo_span_report(dl, preds, ignore_labels=["O"]):
    tokens, labels = [x.tokens[1:-1] for x in dl.dataset], [p[1:-1] for p in preds]
    spans_pred = tokens2spans(tokens, labels)
    labels = [x.labels[1:-1] for x in dl.dataset]
    spans_true = tokens2spans(tokens, labels)
    set_labels = set()
    for idx in range(len(spans_pred)):
        while len(spans_pred[idx]) < len(spans_true[idx]):
            spans_pred[idx].append(("", "O"))
        while len(spans_pred[idx]) > len(spans_true[idx]):
            spans_true[idx].append(("O", "O"))
        set_labels.update([y for x, y in spans_true[idx]])
    set_labels -= set(ignore_labels)
    return flat_classification_report([[y[1] for y in x] for x in spans_true], [[y[1] for y in x] for x in spans_pred],
                                      labels=list(set_labels), digits=3)


def analyze_bert_errors(dl, labels, fn=voting_choicer):
    errors = []
    res_tokens = []
    res_labels = []
    r_labels = [x.labels for x in dl.dataset]
    for f, l_, rl in zip(dl.dataset, labels, r_labels):
        label = fn(f.tok_map, l_)
        label_r = fn(f.tok_map, rl)
        prev_idx = 0
        errors_ = []
        # if len(label_r) > 1:
        # assert len(label_r) == len(f.tokens) - 1
        for idx, (l, rl, t) in enumerate(zip(label, label_r, f.tokens)):
            if l != rl:
                errors_.append({"token: ": t,
                                "real_label": rl,
                                "pred_label": l,
                                "bert_token": f.bert_tokens[prev_idx:f.tok_map[idx]],
                                "real_bert_label": f.labels[prev_idx:f.tok_map[idx]],
                                "pred_bert_label": l_[prev_idx:f.tok_map[idx]],
                                "text_example": " ".join(f.tokens[1:-1]),
                                "labels": " ".join(label_r[1:])})
            prev_idx = f.tok_map[idx]
        errors.append(errors_)
        res_tokens.append(f.tokens[1:-1])
        res_labels.append(label[1:])
    return res_tokens, res_labels, errors


def write_true_and_pred_to_conll(tokens, y_true, y_pred, conll_fpath: Path):
    assert len(tokens) == len(y_true) and len(y_true) == len(y_pred)
    with conll_fpath.open('w') as f:
        for sentence, y_t, y_p in zip(tokens, y_true, y_pred):
            try:
                assert len(sentence) == len(y_t) and len(y_t) == len(y_p)
            except AssertionError:
                print(sentence)
                print(y_t)
                print(y_p)
                print('{} {} {}'.format(len(sentence), len(y_t), len(y_p)))
            for i in range(len(sentence)):
                newline = '{}\t{}\t{}\n'.format(sentence[i], y_t[i], y_p[i])
                f.write(newline)

            f.write('\n')
