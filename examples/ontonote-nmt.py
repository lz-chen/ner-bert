import pandas as pd
import warnings
import os
import sys
import codecs
import torch
from modules import BertNerData as NerData
from modules import NerLearner
from modules.models.bert_models import BertBiLSTMAttnNMT
from modules.data.bert_data import get_bert_data_loader_for_predict
from modules.train.train import validate_step
from modules.utils.plot_metrics import get_bert_span_report, bert_preds_to_ys, bert_preds_to_y, \
    write_true_and_pred_to_conll, flat_classification_report
from pathlib import Path

sys.path.append("../")

warnings.filterwarnings("ignore")

data_path = "/media/liah/DATA/ner_data_other/ontonote/"
train_path = data_path + "onto.train.ner"
dev_path = data_path + "onto.development.ner"
test_path = data_path + "onto.test.ner"

result_conll_path = Path('/media/liah/DATA/log/company_tagging/bert_ontonote.conll')


def read_data(input_file, tkn_field_idx=0, label_field_idx=-1, delim='\t'):
    """Reads a BIO data."""
    with codecs.open(input_file, "r", encoding="utf-8") as f:
        lines = []
        words = []
        labels = []
        for line in f:
            content = line.strip()
            if content.startswith("-DOCSTART-"):
                # words.append('')
                continue
            elif len(content) == 0 and not len(words):
                continue

            elif len(content) == 0:
                # empty line, means a sentence is finished
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                if len(l) != 0 and len(w) != 0:
                    assert len(labels) == len(words)
                    lines.append([l, w])
                words = []
                labels = []
                continue
            word = line.rstrip().split(delim)[tkn_field_idx]
            label = line.rstrip().split(delim)[label_field_idx]
            words.append(word)
            labels.append(label.replace("-", "_"))
        return lines


delim = '\t'
train_f = read_data(train_path, delim=delim)
dev_f = read_data(dev_path, delim=delim)
test_f = read_data(test_path, delim=delim)

train_df = pd.DataFrame(train_f, columns=["0", "1"])
train_df.to_csv(data_path + "train.csv", index=False)

valid_df = pd.DataFrame(dev_f, columns=["0", "1"])

valid_df.to_csv(data_path + "valid.csv", index=False)

test_df = pd.DataFrame(test_f, columns=["0", "1"])
test_df.to_csv(data_path + "test.csv", index=False)

train_path = data_path + "train.csv"
valid_path = data_path + "valid.csv"
test_path = data_path + "test.csv"

model_dir = '/media/liah/DATA/pretrained_models/bert/cased_L-24_H-1024_A-16'
init_checkpoint_pt = os.path.join(model_dir, "pytorch_model.bin")
bert_config_file = os.path.join(model_dir, "bert_config.json")
vocab_file = os.path.join(model_dir, "vocab.txt")

torch.cuda.is_available(), torch.cuda.current_device()

data = NerData.create(train_path=train_path, valid_path=valid_path, vocab_file=vocab_file)

# sup_labels = ['B_ORG', 'B_MISC', 'B_PER', 'I_PER', 'B_LOC', 'I_LOC', 'I_ORG', 'I_MISC']

model = BertBiLSTMAttnNMT.create(len(data.label2idx), bert_config_file, init_checkpoint_pt,
                                 enc_hidden_dim=128,
                                 dec_hidden_dim=128,
                                 dec_embedding_dim=16)
print(model)


def train(model, num_epochs=20):
    learner = NerLearner(model, data,
                         best_model_path=model_dir + "/ontonote/bilstm_attn_cased_en_lr0_0001.cpt",
                         lr=0.0001, clip=1.0,
                         sup_labels=[l for l in data.id2label if l not in ['<pad>', '[CLS]', 'X', 'B_O', 'I_']],
                         t_total=num_epochs * len(data.train_dl))
    print('learner.sup_labels : ')
    print(learner.sup_labels)
    learner.fit(num_epochs, target_metric='f1')

    dl = get_bert_data_loader_for_predict(data_path + "valid.csv", learner)

    learner.load_model()

    preds = learner.predict(dl)

    print(validate_step(learner.data.valid_dl, learner.model, learner.data.id2label, learner.sup_labels))

    clf_report = get_bert_span_report(dl, preds, [])
    print(clf_report)


def pred(model, best_model_path,
         result_conll_path,
         fname="test.csv"):
    learner = NerLearner(model, data,
                         best_model_path=model_dir + "/ontonote/bilstm_attn_cased_en.cpt",
                         lr=0.01, clip=1.0,
                         sup_labels=[l for l in data.id2label if l not in ['<pad>', '[CLS]', 'X', 'B_O', 'I_']], )
    dl = get_bert_data_loader_for_predict(data_path + fname, learner)

    learner.load_model(best_model_path)

    preds = learner.predict(dl)

    tokens, y_true, y_pred, set_labels = bert_preds_to_ys(dl, preds)
    clf_report = flat_classification_report(y_true, y_pred, set_labels, digits=3)

    # clf_report = get_bert_span_report(dl, preds)
    print(clf_report)

    write_true_and_pred_to_conll(tokens=tokens, y_true=y_true, y_pred=y_pred, conll_fpath=result_conll_path)


# pred(model=model,
#      result_conll_path=result_conll_path,
#      best_model_path=model_dir + "conll-2003/bilstm_attn_cased.cpt")

train(model=model)
