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

# data_path = "/media/liah/DATA/ner_data_other/norne/"
# data_path = "/media/liah/DATA/ner_data_acme/datadump_article_no/conll_format_combined/"
data_path = "/media/liah/DATA/ner_data_acme/datadump_article_no/conll/"
train_path = data_path + "train.conll"
dev_path = data_path + "valid.conll"
test_path = data_path + "test.conll"


def read_data(input_file, tkn_field_idx=0, label_field_idx=-1):
    """Reads a BIO data."""
    with codecs.open(input_file, "r", encoding="utf-8") as f:
        lines = []
        words = []
        labels = []
        for line in f:
            contends = line.strip()
            if len(contends) == 0:
                # empty line, means a sentence is finished
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append([l, w])
                words = []
                labels = []
                continue
            word = line.rstrip().split('\t')[tkn_field_idx]
            label = line.rstrip().split('\t')[label_field_idx]
            if contends.startswith("-DOCSTART-"):
                words.append('')
                continue

            if len(contends) == 0 and not len(words):
                words.append("")

            words.append(word)
            labels.append(label.replace("-", "_"))
        return lines


train_f = read_data(train_path)
dev_f = read_data(dev_path)
test_f = read_data(test_path)

train_df = pd.DataFrame(train_f, columns=["0", "1"])
train_df.to_csv(data_path + "train.csv", index=False)

valid_df = pd.DataFrame(dev_f, columns=["0", "1"])

valid_df.to_csv(data_path + "valid.csv", index=False)

test_df = pd.DataFrame(test_f, columns=["0", "1"])
test_df.to_csv(data_path + "test.csv", index=False)

train_path = data_path + "train.csv"
valid_path = data_path + "valid.csv"
test_path = data_path + "test.csv"

model_dir = "/media/liah/DATA/pretrained_models/bert/multi_cased_L-12_H-768_A-12/"
init_checkpoint_pt = os.path.join(model_dir, "pytorch_model.bin")
bert_config_file = os.path.join(model_dir, "bert_config.json")
vocab_file = os.path.join(model_dir, "vocab.txt")

torch.cuda.is_available(), torch.cuda.current_device()

# data = NerData.create(train_path, valid_path, vocab_file)
label2idx = {'<pad>': 0, '[CLS]': 1, 'B_O': 2, 'X': 3, 'B_PROD': 4, 'I_PROD': 5, 'B_LOC': 6, 'B_PER': 7, 'I_PER': 8,
             'B_GPE': 9, 'B_ORG': 10, 'B_DRV': 11, 'I_ORG': 12, 'I_DRV': 13, 'B_MISC': 14, 'I_GPE': 15, 'I_LOC': 16,
             'B_EVT': 17, 'I_EVT': 18, 'I_MISC': 19}
cls2idx = None
data = NerData.create(train_path, valid_path, vocab_file, for_train=False,
                      label2idx=label2idx, cls2idx=cls2idx)

sup_labels = ['B_ORG', 'B_MISC', 'B_PER', 'I_PER', 'B_LOC', 'I_LOC', 'I_ORG', 'I_MISC']

model = BertBiLSTMAttnNMT.create(len(data.label2idx), bert_config_file, init_checkpoint_pt,
                                 enc_hidden_dim=128,
                                 dec_hidden_dim=128,
                                 dec_embedding_dim=16)


def train(model, num_epochs=30, ):
    learner = NerLearner(model, data,
                         best_model_path=model_dir + "conll-2003/bilstm_attn_cased.cpt",
                         lr=0.01, clip=1.0, sup_labels=data.id2label[5:],
                         t_total=num_epochs * len(data.train_dl))

    learner.fit(num_epochs, target_metric='prec')

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
                         best_model_path=model_dir + "conll-2003/bilstm_attn_cased.cpt",
                         lr=0.01, clip=1.0, sup_labels=data.id2label[5:])
    dl = get_bert_data_loader_for_predict(data_path + fname, learner)

    learner.load_model(best_model_path)

    preds = learner.predict(dl)

    tokens, y_true, y_pred, set_labels = bert_preds_to_ys(dl, preds)
    clf_report = flat_classification_report(y_true, y_pred, set_labels, digits=3)

    # clf_report = get_bert_span_report(dl, preds)
    print(clf_report)

    write_true_and_pred_to_conll(tokens=tokens, y_true=y_true, y_pred=y_pred, conll_fpath=result_conll_path)


result_conll_path = Path('/media/liah/DATA/log/company_tagging_no/bert_acme_test.conll')
pred(model=model,
     result_conll_path=result_conll_path,
     best_model_path=model_dir + "conll-2003/bilstm_attn_cased.cpt")
