import pandas as pd
import warnings
import os
import sys
import torch
from modules import BertNerData as NerData
from modules import NerLearner
from modules.models.bert_models import BertBiLSTMAttnNMT

from modules.data.bert_data import get_bert_data_loader_for_predict
from modules.train.train import validate_step
from modules.utils.plot_metrics import get_bert_span_report

sys.path.append("../")

warnings.filterwarnings("ignore")

data_path = "/media/liah/DATA/ner_data_other/norne/"

train_path = data_path + "train.txt"
dev_path = data_path + "valid.txt"
test_path = data_path + "test.txt"

import codecs


def read_data(input_file):
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
            word = line.strip().split('\t')[1]
            label = line.strip().split('\t')[-1]
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

data = NerData.create(train_path, valid_path, vocab_file)
sup_labels = ['B_ORG', 'B_MISC', 'B_PER', 'I_PER', 'B_LOC', 'I_LOC', 'I_ORG', 'I_MISC']

model = BertBiLSTMAttnNMT.create(len(data.label2idx), bert_config_file, init_checkpoint_pt,
                                 enc_hidden_dim=128, dec_hidden_dim=128, dec_embedding_dim=16)

num_epochs = 30
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
