from modules.data.bert_data import get_bert_data_loader_for_predict
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

dl = get_bert_data_loader_for_predict(data_path + "valid.csv", learner)

model = BertBiLSTMAttnNMT.create(len(data.label2idx), bert_config_file, init_checkpoint_pt,
                                 enc_hidden_dim=128, dec_hidden_dim=128, dec_embedding_dim=16)

learner = NerLearner(model, data,
                     best_model_path=model_dir + "conll-2003/bilstm_attn_cased.cpt",
                     lr=0.01, clip=1.0,
                     sup_labels=[l for l in data.id2label if l not in ['<pad>', '[CLS]', 'X', 'B_O', 'I_']],
          t_total = num_epochs * len(data.train_dl))
learner.load_model(best_model_path)

preds = learner.predict(dl)
