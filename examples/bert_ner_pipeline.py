import pandas as pd
import warnings
import os
import sys
import codecs
import torch
from modules import BertNerData as NerData
from modules.models.bert_models import BertBiLSTMAttnNMT
from modules import NerLearner
from modules.data.bert_data import get_bert_data_loader_for_predict, get_data, DataLoaderForPredict
from modules.train.train import validate_step
from modules.utils.plot_metrics import get_bert_span_report, bert_preds_to_ys, bert_preds_to_y, \
    write_true_and_pred_to_conll, flat_classification_report
from pathlib import Path


def df_from_json(text, learner):
    """"""
    # in get_bert_data_loader_for_predict(), the data is read from csv file to pd dataframe,
    # this function should do the same, but reading from a json object representing one article
    # todo: df = read from json
    f, _ = get_data(df, tokenizer=learner.data.tokenizer,
                    label2idx=learner.data.label2idx, cls2idx=learner.data.cls2idx,
                    is_cls=learner.data.is_cls,
                    max_seq_len=learner.data.max_seq_len, is_meta=learner.data.is_meta)
    dl = DataLoaderForPredict(
        f, batch_size=learner.data.batch_size, shuffle=False,
        cuda=True)

    return dl


class BertNERPipeline(object):
    def __init__(self, pretrained_model_dir: Path,
                 torch_model_name: str = "pytorch_model.bin",
                 best_weight_path: str = "",
                 config_file_name: str = "bert_config.json",
                 vocab_file_name: str = "vocab.txt"
                 ):
        self.model_dir = pretrained_model_dir  # "/media/liah/DATA/pretrained_models/bert/multi_cased_L-12_H-768_A-12/"
        self.init_checkpoint_pt = self.model_dir.joinpath(torch_model_name)
        self.bert_config_file = self.model_dir.joinpath(config_file_name)
        self.vocab_file = self.model_dir.joinpath(vocab_file_name)
        self.best_weight_path = best_weight_path
        self.best_model_path = self.model_dir.joinpath(best_weight_path).as_posix()

        self.label2idx = {'<pad>': 0, '[CLS]': 1, 'B_O': 2, 'X': 3, 'B_PROD': 4, 'I_PROD': 5, 'B_LOC': 6, 'B_PER': 7,
                          'I_PER': 8,
                          'B_GPE': 9, 'B_ORG': 10, 'B_DRV': 11, 'I_ORG': 12, 'I_DRV': 13, 'B_MISC': 14, 'I_GPE': 15,
                          'I_LOC': 16,
                          'B_EVT': 17, 'I_EVT': 18, 'I_MISC': 19}
        self.cls2idx = None
        self.data = NerData.create(train_path='',
                                   valid_path='',
                                   vocab_file=self.vocab_file.as_posix(),
                                   for_train=False,
                                   label2idx=self.label2idx,
                                   cls2idx=self.cls2idx)

        self.model = BertBiLSTMAttnNMT.create(len(self.label2idx),
                                              self.bert_config_file.as_posix(),
                                              self.init_checkpoint_pt.as_posix(),
                                              enc_hidden_dim=128,
                                              dec_hidden_dim=128,
                                              dec_embedding_dim=16)

        self.learner = NerLearner(self.model, self.data,
                                  best_model_path=self.best_model_path,
                                  lr=0.01, clip=1.0, sup_labels=self.data.id2label[5:])

        self.learner.load_model(self.best_model_path)

    def extrac_org(self, text):
        dl = get_bert_data_loader_for_predict(text, learner=self.learner)

        preds = self.learner.predict(dl)

        tokens, y_true, y_pred, set_labels = bert_preds_to_ys(dl, preds)

        # todo: pred_to_spans -> span_to_OrgnizationEntity
