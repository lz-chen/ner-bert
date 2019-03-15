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
from modules.utils.utils import bert_labels2tokens, tokens2spans, first_choicer
from pathlib import Path
import json
import re
from company_tagging.evaluator_utils import _clean_text, _bio_tags_to_spans
from exabel.nlp.text_processing.feature_extraction.org_name_matcher import OrganizationEntity

def df_from_text(text, learner):
    """"""
    # in get_bert_data_loader_for_predict(), the data is read from csv file to pd dataframe,
    # this function should do the same, but reading from a json object representing one article
    # todo change tokenizer
    text_tokens = _clean_text(text).split(' ')
    text_tokens = text_tokens[:min(200, len(text_tokens))]
    df = pd.DataFrame([[' '.join(["O" for _ in range(len(text_tokens))]), ' '.join(text_tokens)]], columns=["0", "1"])
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
        self.predefined_set_labels = {'ORG', 'PER', 'DRV', 'GPE', 'PROD', 'MISC', 'EVT', 'LOC'}
        self.ignore_labels = ["O"]
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
        dl = df_from_text(text, learner=self.learner)

        preds = self.learner.predict(dl)

        spans_pred = self._bert_preds_to_span(preds)
        # todo: pred_to_spans -> span_to_OrgnizationEntity


    def _bert_preds_to_span(self, preds):
        tokens, labels = bert_labels2tokens(self.dl, preds, fn=first_choicer)
        spans_pred = tokens2spans(tokens, labels)  # list of (span, pred_tag), span can be more than one token
        tokens, labels = bert_labels2tokens(self.dl, [x.labels for x in self.dl.dataset], fn=first_choicer)
        spans_true = tokens2spans(tokens, labels)
        set_labels = set()
        for idx in range(len(spans_pred)):
            while len(spans_pred[idx]) < len(spans_true[idx]):
                spans_pred[idx].append(("", "O"))
            while len(spans_pred[idx]) > len(spans_true[idx]):
                spans_true[idx].append(("O", "O"))
            set_labels.update([y for x, y in spans_true[idx]])
        set_labels = self.predefined_set_labels if len(set_labels) == 0 else set_labels
        set_labels -= set(self.ignore_labels)
        # y_true, y_pred = [[y[1] for y in x] for x in spans_true], [[y[1] for y in x] for x in spans_pred]
        return spans_pred

    def _span_to_org_ent(self, spans):
        # todo get the start_idx and end_idx for each span
        org_ents = []
        for span in spans:
            if span[1] != 'O':
                org_ents.append(OrganizationEntity(text=span[0],))



if __name__ == "__main__":

    bert_pretrained_dir = Path("/media/liah/DATA/pretrained_models/bert/multi_cased_L-12_H-768_A-12/")
    best_weight_path = 'conll-2003/bilstm_attn_cased.cpt'
    bert_pipeline = BertNERPipeline(pretrained_model_dir=bert_pretrained_dir,
                                    best_weight_path=best_weight_path)

    test_fpath = Path('/media/liah/DATA/ner_data_acme/datadump_article_no/acme_all.json')
    texts = []

    with test_fpath.open('r') as f:
        for line in f:
            text = json.loads(line)['content']
            bert_pipeline.extrac_org(text=text)
