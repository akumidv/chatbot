# -*- coding: utf-8 -*-
"""
14-01-2021 Перенос из экспериментального кода
"""


import os
import json
import logging

import numpy as np
from keras.models import model_from_json
import sentencepiece as spm
import tensorflow as tf

from ruchatbot.bot.relevancy_detector import RelevancyDetector


class NN_RelevancyDetector(RelevancyDetector):
    """ Нейросетевая реализация детектора релевантности предпосылки и вопроса """

    def __init__(self):
        super(NN_RelevancyDetector, self).__init__()
        self.logger = logging.getLogger('NN_RelevancyDetector')

    def load(self, models_folder):
        self.logger.info('Loading NN_RelevancyDetector model files')

        with open(os.path.join(models_folder, 'nn_pq_relevancy.config'), 'r') as f:
            model_config = json.load(f)

        self.max_context_len = model_config['max_context_len']
        self.max_query_len = model_config['max_query_len']
        self.token2index = model_config['token2index']
        self.nb_tokens = model_config['nb_tokens']

        self.arch_filepath = self.get_model_filepath(models_folder, model_config['arch_path'])
        self.weights_filepath = self.get_model_filepath(models_folder, model_config['weights_path'])

        self.bpe_model = spm.SentencePieceProcessor()
        rc = self.bpe_model.Load(self.get_model_filepath(models_folder, model_config['bpe_model_name']+'.model'))
        self.logger.debug('NN_RelevancyDetector.bpe_model loaded with status=%d', rc)

        #self.graph = tf.Graph()
        #self.tf_sess = tf.Session(graph=self.graph)
        #self.tf_sess.__enter__()

        with open(self.arch_filepath, 'r') as f:
            m = model_from_json(f.read())

        m.load_weights(self.weights_filepath)
        self.model = m

        self.graph = tf.compat.v1.get_default_graph() # эксперимент с багом 13-05-2019

        # начало отладки
        #self.model.summary()
        # конец отладки

    def get_most_relevant(self, query_phrase, context_phrases, text_utils, nb_results):
        n = len(context_phrases)
        Xquery = np.zeros((n, self.max_query_len,), dtype=np.int32)
        Xcontext = np.zeros((n, self.max_context_len,), dtype=np.int32)

        query_tx = self.bpe_model.EncodeAsPieces(text_utils.wordize_text(query_phrase).lower())
        for itoken, token in enumerate(query_tx[:self.max_query_len]):
            Xquery[:, itoken] = self.token2index.get(token, 0)

        for icontext, context in enumerate(context_phrases):
            context_str = text_utils.wordize_text(context[0]).lower()
            tx = self.bpe_model.EncodeAsPieces(context_str)
            for itoken, token in enumerate(tx[:self.max_context_len]):
                Xcontext[icontext, itoken] = self.token2index.get(token, 0)

        y = self.model.predict(x={'context_tokens': Xcontext, 'query_tokens': Xquery})
        return y

    def calc_relevancy1(self, premise, question, text_utils):
        raise NotImplemented()
