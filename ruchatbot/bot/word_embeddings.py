# -*- coding: utf-8 -*-

import os
import gensim
import logging
import numpy as np
import math
import scipy.spatial.distance

from ruchatbot.bot.wordchar2vector_model import Wordchar2VectorModel
from ruchatbot.bot.string_constants import PAD_WORD


class WordEmbeddings(object):
    """
    Загрузка и работа с векторными моделями слов.
    """

    def __init__(self):
        self.wc2v = None
        self.wc2v_dims = None
        self.w2v = dict()
        self.w2v_dims = dict()
        self.wordchar2vector_model = None
        self.logger = logging.getLogger('WordEmbeddings')

    def load_models(self, models_folder, wc2v_dir):
        """
        Загружаются нейросетевые модели, позволяющие сгенерировать
        вектор нового слова. Для самых частотных слов готовые вектора
        рассчитаны заранее и сохранены в файле, поэтому они будут
        обработаны объектом self.wc2v.
        """
        if wc2v_dir is not None:
            self.wordchar2vector_model = Wordchar2VectorModel()
            self.wordchar2vector_model.load(models_folder)

    def load_wc2v_model(self, wc2v_path):
        self.logger.info(u'Loading wordchar2vector from "%s"', wc2v_path)
        if wc2v_path.endswith('.kv'):
            self.wc2v = gensim.models.KeyedVectors.load(wc2v_path, mmap='r')
        else:
            self.wc2v = gensim.models.KeyedVectors.load_word2vec_format(wc2v_path, binary=False)
        self.wc2v_dims = len(self.wc2v.vectors[0])

    def load_w2v_model(self, w2v_path):
        w2v_filename = os.path.basename(w2v_path)
        if w2v_filename not in self.w2v:
            self.logger.info(u'Loading word2vector from "%s"', w2v_path)
            if w2v_path.endswith('.kv'):
                w2v = gensim.models.KeyedVectors.load(w2v_path, mmap='r')
            else:
                w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=not w2v_path.endswith('.txt'))

            # При обучении и при предсказании пути к w2v данным могут отличаться, так как
            # тренеры и сами боты работают на разных машинах. Поэтому селектируем по имени файла, без пути.
            self.w2v[w2v_filename] = w2v
            self.w2v_dims[w2v_filename] = len(w2v.vectors[0])

    def vectorize_words(self, w2v_filename, words, X_batch, irow):
        w2v = self.w2v[w2v_filename]
        w2v_dims = self.w2v_dims[w2v_filename]
        for iword, word in enumerate(words[:X_batch.shape[1]]):
            if word != PAD_WORD:
                if word in w2v:
                    X_batch[irow, iword, :w2v_dims] = w2v[word]
                if word in self.wc2v:
                    X_batch[irow, iword, w2v_dims:] = self.wc2v[word]
                else:
                    X_batch[irow, iword, w2v_dims:] = self.wordchar2vector_model.build_vector(word)

    def vectorize_word1(self, w2v_filename, word):
        w2v = self.w2v[w2v_filename]
        w2v_dims = self.w2v_dims[w2v_filename]
        v = np.zeros((self.wc2v_dims+w2v_dims), dtype=np.float32)
        if word in w2v:
            v[:w2v_dims] = w2v[word]
        if word in self.wc2v:
            v[w2v_dims:] = self.wc2v[word]
        else:
            v[w2v_dims:] = self.wordchar2vector_model.build_vector(word)

        return v

    def word_similarity(self, word1, word2):
        w2v = list(self.w2v.values())[0]
        if word1 in w2v and word2 in w2v:
            return 1.0 - abs(scipy.spatial.distance.cosine(w2v[word1], w2v[word2]))
        else:
            return 0.0