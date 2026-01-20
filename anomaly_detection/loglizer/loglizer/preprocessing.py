"""
The interface for data preprocessing.

Authors:
    LogPAI Team

"""


import pandas as pd
import os
import numpy as np
import re
from collections import Counter
from scipy.special import expit
from itertools import compress
import warnings
from pandas.errors import PerformanceWarning
warnings.filterwarnings('ignore', category=PerformanceWarning)


class FeatureExtractor(object):

    def __init__(self):
        self.idf_vec = None
        self.mean_vec = None
        self.events = None
        self.term_weighting = None
        self.normalization = None
        self.oov = None

    def fit_transform(self, X_seq, term_weighting=None, normalization=None, oov=False, min_count=1):
        """ Fit and transform the data matrix

        Arguments
        ---------
            X_seq: ndarray, log sequences matrix
            term_weighting: None or `tf-idf`
            normalization: None or `zero-mean`
            oov: bool, whether to use OOV event
            min_count: int, the minimal occurrence of events (default 0), only valid when oov=True.

        Returns
        -------
            X_new: The transformed data matrix
        """
        print('====== Transformed train data summary ======')
        self.term_weighting = term_weighting
        self.normalization = normalization
        self.oov = oov
        ## train의 각 시퀀스별 event count 저장
        X_counts = []
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)
        
        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)
        # self.events에는 train 시퀀스 event들만 저장
        # X에는 train event count저장
        self.events = X_df.columns
        X = X_df.values
        if self.oov:
            oov_vec = np.zeros(X.shape[0])
            if min_count > 1:
                # 각 X의 event마다 해당 event가 min_count 이상의 시퀀스에 나타났으면 True로 저장 -> 1차원 배열[event수, 1]
                idx = np.sum(X > 0, axis=0) >= min_count
                # X에서 min_count 미만인 열이 시퀀스 별로 몇 개 있는지 oov_vec로 저장 -> 1차원 배열[시퀀스 개수, 1]
                oov_vec = np.sum(X[:, ~idx] > 0, axis=1)
                # X에서 min_count 이상의 시퀀스에 나타난 event에 대해서만 저장
                X = X[:, idx]
                # self.event에는 min_count 이상의 시퀀스에 나타난 event들로 저장
                self.events = np.array(X_df.columns)[idx].tolist()
            # X와 oov를 옆으로 붙인 것을 X에 저장
            X = np.hstack([X, oov_vec.reshape(X.shape[0], 1)])

        
        num_instance, num_event = X.shape
        if self.term_weighting == 'tf-idf':
            df_vec = np.sum(X > 0, axis=0)
            self.idf_vec = np.log(num_instance / (df_vec + 1e-8))
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1)) 
            X = idf_matrix
        if self.normalization == 'zero-mean':
            mean_vec = X.mean(axis=0)
            self.mean_vec = mean_vec.reshape(1, num_event)
            X = X - np.tile(self.mean_vec, (num_instance, 1))
        elif self.normalization == 'sigmoid':
            X[X != 0] = expit(X[X != 0])
        X_new = X
        
        print('Train data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1])) 
        # X와 oov 합친거 return
        return X_new

    def transform(self, X_seq):
        """ Transform the data matrix with trained parameters

        Arguments
        ---------
            X: log sequences matrix
            term_weighting: None or `tf-idf`

        Returns
        -------
            X_new: The transformed data matrix
        """
        print('====== Transformed test data summary ======')
        X_counts = []
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)
        X_df = pd.DataFrame(X_counts)
        # X_df에는 test의 event count 저장
        X_df = X_df.fillna(0)
    
        empty_events = set(self.events) - set(X_df.columns)
        ## test에는 없고, train에만 있는 event를 test에도 컬럼 넣어주기
        for event in empty_events:
            X_df[event] = [0] * len(X_df)
        # X_df = pd.concat((pd.DataFrame([0] * len(X_df), columns=[event]) for event in empty_events), axis=1)   # PerformanceWarning: DataFrame is highly fragmented 때문에 넣어줘야함


        ## test에서 train에서 학습한 event들만의 value를 추출
        X = X_df[self.events].values
       
        ## oov가 False일시, X는 train에서만 학습한 event들, oov가 True일시, X는 oov와 합침(oov feature의 유무가 차이)
        if self.oov:
            ## 각 시퀀스의 test에만 있는 new event의 발생 횟수를 oov_vec에 저장->shape: [시퀀스 갯수, 1]
            oov_vec = np.sum(X_df[X_df.columns.difference(self.events)].values > 0, axis=1)
            ## train의 event만 있는 test행렬에서 new event들의 count vector를 추가함.
            X = np.hstack([X, oov_vec.reshape(X.shape[0], 1)])
        
        
        num_instance, num_event = X.shape
        if self.term_weighting == 'tf-idf':
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1)) 
            X = idf_matrix
        if self.normalization == 'zero-mean':
            X = X - np.tile(self.mean_vec, (num_instance, 1))
        elif self.normalization == 'sigmoid':
            X[X != 0] = expit(X[X != 0])
        X_new = X

        print('Test data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1])) 

        return X_new
