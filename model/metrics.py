import torch
from fastNLP import Metric
import numpy as np
from .metrics_utils import _compute_f_rec_pre, decode

class NERMetric(Metric):
    def __init__(self, matrix_segs, ent_thres, allow_nested=True):
        super(NERMetric, self).__init__()
        self.register_element('tp', 0, aggregate_method='sum')
        self.register_element('pre', 0, aggregate_method='sum')
        self.register_element('rec', 0, aggregate_method='sum')

        assert len(matrix_segs) == 1, "Only support pure entities."
        self.allow_nested = allow_nested
        self.ent_thres = ent_thres
        

    def update(self, ent_target, scores, word_len):
        
        
        # word len은 batch의 우리가 각 문장들을 토큰화했을 때의 각 문장들의 토큰 길이를 tensor로 나타낸 것. ex: [2, 3, 4]: batch는 토큰 길이가 2, 3, 4인 문장들로 이루어짐 
        # 스코어를 sigmoid로 변환하고, 대칭적으로 평균 처리
        ent_scores = scores.sigmoid()  # bsz x max_len x max_len x num_class 여기서 max len은 각 batch에서 우리의 토큰화했을 때 각 토큰의 index의 최댓값(bert의 토큰과 다름)
        ent_scores = (ent_scores + ent_scores.transpose(1, 2)) / 2
        span_pred = ent_scores.max(dim=-1)[0]

        # 스팬(엔티티) 예측
        span_ents = decode(span_pred, word_len, allow_nested=self.allow_nested, thres=self.ent_thres)

        # ents가 정답 span들 (시작 인덱스, 종료 인덱스, 라벨) -> 다 숫자 우리는 object라고 했지만 0이 나옴. 왜냐? 0번째 라벨이란 의미
        # span_ent가 모델이 예측한 엔터티들 -> 시작, 종료, 라벨
        # ent pred는 예측한 점수의 딕셔너리
        for ents, span_ent, ent_pred in zip(ent_target, span_ents, ent_scores.cpu().numpy()):
            pred_ent = set()
            # s는 예측한 엔티티의 시작 인덱스
            # e는 종료 인덱스
            # l은 라벨
            for s, e, l in span_ent:
                # ent_ped[s,e] 를 하면 (s,e) span의 점수가 나옴
                # 이 점수는 각 라벨에 대한 점수 리스트 (sigmoid 결과들)
                # ent_pred는 하나의 문장의 matrix score
                score = ent_pred[s, e]
                
                # 가장 점수가 높은 라벨의 인덱스가 나옴
                ent_type = score.argmax()
                # self.ent_thres 는 예측 thresh
                # 예측 thresh 보다 score[ent_type] 즉, 가장 높은 점수가 threshold보다 높으면 이 라벨로 예측된 것
                # threshold보다 낮으면 어떤 entity에도 속하지 않은 것
                # 그래서 어떤 라벨로 예측이 되면 pred_ent 집합에 포함됨
                if score[ent_type] >= self.ent_thres:
                    pred_ent.add((s, e, ent_type))  #pred_ent는 하나의 문장에 대해 시작 토큰 인덱스, 끝 토큰 인덱스 ent_type이 들어감. 여기서 인덱스는 우리가 토큰화했을때의 index. bert token index 아님. 
            
            # ents의 각 성분들에 대하여 tuple로 바꾸고 집합으로 바꿈
            ents = set(map(tuple, ents))  # 실제 엔티티 (target)
            
            # 교집합을 통해 TP, Precision, Recall 계산

            # tp는 True Positive를 의미, pred_ent와 ents의 교집합의 개수
            # 예를 들어 pred_ent = [(0, 10, 0), (2, 3, 0)], ents = [(0, 10, 0), (2, 4, 0)] 이면 self.tp +=1
            self.tp += len(ents.intersection(pred_ent)) # 각 문장에 대한 tp를 구해서 그것들을 싹 다 더함.
            # 여기서 pre 에 predict ent의 개수 더하는 이유
            # precision = tp / tp + fp
            # 이 때, tp + fp 는 pred_ents의 개수
            # 따라서 TP 와 pred_ent의 개수로 precision 계산할거기 때문에 self.pre += len(pred_ent)
            # 즉, 이 사람들은 마지막에 precision 을 self.tp / self.pre 로 계산하기 위해 이렇게 변수를 둠
            self.pre += len(pred_ent)
            # recall = tp / tp + fn
            # tp + fn 이 ents의 개수, 즉, 실제 정답의 개수
            # 즉, 이 사람들은 마지막에 recall 을 self.tp / self.rec 로 계산
            self.rec += len(ents)
            
            # # 정답 출력
            # print('answer: ', ents)
            # # 예상한 답 출력
            # print('predicted: ', span_ent)
            

    def get_metric(self) -> dict:
        # f1, precision, recall 계산
        f, rec, pre = _compute_f_rec_pre(self.tp, self.rec, self.pre)
        res = {
            'f': f,
            'rec': rec,
            'pre': pre,
        }
        return res