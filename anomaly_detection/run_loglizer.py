from datetime import datetime, timedelta
import pandas as pd
from loglizer.loglizer.models import *
from loglizer.loglizer import dataloader, preprocessing
import numpy as np



## run할 이상탐지 모델
run_models = ['PCA', 'LogClustering', 'LR']
# anomaly type정의
CASE = "flush_continue" 
# 대상 노드
node_list = ["6"]  
# 파서 디렉토리 선택
parser = "LUNAR"
#시퀀스 길이
time_window_size = 20
result_file = open(f'result_time_predict_{CASE}.log', "w")
train_ratio = 0.5

# 이상탐지 실행
def run_AD(x_tr, y_train, x_te, y_test):
    benchmark_results = []
    for _model in run_models:
        print('Evaluating ' + '\033[95m'+  _model  + '\033[0m' + ' on MultiLog:')
        if _model == 'PCA':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf', 
                                                      normalization='zero-mean', oov=True)
            model = PCA(threshold=5.9)
            model.fit(x_train)
        

        elif _model == 'LogClustering':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf', oov=True)
            model = LogClustering(max_dist=0.3, anomaly_threshold=0.04)
            model.fit(x_train[y_train == 0, :]) # Use only normal samples for training

        elif _model == 'LR':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf', oov=True)
            model = LR()
            model.fit(x_train, y_train)
        
        x_test = feature_extractor.transform(x_te)
        # print('\033[96m' + 'Train accuracy:' + '\033[0m')
        # precision, recall, f1 = model.evaluate(x_train, y_train)
        # benchmark_results.append([_model + '-train', precision, recall, f1])
        print('\033[96m' + 'Test accuracy:' + '\033[0m')
        precision, recall, f1 = model.evaluate(x_test, y_test)
        benchmark_results.append([_model + '-test', precision, recall, f1])
    pd.DataFrame(benchmark_results, columns=['Model', 'Precision', 'Recall', 'F1']) \
      .to_csv(f'result_benchmark/{CASE}{train_ratio}{parser}{time_window_size}.csv', index=False)


# #################################### thres 디버깅용0.9 0.2 ###########################################################
# def run_AD(x_tr, y_train, x_te, y_test):
#     benchmark_results = []
#     for _model in run_models:
#         print('Evaluating ' + '\033[95m'+  _model  + '\033[0m' + ' on MultiLog:')
#         if _model == 'LogClustering':
#             score = []
#             for dist in [i * 0.1 for i in range(1, 10)]:
#                 for thres in [j * 0.001 for j in range(1, 50)]:
#                     feature_extractor = preprocessing.FeatureExtractor()
#                     x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf', oov=True)
#                     model = LogClustering(max_dist=dist, anomaly_threshold=thres)
#                     model.fit(x_train[y_train == 0, :]) # Use only normal samples for training



#                     x_test = feature_extractor.transform(x_te)
#                     # print('\033[96m' + 'Train accuracy:' + '\033[0m')
#                     # precision, recall, f1 = model.evaluate(x_train, y_train)
#                     # benchmark_results.append([_model + '-train', precision, recall, f1])
#                     print('\033[96m' + 'Test accuracy:' + '\033[0m')
#                     precision, recall, f1 = model.evaluate(x_test, y_test)
#                     score.append((precision, recall, f1, thres, dist))
#                 print(sorted(score, key = lambda x: x[2], reverse=True))
# #################################### thres 디버깅용 ###########################################################

# 이상주입 구간 리스트 생성
def anomaly_time(CASE):
    inject_file = CASE + "/inject.log"
    # 이상여부 리스트
    inject_anomaly_types = []
    # inject 시간들이 저장
    start_time = []
    end_time = []
    for line in open(inject_file):
        if "start inject" in line:
            anomaly = line.split("start inject ")[1].strip()            # anomaly type 할당
            if "none" not in anomaly:           # anomaly인 경우 1추가
                inject_anomaly_types.append(1)
            else:
                inject_anomaly_types.append(0)
        else:
            if "Recover" not in line and "inject" not in line and ("169" in line or "170" in line) and "." in line: 
                # start 또는 end inject 시간
                time_float = float(line) 
                # start_time과 end_time에는 각 inject start 시간 또는 inject end 시간 주입
                if len(end_time) < len(start_time):
                    end_time.append(time_float)
                else:
                    start_time.append(time_float)
    return inject_anomaly_types, start_time, end_time

# 각 시퀀스 (시작시각, 끝 시각) 생성
def make_seq_time(log_start_time, log_end_time):
    # 로그 파일의 첫 시각을 기준으로 시퀀스 연속적으로 추가
    time_list = []
    curr_start_time = log_start_time
    # 로그 끝 줄까지 반복
    while log_end_time > curr_start_time:
        # window가 time_window_size인 시퀀스의 첫 시각과 끝 시각을 연속적으로 추가
        time_list.append([curr_start_time, curr_start_time + time_window_size])
        curr_start_time = curr_start_time + time_window_size
    print("Sequence Count: ", len(time_list))
    return time_list

# 각 시퀀스 별 event 리스트 생성
def make_event_seq(CASE, node_list):
    # 각 노드마다 시퀀스들의 모음
    node_event_seqs = {}
    for node in node_list:
        # struct_log = "/raid1/eunwoo/MultiLog/sample/flush_continue/GroundTruth/ground_truth_transformed_flush_label6.log_structured.csv"
        struct_log = CASE + "/" + parser + "/transformed_flush_label6.log_structured.csv"
        struct_df = pd.read_csv(struct_log)
        event_seqs = []
        time_seqs = []
        len_df = len(struct_df)
        log_cnt = 0
        struct_df["DateTime"] = struct_df["Date"] + " " + struct_df["Time"]
        struct_df["timestamp"] = struct_df["DateTime"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S,%f").timestamp())   # 유닉스 time으로 변환
        for start, end in time_list:
            mask_idx = (start <= struct_df["timestamp"]) & (struct_df["timestamp"] <= end)
            event_seqs.append(struct_df.loc[mask_idx, 'EventId'].tolist())
            time_seqs.append(struct_df.loc[mask_idx, 'Time'].tolist())
        node_event_seqs[node] = event_seqs

    return node_event_seqs, time_seqs

# 오류주입과 끝이 시퀀스 내에 있는 경우 시퀀스 window size: 50sec 이하 10sec이상
# 오류 주입 기간이 시퀀스와 걸친 경우 시퀀스 window size: 10sec이하
# 각 시퀀스별 이상 여부 리스트 생성
def anomaly_label_list(time_list, inject_anomaly_types, start_time, end_time):
    truth_label_list = [0]*len(time_list)
    overlapping_list = []
    # 매 시퀀스마다 하나라도 inject구간이 포함되었는지 확인
    for i in range(len(time_list)):
        truth_label_list[i] = 0
        for j in range(len(start_time)):
            # 시퀀스와 inject구간이 걸치면 바로 그 시퀀스는 이상으로 간주. 그래서 window는 짧은게 나음
            if not (start_time[j] > time_list[i][1] or end_time[j] < time_list[i][0]):
            # 시퀀스 내에 inject구간이 포함되어야만 그 시퀀스는 이상으로 간주. window size는 50이 적당=> window 길면 시퀀스가 이상일 확률이 높아짐
            # if time_list[i][0] < start_time[j] and end_time[j] < time_list[i][1]:
                if inject_anomaly_types[j] == 1:
                    truth_label_list[i] = 1
                    break
    return truth_label_list




if __name__ == "__main__":
    log_start_time = 999999999999999
    log_end_time = 0
    for node in node_list:
        in_file = CASE + "/label" + node + '.log'

        for line in open(in_file, "r"):
            # 로그 파일에서 time 추출
            time_str = line.split("[")[0][:-1].replace("- ", "")
            # %Y-%m-%d %H:%M:%S,%f형식으로 뽑고 유닉스 계열 time으로 바꾸기
            timestamp = (datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S,%f")).timestamp()
            # 각 로그라인별로 start에는 추출한 시간 중 더 작은값, end에는 더 큰 값을 할당
            log_start_time = min(timestamp, log_start_time)
            log_end_time = max(timestamp, log_end_time)

    # 이상주입 구간 리스트 생성
    inject_anomaly_types, start_time, end_time = anomaly_time(CASE)
    # 각 시퀀스 window (시작시각, 끝 시각) 생성
    time_list = make_seq_time(log_start_time, log_end_time)
    # 각 시퀀스 별 event 리스트 생성
    node_event_seqs, time_seqs = make_event_seq(CASE, node_list)
    # 각 시퀀스별 이상 여부 리스트 생성
    truth_label_list = anomaly_label_list(time_list, inject_anomaly_types, start_time, end_time)

    ## 완전 포함이 되어야 이상으로 간주, 포함이 완전히 배제되어야 정상
    for time_index in range(len(time_list)):
        result_file.write(
            # str(truth_label_list[time_index]) + ":" + str(list(map(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S,%f"), time_list[time_index]))) + ":" + str(time_seqs[time_index]) + "\n")
            str(time_index) + ":" + str(time_list[time_index]) + ":" + str(truth_label_list[time_index]) + "\n")
        

    for node in node_list:
        event_seqs = node_event_seqs[node] 
        train_len = int(len(event_seqs)*train_ratio)
        x_train = np.array(event_seqs[:train_len])
        y_train = np.array(truth_label_list[:train_len])
        print(truth_label_list)
        x_test = np.array(event_seqs[train_len:])
        y_test = np.array(truth_label_list[train_len:])
        run_AD(x_train, y_train, x_test, y_test)