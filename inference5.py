import json
import os
import warnings
import argparse
from transformers import AutoTokenizer
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
from fastNLP import cache_results, prepare_torch_dataloader
# from fastNLP import print
from fastNLP import Trainer, Evaluator
from fastNLP import TorchGradClipCallback, MoreEvaluateCallback
from fastNLP import FitlogCallback
from fastNLP import SortedSampler, BucketedBatchSampler
from fastNLP import TorchWarmupCallback
import fitlog
# fitlog.debug()

from model.model import CNNNer
from model.metrics import NERMetric
# 수정 ner_pipe -> test_ner_pipe
from data.ner_pipe import SpanNerPipe
from data.padder import Torch3DMatrixPadder
from model.metrics_utils import _compute_f_rec_pre, decode
from collections import defaultdict
from collections import deque
import hashlib
import re
import regex
# from mdl_cost import calculate_template_mdl_cost
# from mdl_cost import select_best_templates
from Test.preprocess_test import *
from mdl_cost2 import *
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('-b', '--batch_size', default=48, type=int)
parser.add_argument('-n', '--n_epochs', default=50, type=int)
parser.add_argument('--warmup', default=0.1, type=float)
parser.add_argument('-d', '--dataset_name', default='genia', type=str)
parser.add_argument('--model_name', default=None, type=str)
parser.add_argument('--cnn_depth', default=3, type=int)
parser.add_argument('--cnn_dim', default=200, type=int)
parser.add_argument('--logit_drop', default=0, type=float)
parser.add_argument('--biaffine_size', default=200, type=int)
parser.add_argument('--n_head', default=5, type=int)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--accumulation_steps', default=1, type=int)

args = parser.parse_args()
dataset_name = args.dataset_name
if args.model_name is None:
    if 'genia' in args.dataset_name:
        args.model_name = 'dmis-lab/biobert-v1.1'
    elif args.dataset_name in ('ace2004', 'ace2005'):
        args.model_name = 'roberta-base'
    else:
        args.model_name = 'roberta-base'

model_name = args.model_name
n_head = args.n_head

######hyper
non_ptm_lr_ratio = 100
schedule = 'linear'
weight_decay = 1e-2
size_embed_dim = 25
ent_thres = 0.5
kernel_size = 3
######hyper

allow_nested = True
fitlog.set_log_dir('logs/')
seed = fitlog.set_rng_seed(rng_seed=args.seed)
os.environ['FASTNLP_GLOBAL_SEED'] = str(seed)
fitlog.add_hyper(args)
fitlog.add_hyper_in_file(__file__)


## 정규표현식 16진수, 숫자 정의
BASE10NUM = r"(?<![0-9.+-])(?>[+-]?(?:(?:[0-9]+(?:\.[0-9]+)?)|(?:\.[0-9]+)))"
NUMBER = r"(?:%{BASE10NUM})"
BASE16NUM = r"(?<![0-9A-Fa-f])(?:[+-]?(?:0x)?(?:[0-9A-Fa-f]+))"
BASE16FLOAT = r"\b(?<![0-9A-Fa-f.])(?:[+-]?(?:0x)?(?:(?:[0-9A-Fa-f]+(?:\.[0-9A-Fa-f]*)?)|(?:\.[0-9A-Fa-f]+)))\b"

# hostport 정규표현식 정의 
#Matched: 192.168.1.1:8080
#Matched: example.com:443
HOSTPORT = f"(?:(?:(?:[0-9A-Za-z][0-9A-Za-z-]{0,62})(?:\.(?:[0-9A-Za-z][0-9A-Za-z-]{0,62}))*(\.?|\b))|(?:(?<![0-9])(?:(?:25[0-5]|2[0-4][0-9]|[0-1]?[0-9]{1,2})\.){3}(?:25[0-5]|2[0-4][0-9]|[0-1]?[0-9]{1,2})(?![0-9]))):\b(?:[1-9][0-9]*)\b"

# 정규표현식 정의
USERNAME = r"[a-zA-Z0-9._-]+"
USER = r"{USERNAME}".format(USERNAME=USERNAME)
UUID = r"[A-Fa-f0-9]{8}-(?:[A-Fa-f0-9]{4}-){3}[A-Fa-f0-9]{12}"
HOSTNAME = r"\b(?:[0-9A-Za-z][0-9A-Za-z-]{0,62})(?:\.(?:[0-9A-Za-z][0-9A-Za-z-]{0,62}))*(\.?|\b)"
IPV4 = r"(?<![0-9])(?:(?:25[0-5]|2[0-4][0-9]|[0-1]?[0-9]{1,2})[.](?:25[0-5]|2[0-4][0-9]|[0-1]?[0-9]{1,2})[.](?:25[0-5]|2[0-4][0-9]|[0-1]?[0-9]{1,2})[.](?:25[0-5]|2[0-4][0-9]|[0-1]?[0-9]{1,2}))(?![0-9])" 
IPV6 = r"((([0-9A-Fa-f]{1,4}:){7}([0-9A-Fa-f]{1,4}|:))|(([0-9A-Fa-f]{1,4}:){6}(:[0-9A-Fa-f]{1,4}|((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})|:))|(([0-9A-Fa-f]{1,4}:){5}(((:[0-9A-Fa-f]{1,4}){1,2})|:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})|:))|(([0-9A-Fa-f]{1,4}:){4}(((:[0-9A-Fa-f]{1,4}){1,3})|((:[0-9A-Fa-f]{1,4})?:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){3}(((:[0-9A-Fa-f]{1,4}){1,4})|((:[0-9A-Fa-f]{1,4}){0,2}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){2}(((:[0-9A-Fa-f]{1,4}){1,5})|((:[0-9A-Fa-f]{1,4}){0,3}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){1}(((:[0-9A-Fa-f]{1,4}){1,6})|((:[0-9A-Fa-f]{1,4}){0,4}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(:(((:[0-9A-Fa-f]{1,4}){1,7})|((:[0-9A-Fa-f]{1,4}){0,5}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:)))(%.+)?"
IP = r"(?:{IPV6}|{IPV4})".format(IPV4 = IPV4, IPV6 = IPV6)
POSINT = r"\b(?:[1-9][0-9]*)\b"
IPPORT = r"{IP}:{PORT}".format(IP=IP, PORT=POSINT)
IPORHOST = r"(?:{HOSTNAME}|{IP})".format(HOSTNAME=HOSTNAME, IP=IP)

# paths
UNIXPATH = r"(?>/(?>[\w_%!$@.-]+|\\.)*)+"
WINPATH = r"(?>[A-Za-z]+:|\\)(?:\\[^\\?*]*)+"
BGLPATH = r"^(/[a-zA-Z0-9\-_\.]+)+(/[a-zA-Z0-9\-_\.]+)*$"
PATH = r"(?:{UNIXPATH}|{WINPATH}|{BGLPATH})".format(UNIXPATH=UNIXPATH, WINPATH=WINPATH, BGLPATH=BGLPATH)
PATHPORT = r"{PATH}:{PORT}".format(PATH=PATH, PORT=POSINT)
URIPATH = r"(?:/[A-Za-z0-9$.+!*'(){},~:;=@#%_\-]*)+"
URIPARAM = r"\?[A-Za-z0-9$.+!*'|(){},~@#%&/=:;_?\-\[\]]*"
URIPATHPARAM =r"{URIPATH}(?:{URIPARAM})?".format(URIPATH=URIPATH, URIPARAM=URIPARAM)
URIHOST = r"{IPORHOST}(?::{POSINT})?".format(IPORHOST=IPORHOST, POSINT=POSINT)
URIPROTO = r"[A-Za-z]+(\+[A-Za-z+]+)?"
URI = r"{URIPROTO}://(?:{USER}(?::[^@]*)?@)?(?:{URIHOST})?(?:{URIPATHPARAM})?".format(URIPROTO=URIPROTO, USER=USER, URIHOST=URIHOST, URIPATHPARAM=URIPATHPARAM)


log_file = "/home/eunwoo/logdeep/data/sampling_example/bgl/bgl2_100k"
headers, logformat_regex = generate_logformat_regex(benchmark["BGL"]["log_format"])
df_log = log_to_dataframe(log_file, logformat_regex, headers, benchmark["BGL"]["log_format"])

logs =  list(df_log["Content"])
# logs = ["problem state (0=sup,1=usr).......0"]
print(len(logs))

template_log_match_dic = defaultdict(list)
log_template_match_dic = defaultdict(list)
wildcard_match_length = defaultdict(list)
original_logs = []


def add_space_before_commas(text):
# 쉼표 앞에 문자가 붙어있으면 공백을 추가
    return re.sub(r'(?<=[^\s]),', ' ,', text)

def add_space_before_comma(text):
    # 쉼표 앞에 공백이 없으면 공백 추가
    return re.sub(r'(\S),', r'\1 ,', text)

# forward 이후 (s, e, score)와 토큰화된 문장이 필요
def tokenizing(log):
    seps = [':', '(', ')', ';', '=', ' ', '[', ']', '{', '}', ',', '$', '<', '>', '"', "'", "@", "|"]
    tokens = []
    token = ""
    length = len(log)
    # 토크나이징 기준 1: 특수문자는 다 따로
    # 토크나이징 기준 2: 앞의 토큰이 모두 숫자인데 뒤에 숫자 아닌게 나오면 토큰
    token_start = 0

    for i in range(length):
        ch = log[i]
        # ch 가 seperator면 앞의 토큰을 저장하고, ch는 따로 또 저장해야 한다
        if ch in seps:
            # seperator 중 따옴표는 기존 토큰들에 붙일 것
            # if ch == '"':
                # ddaom_count += 1
                # 끝 따옴표
                # if ddaom_count % 2 == 0:
                    # token += ch
                    # tokens.append(token)
                    # token = ""
                    # token_start = i + 1
                    # continue
                # 시작 따옴표
                # else:
                    # if len(token) != 0:
                        # tokens.append(token)
                    # token = ch
                    # token_start = i
                    # continue

            if len(token) != 0:
                tokens.append(token)
                token_start = i
            if ch != ' ':
                tokens.append(ch)
            token = ""
            token_start = i + 1
            continue
        # # ch가 seperator가 아니고 앞에 토큰이 있으면
        # if len(token) != 0:
        #     # 숫자인지 아닌지 체크할 필요가 있음
        #     if token.isnumeric() and not ch.isdigit() and ch != '.':
        #         tokens.append(token)
        #         token = ""
        if ch != ' ':
            token += ch
        # 문장의 마지막에 도달했고 저장해야 할 token이 있다면
        if i == length - 1 and len(token) != 0:
            tokens.append(token)
    return tokens
             
# 메모리 이슈를 막기 위해 arr은 전역 변수를 사용
def deleting_overlapping():
    global span_list
    span_list.sort(key=lambda x: (x[0], -x[1]))
    s = set()

    for i in range(len(span_list)):
        for j in range(i + 1, len(span_list)):
            # (1, 3) (2, 4) / (1, 3) (3, 4)
            if span_list[j][0] <= span_list[i][1] < span_list[j][1]:
                s.add(span_list[i])
                s.add(span_list[j])

    if len(s) == 0:
        return []

    else:
        brr = []
        arr = list(s)
        # sigmoid가 가장 작은 값의 인덱스를 찾아서 그 값을 삭제
        brr.append(min(span_list, key=lambda x: x[2]))
        arr.remove(brr[-1])
        # 정렬
        arr.sort(key=lambda x: (x[0], -x[1]))
        # while 문
        # 1. overlapping 있는지 여부 확인, 없으면 반복문 종료 및 함수 종료
        # 2. overlapping 있으면 sigmoid가 가장 작은 값 찾아서 삭제
        # 1,2 를 반복
        while True:
            overlapping = False
            for i in range(len(arr)):
                for j in range(i + 1, len(arr)):
                    if arr[j][0] <= arr[i][1] < arr[j][1]:
                        overlapping = True
                        break
                if overlapping: break

            if not overlapping: break
            else:
                brr.append(min(arr, key=lambda x: x[2]))
                arr.remove(brr[-1])
        return brr
                     
# find_hier을 위해 완전 동일한 구간은 작은 sigmoid를 가지는 구간을 없애는 코드
def identical_remove(span_list):
    rng_dic = defaultdict(float)
    output = []
    for s, e, sigmoid in span_list:
        if rng_dic[(s, e)] < sigmoid:
            rng_dic[(s, e)] = sigmoid
    for s, e in rng_dic.keys():
        sigmoid = rng_dic[(s, e)]
        output.append((s, e, sigmoid))
    return output

def deleting_overlapping2():
    global span_list
    span_list = identical_remove(span_list)
    span_list.sort(key=lambda x: (x[0], -x[1]))
    s = set()

    for i in range(len(span_list)):
        for j in range(i + 1, len(span_list)):
            # (1, 3) (2, 4) / (1, 3) (3, 4)
            if span_list[j][0] <= span_list[i][1] < span_list[j][1]:
                s.add(span_list[i])
                s.add(span_list[j])
    # print(s)
    if len(s) == 0:
        return []

    else:
        brr = []
        arr = list(s)
        # sigmoid가 가장 작은 값의 인덱스를 찾아서 그 값을 삭제
        brr.append(min(span_list, key=lambda x: x[2]))
        arr.remove(brr[-1])
        # 정렬
        arr.sort(key=lambda x: (x[0], -x[1]))
        # while 문
        # 1. overlapping 있는지 여부 확인, 없으면 반복문 종료 및 함수 종료
        # 2. overlapping 있으면 sigmoid가 가장 작은 값 찾아서 삭제
        # 1,2 를 반복
        while True:
            overlapping = False
            for i in range(len(arr)):
                for j in range(i + 1, len(arr)):
                    if arr[j][0] <= arr[i][1] < arr[j][1]:
                        overlapping = True
                        break
                if overlapping: break

            if not overlapping: break
            else:
                brr.append(min(arr, key=lambda x: x[2]))
                arr.remove(brr[-1])
        return brr
                          
def find_parents(cur, parent):
    for child in childs[parent]:
        if child[1] >= cur[1]:
            find_parents(cur, child)
            return
    childs[parent].append(cur)


def dfs(cur, depth):
    global max_depth
    global root

    if not childs[cur]:
        if cur == root:
            return
        leafs.append((cur, depth))
        if max_depth < depth:
            max_depth = depth
    else:
        for child in childs[cur]:
            parents[child] = cur
            dfs(child, depth + 1)


def dfs2(cur):
    visited.add(cur)
    for child in childs[cur]:
        dfs2(child)

# json데이터를 만들기 위한 것
def make_test_json_list(logs):

    json_list = []

    for log in logs:
        d = dict()

        tokens = []
        sep_list = [':', '(', ')', ';', '=', ' ', '[', ']', '{', '}', ',', '$', '<', '>', '"', "'", "@", "|"]

        token = ""

        length = len(log)

        # 토크나이징 기준 1: 특수문자는 다 따로
        # 토크나이징 기준 2: 앞의 토큰이 모두 숫자인데 뒤에 숫자 아닌게 나오면 토큰

        for i in range(length):
            ch = log[i]

            # ch 가 seperator면 앞의 토큰을 저장하고, ch는 따로 또 저장해야 한다
            if ch in sep_list:
                if len(token) != 0:
                    tokens.append(token)
                if ch != ' ':
                    tokens.append(ch)
                token = ""
                continue

            # # ch가 seperator가 아니고 앞에 토큰이 있으면
            # if len(token) != 0:
            #     # 숫자인지 아닌지 체크할 필요가 있음
            #     if token.isnumeric() and not ch.isdigit() and ch != '.':
            #         tokens.append(token)
            #         token = ""

            if ch != ' ':
                token += ch

            # 문장의 마지막에 도달했고 저장해야 할 token이 있다면
            if i == length - 1 and len(token) != 0:
                tokens.append(token)

        d['tokens'] = tokens
        d['entity_mentions'] = []
        json_list.append(d)

    return json_list


# 8자리 16진수 확인 (0-9, a-f, A-F로 이루어진 8자리)
def is_8_digit_hex(value):
    return bool(re.match(r'^[0-9a-fA-F]{8}$', value))

# 0x로 시작하고 뒤에 16진수 숫자(0-9, a-f, A-F)가 있는지 확인
def is_hexadecimal(value):
    if isinstance(value, str) and value.lower().startswith('0x'):
        # 정규 표현식으로 0x 이후에 16진수 문자만 있는지 확인
        if bool(re.match(r'^0x[0-9a-fA-F]+$', value)):
            return True
    return False


def find_regex(tokenized_log):
    regex_rng=[] 
    for i in range(len(original_sentence)):
        for j in range(i, len(original_sentence)):
            match_URI = re.fullmatch(URI, log_span[(i, j)])
            if match_URI:
                # print(f"URI: {log_span[(i, j)]}")
                regex_rng.append((i, j, 0.999))
                continue

            # HDFS돌릴때만 주석
            # match_IP = re.fullmatch(IP, log_span[(i, j)])
            # if match_IP:
            #     # print(f"IP: {log_span[(i, j)]}")
            #     regex_rng.append((i, j, 0.999))
            #     if j+2<len(original_sentence) and original_sentence[j+1] == ":" and re.fullmatch(POSINT, original_sentence[j+2]):
            #         # print(f"PORT: {original_sentence[j+2]}")
            #         regex_rng.append((j+2, j+2, 0.999))
            #         # print(f"IPPORT: {log_span[(i, j+2)]}")
            #         regex_rng.append((i, j+2, 0.999))
            #     continue
            
            # match_IPPORT = re.fullmatch(IPPORT, log_span[(i, j)])
            # if match_IPPORT:
            #     print(f"IP: {log_span[(i, j-2)]}  PORT: {log_span[(j, j)]}")
            #     # regex_rng.append((i, j-2, 0.999))
            #     regex_rng.append((j, j, 0.999))
            #     continue


            match_PATH = regex.fullmatch(PATH, log_span[(i, j)])
            if match_PATH:
                # print(f"PATH: {log_span[(i, j)]}")
                regex_rng.append((i, j, 0.999))
                if j+2<len(original_sentence) and original_sentence[j+1] == ":" and re.fullmatch(POSINT, original_sentence[j+2]):
                    # print(f"PORT: {original_sentence[j+2]}")
                    regex_rng.append((j+2, j+2, 0.999))
                    # print(f"PATHPORT: {log_span[(i, j+2)]}")
                    regex_rng.append((i, j+2, 0.999))
                continue
            
            match_PATHPORT = regex.fullmatch(PATHPORT, log_span[(i, j)])
            if match_PATHPORT:
                print(f"PATH: {log_span[(i, j-2)]}  PORT: {log_span[(j, j)]}")
                # regex_rng.append((i, j-2, 0.999))
                regex_rng.append((j, j, 0.999))
                continue

            match_UUID = re.fullmatch(UUID, log_span[(i, j)])
            if match_UUID:
                # print(f"UUID: {log_span[(i, j)]}")
                regex_rng.append((i, j, 0.999))
                continue

            match_NUMBER = regex.fullmatch(NUMBER, log_span[(i, j)])
            if match_NUMBER:
                # print(f"NUMBER: {log_span[(i, j)]}")
                regex_rng.append((i, j, 0.999))
                continue

            match_BASE16NUM = regex.fullmatch(BASE16NUM, log_span[(i, j)])
            if match_BASE16NUM:
                # print(f"BASE16NUM: {log_span[(i, j)]}")
                regex_rng.append((i, j, 0.999))
                continue

            if is_8_digit_hex(log_span[(i, j)]):
                # print(f"8자리 16진수값: {log_span[(i, j)]}")
                regex_rng.append((i, j, 0.999))
                continue

            if is_hexadecimal(log_span[(i, j)]):
                # print(f"0x로 시작하는 16진수값: {log_span[(i, j)]}")
                regex_rng.append((i, j, 0.999))
                continue
            
    return regex_rng

def find_hier(tokenized_log):
    hier_dict = {"}": "{", "]": "[", ")": "("}
    hier_stack = deque([])
    hier_rng = []
    for idx, token in enumerate(tokenized_log):
        if token in hier_dict.values():
            hier_stack.append((token, idx))
        elif token in hier_dict.keys():
            if len(hier_stack):
                left_chr, left_idx = hier_stack.pop()
                hier_rng.append((left_idx, idx, 1))
    return hier_rng


@cache_results('caches/ner_caches.pkl', _refresh=False)
def get_data(dataset_name, model_name):
    # 以下是我们自己的数据
    if dataset_name == 'ace2004':
        paths = 'preprocess/outputs/ace2004'
    elif dataset_name == 'ace2005':
        paths = 'preprocess/outputs/ace2005'
    elif dataset_name == 'genia':
        paths = 'preprocess/outputs/genia'
    
    else:
        paths = 'preprocess/outputs/inference'
    pipe = SpanNerPipe(model_name=model_name)

    dl = pipe.process_from_file(paths)

    return dl, pipe.matrix_segs

def densify(x):
    x = x.todense().astype(np.float32)
    return x

## 로그 입력하면 jsonlines파일 생성
with open("preprocess/outputs/inference/test.jsonlines", 'w') as infer_file:
    input_json_list = make_test_json_list(logs) # json형식으로 변환
    # print(input_json_list)
    for dic in input_json_list:
        json_data = json.dumps(dic)
        infer_file.write(json_data + '\n')

dl, matrix_segs = get_data(dataset_name, model_name) 

dl.apply_field(densify, field_name='matrix', new_field_name='matrix', progress_bar='Densify')
print(dl)
label2idx = getattr(dl, 'ner_vocab') if hasattr(dl, 'ner_vocab') else getattr(dl, 'label2idx')
print(f"{len(label2idx)} labels: {label2idx}, matrix_segs:{matrix_segs}")
dls = {}
for name, ds in dl.iter_datasets():
    ds.set_pad('matrix', pad_fn=Torch3DMatrixPadder(pad_val=ds.collator.input_fields['matrix']['pad_val'],
                                                    num_class=matrix_segs['ent'],
                                                    batch_size=args.batch_size))

    if name == 'train':
        _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=1,
                                       batch_sampler=BucketedBatchSampler(ds, 'input_ids',
                                                                          batch_size=args.batch_size,
                                                                          num_batch_per_bucket=30),
                                       pin_memory=True, shuffle=True)

    if name == 'dev':
        _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=1,
                                      sampler=SortedSampler(ds, 'input_ids'), pin_memory=True, shuffle=False)
    if name == 'test':
        _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=1,
                                      sampler=None, pin_memory=True, shuffle=False)

    dls[name] = _dl


evaluate_dls = {}
if 'dev' in dls:
    evaluate_dls = {'dev': dls.get('dev')}
if 'test' in dls:
    evaluate_dls['test'] = dls['test']
for batch in evaluate_dls['test']:
    input_ids = batch['input_ids']  # 데이터의 'input_ids' 키에 따라 접근
             # 레이블도 필요하다면 이렇게 접근
    
    # input_ids를 사용한 처리
    # print("batch:", batch.keys())

# 전체 모델 로드
model = torch.load("/home/eunwoo/CNN_Nested_NER/model_cache/union.pth")
# model = torch.load("/home/eunwoo/CNN_Nested_NER/model_cache/union.pth")
# # 훈련이 끝난 후 바로 추론 수행
# model.eval()  # 모델을 평가 모드로 전환
# # 추론할 데이터 준비 (예시 데이터)
# # 실제로는 데이터 전처리 및 토크나이저를 통해 input_ids를 준비해야 함
# sample_input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # 예시로 입력 데이터
# sample_word_len = torch.tensor([5])  # 예시로 입력 데이터의 길이
# with torch.no_grad():
#     outputs = model(sample_input_ids, word_len=sample_word_len)  # 모델에 데이터 입력
#     # 여기서 outputs를 사용하여 예측 결과를 처리
#     print(outputs)  # 예측 결과 출력

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
# for batch in evaluate_dls['test']:  #test 데이터셋에서 3개의 batch가 있음
#     for i in range(len(batch["indexes"])):  #각 batch마다 8개(batch_size)의 로그메시지가 있음. 끝은 3개
#         original_sentence = tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)
#         print("original_log: ", original_sentence)  # 출력: "Hello, how are you?"
#         print(model.forward(batch["input_ids"][i], batch["bpe_len"][i], batch["indexes"][i], batch["matrix"][i]))

model.eval()
fitlog.finish()


temp_lst = []
event_id_list = []
log_count=0
for batch in evaluate_dls['test']:
    scores = model.forward(batch["input_ids"].cuda(), batch["bpe_len"].cuda(), batch["indexes"].cuda(), batch["matrix"].cuda())["scores"]  #indexes 값은 각 bert 토크나이징된 토큰들의 우리의 train, dev, test 데이터의 token index로 보여줌
    # word len은 batch의 우리가 각 문장들을 토큰화했을 때의 각 문장들의 토큰 길이를 tensor로 나타낸 것. ex: [2, 3, 4]: batch는 토큰 길이가 2, 3, 4인 문장들로 이루어짐 
    # 스코어를 sigmoid로 변환하고, 대칭적으로 평균 처리
    ent_scores = scores.sigmoid()  # bsz x max_len x max_len x num_class 여기서 max len은 각 batch에서 우리의 토큰화했을 때 각 토큰의 index의 최댓값(bert의 토큰과 다름)
    ent_scores = (ent_scores + ent_scores.transpose(1, 2)) / 2
    span_pred = ent_scores.max(dim=-1)[0]

    # print("bpe_len: ", batch["bpe_len"])
    # print("word_len: ", batch["word_len"])
    # 스팬(엔티티) 예측
    span_ents = decode(span_pred, batch["word_len"], allow_nested=allow_nested, thres=ent_thres)
    
        # ents가 정답 span들 (시작 인덱스, 종료 인덱스, 라벨) -> 다 숫자 우리는 object라고 했지만 0이 나옴. 왜냐? 0번째 라벨이란 의미
        # span_ent가 모델이 예측한 엔터티들 -> 시작, 종료, 라벨
        # ent pred는 예측한 점수의 딕셔너리
    for idx, (ents, span_ent, ent_pred) in enumerate(zip(batch["ent_target"], span_ents, ent_scores.detach().cpu().numpy())):
            pred_ent = set()    
            pred_score = []
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
                if score[ent_type] >= ent_thres:
                    pred_ent.add((s, e, ent_type))  #pred_ent는 하나의 문장에 대해 시작 토큰 인덱스, 끝 토큰 인덱스 ent_type이 들어감. 여기서 인덱스는 우리가 토큰화했을때의 index. bert token index 아님. 
                    pred_score.append((s, e, score[score.argmax()]))

            # decode하는 것보다 tokenizing(log)를 ' '.join()하는 게 나을듯
            # original_sentence = tokenizer.decode(batch["input_ids"][idx], skip_special_tokens=True)
            print("log count: ", log_count)
            original_sentence = ' '.join(tokenizing(logs[log_count]))
            log_count+=1
            import re
            # original_log = add_space_before_comma(original_sentence).strip()
            original_log = original_sentence.strip()
            # print(original_log)
            original_logs.append(original_log)
            # print()
            original_sentence = tokenizing(original_sentence)
            # print(original_sentence)
            ####################grok#######################################
            log_span = defaultdict(str)
            for i in range(len(original_sentence)):
                for j in range(i, len(original_sentence)):
                    if i < j:
                        log_span[(i, j)] = log_span[(i, j-1)] + " " + original_sentence[j]
                    else:
                        log_span[(i, j)] = log_span[(i, j)] + original_sentence[j]


            # 규칙 기반 계층 찾기 [(s, e, 1), ...]
            hier_rng = find_hier(original_sentence)
            # grok 계층 찾기
            regex_rng = find_regex(original_sentence)
            dic = {i: t for i, t in enumerate(original_sentence)}
            # print("\033[95m" + "original log: " + "\033[0m", dic)
            # print()
            # print("scores: ", pred_score)
            # it_dict = dict()
            # for it_idx, it in enumerate():
            #     it_dict[it_idx] = it
            # print()
            # print(it_dict)
            # print(pred_ent)
            # print()
            # for pred_s, pred_e, pred_l in pred_ent:
            #     if pred_e >= len(original_sentence):
            #         continue
            #     for pred_i in range(pred_s, pred_e+1):
            #         print(origina[pred_i], end=" ") 
            #     print()
            # print("------------------------------------")
            # print()

            # input은 start index, end index, sigmoid (이 떄, index는 토큰 기준)

## 여기서 민엽이형의 코드 
            sentence = original_sentence
            span_list = pred_score
            # span_list = []
            max_depth = 0
            # hier_rng와 기존 span_list 합치기
            span_list.extend(hier_rng)
            # regex_rng와 기존 span_list 합치기
            span_list.extend(regex_rng)
            # overlapping 제거
            brr = deleting_overlapping2()

            # 너무 시간을 많이 잡아먹으면?
            # set으로 바꾼뒤 차집합 구하면 O(n)
            for x in brr:
                span_list.remove(x)
            # print("final span: ", span_list)
            # # span_list에서 sigmoid값을 제거
            # span_list = [(s, e) for s, e, sigmoid in span_list]
            # # span_list에서 중복 제거
            # span_list = list(set(span_list))
            # overlapping entity 가 없는 span_list를 이용해 트리 생성
            childs = defaultdict(list)
            # 실제로는 받은 토큰 리스트의 마지막 인덱스를 이용해 생성
            root = (0, len(sentence)-1)

            for span in span_list:
                find_parents(span, root)

            # 만들어진 트리의 결과 확인
            # print('root:', childs[root])

            # for span in span_list:
            #     print(f'{span}:', childs[span])

            # 리프 노드 구하기

            leafs = []
            parents = dict()
            wilds = []

            dfs(root, 0)
            

            # depth 낮은 leaf 노드 먼저
            # leafs의 원소들은 leaf 노드 ((시작 index, 종료 index), depth)
            # (종료 인덱스 + 1) 이 아님
            leafs.sort(key=lambda y: y[1])

            # 이 리스트에 있는 변수들을 전부 *로 바꿀 것
            variables = leafs.copy()
            # leaf들도 안전빵이라면 set에 추가하는게 좋겠으나... 안 해도 될듯?
            # 부모를 찾으면서 중복 방문을 방지하기 위한 건데... leaf 는 자식이 없으니까
            visited = set()

            # max_depth 이용해서 height 별 템플릿 만들기!!!!
            for depth in range(max_depth, 0, -1):
                # 부모 노드로 갈 때, 이미 방문한 부모 노드는 방문하지 않게 하기 위한 방칙

                # 이번에 변수 *로 치환할 변수들의 목록
                new_variables = []
                for v in variables:
                    # v[1] 은 이 변수의 depth
                    # 이 depth가 현재 설정해놓은 depth보다 크면, 그 부모를 찾아야 함
                    if v[1] > depth:
                        # parents 딕셔너리는 key와 value를 (시작 인덱스, 종료 인덱스) 로 가짐
                        parent = parents[v[0]]
                        # 원래는 new_variables도 depth를 계산해야 하나, max_depth부터 하나씩 올라가므로, parent는 depth가 목표 depth 이하라는 보장이 있음
                        # parent를 이미 방문했으면 리스트에 이를 추가할 필요 없음
                        if parent not in visited:
                            # 부모이므로 depth는 당연히 -1됨
                            new_variables.append((parent, v[1] - 1))
                            visited.add(parent)
                    else:
                        new_variables.append(v)

                # new_variables 부분을 이용하여 변수별 길이 계산하기
                # ex. [1, 2, 5]가 변수리스트이면 변수들의 길이는 [1번째 토큰의 길이 + 2번째 토큰의 길이 + 공백의 길이(1), 5번째 토큰의 길이]
                # mdl_cost 계산에 사용할 것 
                variable_lengths = []

                for v in new_variables:
                    v_len = 0
                    start = v[0][0]
                    end = v[0][1]
                    for i in range(start, end + 1):
                        v_len += len(sentence[i])   # 변수 길이들 더하기
                    v_len += (end - start)  # 공백 길이 더하기
                    variable_lengths.append(v_len)



                # new_variables로 템플릿을 만들어야 함
                template_tokens = sentence.copy()
                variables = new_variables
                # 변수의 인덱스들을 모두 * 로 바꾸기
                for v in variables:
                    start = v[0][0]
                    end = v[0][1]
                    for i in range(start, end + 1):
                        template_tokens[i] = '.*'
                # 연속되는 *을 하나의 *로 바꾸기
                template = []
                before = ' '
                for word in template_tokens:
                    # 연속되는 * 은 추가 X
                    if word == '.*' and before == '.*':
                        continue
                    else:
                        template.append(word)
                        before = word
                candidate_template = ' '.join(template).strip()
                # print('template: ', candidate_template)
                template_log_match_dic[candidate_template].append(original_log)
                log_template_match_dic[original_log].append(candidate_template)
                wildcard_match_length[candidate_template].append((original_log, variable_lengths))
            # 변수가 없는 경우
            # 아예 원본 그대로 출력

            if max_depth == 0:
                candidate_template = ' '.join(sentence).strip()
                # print('template: ', candidate_template)
                template_log_match_dic[candidate_template].append(original_log)
                log_template_match_dic[original_log].append(candidate_template)
                wildcard_match_length[candidate_template].append((original_log, [0]))


# 트리 만들 때, MDL cost 계산할 때 등등에 사용해야 하는 자료구조
# 딕셔너리1[템플릿] = 딕셔너리2
# 딕셔너리2[로그] = 변수 길이 합   #### 너무 나이브한 구현이라 나중에 수정 필요할 수 있음 ####
# 위에서 템플릿에 매칭되는 로그의 범위를 조정할 필요가 있어, O(1)로 탐색 및 삭제를 하기 위해 로그 리스트를 딕셔너리로 변경
# template_to_log_set은 딕셔너리에 key로 템플릿을 주면 그 템플릿에 매칭되는 로그의 set을 리턴
template_to_log_variable_dict = dict()
template_to_log_set = dict()
root = '*'

# with open('data.p', 'wb') as f:
#     pickle.dump(wildcard_match_length, f)
for t in wildcard_match_length:
    template_to_log_variable_dict[t] = dict()
    template_to_log_set[t] = set()

    temp1 = template_to_log_variable_dict[t]
    temp2 = template_to_log_set[t]
    for log_list in wildcard_match_length[t]:
        temp_log = log_list[0]
        variable_lengths = log_list[1]

        temp1[temp_log] = sum(variable_lengths)
        temp2.add(temp_log)


# 메모리 사용량을 줄이기 위하여 불필요한 객체 삭제
del wildcard_match_length
gc.collect()

print('전처리 완료')

# 딕셔너리로 템플릿 트리 구현
tree = defaultdict(list)
# root로 사용할 것
# 모든 로그가 포함되는 것 처럼 동작하게 해야함
# 어떤 로그 템플릿이 트리에 들어갔는지 찾는 visited set이 있을 필요는 없음, 딕셔너리 안에 있는지 확인하면 됨
tree['*'] = []

template_list = list(template_to_log_set.keys())
# 템플릿들을 템플릿에 매칭되는 로그의 수의 개수 역순으로 정렬
# 이렇게 하여 general한 것부터 먼저 트리에 들어가게 함
# #### 제대로 짠 거 맞는지 검증 필요 ####
template_list.sort(key=lambda x: len(template_to_log_set[x]), reverse=True)

# 트리에 템플릿 삽입
for t in template_list:
    insert_template(t, tree, template_to_log_set, template_to_log_variable_dict)

print("트리 생성 완료")

# 템플릿 트리의 각 노드의 마킹 정보
# 선택한 template set을 찾는 데에 필요함
node_mark = dict()

# find_best_mdl_set에서 사용
# parent에 매칭되는 로그가 child에 매칭되는지 여부를 알아보기 위해 사용
visited_logs = set()

# 최적의 mdl_cost를 DP로 계산하며, 최적의 템플릿 set에 마킹
# 마킹 정보는 marking 딕셔너리에 담겨있음
best_cost = find_best_mdl_cost(tree, root, node_mark, visited_logs, template_to_log_set, template_to_log_variable_dict)
print("최적의 mdl cost:", best_cost)
print("최적의 mdl cost 계산 완료")



template_result = []

# 위에 선언한 template_list 배열에 최적의 template set 저장
find_best_template_set(root, node_mark, tree, template_result)

# 템플릿 리스트를 specific한 것부터 출력
# #### 이후 Occurence 오름차순 또는 내림차순으로 정렬 ####
template_result.sort(key= lambda x: len(x), reverse=True)
# 이거는 단순 출력문
# 결과물은 template_list 배열
print("최적의 템플릿 세트")
for t in template_result:
    print(t)






## 밑은 주석 ##


     
# ## mdl cost 계산 후, 최적의 template 반환
# # input으로는 key가 템플릿, value가 매칭되는 로그 및 변수들의 길이인 딕셔너리를 이용
# # output으로는 key가 템플릿, value가 mdl_cost인 딕셔너리

# # 혹시 RAM 용량이 부족하면 input으로 key가 템플릿, value가 매칭되는 로그의 변수들의 길이인 딕셔너리를 이용 가능 (즉, 로그는 없어도 됨)
# template_to_cost_dict = calculate_template_mdl_cost(wildcard_match_length)

# # input은 key가 template, value가 mdl_cost 인 딕셔너리, key가 log, value가 template인 딕셔너리
# # output은 key가 로그, value가 매칭되는 best template인 딕셔너리, key가 template, value가 event_id인 딕셔너리
# log_match_bestTemplate = select_best_templates(template_to_cost_dict, log_template_match_dic)

# template_list = list(map(lambda x: log_match_bestTemplate[x], original_logs))  ## log_match_bestTemplate[x].get(x, -1)하면 딕셔너리에 키값이 없어도 error를 반환하지 않고 -1을 반환환

# ############## BGL ##################################
# # structured_log = pd.DataFrame({
# #     "label": df_log["Label"],
# #     "time": df_log["Time"],  
# #     "log": original_logs, 
# #     "event_template": list(map(lambda x: log_match_bestTemplate[x], original_logs))       ## log_match_bestTemplate[x].get(x, -1)하면 딕셔너리에 키값이 없어도 error를 반환하지 않고 -1을 반환환
# # }, index=None)
# # print(set(structured_log["event_template"]))
# ######################################################


# ############## OpenStack ##################################
# # structured_log = pd.DataFrame({
# #     "LineId": list(range(1, len(original_logs)+1)),
# #     "Logrecord": df_log["Logrecord"],
# #     "Date": df_log["Date"],
# #     "Time": df_log["Time"],
# #     "Pid": df_log["Pid"],
# #     "Level": df_log["Level"],
# #     "Component": df_log["Component"],
# #     "ADDR": df_log["ADDR"], 
# #     "Content": original_logs, 
# #     "EventId": list(map(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()[0:8], original_logs)),   
# #     "EventTemplate": template_list      
# # }, index=None)
# ######################################################


# ############## HDFS ##################################
# structured_log = pd.DataFrame({
#     "LineId": list(range(1, len(original_logs)+1)),
#     "Date": df_log["Date"],
#     "Time": df_log["Time"],
#     "Pid": df_log["Pid"],
#     "Level": df_log["Level"],
#     "Component": df_log["Component"],
#     "Content": original_logs, 
#     "EventId": list(map(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()[0:8], original_logs)),   
#     "EventTemplate": template_list      
# }, index=None)
# ######################################################


# # print(set(structured_log["event_template"]))

# print("유니크 템플릿 수: ", len(set(structured_log["EventTemplate"])))
# # ## 템플릿 매칭용 딕셔너리를 json으로 구현
# # with open('structured.json', 'w', encoding='utf-8') as json_file:
# #     json.dump(log_match_bestTemplate, json_file, indent=4, ensure_ascii=False)

# structured_log.to_csv('hdfs_our_structured.csv', index=False)