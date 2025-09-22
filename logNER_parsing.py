import json
import os
import warnings
import argparse
import math
from transformers import AutoTokenizer
from tqdm import tqdm
import random
from utils.find_template_by_candidates import find_best_template_by_log_and_candidate_templates

# if 'p' in os.environ:
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# os.environ['MKL_THREADING_LAYER'] = 'GNU'
# warnings.filterwarnings('ignore')

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

# 수정 ner_pipe -> test_ner_pipe
# inference_dictionary.py 버전에 비해 v_len 구하는게 변수길이가 아닌 변수의 시작 인덱스와 끝 인덱스 구하는 걸로 바뀜
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
# from mdl_cost_dictionary import *
import pickle
import time
from torch.cuda.amp import autocast

parser = argparse.ArgumentParser()
parser.add_argument('--log_file', default='log_file/hadoop.log', type=str, required=True)
parser.add_argument('-b', '--batch_size', default=8, type=int)
parser.add_argument('--model_name', default=None, type=str)
args = parser.parse_args()

if args.model_name is None:
    args.model_name = 'roberta-base'
        

model_name = 'roberta-base'
log_file = args.log_file
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
seed = fitlog.set_rng_seed(rng_seed=None)
os.environ['FASTNLP_GLOBAL_SEED'] = str(seed)
fitlog.add_hyper(args)
fitlog.add_hyper_in_file(__file__)

# ANSI 코드 정의
RED = "\033[91m"
END = "\033[0m"

## 정규표현식 16진수, 숫자 정의
BASE10NUM = r"(?<![0-9.+-])(?>[+-]?(?:(?:[0-9]+(?:\.[0-9]+)?)|(?:\.[0-9]+)))"
NUMBER = r"(?:%{BASE10NUM})"
BASE16NUM = r"(?<![0-9A-Fa-f])(?:[+-]?(?:0x)?(?:[0-9A-Fa-f]+))"
BASE16FLOAT = r"\b(?<![0-9A-Fa-f.])(?:[+-]?(?:0x)?(?:(?:[0-9A-Fa-f]+(?:\.[0-9A-Fa-f]*)?)|(?:\.[0-9A-Fa-f]+)))\b"
FLOAT = r"\b\d+\.\d*\b"

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
UNIT = r"^\d+(\.\d+)?(K|M|G|T|P|E|Z|Y)?(iB|B)$"
HEX8 = r'^[0-9a-fA-F]{8}$'
NODE = r"^[a-z]+_[a-z0-9_]+$|^[A-Z_]+$" # 소문자로 시작하고, _,숫자,문자 조합이 하나 이상 반복되는 문자열 또는 대문자 또는 밑줄로만 구성된 문자열

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
PROPERTY = r"(?:[a-zA-Z0-9_-]+\.)+[a-zA-Z0-9_-]+"
# URI = re.compile(r"{URIPROTO}://(?:{USER}(?::[^@]*)?@)?(?:{URIHOST})?(?:{URIPATHPARAM})?".format(URIPROTO=URIPROTO, USER=USER, URIHOST=URIHOST, URIPATHPARAM=URIPATHPARAM))

# pattern
UUID_p = regex.compile(UUID)
IP_p = re.compile(IP)
HEX8_p = re.compile(HEX8)
PATH_p = regex.compile(PATH)
POSINT_p = re.compile(POSINT)
IPPORT_p = re.compile(IPPORT)
PATHPORT_p = regex.compile(PATHPORT)
BASE10NUM_p = regex.compile(BASE10NUM)
BASE16NUM_p = regex.compile(BASE16NUM)
FLOAT_p = regex.compile(FLOAT)
PROPERTY_p = regex.compile(PROPERTY)
NODE_p = regex.compile(NODE)
# log_file = "/raid1/eunwoo/lognroll/logs/cassandra_clean.log"



def variable_bit(variable):
    bit = 0
    for i in range(len(variable)):
        if variable[i].isdigit():
            bit+=3
        else:
            bit+=16
    return bit


def add_space_before_commas(text):
# 쉼표 앞에 문자가 붙어있으면 공백을 추가
    return re.sub(r'(?<=[^\s]),', ' ,', text)

def add_space_before_comma(text):
    # 쉼표 앞에 공백이 없으면 공백 추가
    return re.sub(r'(\S),', r'\1 ,', text)

# forward 이후 (s, e, score)와 토큰화된 문장이 필요
def tokenizing(log):
    seps = [':', '(', ')', ';', '=', ' ', '[', ']', '{', '}', ',', '$', '<', '>', '"', "'", "@", "|", "&"]
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
             
def compileTemplate(template):
    template = template.replace('\\', '\\\\')
    result = (template.replace("[", r"\[").replace("]", r"\]").replace("(", r"\(").replace(")", r"\)")
              .replace("$", r"\$").replace("+", r"\+").replace("?", r"\?")).replace("|", r"\|").replace(".", r"\.")
    # \가 추가된 .\* 처리 완료
    result = result.replace("*", r"\*")
    result = result.replace(r"\.\*", "(.*)")

    result = re.compile(result)

    return result

def delete_continuous_wildcard(temp):
    while '.* .*' in temp:
        temp = temp.replace('.* .*', '.*')
    while '.*..*' in temp:
        temp = temp.replace('.*..*', '.*')
    while '.* , .*' in temp:
        temp = temp.replace('.* , .*', '.*')
    while '.* : .*' in temp:
        temp = temp.replace('.* : .*', '.*')
    while '.* | .*' in temp:
        temp = temp.replace('.* | .*', '.*')
    while 'at .* at .*' in temp:
        temp = temp.replace('at .* at .*', '.*')
    return temp
                     
# # find_hier을 위해 완전 동일한 구간은 작은 sigmoid를 가지는 구간을 없애는 코드
# def identical_remove(span_list):
#     rng_dic = defaultdict(float)
#     output = []
#     # identical 제거
#     for s, e, sigmoid in span_list:
#         if rng_dic[(s, e)] < sigmoid:
#             rng_dic[(s, e)] = sigmoid

#     for s, e in rng_dic.keys():
#         sigmoid = rng_dic[(s, e)]
#         output.append((s, e, sigmoid))
#     return output

# find_hier을 위해 완전 동일한 구간은 작은 sigmoid를 가지는 구간을 없애는 코드
def identical_remove(span_list):
    rng_dic = defaultdict(float)
    output = []
    # identical 제거
    for s, e, sigmoid in span_list:
        if rng_dic[(s, e)] < sigmoid:
            rng_dic[(s, e)] = sigmoid
    

    # contains_digit가 아닌 경우(ex: PATH)에는 s, s-1  삭제
    remove_list = set()
    for s, e in rng_dic.keys():
        if rng_dic[(s, e)] > ent_thres:
            if (s, s-1) in rng_dic.keys():
                remove_list.add((s, s-1))
    
    for key in remove_list:
        del rng_dic[key]

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


# json데이터를 만들기 위한 것
def make_test_json_list(logs):

    json_list = []

    for log in logs:
        d = dict()
        d['tokens'] = log.strip().split()
        d['entity_mentions'] = []
        json_list.append(d)

    return json_list

def is_number_with_unit(s):
    return bool(re.search(r"\d+ms", s)) or bool(re.search(r"\d+/s", s)) or bool(re.search(r"\d+%", s))

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

def is_contains_digit(s):
    return any(char.isdigit() for char in s)
    
def find_regex(original_sentence):
    regex_rng=[] 
    for i in range(len(original_sentence)):
        # match_URI = URI.fullmatch(log_span[(i, j)])
        # if match_URI:
        #     # print(f"URI: {log_span[(i, j)]}")
        #     regex_rng.append((i, j, 0.999))
        #     continue

        # HDFS돌릴때만 주석
        match_IP = IP_p.fullmatch(original_sentence[i])
        if match_IP:
            # print(f"IP: {log_span[(i, j)]}")
            regex_rng.append((i, i, 0.999))
            if i+2<len(original_sentence) and original_sentence[i+1] == ":" and POSINT_p.fullmatch(original_sentence[i+2]):
                # print(f"PORT: {original_sentence[j+2]}")
                regex_rng.append((i+2, i+2, 0.999))
                # print(f"IPPORT: {log_span[(i, j+2)]}")
                regex_rng.append((i, i+2, 0.999))
            continue
            


        match_PATH = PATH_p.fullmatch(original_sentence[i])
        if match_PATH:
            # print(f"PATH: {log_span[(i, j)]}")
            regex_rng.append((i, i, 0.999))
            if i+2<len(original_sentence) and original_sentence[i+1] == ":" and POSINT_p.fullmatch(original_sentence[i+2]):
                # print(f"PORT: {original_sentence[j+2]}")
                regex_rng.append((i+2, i+2, 0.999))
                # print(f"PATHPORT: {log_span[(i, j+2)]}")
                regex_rng.append((i, i+2, 0.999))
            continue

        match_UUID = UUID_p.fullmatch(original_sentence[i])
        if match_UUID:
            # print(f"UUID: {log_span[(i, j)]}")
            regex_rng.append((i, i, 0.999))
            continue

        match_BASE10NUM = BASE10NUM_p.fullmatch(original_sentence[i])
        if match_BASE10NUM:
            # print(f"NUMBER: {log_span[(i, j)]}")
            regex_rng.append((i, i, 0.999))
            continue

        match_BASE16NUM = BASE16NUM_p.fullmatch(original_sentence[i])
        if match_BASE16NUM:
            # print(f"BASE16NUM: {log_span[(i, j)]}")
            regex_rng.append((i, i, 0.999))
            continue

        if HEX8_p.match(original_sentence[i]):
            # print(f"8자리 16진수값: {log_span[(i, j)]}")
            regex_rng.append((i, i, 0.999))
            regex_rng.append((i, i-1, 0.999))
            continue

        if is_hexadecimal(original_sentence[i]):
            # print(f"0x로 시작하는 16진수값: {log_span[(i, j)]}")
            regex_rng.append((i, i, 0.999))
            continue

        match_FLOAT = FLOAT_p.fullmatch(original_sentence[i])
        if match_FLOAT:
            # print(f"FLOAT: {log_span[(i, j)]}")
            regex_rng.append((i, i, 0.999))
            continue
            


        ## cassandra 54.3KiB 같은거 처리를 위해 
        if "#" in original_sentence[i] or "MB" in original_sentence[i] or "KB" in original_sentence[i] or "KiB" in original_sentence[i] or original_sentence[i][0] == "~"or original_sentence[i].lower()=="null" or original_sentence[i].lower()=="true" or original_sentence[i].lower()=="false" or is_number_with_unit(original_sentence[i]):
            regex_rng.append((i, i, 0.999))
            continue

        ## $ RMFatalEventDispatcher 처럼 $ 뒤에 붙는거 .*로
        if original_sentence[i-1] == "$":
            regex_rng.append((i, i, 0.999))
            continue


        ## MultiLog 처리 set raft.server.log.queue.element-limit = .* 처리 위해 
        if 2 <= len(original_sentence) and original_sentence[i] == original_sentence[-1] and (":" == original_sentence[-2] or "=" == original_sentence[-2]):
            regex_rng.append((i, i, 0.999))
            continue

        ## MultiLog new SynchronousQueue thread pool : ConfigNodeRPC-Processor처리 위해
        if "-" in original_sentence[i]:
            regex_rng.append((i, i, 0.999))
            continue

        ## MultiLog set raft.server.log.queue.element-limit 같은 property 처리하기 위해
        match_PROPERTY = PROPERTY_p.fullmatch(original_sentence[i])
        if match_PROPERTY:
            regex_rng.append((i, i, 0.999))
            continue

        ## Spark dfs.namenode.safemode.extension = 30000  처리 위해(.* = .*로로) 
        if 3 == len(original_sentence) and (":" == original_sentence[-2] or "=" == original_sentence[-2]):
            regex_rng.append((0, 0, 0.999))
            regex_rng.append((2, 2, 0.999))
            continue

        ## spark CID-UUID 형식 처리하기 위해
        contain_UUID = UUID_p.search(original_sentence[i])
        if contain_UUID:
            regex_rng.append((i, i, 0.999))
            continue

        match_NODE = NODE_p.fullmatch(original_sentence[i])
        if match_NODE:
            regex_rng.append((i, i, 0.999))
            continue

        if 100 < len(original_sentence[i]):
            regex_rng.append((i, i, 0.999))
            continue

        # ## cassandra 54.3KiB 같은거 처리를 위해 
        # if is_contains_digit(original_sentence[i]):
        #     # print(f"8자리 16진수값: {log_span[(i, j)]}")
        #     regex_rng.append((i, i, ent_thres))
        #     regex_rng.append((i, i-1, ent_thres))
        #     continue

        # if i+1<len(original_sentence) and (original_sentence[i] == '=' or original_sentence[i] ==':'):
        #     print(original_sentence[i])
        #     regex_rng.append((i+1, i+1, ent_thres))
        #     regex_rng.append((i+1, i, ent_thres))
        #     continue
    return regex_rng

## find_hier을 괄호가 비대칭인 경우 케이스 고려로 수정
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

    ## 가끔 괄호가 비대칭인 경우가 있음. 그럴땐 맨 마지막 토큰을 닫는 괄호로 취급
    if hier_stack:
        for i in reversed(range(len(hier_stack))):
            hier_rng.append((hier_stack[i][1], len(tokenized_log)-1, 1))

    return hier_rng


# @cache_results('caches/ner_caches.pkl', _refresh=False)
def get_data(model_name):
    paths = 'preprocess/outputs/inference'
    pipe = SpanNerPipe(model_name=model_name)

    dl = pipe.process_from_file(paths)

    return dl, pipe.matrix_segs

def densify(x):
    x = x.todense().astype(np.float32)
    return x

def log_file_tag(log_file):
    if "hadoop" in log_file:
        return "Hadoop"

    if "spark" in log_file:
        return "Spark"

    if "iot" in log_file:
        return "MultiLog"

def make_templates_csv(total_template_occurrence):
    templates_log = pd.DataFrame({
    "EventId": list(map(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()[0:8], total_template_occurrence)), 
    "EventTemplate": total_template_occurrence.keys(),
    "Occurrences": total_template_occurrence.values()
    })
    templates_log.to_csv('./parsing_result/result_templates.csv', index=False)


def make_struct_csv(log_file, total_log_list, df_log, template_list):
    if "spark" in log_file:
        structured_log = pd.DataFrame({
            "LineId": list(range(1, len(total_log_list)+1)),
            "Level": df_log["Level"],
            "Component": df_log["Component"],
            "Content": total_log_list, 
            "EventId": list(map(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()[0:8], template_list)), 
            "EventTemplate": template_list       ## log_match_bestTemplate[x].get(x, -1)하면 딕셔너리에 키값이 없어도 error를 반환하지 않고 -1을 반환환
        }, index=None)
    
    if "hadoop" in log_file:
        structured_log = pd.DataFrame({
            "LineId": list(range(1, len(total_log_list)+1)),
            "Date": df_log["Date"],
            "Time": df_log["Time"],
            "Level": df_log["Level"],
            "Process": df_log["Process"],
            "Component": df_log["Component"],
            "Content": total_log_list, 
            "EventId": list(map(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()[0:8], template_list)), 
            "EventTemplate": template_list       ## log_match_bestTemplate[x].get(x, -1)하면 딕셔너리에 키값이 없어도 error를 반환하지 않고 -1을 반환환
        }, index=None)

    if "iot" in log_file: 
        structured_log = pd.DataFrame({
            "LineId": list(range(1, len(total_log_list)+1)),
            "Date": df_log["Date"],
            "Time": df_log["Time"],
            "Content": total_log_list, 
            "EventId": list(map(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()[0:8], template_list)), 
            "EventTemplate": template_list       ## log_match_bestTemplate[x].get(x, -1)하면 딕셔너리에 키값이 없어도 error를 반환하지 않고 -1을 반환환
        }, index=None)

    structured_log.to_csv('./parsing_result/result_structured.csv', index=False)



# 전체 모델 로드
model = torch.load("./model_cache/union.pth")
###################### Load Data ################################################
headers, logformat_regex = generate_logformat_regex(benchmark[log_file_tag(log_file)]["log_format"])
df_log = log_to_dataframe(log_file, logformat_regex, headers, benchmark[log_file_tag(log_file)]["log_format"])
regex_mode = True

## 길이 제한을 둘 경우
# df_log = df_log[df_log["Content"].str.len()>700]
# df_log = df_log.iloc[:int(150000 * 0.1)]

## 몇개로 토막내서 resample할건지 
filtering_parameter = 1000

## unmatch log가 줄어드는 수가 filtering_threshold 미만일시 멈춤
filtering_threshold = len(df_log)

# 사용하는 플래그 및 상수들
# THRESHOLD를 넘어서면 그룹의 성분 수가 너무 많다고 판단, 모든 경우의 수를 고려하지 않고 다른 방법을 사용
GROUP_ELEMENT_THRESHOLD = 10
# 그룹핑 사용 여부에 대한 플래그
using_grouping = True
# drc를 각 변수에 대하여 계산하는지, 모든 변수에 대하여 계산하는지에 대한 플래그
drc_each_variable = False
# 그룹의 성분 수가 많을 때, 템플릿 트리를 사용한 경우의 수를 줄이는 방법을 사용할지에 대한 플래그
using_template_tree = True
###################################################################################

if __name__=="__main__":

    logs =  list(df_log["Content"])

    # logs = ["Shuffle index for mapId 23 : [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0"]
    print(len(logs))


    total_log_list = [" ".join(tokenizing(x)).strip() for x in logs]
    remain_logs = total_log_list
    ## 줄어드는 개수가 K개 미만이면 resample 종료 
    unmatch_arr = [len(remain_logs)]

    start_time = time.time()
    total_log_template_dict = {}
    total_template_occurrence = defaultdict(int)
    resample_log = True


    while len(remain_logs):
        template_log_match_dic = defaultdict(list)
        log_template_match_dic = defaultdict(list)
        wildcard_match_length = defaultdict(list)
        original_logs = []

        if resample_log:
            log_list = random.sample(remain_logs, filtering_parameter)
            print("log_list: ", len(log_list))
        else:
            log_list = remain_logs
            print("log_list: ", len(log_list))
        ## 로그 입력하면 jsonlines파일 생성
        with open("preprocess/outputs/inference/test.jsonlines", 'w') as infer_file:
            input_json_list = make_test_json_list(log_list) # json형식으로 변환
            # print(input_json_list)
            for dic in input_json_list:
                json_data = json.dumps(dic)
                infer_file.write(json_data + '\n')

        dl, matrix_segs = get_data(model_name) 

        dl.apply_field(densify, field_name='matrix', new_field_name='matrix', progress_bar='Densify')
        ## 디버깅용
        # print(dl)
        label2idx = getattr(dl, 'ner_vocab') if hasattr(dl, 'ner_vocab') else getattr(dl, 'label2idx')
        ## 디버깅용
        # print(f"{len(label2idx)} labels: {label2idx}, matrix_segs:{matrix_segs}")
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

            
        # for batch in evaluate_dls['test']:
        #     input_ids = batch['input_ids']  # 데이터의 'input_ids' 키에 따라 접근
        #              # 레이블도 필요하다면 이렇게 접근
            
            # input_ids를 사용한 처리
            # print("batch:", batch.keys())

        # dirPath = "./caches"

        # if os.path.exists(dirPath):
        #     for file in os.scandir(dirPath):
        #     	print("Remove File: ",file)
        #         os.remove(file)



        # model = torch.load("./model_cache/union.pth")
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
        print("로그 후보 템플릿 생성 중")
        for batch in tqdm(evaluate_dls['test']):
            with torch.no_grad():
                try:
                    scores = model.forward(batch["input_ids"].cuda(), batch["bpe_len"].cuda(), batch["indexes"].cuda(), batch["matrix"].cuda())["scores"]  #indexes 값은 각 bert 토크나이징된 토큰들의 어떤 원래 단어에 속하는지를 나타내는 그룹 인덱스입니다.
                except RuntimeError as e:
                    print("Bug Cause")
                    span_pred_shape = (8, batch["word_len"].max(), batch["word_len"].max())
                    span_pred = torch.zeros(span_pred_shape)
                else:
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
                    ## 디버깅용
                    # print("log count: ", log_count)
                    original_sentence = log_list[log_count]
                    # print(original_sentence)
                    ##For cassandra log, if empty: continue
                    if batch["bpe_len"][idx].item()==2:
                        continue
                    
                    log_count+=1
                
                    import re
                    # original_log = add_space_before_comma(original_sentence).strip()
                    original_log = original_sentence.strip()
                    # print(original_log)
                    original_logs.append(original_log)
                    # print()
                    original_sentence = list(original_log.split())
                    # print(original_sentence)
                    ####################grok#######################################
                    if regex_mode:
                        # log_span = defaultdict(str)
                        # for i in range(len(original_sentence)):
                        #     for j in range(i, len(original_sentence)):
                        #         if i < j:
                        #             log_span[(i, j)] = log_span[(i, j-1)] + " " + original_sentence[j]
                        #         else:
                        #             log_span[(i, j)] = log_span[(i, j)] + original_sentence[j]

                        
                        # 규칙 기반 계층 찾기 [(s, e, 1), ...]
                        hier_rng = find_hier(original_sentence)
                        
                        # grok 계층 찾기
                        regex_rng = find_regex(original_sentence)
                    else:
                        hier_rng = []
                        regex_rng = []
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

                    if regex_mode:
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
                        # variable_lengths = []

                        # for v in new_variables:
                        #     start = v[0][0]
                        #     end = v[0][1]
                        #     v_len = (start, end)
                        #     # for i in range(start, end + 1):
                        #     #     v_len += len(sentence[i])
                        #     #     # v_len += variable_bit(sentence[i])
                        #     # if end >= start:
                        #     #     v_len += (end - start)  # 공백 길이 더하기
                        #     variable_lengths.append(v_len)



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
                        # True면 *
                        labels = []
                        # variable length 계산에 쓰는건 이 배열에 추가
                        variable_lengths = []
                        # before = ' '
                        # for word in template_tokens:
                        #     # 연속되는 * 은 추가 X
                        #     if word == '.*' and before == '.*':
                        #         continue
                        #     else:
                        #         template.append(word)
                        #         before = word
                        for i in range(len(template_tokens)):
                            if i > 0:
                                if template_tokens[i] == '.*' and template_tokens[i - 1] == '.*':
                                    labels.append(True)
                                    continue
                            if i > 0 and i < len(template_tokens) - 1:
                                if template_tokens[i] == ',' and template_tokens[i - 1] == '.*' and template_tokens[
                                    i + 1] == '.*':
                                    labels.append(True)
                                    continue
                            if i > 1:
                                if template_tokens[i] == '.*' and template_tokens[i - 1] == ',' and template_tokens[
                                    i - 2] == '.*':
                                    labels.append(True)
                                    continue
                            template.append(template_tokens[i])
                            if template_tokens[i] == '.*':
                                labels.append(True)
                            else:
                                labels.append(False)

                        # print(' '.join(template))

                        before = False
                        start = -1
                        end = -1

                        for i in range(len(labels)):
                            if labels[i]:
                                # 이전은 .* 이 아니었으므로
                                if not before:
                                    start = i
                                if i + 1 >= len(labels) or not labels[i + 1]:
                                    end = i
                                    variable_lengths.append((start, end))
                            before = labels[i]
                                
                        candidate_template = ' '.join(template).strip()
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
                        wildcard_match_length[candidate_template].append((original_log, []))





        # 해당되는 코드만 주석 제거
        ################################################## 1. NER 이용시 (slow) 사용코드 
        log_list = original_logs
        print("log_list: ", len(log_list))
        log_list = [cur_log.strip() for cur_log in log_list]


        template_occurrence, log_template_dict = find_best_template_by_log_and_candidate_templates(log_list, wildcard_match_length, GROUP_ELEMENT_THRESHOLD, using_grouping, drc_each_variable, using_template_tree)

        print(len(log_template_dict.keys()))
        # 너무 general한 템플릿을 제외한 1차로 생성된 템플릿들
        
        if resample_log:
            pre_templates = [" ".join(tokenizing(t)).strip() for t in template_occurrence if t.strip() not in [".*", ".* : .*", ".* = .*", ".* = .* ( .* )"]]
            if len(pre_templates) == 0:
                pre_templates = [" ".join(tokenizing(t)).strip() for t in template_occurrence]
                resample_log = False
        else:
            pre_templates = [" ".join(tokenizing(t)).strip() for t in template_occurrence]
            
        # specific한 것에 먼저 매칭되도록 함
        pre_templates.sort(reverse=True, key=lambda x:len(x))
        # print("\n".join(pre_templates))
        compiled_pre_templates = []


        # 미리 컴파일시켜서 속도 높임
        for t in pre_templates:
            t = delete_continuous_wildcard(t)
            t = compileTemplate(t)
            compiled_pre_templates.append(t)
        
        unmatch_logs = []
        # 매칭된 로그는 log_template_dict에 추가, template_occurrence 에 템플릿에 매칭된 로그수 + 1
        # 매칭 안 된 로그는 unmatch_logs 에 추가

        ## 중복되는 로그를 빠르게 처리하는 코드 나중에 추가 가능
        
        for log in tqdm(remain_logs):
            matched = False
            for i in range(len(compiled_pre_templates)):
                ct = compiled_pre_templates[i]
                t = pre_templates[i]
                if not resample_log and 'PrivilegedAction' in t:
                    continue
                if ct.fullmatch(log):
                    total_log_template_dict[log] = t
                    total_template_occurrence[t] += 1
                    matched = True
                    break
            if not matched:
                unmatch_logs.append(log)


        
        print(f"{RED}unmatch log의 로그수는 {len(unmatch_logs)}{END}")
        # print("\n".join(unmatch_logs))
        remain_logs = unmatch_logs
        unmatch_arr.append(len(unmatch_logs))

        if unmatch_arr[-2] - unmatch_arr[-1] < filtering_threshold:
            resample_log = False
    template_list = list(map(lambda x: total_log_template_dict[x], total_log_list))


    # best_mean_src = whole_src / len(set([t for t in template_occurrence]))
    # best_mean_drc = whole_drc / len(log_list)

    end_time = time.time()
    execution_time = end_time - start_time  # 실행 시간 계산
    print(f"execution time: {execution_time:.4f}sec\n")
    # print('\033[31m' + f'best mdl cost: {best_mean_src + best_mean_drc}' + '\033[0m')
    # print('\033[31m' + f'best mean src: {best_mean_src}' + '\033[0m')
    # print('\033[31m' + f'best mean drc: {best_mean_drc}' + '\033[0m')

    make_templates_csv(total_template_occurrence)





