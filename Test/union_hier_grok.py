from collections import deque 
from collections import defaultdict
import re
import regex

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

log = ['command', 'university.student', 'command', ':', 'insert', '{', 'insert', ':', "'", 'student', "'", ',', 'ordered', ':', 'true', ',', 'lsid', ':', '{', 'id', ':', 'UUID', '(', "'", '49da9317-2afd-4ad4-8e53-58ea21d6ac88', "'", ')', '}', ',', '$', 'db', ':', "'", 'university', "'", '}', 'ninserted', ':', '1', 'keysInserted', ':', '1', 'numYields', ':', '0', 'reslen', ':', '44', 'locks', ':', '{', 'Global', ':', '{', 'acquireCount', ':', '{', 'r', ':', '1', ',', 'w', ':', '1', '}', '}', ',', 'MMAPV1Journal', ':', '{', 'acquireCount', ':', '{', 'w', ':', '2', '}', '}', ',', 'Database', ':', '{', 'acquireCount', ':', '{', 'w', ':', '1', '}', '}', ',', 'Collection', ':', '{', 'acquireCount', ':', '{', 'W', ':', '1', '}', '}', '}', 'protocol', ':', 'op_query', '0ms']

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
    print(s)
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


###############################grok_test###################################################
log_span = defaultdict(str)
for i in range(len(log)):
    for j in range(i, len(log)):
        if i < j:
            log_span[(i, j)] = log_span[(i, j-1)] + log[j]
        else:
            log_span[(i, j)] = log_span[(i, j)] + log[j]
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
PATH = r"(?:{UNIXPATH}|{WINPATH})".format(UNIXPATH=UNIXPATH, WINPATH=WINPATH)
PATHPORT = r"{PATH}:{PORT}".format(PATH=PATH, PORT=POSINT)
URIPATH = r"(?:/[A-Za-z0-9$.+!*'(){},~:;=@#%_\-]*)+"
URIPARAM = r"\?[A-Za-z0-9$.+!*'|(){},~@#%&/=:;_?\-\[\]]*"
URIPATHPARAM =r"{URIPATH}(?:{URIPARAM})?".format(URIPATH=URIPATH, URIPARAM=URIPARAM)
URIHOST = r"{IPORHOST}(?::{POSINT})?".format(IPORHOST=IPORHOST, POSINT=POSINT)
URIPROTO = r"[A-Za-z]+(\+[A-Za-z+]+)?"
URI = r"{URIPROTO}://(?:{USER}(?::[^@]*)?@)?(?:{URIHOST})?(?:{URIPATHPARAM})?".format(URIPROTO=URIPROTO, USER=USER, URIHOST=URIHOST, URIPATHPARAM=URIPATHPARAM)
# # 정규표현식을 이용한 추출
# for span in log_span.keys():
#     match = re.fullmatch(IP, log_span[span])
#     if match:
#         print(f"Matched: {log_span[span]}")
#         if span[1]+1

def find_regex(tokenized_log):
    regex_rng=[] 
    for i in range(len(tokenized_log)):
        for j in range(i, len(tokenized_log)):
            match_URI = re.fullmatch(URI, log_span[(i, j)])
            if match_URI:
                print(f"URI: {log_span[(i, j)]}")
                regex_rng.append((i, j, 0.999))
                continue
        
            match_IP = re.fullmatch(IP, log_span[(i, j)])
            if match_IP:
                print(f"IP: {log_span[(i, j)]}")
                regex_rng.append((i, j, 0.999))
                continue
            
            match_IPPORT = re.fullmatch(IPPORT, log_span[(i, j)])
            if match_IPPORT:
                print(f"IP: {log_span[(i, j-2)]}  PORT: {log_span[(j, j)]}")
                # regex_rng.append((i, j-2, 0.999))
                regex_rng.append((j, j, 0.999))
                continue

            match_PATH = regex.fullmatch(PATH, log_span[(i, j)])
            if match_PATH:
                print(f"PATH: {log_span[(i, j)]}")
                regex_rng.append((i, j, 0.999))
                continue
            
            match_PATHPORT = regex.fullmatch(PATHPORT, log_span[(i, j)])
            if match_PATHPORT:
                print(f"PATH: {log_span[(i, j-2)]}  PORT: {log_span[(j, j)]}")
                # regex_rng.append((i, j-2, 0.999))
                regex_rng.append((j, j, 0.999))
                continue

            match_UUID = re.fullmatch(UUID, log_span[(i, j)])
            if match_UUID:
                print(f"UUID: {log_span[(i, j)]}")
                regex_rng.append((i, j, 0.999))
                continue
    return regex_rng

regex_rng = find_regex(log)
print(regex_rng)
###############################grok_test###################################################

span_list = find_hier(log)
span_list.extend(regex_rng)
result = deleting_overlapping2()
for x in result:
    span_list.remove(x)
print(span_list)


