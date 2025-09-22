#### 2k 데이터 자동 라벨링할 때 쓴 소스코드 ####

import json
import re
import sys
from os import path
import pandas as pd
sys.path.append(path.dirname(path.dirname(path.abspath('__main__'))))



logfile = "/home/eunwoo/CNN_Nested_NER/Test/data/additional/spark_additional.log_structured.csv"
jsonfile = "spark_additional.jsonlines"

instance_in_head = False

## <*>가 .나 -로 연속되게 이어진 걸을 하나의 <*>로 치환
def merge_placeholders(text):
    # Define a regex pattern to match "<*>" followed by "-" or "." and another "<*>"
    pattern = r'<\*>[-\.]<\*>'
    
    # Continuously replace the pattern until no more matches are found
    while re.search(pattern, text):
        text = re.sub(pattern, '<*>', text)
    
    return text

## DFSClient_NONMAPREDUCE_<*>_<*>, core.323, i-fetch......................0, processor.........................0 와 같이 문자열이나 -.숫자,<*>로 이어진것을 <*>로 치환
def replace_placeholder_pattern_with_asterisk(text):
    # 정규식: "문자열(알파벳과 '-') + '.' + '<*>'" 형식 매칭
    return re.sub(r'[a-zA-Z\-_<\*>]+\.*<\*>', '<*>', text)

def replace_point_with_wildcard(text):
    # <*>앞에 .이 1개이상 붙은 문자열 <*>로 치환환
    return re.sub(r'\.{1,}<\*>', '<*>', text)

# 템플릿에 그냥 *이 들어가있는경우
def remove_stars_except_placeholders(text):
    # Match `*` outside of `<*>` and remove them
    return re.sub(r'\*(?![^<]*>)', '', text)

## 마침점 없애주는 코드, 라벨 갯수와 entity 개수가 다르게 되는 문제 예방.(ip address나 class의 점을 없애지 않기 위해 주의!)
def remove_dots_after_chars(text):
    # 숫자, 문자, 특수문자 뒤에 점(.)이 있는 경우 매칭
    pattern = r'(?<=[\w\W])\.(?=\s|$)'
    return re.sub(pattern, '', text)

## labeling tokenizing함수 전용 기존 토크나이징 함수와 다름 
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
            if len(token) != 0:
                tokens.append((token, token_start, i))
                token_start = i
            if ch != ' ':
                tokens.append((ch, i, i + 1))
            token = ""
            token_start = i + 1
            continue

        if ch != ' ':
            token += ch

        # 문장의 마지막에 도달했고 저장해야 할 token이 있다면
        if i == length - 1 and len(token) != 0:
            tokens.append((token, token_start, i + 1))

    return tokens

# .* 들을 ()로 둘러싸 캡쳐한 뒤, 각 그룹의 시작 인덱스, 종료 인덱스를 리턴
# 또한 매칭되는 entity들을 리턴
def finding_indexes(temp, log):
    match = temp.search(log)
    indexes = []
    entities = []

    if instance_in_head:
        for i in range(2, len(match.groups()) + 1):
            indexes.append((match.start(i), match.end(i)))
            entities.append(match.group(i))

    else:
        for i in range(1, len(match.groups()) + 1):
            if match.group(i):
                indexes.append((match.start(i), match.end(i)))                
                entities.append(match.group(i))

    return indexes, entities


def find_entity_indexes(tokens, indexes):

    labels = []

    if not indexes: return []

    current_label_start = -1
    current_label_end = -1

    cur = 0
    start = indexes[0][0]
    end = indexes[0][1]

    # 예를 들어 start = 10, end = 100 (indexes의 start와 ent)
    # 각 토큰들은 (0, 10), (10, 20), (20, 30), ..... (90, 101)

    for i in range(len(tokens)):
        token = tokens[i]
        token_start = token[1]
        token_end = token[2]

        if start == token_start:
            current_label_start = i

        if end == token_end:
            current_label_end = i + 1

            if current_label_start == -1:
                # print("tokens: ", tokens)
                # print("indexes: ", indexes)
                print("why label start does not exist??")
                return []
            labels.append((current_label_start, current_label_end))

            current_label_start = -1
            current_label_end = -1
            cur += 1
            if cur >= len(indexes):
                break
            start = indexes[cur][0]
            end = indexes[cur][1]

    return labels


def make_json(tokens, labels, entities):
    d = dict()
    d['tokens'] = [token[0] for token in tokens]

    ## '' 나 None 값이 포함되어서 라벨 개수와 entity 개수가 맞지 않을 때가 있는데, 결국 label 수가 맞기에 그거 기준대로 하면됨. 보통 path가 /<*>/blk_<*> 이런식으로 되어있을 때 생김.
    if len(labels) != len(entities):
        print("토크나이징이 잘못 되었습니다. 라벨의 개수와 entity의 개수가 다릅니다.")
        print("tokens: ", tokens)
        print("labels: ", labels)
        print("entities: ", entities)
        return -1
    d['entity_mentions'] = list()

    for i in range(len(labels)):
        mention = dict()
        mention["entity_type"] = 'object'
        mention['start'] = labels[i][0]
        mention['end'] = labels[i][1]
        mention['text'] = entities[i]
        d['entity_mentions'].append(mention)

    s = json.dumps(d)
    return s



# with open(logfile, 'r') as f, open(jsonfile, 'w') as f2:
#     while True:
#         line = f.readline()
#         if not line: break
#         line = line.strip()

#         # 문자열 전처리
#         # 라벨링을 위해서 로그 메시지만 남겨놓고 다른건 전처리를 해야함
#         line = preprocess(line)
#         if not line: continue

#         # 3. 토큰별 문장에서의 세부 인덱스 저장 및 각 entity의 토큰 시작인덱스, 종료 인덱스 찾기
#         # 4. 라벨링

#         # preprocessing 된 로그가 매칭되는 로그 템플릿 찾기
#         # template은 compile된 매칭되는 로그 템플릿
#         template = return_matching_template_format(line)

#         # 매칭되는 로그의 시작 인덱스와 종료 인덱스 찾기
#         # index_list 의 각 성분은 다음과 같음
#         # (시작 index, 종료 index + 1)
#         # entity_list 의 각 성분은 entity 문자열
#         index_list, entity_list = finding_indexes(template, line)

#         # 토크나이징 결과
#         # token_list의 각 성분은 다음과 같이 이루어짐
#         # (토큰, 토큰의 시작 인덱스, 토큰의 종료 인덱스 + 1)
#         token_list = tokenizing.tokenizing(line)
#         print(token_list)
#         # 토크나이징 결과와 index_list를 비교, 어느 부분이 라벨링되어야 하는지 확인
#         # 리턴되는 결과는 다음과 같음
#         # labeling_list의 각 성분은 (시작 토큰 인덱스, 종료 토큰 인덱스)
#         labeling_list = find_entity_indexes(token_list, index_list)

#         # json으로 만들기
#         temp_json = make_json(token_list, labeling_list, entity_list)

#         f2.write(temp_json + '\n')
log_df = pd.read_csv(logfile)
print(len(log_df))
with open(jsonfile, 'w') as f2:
    for i in range(len(log_df)):
        line = log_df["Content"][i]
        template = log_df["EventTemplate"][i]

        ## 
        if template == "com.apple.<*>: scheduler_evaluate_activity told me to run this job; however, but the start time isn't for <*> seconds.  Ignoring.":
            template = "<*>: scheduler_evaluate_activity told me to run this job; however, but the start time isn't for <*> seconds.  Ignoring."
        if "CCFile::captureLog Received Capture notice id:" in template:
            continue
        ## Spark에서 토큰기준때문에 다음으로 변경
        if template == "Input split: hdfs://<*>":
            template = "Input split: hdfs:<*>"
        if template == "Saved output of task 'attempt_<*>' to hdfs://<*>":
            template = "Saved output of task '<*>' to hdfs:<*>"
        ## Thunderbird에서 토큰기준때문에 다음으로 변경
        if template == "<*>: from=root, size=<*>, class=<*>, nrcpts=<*>, msgid=<<*>>, relay=#<*>#@localhost":
            template = r"<*>: from=root, size=<*>, class=<*>, nrcpts=<*>, msgid=\<<*>\>, relay=<*>@localhost"
        ## Zookeeper에서 토큰기준때문에 다음으로 변경
        if template == "Received connection request /<*>:<*>":
            template = "Received connection request <*>:<*>"
        if template == "Expiring session <*>, timeout of <*>ms exceeded":
            template = "Expiring session <*>, timeout of <*> exceeded"
        # 템플릿과 로그에 마침점(.)을 없애줌. 토크나이징과 match의 index가 다르게 나오는 문제점 예방.
        line = remove_dots_after_chars(line)
        template = remove_dots_after_chars(template)
        
        # 템플릿과 로그에 그냥 \이 들어가있는 경우 \ 제거
        line = line.replace("\\", "")
        template = template.replace("\\", "")

        # 템플릿과 로그에 그냥 *이 들어가있는 경우 * 제거
        line = line.replace("*", "")
        template = remove_stars_except_placeholders(template)
        if "<*>" not in template:
            print(template)
            continue
        ## hadoop 경우에 msra-sa-41/10.190.173.170가 많음
        if "<*>/<*>" in template:
            template = template.replace("<*>/<*>", "<*>")
        ## ----------BGL,HDFS 데이터인 경우에 /example/com을 /<*>로 표현해놔서 다음과 같이 코드 실행 딴 데이터는 주석처리하면 됨------------
        if "/<*>" in template:
            template = template.replace("/<*>", "<*>")
        
        ## 백스페이스로 붙여야만 정규표현식으로 나타나지는 토큰 정의
        template = template.replace("[", "\[").replace("]", "\]").replace("(", "\(").replace(")", "\)").replace("$", "\$").replace("+", "\+").replace("?", "\?")

        ## "i-fetch......<*>" "generating core.<*>"와 같은 로그 <*>로 처리리
        template = replace_placeholder_pattern_with_asterisk(template)

        ## <*>-<*>-<*>나 <*>.<*>처럼 -나 .로 연속된 거 <*>로 처리리
        template = merge_placeholders(template)

        ## ....<*> 이나 .<*>을 <*>로 처리
        template = replace_point_with_wildcard(template)
        


        template = template.replace("<*>", "(.*)")
        template = re.compile(template)
        index_list, entity_list = finding_indexes(template, line)
        token_list = tokenizing(line)
        labeling_list = find_entity_indexes(token_list, index_list)
        if len(labeling_list)==0:
            # print(line)
            # print(template)
            # print("indexs: ", index_list)
            # print("labels: ", labeling_list)
            # print("entity: ", entity_list)
            continue
        temp_json = make_json(token_list, labeling_list, entity_list)
        if temp_json==-1:
            continue
        f2.write(temp_json + '\n')
            

