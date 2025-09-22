import json
import re
import sys
from os import path
import pandas as pd

instance_in_head = False

def merge_placeholders(text):
    # Define a regex pattern to match "<*>" followed by "-" or "." and another "<*>"
    pattern = r'<\*>[-\.]<\*>'
    
    # Continuously replace the pattern until no more matches are found
    while re.search(pattern, text):
        text = re.sub(pattern, '<*>', text)
    
    return text

def remove_stars_except_placeholders(text):
    # Match `*` outside of `<*>` and remove them
    return re.sub(r'\*(?![^<]*>)', '', text)

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
    print(temp)
    match = temp.search(log)
    indexes = []
    entities = []

    if instance_in_head:
        for i in range(2, len(match.groups()) + 1):
            indexes.append((match.start(i), match.end(i)))
            entities.append(match.group(i))

    else:
        for i in range(1, len(match.groups()) + 1):
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
                print("why label start does not exist??")

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

    if len(labels) != len(entities):
        print("토크나이징이 잘못 되었습니다. 라벨의 개수와 entity의 개수가 다릅니다.")

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


log2 = "remainingBlocks: Set ( shuffle_9_94_2 , shuffle_9_93_2 , shuffle_9_91_2 , shuffle_9_92_2 , shuffle_9_95_2 )"
template2 = "remainingBlocks : Set ( * , * , * , * , * )"

template2 = template2.replace("[", "\[").replace("]", "\]").replace("(", "\(").replace(")", "\)").replace("$", "\$")
template2 = template2.replace("*", "(.*)")
print(template2)
template2 = re.compile(template2)
print(template2.search(log2))
# template2 = merge_placeholders(template2)
# template2 = remove_stars_except_placeholders(template2)
# template2 = template2.replace("<*>", "(.*)")
# print("log: ", log2)
# print("templte: ", template2)
# template2 = re.compile(template2)

# index_list, entity_list = finding_indexes(template2, log2)
# token_list = tokenizing(log2)
# labeling_list = find_entity_indexes(token_list, index_list)
# print("index list: ", index_list)
# print("token list: ", token_list)
# print("label list: ", labeling_list)
# print("entity list: ", entity_list)
# temp_json = make_json(token_list, labeling_list, entity_list)

# print(temp_json)