from collections import defaultdict
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath('__main__'))))


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


def find_parents(cur, parent):
    for child in childs[parent]:
        if child[1] >= cur[1]:
            find_parents(cur, child)
            return
    childs[parent].append(cur)


def print_tree():
    # 만들어진 트리의 결과 확인
    # childs[root] 는 root의 child tuple들의 list
    # 이 때 tuple 들은 (시작 인덱스, 종료 인덱스)
    print('root:', [" ".join(sentence[child[0]:child[1]+1]) for child in childs[root]])

    for span in span_list:
        print(f'"{" ".join(sentence[span[0]:span[1]+1])}":', [" ".join(sentence[child[0]:child[1]+1]) for child in childs[span]])


# input은 start index, end index, sigmoid (이 떄, index는 토큰 기준)
sentence = ["D", "REPL", "[", "conn2", "]", "Waiting", "for", "write", "concern.", "OpTime", ":", "{", "ts", ":", "Timestamp", "(", "0", ",", "0", ")", ",", "t", ":", "-1", "}", ",", "write", "concern", ":", "{", "w", ":", "1", ",", "wtimeout", ":", "0", "}"]
span_list = [(11, 24, 0.9), (29, 37, 0.9), (14, 19, 0.9)]
raw_log = r"D REPL     [conn2] Waiting for write concern. OpTime: { ts: Timestamp(0, 0), t: -1 }, write concern: { w: 1, wtimeout: 0 }"

# Grok pattern 들은 여기서 잡아서 찾아내야함
# 나중에 필요하면 코드 짜서 여기에 new_patterns = grok_function(raw_log) 해서 찾을 수 있음
# 그리고 new_patterns를 span_list에 추가한 후 아래의 과정들을 수행하면 됨
# 즉, 우리가 추가해야 할 것은 new_patterns = grok_function(raw_log) 와 for x in new_patters: span_list,append(x)

# overlapping 제거
brr = deleting_overlapping()

# 너무 시간을 많이 잡아먹으면?
# set으로 바꾼뒤 차집합 구하면 O(n)
for x in brr:
    span_list.remove(x)

# overlapping entity 가 없는 span_list를 이용해 트리 생성
childs = defaultdict(list)
# 실제로는 받은 토큰 리스트의 마지막 인덱스를 이용해 생성
root = (0, len(sentence) - 1)

span_set = {span[:2] for span in span_list}

for i in range(0, len(sentence)):
    span_set.add((i, i))

span_list = list(span_set)
span_list.sort(key=lambda x: (x[0], -x[1]))

for span in span_list:
    find_parents(span, root)

print_tree()







