from collections import deque 
from collections import defaultdict

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

span_list = find_hier(log)
result = deleting_overlapping2()
for x in result:
    span_list.remove(x)
print(span_list)


