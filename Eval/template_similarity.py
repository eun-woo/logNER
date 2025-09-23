import re
from Levenshtein import distance as lev
import pandas as pd
from tqdm import tqdm

ground_truth_path = '/raid1/eunwoo/CNN_Nested_NER/Eval/MultiLog/ground_truth_MultiLog.csv'
parser_path = '/raid1/eunwoo/CNN_Nested_NER/Eval/MultiLog/Lognroll_processed_log_templates.csv'

def sim(truth, template):
    edit_distance_sim = 1 - lev(truth, template) / max(len(truth), len(template))
    return edit_distance_sim

def delete_continuous_wildcard(temp):
    while '<*> <*>' in temp:
        temp = temp.replace('<*> <*>', '<*>')
    while '<*>.<*>' in temp:
        temp = temp.replace('<*>.<*>', '<*>')
    while '<*> , <*>' in temp:
        temp = temp.replace('<*> , <*>', '<*>')
    while '<*> : <*>' in temp:
        temp = temp.replace('<*> : <*>', '<*>')
    while '<*> | <*>' in temp:
        temp = temp.replace('<*> | <*>', '<*>')
    while '""' in temp:
        temp = temp.replace('""', '"')
    return temp

def tokenizing(log):
    seps = [':', '(', ')', ';', '=', ' ', '[', ']', '{', '}', ',', '$', '"', "'", "@", "|", "<", ">", "&", ";"]
    tokens = []
    token = ""
    length = len(log)
    # 토크나이징 기준 1: 특수문자는 다 따로
    # 토크나이징 기준 2: 앞의 토큰이 모두 숫자인데 뒤에 숫자 아닌게 나오면 토큰
    for i in range(length):
        ch = log[i]
        # ch 가 seperator면 앞의 토큰을 저장하고, ch는 따로 또 저장해야 한다
        if ch in seps:
            # seperator 중 따옴표는 기존 토큰들에 붙일 것
            if len(token) != 0:
                # .*이 포함되면 그냥 .*로
                if ".*" in token:
                    tokens.append(".*")
                else:
                    tokens.append(token)
            if ch != ' ':
                if ".*" in token:
                    tokens.append(".*")
                else:
                    tokens.append(ch)
            token = ""
            continue
        if ch != ' ':
            token += ch
        # 문장의 마지막에 도달했고 저장해야 할 token이 있다면
        if i == length - 1 and len(token) != 0:
            if ".*" in token:
                tokens.append(".*")
            else:
                tokens.append(token)
    return tokens

# def in_wildcard(tokens):
#     ret = []
#     for token in tokens:
#         if "*" in token:
#             token = ".*"
#         ret.append(token)
#     return ret

def preprocessingTemplate(template, mod):
    if mod == '2':
        template = template.replace('<*>', '.*')
        template = " ".join(tokenizing(template)).strip()
    else:
        template = " ".join(tokenizing(template)).strip()
    return template


def compileTemplate(template, mod):
    if mod == '2':
        template = template.replace('<*>', '.*')
        template = " ".join((tokenizing(template)))

    template = template.replace('\\', '\\\\')
    result = (template.replace("[", r"\[").replace("]", r"\]").replace("(", r"\(").replace(")", r"\)")
              .replace("$", r"\$").replace("+", r"\+").replace("?", r"\?")).replace("|", r"\|").replace(".", r"\.")
    # \가 추가된 .\* 처리 완료
    result = result.replace("*", r"\*")
    result = result.replace(r"\.\*", "(.*)")

    result = re.compile(result)

    return result


if __name__ == '__main__':
    print('모드를 선택해주세요')
    print("1. .*, 2. <*>")
    mode = input().strip()

    if mode != '1' and mode != '2':
        print("모드 입력이 잘못 되었습니다.")
        exit(-1)
    
    # .* 고정
    f1 = pd.read_csv(ground_truth_path)
    ground_truth = list(f1['EventTemplate'])
    ground_truth = [x.replace('"', '').strip() for x in ground_truth]

    # 컴파일
    ground_truth_compiled = [compileTemplate(delete_continuous_wildcard(x), '1') for x in ground_truth]
    ground_truth = [preprocessingTemplate(x, '1') for x in ground_truth]
    print("ground_truth 불러오기 완료, 컴파일 완료")

    # .* 또는 <*>
    # <*> 이면 토크나이징 후 디코딩
    f2 = pd.read_csv(parser_path)
    raw_parser = list(f2['EventTemplate'])
    test = [x.replace('"', '').strip() for x in raw_parser]
    # 컴파일
    test_compiled = [compileTemplate(delete_continuous_wildcard(x), mode) for x in test]
    test = [preprocessingTemplate(x, mode) for x in test]

    print("test 불러오기 완료, 컴파일 완료")

    avg_similarities = []

    for i in tqdm(range(len(test))):
        temp = test[i]
        if temp == ".*" or temp == ".* : .*" or temp == ".* = .*":
            continue
        temp_compiled = test_compiled[i]

        cur_match_cnt = 0
        cur_total_similarity = 0

        for j in range(len(ground_truth)):
            answer = ground_truth[j]
            if answer == ".*" or answer == ".* : .*" or temp == ".* = .*":
                continue
            answer_compiled = ground_truth_compiled[j]
            # print("answer: ", answer_compiled)
            # print("temp: ", temp_compiled)
            if ("PrivilegedAction" in temp and "PrivilegedAction" in answer) or answer_compiled.fullmatch(temp) or temp_compiled.fullmatch(answer):
                cur_match_cnt += 1
                cur_total_similarity += sim(answer, temp)

        if cur_match_cnt:
            avg_similarities.append(cur_total_similarity / cur_match_cnt)

        else:
            avg_similarities.append(0)
            # 매칭안되는 로그 출력하기
            print(raw_parser[i])
    # 평균 유사도의 합 / 파싱한 템플릿 개수
    print(f'similarity: {sum(avg_similarities) / len(test)}')



