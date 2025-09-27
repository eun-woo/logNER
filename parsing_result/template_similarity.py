import re
from Levenshtein import distance as lev
import pandas as pd
from tqdm import tqdm
import argparse





def sim(truth, template):
    edit_distance_sim = 1 - lev(truth, template) / max(len(truth), len(template))
    return edit_distance_sim

def delete_continuous_wildcard(temp):
    while '""' in temp:
        temp = temp.replace('""', '"')
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


def open_files_and_compile():
    # .* 고정
    file1 = pd.read_csv(ground_truth_path)
    gt = list(file1['EventTemplate'])
    gt = [x.replace('"', '').strip() for x in gt]

    # 컴파일
    gtc = [compileTemplate(delete_continuous_wildcard(x), '1') for x in gt]
    gt = [preprocessingTemplate(x, '1') for x in gt]
    print("ground_truth 불러오기 완료, 컴파일 완료")

    # .* 또는 <*>
    # <*> 이면 토크나이징 후 디코딩
    file2 = pd.read_csv(parser_path)
    rp = list(file2['EventTemplate'])
    t = [x.replace('"', '').strip() for x in rp]
    # 컴파일
    tc = [compileTemplate(delete_continuous_wildcard(x), mode) for x in t]
    t = [preprocessingTemplate(x, mode) for x in t]

    print("test 불러오기 완료, 컴파일 완료")

    return gt, gtc, t, tc, rp


if __name__ == '__main__':
    scores = []
    for filtering_size in range(1000, 51000, 1000):
        ground_truth_path = f'/raid1/eunwoo/logNER/Eval/multilog/ground_truth_multilog_evaluation.log_templates.csv'
        parser_path = f'/raid1/eunwoo/logNER/parsing_result/{filtering_size}_result_templates.csv'
        mode = '1'
        print(filtering_size, "크기 filtering size")
        ground_truth, ground_truth_compiled, test, test_compiled, raw_parser = open_files_and_compile()

        # 템플릿들은 현재 서로 매칭 가능하도록 준비된 상태

        # 각 템플릿의 평균 유사도
        avg_similarities = []

        # 템플릿들의 매칭 정보를 적기 위한 딕셔너리
        # 그라운드 트루쓰의 어떤 템플릿이 매칭이 몇번 되었는지를 기록
        gt_match_count = dict()

        # 어떤 파서 템플릿에 어떤 그라운드 템플릿이 매칭되었고 유사도는 얼마인지 기록
        ft_match_pt_similarity = dict()

        not_matching_cnt = 0

        # test는 토크나이징을 완료한 파서가 파싱한 템플릿들
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
                # 둘 이 서로 매치된다면
                if ("PrivilegedAction" in temp and "PrivilegedAction" in answer) or answer_compiled.fullmatch(temp) or temp_compiled.fullmatch(answer):
                    cur_match_cnt += 1
                    cur_total_similarity += sim(answer, temp)

                    # 개선된 스코어를 계산하기 위한 정보 추가
                    if answer not in gt_match_count:
                        gt_match_count[answer] = 0
                    gt_match_count[answer] += 1

                    if temp not in ft_match_pt_similarity:
                        ft_match_pt_similarity[temp] = dict()
                    ft_match_pt_similarity[temp][answer] = sim(answer, temp)

            if cur_match_cnt:
                avg_similarities.append(cur_total_similarity / cur_match_cnt)

            else:
                avg_similarities.append(0)
                not_matching_cnt += 1
                # 매칭안되는 로그 출력하기`
                print(raw_parser[i])
        # 평균 유사도의 합 / 파싱한 템플릿 개수
        print(f'similarity: {sum(avg_similarities) / len(test)}')

        # 개선된 스코어 계산
        total_similarity = 0

        for temp in ft_match_pt_similarity:
            cur_cnt = 0
            cur_total_sim = 0
            for gt in ft_match_pt_similarity[temp]:
                cur_sim = ft_match_pt_similarity[temp][gt]
                # ground truth 에 매칭된 템플릿 개수로 우선 스코어를 나누기
                cur_sim /= gt_match_count[gt]
                cur_total_sim += cur_sim
            # cur_total_sim을 현재 템플릿에 매칭된 ground_truth 수로 나누기
            cur_total_sim /= (len(ft_match_pt_similarity[temp]) ** 2)
            # 이를 total_similarity에 더하기
            total_similarity += cur_total_sim

        # total_similarity 평균내기
        print(f'new similarity: {total_similarity / (len(test) - not_matching_cnt)}')
        scores.append((f"{filtering_size}크기 score: {total_similarity / (len(test) - not_matching_cnt)}"))
    print("\n".join(list(scores)))





