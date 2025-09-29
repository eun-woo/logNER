import re
import math
from tqdm import tqdm
import csv

############################################## Load Data ############################################################
# Parser  Structured_log.csv 파일
Dataset = "MultiLog"
parsers = ["AEL"]

output_dir = 'MDL_cost/'
# # LogNER Structured_log.csv 파일
# input_struct_file = "/raid1/eunwoo/CNN_Nested_NER/parsing_result/Spark/spark_evaluation_switched_our_structured.csv"
# ###########################################################################################################


# forward 이후 (s, e, score)와 토큰화된 문장이 필요
def tokenizing(log):
    seps = [':', '(', ')', ';', '=', ' ', '[', ']', '{', '}', ',', '$', '"', "'", "@", "|", "&"]
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
                tokens.append(token)
            if ch != ' ':
                tokens.append(ch)
            token = ""
            continue
        if ch != ' ':
            token += ch
        # 문장의 마지막에 도달했고 저장해야 할 token이 있다면
        if i == length - 1 and len(token) != 0:
            tokens.append(token)
    return tokens

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
    while '( <*> ) , ( <*> )' in temp:
        temp = temp.replace('( <*> ) , ( <*> )', '( <*> )')
    return temp


def compile_template(temp, mod):
    if mod == '1':
        temp = ' '.join(tokenizing(temp).strip())
        temp = delete_continuous_wildcard(temp)
        temp = temp.replace('\\', '\\\\')
        temp = (temp.replace("[", r"\[").replace("]", r"\]").replace("(", r"\(").replace(")", r"\)")
              .replace("$", r"\$").replace("+", r"\+").replace("?", r"\?")).replace("|", r"\|").replace(".", r"\.")
        # \가 추가된 .\* 처리 완료
        temp = temp.replace("*", r"\*")
        temp = temp.replace(r"\.\*", "(.*)")

    if mod == '2':
        token_list = tokenizing(temp)
        for j in range(len(token_list)):
            if '<*>' in token_list[j]:
                token_list[j] = '<*>'
        temp = ' '.join(token_list)

        temp = delete_continuous_wildcard(temp)
        temp = temp.replace('\\', '\\\\')
        temp = (temp.replace("[", r"\[").replace("]", r"\]").replace("(", r"\(").replace(")", r"\)")
                  .replace("$", r"\$").replace("+", r"\+").replace("?", r"\?")).replace("|", r"\|").replace(".", r"\.")
        # \가 추가된 <\*> 처리 완료
        temp = temp.replace("*", r"\*")
        temp = temp.replace(r"<\*>", "(.*)")

    return re.compile(temp)


# variable_length는 나중에 쓰려면 각 variable의 length를 이용해야 함
# 한 템플릿에서 변수의 개수는 고정이므로 리스트를 쓰는게 합리적일듯
def get_variance_info(cur_log, cur_temp):
    # print(cur_temp)
    match = cur_temp.search(cur_log)
    variable_length_list = []
    char_set = set()

    if match==None:
        return 0, 0
    for group in match.groups():
        variable_length_list.append(len(group))
        char_set |= set(group)

    return variable_length_list, char_set


# drc 계산식 이 바뀌면 이 부분을 수정하면 됨
def calculate_drc(variable_lengths, character_set):
    if len(variable_lengths) == 0:
        return 0
    else:
        total_length = sum(variable_lengths)
        bit_per_character = math.ceil(math.log2(len(character_set)))
        return total_length * bit_per_character


def calculate_src(cur_template, mod):
    if mod == '1':
        cur_template = cur_template.replace('.*', '*')
    elif mod == '2':
        cur_template = cur_template.replace('<*>', '*')
    else:
        print('calculate_src에서 mode가 잘못 됨')
        exit()

    # bit_per_character = math.ceil(math.log2(len(set(cur_template))))
    bit_per_character = 7
    return bit_per_character * len(cur_template)


if __name__ == '__main__':
    for parser in parsers:
        input_struct_file = f"/raid1/eunwoo/CNN_Nested_NER/Eval/{Dataset}/{parser}_transformed_flush_label6.log_structured.csv"
        # 템플릿 캐싱
        compiled_templates = dict()
        # drc 를 계산하는데 필요한 정보들을 저장해 놓은 딕셔너리
        # key 는 템플릿
        # value 는 (character set, variance_length_list)
        # variance_length_list를 쓰는 이유는 이후에 MDLcost 계산 방법이 수정되어도 함수만 수정하면 되도록 하기 위해서임
        drc_dict = dict()
        total_log_count = 0

        print('wildcard에 따라서 모드를 선택하시오')
        print('1. .*')
        print('2. <*>')

        mode = input().strip()

        if mode != '1' and mode != '2':
            print('mode의 설정이 잘못 되었습니다.')
            exit()

        # csv 파일을 읽어와야 함
        # 어디가 Content 인지, 어디가 EventTemplate 인지 확인하고 읽어와야 함
        # 우선 메모리가 충분하다는 가정 하에 한번에 읽어오도록 하겠음



        f = open(input_struct_file, 'r', encoding="UTF8")
        rdr = csv.reader(f)

        # 각각 원본 로그가 csv에서 몇번째 칼럼인지, 템플릿이 csv에서 몇번째 칼럼인지를 알려줌
        content_idx = -1
        template_idx = -1

        # 첫번째 라인을 활용하여 어디가 'Content' 인지 어디가 'EventTemplate' 인지 확인
        columns = next(rdr)
        for i in range(len(columns)):
            if columns[i].strip() == 'Content':
                content_idx = i
            if columns[i].strip() == 'EventTemplate':
                template_idx = i
                
        if content_idx < 0 or template_idx < 0:
            print('Content 칼럼 또는 EventTemplate 칼럼을 찾을 수 없음')
            exit()

        print('각 라인에서 원본 로그 및 템플릿 추출 중')

        # 각 라인에서 원본 로그 및 템플릿 추출
        for line in tqdm(rdr):
            log = line[content_idx].strip()
            log = ' '.join(tokenizing(log))
            template = line[template_idx].strip()

            total_log_count += 1

            # 캐시에 템플릿이 있는지 확인
            # 없다면 템플릿을 컴파일해야함
            if template not in compiled_templates:
                cur_compiled_template = compile_template(template, mode)
                compiled_templates[template] = cur_compiled_template
            else:
                cur_compiled_template = compiled_templates[template]

            cur_variable_list, cur_char_set = get_variance_info(log, cur_compiled_template)
            if cur_variable_list == 0 and cur_char_set == 0:
                continue
            # 각 변수의 변수 길이 합을 리스트로 저장
            if template not in drc_dict:
                drc_dict[template] = [cur_variable_list, cur_char_set]
            else:
                variable_list, char_set = drc_dict[template]
                for i in range(len(variable_list)):
                    variable_list[i] += cur_variable_list[i]
                char_set |= cur_char_set

        total_src = 0
        total_drc = 0
        total_template_count = len(drc_dict)

        print('mdl cost 계산 중')

        # 만들어진 drc_dict를 이용해 src, drc를 계산해야 함
        for template in tqdm(drc_dict):
            total_src += calculate_src(template, mode)
            total_variable_length, current_char_set = drc_dict[template]
            total_drc += calculate_drc(total_variable_length, current_char_set)

        mean_src = total_src / total_template_count
        mean_drc = total_drc / total_log_count

        print(f'전체 템플릿 개수는 {total_template_count}')

        with open(output_dir + f"{Dataset}_{parser}_MDL_cost.txt", "w") as output_file:
            output_file.write(f"{parser}: [{mean_src}, {mean_drc}, {mean_src + mean_drc}]\n")
        print(f'mdl_cost는 {mean_src + mean_drc}')
        print(f'평균 SRC는 {mean_src}')
        print(f'평균 DRC는 {mean_drc}')



