from tqdm import tqdm
import math

# 각 로그의 개수를 카운팅하는 함수
def counting_logs(logs):
    print("로그 카운팅 시작")
    counting_dict = dict()
    for cur_log in tqdm(logs):
        if cur_log in counting_dict:
            counting_dict[cur_log] += 1
        else:
            counting_dict[cur_log] = 1
    print("로그 카운팅 완료")
    return counting_dict


# 템플릿이 key, 템플릿 인덱스가 value인 딕셔너리를 만드는 함수
def template_list_to_dict(templates):
    index_template_dict = dict()
    for i in range(len(templates)):
        index_template_dict[templates[i]] = i
    return index_template_dict

    # drc를 계산하는 함수는 각 변수당 drc를 계산
# drc를 변수별로 계산 안 할 경우, 리스트에 정보 하나만 담음
def get_var_info(current_log, var_info, drc_calculated_by_each_variable, log_count):
    tokens = current_log.split()
    var_info_list = []
    # 만약 drc를 변수별로 계산하면 각 변수 별로 character set과 변수 길이를 구해놓음
    for start, end in var_info:
        # 각 변수의 시작, 종료 인덱스를 이용해 복원한 변수
        cur_variable = ' '.join(tokens[start:end+1])
        # 각 변수들의 유일한 알파벳들과 변수의 길이를 통해 DRC를 계산해야 하므로 이와 같은 정보를 저장
        # 이 로그를 선택했을 때의 drc를 저장하는 것이므로 로그의 개수를 고려해야 함 (앞에서 중복 로그는 삭제했음)
        var_info_list.append([set(cur_variable), len(cur_variable) * log_count[current_log]])
    # drc 를 변수별로 계산하지 앟으면 모든 변수에 대하여 character set 을 구하고 변수 길이를 합함
    # 이도 리스트로 리턴
    if not drc_calculated_by_each_variable and len(var_info) != 0:
        temp_list = [set(), 0]
        for char_set, var_len in var_info_list:
            temp_list[0] = temp_list[0].union(char_set)
            temp_list[1] += var_len
        var_info_list = [temp_list]
    # 변수가 없으면 빈 리스트를 리턴
    return var_info_list


# 각 로그에 해당하는 후보 템플릿 및 변수들에 대한 (drc를 계산하기 위한) 정보를 저장
# 템플릿에 매칭되는 로그 및 로그의 변수 범위 정보를 활용하여 drc를 계산하기 위한 정보를 생성
# 리턴되는 것은 키는 로그, 벨류는 딕셔너리(키가 템플릿, 벨류는 (유일한 문자 집합(character set), 변수 길이)들의 리스트
def find_log_matching_template_info(template_variable_info, drc_calculated_by_each_variable, template_to_template_index, log_count):
    log_matching_template = dict()
    for template in template_variable_info:
        for cur_log, var_info in template_variable_info[template]:
            if cur_log not in log_matching_template:
                log_matching_template[cur_log] = dict()
            log_matching_template[cur_log][template_to_template_index[template]] = get_var_info(cur_log, var_info, drc_calculated_by_each_variable, log_count)
    return log_matching_template
