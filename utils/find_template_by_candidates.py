
from tqdm import tqdm
import random
from preprocessing import *
from grouping import *
from find_best_templates import *
from output import *


def test_data():
    tokenized_joined_logs = ['READ [ 1 , 2 ]', 'READ [ 4 , 5 , 6 ]', 'READ [ 3 , 4 ]']
    template_variable_info = dict()
    template_variable_info['READ *'] = [['READ [ 1 , 2 ]', [(1, 5)]], ['READ [ 4 , 5 , 6 ]', [(1, 7)]],
                                        ['READ [ 3 , 4 ]', [(1, 5)]]]
    template_variable_info['READ [ * , * ]'] = [['READ [ 1 , 2 ]', [(2, 2), (4, 4)]],
                                                ['READ [ 3 , 4 ]', [(2, 2), (4, 4)]]]
    template_variable_info['READ [ * , * , * ]'] = [['READ [ 4 , 5 , 6 ]', [(2, 2), (4, 4), (6, 6)]]]
    return template_variable_info, tokenized_joined_logs


def find_best_template_by_log_and_candidate_templates(logs, candidate_templates_info, group_threshold, grouping, drc_by_each_variable):
    # 완전히 중복되는 로그들이 많음
    # 이러한 로그들의 중복 계산을 막기 위해 완전 중복되는 로그들의 개수를 카운팅
    log_count_dict = counting_logs(logs)

    # 현재 wildcard_match_length 는 템플릿에 매칭되는 로그들과 그 로그들의 wildcard 부분에 대한 정보
    # 템플릿 정보가 계속 중복 저장되므로 공간, 시간적 성능 개선을 위해 템플릿 정보를 압출
    # 뒤에 활용하기 쉽게 길이 오름차순으로 정렬(general한 순으로 정렬)
    template_list = [template for template in candidate_templates_info]
    template_list.sort(key=lambda temp: len(temp))

    # 앞으로 템플릿을 직접 자료구조에 저장하기보다는 인덱스로 저장
    # 이 때, 어떤 템플릿이 어떤 인덱스인지 알 필요가 있으므로, 이를 저장한 자료구조가 필요
    # template_template_index_dict는 template을 키로 조회하면 template의 인덱스를 얻을 수 있음
    template_template_index_dict = template_list_to_dict(template_list)

    # 로그가 키, 벨류는 딕셔너리
    # 벨류 딕셔너리는 키가 템플릿 인덱스, 벨류가 이 로그가 이 템플릿을 선택했을 때의 DRC
    log_matching_template_drc_dict = find_log_matching_template_info(wildcard_match_length, drc_each_variable,
                                                                     template_template_index_dict, log_count_dict)

    # 함수 내에서 플래그 이용해서 그룹핑하지 않는 경우와 하는 경우 관리
    first_grouping_result, first_grouping_log_counts, first_grouping_logs = grouping_by_whole_candidate_template(
        log_matching_template_drc_dict, log_count_dict, using_grouping)
    second_grouping_result, second_grouping_log_counts = grouping_by_most_general_template(first_grouping_result,
                                                                                           first_grouping_log_counts,
                                                                                           using_grouping)

    # best_template_set을 구하는 함수
    min_mdl_cost, best_template_set, candidate_tuple_matching_templates = find_best_template_set(first_grouping_result,
                                                                                                 second_grouping_result,
                                                                                                 second_grouping_log_counts,
                                                                                                 template_list,
                                                                                                 GROUP_ELEMENT_THRESHOLD)

    template_occurrences = get_template_occurence(first_grouping_log_counts, candidate_tuple_matching_templates,
                                                 template_list)
    log_matching_template_dict = get_log_matching_best_template(first_grouping_logs, candidate_tuple_matching_templates,
                                                       template_list)

    return template_occurrences, log_matching_template_dict


if __name__=="__main__":
    # 사용하는 플래그 및 상수들
    # THRESHOLD를 넘어서면 그룹의 성분 수가 너무 많다고 판단, 모든 경우의 수를 고려하지 않고 다른 방법을 사용
    GROUP_ELEMENT_THRESHOLD = 10
    # 그룹핑 사용 여부에 대한 플래그
    using_grouping = True
    # drc를 각 변수에 대하여 계산하는지, 모든 변수에 대하여 계산하는지에 대한 플래그
    drc_each_variable = False
    ###################################################################################

    wildcard_match_length, log_list = test_data()

    tenmplate_occurrence, log_template_dict = find_best_template_by_log_and_candidate_templates(log_list, wildcard_match_length, GROUP_ELEMENT_THRESHOLD, using_grouping, drc_each_variable)

    print(log_template_dict)
    print(len(log_template_dict.keys()))

