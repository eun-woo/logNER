from tqdm import tqdm
import math
def src(template):
    return len(template) * math.ceil(math.log2(len(set(template))))


def drc(var_info_list):
    drc_total = 0
    for var_info in var_info_list:
        # 각 변수의 unique한 character 수에 로그를 취하고 올림한 다음(문자당 필요 비트 수) 거기에 변수 길이를 곱함
        drc_total += math.ceil(math.log2(len(var_info[0]))) * var_info[1]
    return drc_total


# 모든 경우의 수를 고려하여 MDL을 계산 및 비교
# drc는 중복 계산이 많음, 시간 성능이 문제라면 이 부분을 어느 정도 개선 가능
# 매개변수의 의미: 현재 인덱스, 현재 선택한 템플릿들, 최소 mdl, 현재 계산된 템플릿 집합들 중 최소 mdl을 가지는 템플릿 집합, 현재 그룹에 속하는 candidate_tuple 리스트 (first_group), 각 candidate_tuple이 특정 템플릿 인덱스를 선택했을 때의 drc, 템플릿 리스트(SRC 계산용)
def find_best_template_sub_set_by_bruteforce(cur_idx, cur_templates, min_mdl, min_mdl_templates, candidate_tuple_list, candidate_tuple_template_drc_info, templates, group_log_count):
    if cur_idx == len(candidate_tuple_list):
        cur_template_set = set(cur_templates)
        # 현재 선택된 템플릿들의 SRC, src는 템플릿 개수로 평균
        cur_src = (sum(src(templates[idx]) for idx in cur_template_set)) / len(cur_template_set)
        cur_drc = 0
        # 각 candidate tuple이 각 템플릿을 선택했을 때의 DRC를 계산해서 더함
        for i in range(len(candidate_tuple_list)):
            cur_var_info = candidate_tuple_template_drc_info[candidate_tuple_list[i]][cur_templates[i]]
            cur_drc += drc(cur_var_info)
        # drc는 로그 개수로 평균
        cur_drc /= group_log_count
        # 새로운 min_mdl 과 min_mdl을 가진 템플릿 리스트를 리턴
        if min_mdl > cur_src + cur_drc:
            return (cur_src + cur_drc), cur_templates.copy()
        # 아니면 기존 것을 리턴
        else:
            return min_mdl, min_mdl_templates

    else:
        # 현재 순서의 candidate tuple에 해당하는 template들
        for template_idx in candidate_tuple_template_drc_info[candidate_tuple_list[cur_idx]]:
            # 이번엔 이걸 선택
            cur_templates[cur_idx] = template_idx
            min_mdl, min_mdl_templates = find_best_template_sub_set_by_bruteforce(cur_idx+1, cur_templates, min_mdl, min_mdl_templates, candidate_tuple_list, candidate_tuple_template_drc_info, templates, group_log_count)
        return min_mdl, min_mdl_templates


# 첫번째 그룹핑 결과와 두번째 그룹핑 결과를 매개변수로 받아야 함
def find_best_template_set(first_grouping, second_grouping, second_grouping_log_count, templates, threshold):
    # 지금까지 나온 모든 DRC
    total_drc = 0
    cur_template_set = set()

    for group_idx in second_grouping:
        candidate_tuple_list = second_grouping[group_idx]
        temp_template_list = [-1] * len(candidate_tuple_list)

        # 그룹의 성분 개수가 많으면 브루트포스 하지 않는 함수 추가 필요
        cur_min_mdl, cur_best_templates = find_best_template_sub_set_by_bruteforce(0, temp_template_list, float("inf"), None, candidate_tuple_list, first_grouping, templates, second_grouping_log_count[group_idx])

        cur_best_templates = set(cur_best_templates)
        cur_src = (sum(src(temp) for temp in cur_best_templates)) / len(cur_best_templates)

        # 지금은 로그 개수를 곱해서 모두 더하고, 나중에 한꺼번에 나눌 것
        total_drc += (cur_min_mdl - cur_src) * second_grouping_log_count[group_idx]
        cur_template_set.union(cur_best_templates)
    # 템플릿 개수 및 로그 개수로 평균
    total_src = sum(src(temp) for temp in cur_template_set) / len(cur_template_set)
    total_drc /= sum(second_grouping_log_count[group_idx] for group_idx in second_grouping_log_count)

    return (total_src + total_drc), cur_template_set