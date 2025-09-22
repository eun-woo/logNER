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


# 인풋은 candidate_tuple들의 리스트
# 아웃풋은 어떤 템플릿에 어떤 candidate_tuple들이 매칭되는지 키와 벨류 셋으로 저장된 딕셔너리
# 집합인 이유는 포함 관계를 판단하기 쉽게 하기 위해서
def get_template_matched_candidate_tuple_info(candidate_tuple_list):
    # 템플릿과 템플릿에 매칭되는 candidate_tuple(1차로 그룹핑된 로그들)에 대한 정보를 저장
    # 이를 통해 템플릿들간의 포함 관계를 알아낼 것
    template_candidate_tuple_info = dict()

    for candidate_tuple in candidate_tuple_list:
        for template in candidate_tuple:
            if template not in template_candidate_tuple_info:
                template_candidate_tuple_info[template] = set()
            template_candidate_tuple_info[template].add(candidate_tuple)

    return template_candidate_tuple_info


# cur_node: 현재 템플릿이 삽입될 후보 노드, 현재 템플릿을 포함함이 확인됨
# visited: 한번 삽입된 템플릿에 다시 삽입 안 되도록 하는 visited
# cur_template: 현재 삽입되는 템플릿
def insert_into_tree(cur_node, visited, cur_template, tree, template_candidate_tuple_info):
    child_include = False
    cur_template_matching_set = template_candidate_tuple_info[cur_template]
    
    for child in tree[cur_node]:
        if child in visited:
            continue
        else:
            visited.add(child)
            child_matching_set = template_candidate_tuple_info[child]
            if cur_template_matching_set.issubset(child_matching_set):
                child_include = True
                insert_into_tree(child, visited, cur_template, tree, template_candidate_tuple_info)

    # 더 내려갈 곳이 없음
    if not child_include:
        tree[cur_node].append(cur_template)
        tree[cur_template] = list()

    return


def generating_template_tree(template_candidate_tuple_info):
    tree = dict()
    root = '*'
    
    tree[root] = list()
    
    # 템플릿에 매칭되는 candidate_tuple 수의 내림차순으로 template 리스트를 정렬
    template_sorted_by_generality = [template for template in template_candidate_tuple_info]
    template_sorted_by_generality.sort(reverse=True, key=lambda x : len(template_candidate_tuple_info[x]))
    
    # general한 순으로 먼저 트리에 삽입
    for template in template_sorted_by_generality:
        visited = set()
        insert_into_tree(root, visited, template, tree, template_candidate_tuple_info)

    return tree


def find_best_template_set_by_group_fast(candidate_tuple_list, first_grouping):
    template_candidate_tuple_info = get_template_matched_candidate_tuple_info(candidate_tuple_list)
    template_tree = generating_template_tree(template_candidate_tuple_info)

    return


# 첫번째 그룹핑 결과와 두번째 그룹핑 결과를 매개변수로 받아야 함
def find_best_template_set(first_grouping, second_grouping, second_grouping_log_count, templates, threshold):
    # 지금까지 나온 모든 DRC
    total_drc = 0
    cur_template_set = set()

    candidate_tuple_template_match = dict()

    for group_idx in tqdm(second_grouping):
        candidate_tuple_list = second_grouping[group_idx]
        temp_template_list = [-1] * len(candidate_tuple_list)

        # 그룹의 성분 개수가 많으면 브루트포스 하지 않는 함수 추가 필요
        cur_min_mdl, cur_best_templates = find_best_template_sub_set_by_bruteforce(0, temp_template_list, float("inf"), None, candidate_tuple_list, first_grouping, templates, second_grouping_log_count[group_idx])

        cur_best_template_set = set(cur_best_templates)
        cur_src = (sum(src(templates[idx]) for idx in cur_best_templates)) / len(cur_best_templates)

        # 지금은 로그 개수를 곱해서 모두 더하고, 나중에 한꺼번에 나눌 것
        total_drc += (cur_min_mdl - cur_src) * second_grouping_log_count[group_idx]
        cur_template_set = cur_template_set.union(cur_best_template_set)

        # 각 candidate tuple에 매칭되는 best_template을 구함
        # 이를 이용해 template_occurence 나 각 로그에 매칭되는 템플릿을 알 수 있음
        for i in range(len(candidate_tuple_list)):
            cur_candidate_tuple = candidate_tuple_list[i]
            cur_best_template = cur_best_templates[i]
            candidate_tuple_template_match[cur_candidate_tuple] = cur_best_template
            
    # 템플릿 개수 및 로그 개수로 평균
    total_src = sum(src(templates[idx]) for idx in cur_template_set) / len(cur_template_set)
    total_drc /= sum(second_grouping_log_count[group_idx] for group_idx in second_grouping_log_count)

    return (total_src + total_drc), cur_template_set, candidate_tuple_template_match
