import json
import gc
from collections import defaultdict


# 트리의 root
root = '*'


# Schema Representation Cost
# naive하게 글자 수에 비례하게 계산
# ####이후 수정이 필요하다면 수정 ####
def src(template):
    return 8 * len(template)


# 주어진 template의 주어진 로그 log에 대한 drc
# variable length도 naive하게 구현되어 있음
# 이것 역시 naive하게 변수 길이에 비례하게 계산
# #### 이후 수정이 필요하다면 수정 ####
def drc(template, log, template_variable_dict):
    variable_length = template_variable_dict[template][log]
    return 8 * variable_length


# x는 부모 candidate
# y는 자식 candidate
def include(template1, template2, template_log_set):
    set1 = template_log_set[template1]
    set2 = template_log_set[template2]
    if set2.issubset(set1):
        return True
    else:
        return False


# 템플릿간의 교집합을 리턴하는 함수
def intersection(template1, template2, template_log_set):
    set1 = template_log_set[template1]
    set2 = template_log_set[template2]
    c = set1.intersection(set2)
    return c


# general한걸 먼저 삽입
# 따라서 나중에 들어온 템플릿이 먼저 들어온 템플릿의 부모가 될 경우를 생각하지 않음
def insert_template(template, tree_dict, template_log_set, template_variable_dict):
    # 이미 트리에 추가된 템플릿은 추가할 필요 없음
    if template in tree_dict:
        return

    # root에서부터 시작
    current_node = root
    # template 의 위치를 찾았는지에 대한 플래그
    found_insert_position = False

    # 여기서 포함 관계를 확인하기 위하여 로그 템플릿 - 로그 집합 딕셔너리가 필요
    # 로그 집합이 set이어야 연산이 수월
    while not found_insert_position:
        broken = False
        for child in tree_dict[current_node]:
            # child가 템플릿의 부모인 경우
            if include(child, template, template_log_set):
                current_node = child
                broken = True
                break
            # child와 템플릿 사이에 intersection이 있는 경우, 단 부모 자식 관계는 아닌 경우
            # 이 함수는 부모 자식 관계인지는 판단하지 않고 교집합이 있는지만 판단함
            # 들어온 쪽에서 겹치는 로그를 삭제하는게 안전
            # #### 잘못 만들어진 템플릿일 수도 있으므로 나중에 삭제하는 방법도 생각 가능, 교집합에 대한 처리를 다르게 할 수도 있다 ####
            else:
                c = intersection(child, template, template_log_set)
                # 템플릿에서 로그로 매칭되는 set과 딕셔너리 모두에서 교집합에 있는 로그들을 제거
                for log in c:
                    template_log_set[template].remove(log)
                    del template_variable_dict[template][log]

        if not broken:
            found_insert_position = True

    # insert position을 찾았으므로 트리에 노드 삽입
    if found_insert_position:
        tree_dict[current_node].append(template)
        tree_dict[template] = []


# current_node는 현재 노드의 '템플릿' 을 의미
def find_best_mdl_cost(tree_dict, current_node, marking, visited, template_log_set, template_variable_dict):

    mdl_child = 0
    for child in tree_dict[current_node]:
        mdl_child += find_best_mdl_cost(tree_dict, child, marking, visited, template_log_set, template_variable_dict)
    # root의 * 은 실제 사용 X
    # #### 나중에 사용하도록 구현될 수도 있음 ####
    if current_node == root:
        marking[root] = 1
        return mdl_child

    current_node_set = template_log_set[current_node]
    src_current = src(current_node)
    # child에 매칭되는 로그들에 대한 cost
    mdl_current1 = src_current
    # child에 매칭되지 않는 로그들에 대한 cost
    mdl_current2 = src_current

    # child가 없으면 mdl_child 가 0으로 계산되는 문제 개선
    if len(tree_dict[current_node]) == 0:
        mdl_child = float("inf")

    # child에 매칭되지 않은 로그의 카운트
    not_child_matching_cnt = 0

    # child에서 방문되지 않은 로그들은 childs에는 매칭되지 않고 current_node에는 매칭되는 로그들
    for log in current_node_set:
        if log not in visited:
            not_child_matching_cnt += 1
            mdl_current2 += drc(current_node, log, template_variable_dict)
            visited.add(log)
        else:
            mdl_current1 += drc(current_node, log, template_variable_dict)

    # 비교 대상은 mdl_current1 과 mdl_child
    # chiid와 current_node 둘 다에 매칭되는 로그들로 비교해야 child와 current_node 중 어느 쪽이 나은지에 대한 선택이 가능

    # child가 더 낫다는 선택
    if mdl_current1 > mdl_child:
        # current_node와 child node 들간의 매칭되는 로그 집합이 같음
        if not_child_matching_cnt == 0:
            marking[current_node] = 1  # 1은 이 템플릿은 포함되지 않는다는 듰
            return mdl_child
        # current_node와 child node 들간의 매칭되는 로그 집합이 같지 않음. 즉, 이 템플릿을 사용은 해야 함
        else:
            marking[current_node] = 2
            return mdl_child + mdl_current2
    # current node 가 더 낫다는 선택
    else:
        marking[current_node] = 3
        # src_current가 두번 중복하여 더해졌으므로, src_current는 생각 X
        return mdl_current1 + mdl_current2 - src_current


# 마킹 정보를 이용하여 최적의 템플릿 세트를 main에 선언된 template_list 배열에 저장
def find_best_template_set(current_node, marking, tree_dict, template_list):
    for child in tree_dict[current_node]:
        if marking[child] == 1:
            find_best_template_set(child, marking, tree_dict, template_list)
        # child가 자신과 매칭되는 로그들 중 일부를 커버
        elif marking[child] == 2:
            template_list.append(child)
            find_best_template_set(child, marking, tree_dict, template_list)
        # child가 자신과 매칭되는 로그들 전체를 커버
        # 더 이상 아래로 내려가지 않아도 됨
        else:
            template_list.append(child)
