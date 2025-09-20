from tqdm import tqdm
import math

# 첫번째 그룹핑
# 후보 템플릿이 같은 것들끼리 그룹핑
# 로그_템플릿_drc 정보를 활용
# 같은 그룹의 로그들은 같은 템플릿들을 선택
# 각 그룹이 어떤 템플릿을 선택하면 DRC가 얼마일지 계산하기 위해 정보를 저장
# 리턴되는 것은 키는 후보 템플릿들의 인덱스로 이루어진 튜플, 벨류는 딕셔너리
# 벨류 틱셔너리는 키는 템플릿 인덱스, 벨류는 (character_set, 변수 길이)의 리스트
def grouping_by_whole_candidate_template(log_template_variable_info, log_count):
    print("첫번째 그룹핑 중")

    # 딕셔너리의 키는 후보 템플릿 튜플
    first_group_info = dict()
    # 각 그룹의 로그 개수를 저장 (DRC 계산에 필요)
    first_group_log_count = dict()

    # cur_log는 매칭되는 로그
    # var_info는 cur_log가 이 템플릿에 매칭되었을 때 매칭되는 변수들의 (unique한 chararecter set, 변수 길이) 리스트
    for cur_log in log_template_variable_info:
        candidates = []
        for template_idx in log_template_variable_info[cur_log]:
            # candidates에 템플릿들의 인덱스를 저장
            candidates.append(template_idx)
        # 튜플의 유일성을 보장하기 위해 sort하고 튜플로 만듬
        candidates.sort()
        candidate_tuple = tuple(candidates)

        # 초기화
        if candidate_tuple not in first_group_info:
            first_group_info[candidate_tuple] = dict()
            # 각 그룹의 로그 개수를 관리
            first_group_log_count[candidate_tuple] = log_count[cur_log]
            for template_idx in log_template_variable_info[cur_log]:
                first_group_info[candidate_tuple][template_idx] = log_template_variable_info[cur_log][template_idx].copy()
        # 정보 추가
        else:
            # 각 그룹의 로그 개수를 관리
            first_group_log_count[candidate_tuple] += log_count[cur_log]
            for template_idx in log_template_variable_info[cur_log]:
                var_info_list = log_template_variable_info[cur_log][template_idx]
                cur_info_list = first_group_info[candidate_tuple][template_idx]

                # 각각의 cur_info_list[i]는 character set 과 변수 길이로 이루어짐
                for i in range(len(cur_info_list)):
                    cur_info_list[i][0].union(var_info_list[0])
                    cur_info_list[i][1] += var_info_list[1]

    print("첫번째 그룹핑 완료")
    return first_group_info, first_group_log_count


# 가장 general한 템플릿을 기준으로 그룹핑
def grouping_by_most_general_template(candidates_template_idx_drc_info, candidate_templates_log_count):
    print("두번째 그룹핑 중")
    second_group_info = dict()
    second_group_log_count = dict()

    # 템플릿 리스트가 길이 오름차순으로 정렬되었고, candidate tuple은 오름차순으로 정렬되었으므로 0번째 성분이 가장 general한 템플릿
    for candidate_tuple in candidates_template_idx_drc_info:
        if candidate_tuple[0] not in second_group_info:
            second_group_info[candidate_tuple[0]] = list()
            second_group_log_count[candidate_tuple[0]] = 0
        second_group_info[candidate_tuple[0]].append(candidate_tuple)
        second_group_log_count[candidate_tuple[0]] += candidate_templates_log_count[candidate_tuple]

    print("두번째 그룹핑 완료")
    return second_group_info, second_group_log_count


# 브루트 포스 함수의 통일을 위해, 그룹핑하지 않은 로그들에 대해서도 더미 딕셔너리 만들기
# most general template 의 인덱스 대신 -1을 키로 넣고, 이 키 하나에 모든 로그들을 넣음
# 브루트포스를 할 때는 그룹핑 한 경우, candidate_tuple을 이용해 그 candidate tuple이 템플릿을 선택했을 때의 drc를 계산
# 이 경우는 log를 이용해 그 log가 템플릿을 선택했을 떄의 drc를 계산
def make_dummy_group(log_template_idx_dict, log_counts):
    dummy_group = dict()
    dummy_count = dict()
    dummy_group[-1] = list()
    dummy_count[-1] = 0
    for log in log_template_idx_dict:
        dummy_group[-1].append(log)
        dummy_count[-1] += log_counts[log]
    return dummy_group, dummy_count