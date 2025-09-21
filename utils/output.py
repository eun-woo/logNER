# 각 템플릿에 매칭되는 후보 템플릿 튜플과 후보 템플릿 튜플의 로그수를 이용
def get_template_occurence(candidate_tuple_count, candidate_tuple_best_templates):
    best_template_occurrences = dict()
    for candidate_tuple in candidate_tuple_best_templates:
        best_template = candidate_tuple_best_templates[candidate_tuple]
        if best_template not in best_template_occurrences:
            best_template_occurrences[best_template] = 0
        best_template_occurrences[best_template] += candidate_tuple_count[candidate_tuple]
    return best_template_occurrences


# 각 후보 템플릿 튜플에 속하는 로그 정보 및 각 템플릿에 매칭되는 후보 템플릿 튜플을 이용
def get_log_matching_best_template(candidate_tuple_log_info, candidate_tuple_best_templates):
    log_matching_best_templates = dict()
    for candidate_tuple in candidate_tuple_best_templates:
        best_template = candidate_tuple_best_templates[candidate_tuple]
        for cur_log in candidate_tuple_log_info[candidate_tuple]:
            log_matching_best_templates[cur_log] = best_template

    return log_matching_best_templates
