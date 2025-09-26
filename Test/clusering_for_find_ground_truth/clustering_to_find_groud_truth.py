import re


def tokenizing(log):
    seps = [':', '(', ')', ';', '=', ' ', '[', ']', '{', '}', ',', '$', '<', '>', '"', "'", "@", "|", "&"]
    tokens = []
    token = ""
    length = len(log)
    # 토크나이징 기준 1: 특수문자는 다 따로
    # 토크나이징 기준 2: 앞의 토큰이 모두 숫자인데 뒤에 숫자 아닌게 나오면 토큰
    token_start = 0

    for i in range(length):
        ch = log[i]
        # ch 가 seperator면 앞의 토큰을 저장하고, ch는 따로 또 저장해야 한다
        if ch in seps:
            # seperator 중 따옴표는 기존 토큰들에 붙일 것
            # if ch == '"':
                # ddaom_count += 1
                # 끝 따옴표
                # if ddaom_count % 2 == 0:
                    # token += ch
                    # tokens.append(token)
                    # token = ""
                    # token_start = i + 1
                    # continue
                # 시작 따옴표
                # else:
                    # if len(token) != 0:
                        # tokens.append(token)
                    # token = ch
                    # token_start = i
                    # continue

            if len(token) != 0:
                tokens.append(token)
                token_start = i
            if ch != ' ':
                tokens.append(ch)
            token = ""
            token_start = i + 1
            continue
        # # ch가 seperator가 아니고 앞에 토큰이 있으면
        # if len(token) != 0:
        #     # 숫자인지 아닌지 체크할 필요가 있음
        #     if token.isnumeric() and not ch.isdigit() and ch != '.':
        #         tokens.append(token)
        #         token = ""
        if ch != ' ':
            token += ch
        # 문장의 마지막에 도달했고 저장해야 할 token이 있다면
        if i == length - 1 and len(token) != 0:
            tokens.append(token)
    return tokens


regex_list = [
                r"(?<![0-9.+-])(?>[+-]?(?:(?:[0-9]+(?:\.[0-9]+)?)|(?:\.[0-9]+)))",
                r"(?<![0-9A-Fa-f])(?:[+-]?(?:0x)?(?:[0-9A-Fa-f]+))",
                r"[A-Fa-f0-9]{8}-(?:[A-Fa-f0-9]{4}-){3}[A-Fa-f0-9]{12}",
                r"(?<![0-9])(?:(?:25[0-5]|2[0-4][0-9]|[0-1]?[0-9]{1,2})[.](?:25[0-5]|2[0-4][0-9]|[0-1]?[0-9]{1,2})[.](?:25[0-5]|2[0-4][0-9]|[0-1]?[0-9]{1,2})[.](?:25[0-5]|2[0-4][0-9]|[0-1]?[0-9]{1,2}))(?![0-9])",
                r'^[0-9a-fA-F]{8}$',
                r"(?:/[A-Za-z0-9$.+!*'(){},~:;=@#%_\-]*)+"
                r'blk_(|-)[0-9]+' , # block id
                r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
                r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
                ]

compiled_regex_list = [re.compile(regex) for regex in regex_list]


def is_variable(token):
    for cr in compiled_regex_list:
        if cr.fullmatch(token):
            return True
    return False


with open('../../log_file/spark.log') as f:
    lines = f.readlines()
    logs = [log.strip() for log in lines]
    # 중복되는 로그 제거
    logs = list(set(logs))

    # 앞의 토큰 두개를 이용해 클러스터링
    clustering_dict = dict()
    short_logs = list()

    for log in logs:
        log_tokens = tokenizing(log)

        if len(log_tokens) <= 1:
            short_logs.append(log)
        
        # 앞의 토큰이 wildcard면 *로 치환
        if is_variable(log_tokens[0]):
            log_tokens[0] = '*'
        if is_variable(log_tokens[1]):
            log_tokens[1] = '*'

        # 앞의 토큰들이 일치하는 로그들끼리 클러스터링
        if log_tokens[0] not in clustering_dict:
            clustering_dict[log_tokens[0]] = dict()
        first_grouped_logs = clustering_dict[log_tokens[0]]
        if log_tokens[1] not in first_grouped_logs:
            first_grouped_logs[log_tokens[1]] = list()
        clustered_logs = first_grouped_logs[log_tokens[1]]

        clustered_logs.append(log)
        
    cnt = 1

    # 클러스터링된 로그들 출력
    for first_token in clustering_dict:
        for second_token in clustering_dict[first_token]:
            print(f'{cnt}th cluster')
            
            for log in clustering_dict[first_token][second_token]:
                print(log)
            
            # 공백 추가
            print()
            cnt += 1

    print(f'{cnt}th cluster')
    # 마지막으로 short log들 출력
    for log in short_logs:
        print(log)


