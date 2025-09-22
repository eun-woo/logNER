import pandas as pd
import re
logfile = "/home/eunwoo/CNN_Nested_NER/parsing_result/Spark0100000_our_structured.csv"

log_df = pd.read_csv(logfile)

if "Cassandra" in logfile:
    data = []
    LineId = 1
    pattern = r"^\d+$|^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    pattern2 = r"^\d+\.$"
    for i in range(len(log_df)):
        if "Initializing" in log_df.iloc[i]["Content"] or "Compacted" in log_df.iloc[i]["Content"] or "forceFlush requested but everything is clean in" in log_df.iloc[i]["Content"]:
            tokens = log_df.iloc[i]["Content"].split()
            for idx, token in enumerate(tokens):
                if "KiB" in token or "/s" in token or "~" in token or "%" in token or "ms" in token:
                    tokens[idx] = ".*"

                if re.fullmatch(pattern, token) or re.fullmatch(pattern2, token):
                    tokens[idx] = ".*"

                # Initializing system.batches 같은 경우
                if token.strip()=="Initializing":
                    tokens[1] = ".*"

                # forceFlush requested but everything is clean in .* 같은 경우
                if token.strip()=="clean":
                    tokens[idx+2] = ".*"
            log_df.at[i, "EventTemplate"] = " ".join(tokens)
            log_df.at[i, "EventTemplate"] = log_df.iloc[i]["EventTemplate"].replace(".*", "<*>")
            log_df.at[i, "LineId"] = int(LineId)
            LineId+=1
            data.append(log_df.iloc[i])


    df = pd.DataFrame(data, columns=["LineId", "Content", "EventId", "EventTemplate"])
    df.to_csv("/home/eunwoo/CNN_Nested_NER/Test/data/cassandra_additional.log_structured.csv", index=False)

def contains_shuffle_pattern(text):
    """shuffle 뒤에 _ 또는 숫자가 있는 문자열인지 확인"""
    pattern = r"shuffle[\d_]+"
    return bool(re.search(pattern, text))

if "Spark" in logfile:
    data = []
    for i in range(len(log_df)):  
        tokens = log_df.iloc[i]["EventTemplate"].split()
        contain_shuffle = False
        for idx, token in enumerate(tokens):
            if contains_shuffle_pattern(token):
                print(i)
                tokens[idx] = ".*"
                contain_shuffle = True
        if contain_shuffle:
            log_df.at[i, "EventTemplate"] = " ".join(tokens)
            log_df.at[i, "EventTemplate"] = log_df.iloc[i]["EventTemplate"].replace(".*", "<*>")
            data.append(log_df.iloc[i])
        
    df = pd.DataFrame(data, columns=["LineId", "Content", "EventId", "EventTemplate"])
    df.to_csv("/home/eunwoo/CNN_Nested_NER/Test/data/additional/spark_additional.log_structured.csv", index=False)
# if "BGL" in logfile:
#     for i in range(len(log_df)):
#         if "Ido chip status changed" in log_df.iloc[i]["Content"]:

