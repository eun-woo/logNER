import json
import os
import pandas as pd
import re
import string
# from sklearn.utils import shuffle
# import textdistance
# import random
# import heapq
# from collections import Counter, defaultdict, deque, OrderedDict
# from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
# import time
# import calendar
# import argparse
# import numpy as np

log_file = "/home/eunwoo/logdeep/data/sampling_example/bgl/bgl2_100k"

datasets = [
    "Apache",
    "BGL",
    "Hadoop",
    "HDFS",
    "HealthApp",
    "HPC",
    "Linux",
    "Mac",
    "OpenSSH",
    "OpenStack",
    "Proxifier",
    "Spark",
    "Thunderbird",
    "Zookeeper"
]

benchmark = {
    'HDFS': {
        'log_file': 'HDFS/HDFS_full.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>'
    },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_full.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>'
    },

    'Spark': {
        'log_file': 'Spark/Spark_full.log',
        'log_format': '<Level> <Component>: <Content>'
    },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_full.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>'
    },

    'BGL': {
        'log_file': 'BGL/BGL_full.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'
    },

    'HPC': {
        'log_file': 'HPC/HPC_full.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>'
    },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_full.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>'
    },

    'Windows': {
        'log_file': 'Windows/Windows_full.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>'
    },

    'Linux': {
        'log_file': 'Linux/Linux_full.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>'
    },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_full.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>'
    },

    'Apache': {
        'log_file': 'Apache/Apache_full.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>'
    },

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_full.log',
        'log_format': '\[<Time>\] <Program> - <Content>'
    },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_full.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>'
    },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_full.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>'
    },

    'Mac': {
        'log_file': 'Mac/Mac_full.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>'
    },

    'Cassandra': {
            'log_file': 'cassandra/cassandra_full.log',
            'log_format': '<Level> <Name> <Date> <Time> <File> - <Content>'
    },

    'MultiLog': {
        'log_file': 'multilog/multilog_full.log',
        'log_format': '<Date> <Time> <Content>'
    }
}


def generate_logformat_regex(logformat):
    """Function to generate regular expression to split log messages"""
    headers = []
    splitters = re.split(r"(<[^<>]+>)", logformat)
    regex = ""
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(" +", "\\\s+", splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip("<").strip(">")
            regex += "(?P<%s>.*?)" % header
            headers.append(header)
    regex = re.compile("^" + regex + "$")
    return headers, regex

def log_to_dataframe(log_file, regex, headers, logformat):
    """Function to transform log file to dataframe"""
    log_messages = []
    linecount = 0
    with open(log_file, "r") as fin:
        for line in fin.readlines():
            try:
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]
                log_messages.append(message)
                linecount += 1
            except Exception as e:
                print("[Warning] Skip line: " + line)
    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(0, "LineId", None)
    logdf["LineId"] = [i + 1 for i in range(linecount)]
    print("Total lines: ", len(logdf))
    return logdf

if __name__=="__main__":
    headers, regex = generate_logformat_regex(benchmark["Cassandra"]["log_format"])
    df_log = log_to_dataframe(log_file, regex, headers, benchmark["Cassandra"]["log_format"])

    print(df_log['Time'])
    print(len(set(list(df_log["Content"]))))
