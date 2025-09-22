import re 
import math
log = "Adding org.apache.cassandra.config.CFMetaData@83d1ca3[cfId=0ecdaa87-f8fb-3e60-88d1-74fb36fe5c0d,ksName=system_auth,cfName=role_members,flags=[COMPOUND],params=TableParams{comment=role memberships lookup table, read_repair_chance=0.0, dclocal_read_repair_chance=0.0, bloom_filter_fp_chance=0.01, crc_check_chance=1.0, gc_grace_seconds=7776000, default_time_to_live=0, memtable_flush_period_in_ms=3600000, min_index_interval=128, max_index_interval=2048, speculative_retry=99PERCENTILE, caching={'keys' : 'ALL', 'rows_per_partition' : 'NONE'}, compaction=CompactionParams{class=org.apache.cassandra.db.compaction.SizeTieredCompactionStrategy, options={max_threshold=32, min_threshold=4}}, compression=org.apache.cassandra.schema.CompressionParams@54491733, extensions={}, cdc=false},comparator=comparator(org.apache.cassandra.db.marshal.UTF8Type),partitionColumns=[[] | []],partitionKeyColumns=[role],clusteringColumns=[member],keyValidator=org.apache.cassandra.db.marshal.UTF8Type,columnMetadata=[role, member],droppedColumns={},triggers=[],indexes=[]] to cfIdMap"

templates = ["Adding * to cfIdMap", 
"Adding org.apache.cassandra.config.CFMetaData@83d1ca3* to cfIdMap",
"Adding org.apache.cassandra.config.CFMetaData@83d1ca3[cfId=*,ksName=system_auth,cfName=*,flags=*,params=*,comparator=*,partitionColumns=*,partitionKeyColumns=*,clusteringColumns=*,keyValidator=*,columnMetadata=*,droppedColumns=*,triggers=*,indexes=*] to cfIdMap",
"Adding org.apache.cassandra.config.CFMetaData@83d1ca3[cfId=*,ksName=*,cfName=*,flags=[*],params=TableParams{comment=*, read_repair_chance=*, dclocal_read_repair_chance=*, bloom_filter_fp_chance=*, crc_check_chance=*, gc_grace_seconds=*, default_time_to_live=*, memtable_flush_period_in_ms=*, min_index_interval=*, max_index_interval=*, speculative_retry=*, caching=*, compaction=*, compression=*, extensions=*, cdc=*},comparator=comparator(org.apache.cassandra.db.marshal.UTF8Type),partitionColumns=[* | *],partitionKeyColumns=[*],clusteringColumns=[*],keyValidator=org.apache.cassandra.db.marshal.UTF8Type,columnMetadata=[*, *],droppedColumns={},triggers=[],indexes=[]] to cfIdMap",
"Adding org.apache.cassandra.config.CFMetaData@83d1ca3[cfId=*,ksName=*,cfName=*,flags=[*],params=TableParams{comment=*, read_repair_chance=*, dclocal_read_repair_chance=*, bloom_filter_fp_chance=*, crc_check_chance=*, gc_grace_seconds=*, default_time_to_live=*, memtable_flush_period_in_ms=*, min_index_interval=*, max_index_interval=*, speculative_retry=*, caching={'keys' : *, 'rows_per_partition' : *}, compaction=CompactionParams*, compression=org.apache.cassandra.schema.CompressionParams@54491733, extensions={}, cdc=false},comparator=comparator(org.apache.cassandra.db.marshal.UTF8Type),partitionColumns=[[] | []],partitionKeyColumns=[*],clusteringColumns=[*],keyValidator=org.apache.cassandra.db.marshal.UTF8Type,columnMetadata=[*, *],droppedColumns={},triggers=[],indexes=[]] to cfIdMap",
"Adding org.apache.cassandra.config.CFMetaData@83d1ca3[cfId=*,ksName=*,cfName=*,flags=[*],params=TableParams{comment=*, read_repair_chance=*, dclocal_read_repair_chance=*, bloom_filter_fp_chance=*, crc_check_chance=*, gc_grace_seconds=*, default_time_to_live=*, memtable_flush_period_in_ms=*, min_index_interval=*, max_index_interval=*, speculative_retry=*, caching={'keys' : *, 'rows_per_partition' : *}, compaction=CompactionParams{class=org.apache.cassandra.db.compaction.SizeTieredCompactionStrategy, options=*}, compression=org.apache.cassandra.schema.CompressionParams@54491733, extensions={}, cdc=false},comparator=comparator(org.apache.cassandra.db.marshal.UTF8Type),partitionColumns=[[] | []],partitionKeyColumns=[*],clusteringColumns=[*],keyValidator=org.apache.cassandra.db.marshal.UTF8Type,columnMetadata=[*, *],droppedColumns={},triggers=[],indexes=[]] to cfIdMap"
]

templates2 = []
for t in templates:
    t = t.replace("*", ".*")
    templates2.append(t)

# TRC: 템플릿이 간결할수록 값이 작아짐
def template_encoding_cost(log, template):
    n = len(template)
    N = len(set(log))
    M = 2 if ".*" in template else 0
    # print(n, N, M)
    return n * math.log2(N+M)


def extract_wildcards(template, log):
    # 정규식으로 템플릿을 변환
    regex_pattern = re.escape(template).replace(r"\.\*", "(.*)")
    # 문자열에서 추출
    match = re.match(regex_pattern, log)
    if match:
        return match.groups()  # 캡처된 그룹 반환
    else:
        return None

# # 로그 -> 템플릿로 인코딩하는데 드는 비용
# def sequence_encoding_cost(log, template, cost):
#     wildcards = extract_wildcards(template, log)
#     encoding_cost = 0
#     encoding_cost+=(len(wildcards)*cost)
#     for w in wildcards:
#         k = len(w)
#         encoding_cost+=k*cost
#     return encoding_cost


## DRC: 데이터가 구체화될수록 비용이 적어짐짐
def sequence_encoding_cost2(log, template):
    wildcards = extract_wildcards(template, log)
    encoding_cost = 0
    for w in wildcards:
        k_bit = len(bin(len(w))[2:])
        k_cost = 2*k_bit+1
        idx = len(set(log))
        idx_bit = len(bin(idx)[2:])    
        encoding_cost=(encoding_cost+k_cost+idx_bit*len(w))
    return encoding_cost
# .*이 한 개는 있다는 가정
def cal_MDL_cost(log, template, cost):
    print(1.1*template_encoding_cost(log, template), sequence_encoding_cost2(log, template), end = ' ')
    return 1.1*template_encoding_cost(log, template) + sequence_encoding_cost2(log, template)

for i, temp in enumerate(templates2):
    print('\033[95m' + "템플릿" + str(i+1) + ": " + '\033[0m' + temp)
    print('\033[96m' + "MDL 비용: " + '\033[0m' +  str(cal_MDL_cost(log, temp, 9)))
    print()

