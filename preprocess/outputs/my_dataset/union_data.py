datasets = ["android.jsonlines", "apache.jsonlines", "bgl.jsonlines", "cassan_spark_hadoop680.jsonlines", "hadoop.jsonlines", "hdfs.jsonlines", "hpc.jsonlines", "linux.jsonlines", 
"mac.jsonlines", "openssh.jsonlines", "openstack.jsonlines", "proxifier.jsonlines", "spark.jsonlines", "thunderbird.jsonlines", "zookeeper.jsonlines", "cassandra_additional.jsonlines"]


with open("union_datasets+cassandra_additional.jsonlines", "w") as wf:
    for dataset in datasets:
        with open(dataset, "r") as rf:
            while True:
                line = rf.readline()
                if not line:
                    break
                wf.write(line)
    