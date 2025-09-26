#!/bin/bash

for i in $(seq 1000 1000 50000)
do
    echo "$i크기 필터링 수행"
    python logNER_parsing.py --log_file ./log_file/apache_iot.log --filtering_size $i
done
