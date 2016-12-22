#!/bin/bash

source /home/v-user/.bash_profile
cd /media/sf_artesto/temporal/Pine 

echo "-----------------------------" &>> log.txt 2>&1
for r in `seq 0 250`
do
echo $r
#time (python ./main.py) &>> Flog.txt 2>&1
python ./main.py
done
echo "-----------------------------" &>> log.txt 2>&1
