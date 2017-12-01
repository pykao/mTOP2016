#!/bin/bash
touch log.CNNFeatureExtraction.txt
for i in {0..26}
do
     python mTOP2016_CNNFeatureExtraction.py $i  | tee -a log.CNNFeatureExtraction.txt
   done
