#!/bin/bash
touch log.clustering.unbalanced.txt
for i in {0..20}
do
     python mTOP2016_clusteringUnbalanced.py  | tee -a log.clustering.unbalanced.txt
   done
