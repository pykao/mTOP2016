#!/bin/bash
touch log.clustering.balanced.prob.txt
for i in {0..30}
do
     python mTOP2016_clusteringBalancedProbability.py  | tee -a log.clustering.balanced.prob.txt
   done
