#!/bin/bash
touch log.clustering.balanced.distance.txt
for i in {0}
do
     python clustering_distance_searching.py  | tee -a log.clustering.balanced.distance.txt
   done
