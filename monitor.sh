#!/bin/sh  
i=0
while true  
do
  i=$((i+1))
  echo "$i loop:"  
  klee-stats klee-last
  sleep 600  
done