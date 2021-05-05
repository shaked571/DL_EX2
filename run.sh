#!/bin/bash
which python

for part in 1 3 4
do
 for task in ner pos
 do
   for embedding_dim in 10 20 50
   do
    for hidden_dim in 100 200 300
    do
     for l_r in 0.01 0.001 0.02
     do
      for batch_size in 1 2 8 32 128
      do
      echo "Output:"
       echo "part${part}_task${task}_hiddendim${hidden_dim}_embeddingdim${embedding_dim}_lr${l_r}_batch_size${batch_size}"

        python main.py \
        --task ${task} \
        --part $part \
        --embedding_dim $embedding_dim \
        --batch_size $batch_size \
        --hidden_dim $hidden_dim \
        --l_r $l_r

      done
     done
    done
   done
  done
 done