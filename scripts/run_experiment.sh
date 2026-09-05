#!/bin/bash

echo "Operation mode: $1"
echo "Total number of instances: $2"
echo "Graph size/variant: $3"

gen_merge_graphs() {
    for i in $(seq $1); do
        instance_id=$(($i - 1))
        echo "[BATCH] Instance ID: $instance_id -- generating MergeGraphs..."
        nohup python3 -u utils.py --task gen-merge-graphs --graph-size $2 --num-instances $1 --instance-id $instance_id >../log/gen-merge-graphs/$instance_id.log 2>&1 &
    done
}

gen_programl_graphs() {
    for i in $(seq $1); do
        instance_id=$(($i - 1))
        echo "[BATCH] Instance ID: $instance_id -- generating ProGraML graphs..."
        nohup python3 -u utils.py --task gen-programl-graphs --num-instances $1 --instance-id $instance_id >../log/gen-programl-graphs/$instance_id.log 2>&1 &
    done
}

build_klee() {
    echo "[MISC] Running KLEE..."
    python3 utils.py --task batch-run-klee-exp
}

batch_run_klee() {
    echo "[MISC] Running KLEE..."
    python3 utils.py --task batch-run-klee-exp
}

kill_klee() {
    echo "[MISC] Killing all KLEE processes..."
    python3 utils.py --task kill-klee
}

clean_up() {
    echo "[MISC] Cleaning up merge records..."
    python3 utils.py --task clean-up-logs
}

if [ "$1" = "gen-merge-graphs" ]; then
    gen_merge_graphs $2 $3
elif [ "$1" = "build-klee" ]; then
    build_klee
elif [ "$1" = "batch-run-klee" ]; then
    batch_run_klee
elif [ "$1" = "kill-klee" ]; then
    kill_klee
elif [ "$1" = "clean-up-merge-records" ]; then
    clean_up
elif [ "$1" = "gen-programl-graphs" ]; then
    gen_programl_graphs $2
fi
