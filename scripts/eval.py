import subprocess
import datetime
import multiprocessing
from sys import stderr
import traceback
import os
import time
import sys


def command(cmd):
    p = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.stdout.readlines()


def print_traceback(e):
    print(traceback.print_exception(type(e), e, e.__traceback__))


def exec(task, bc_name, arg_index):
    arg = str((arg_index+1)*5)
    cmd = f"time /data/{user_name}/NeuSE/klee-based/build/klee-dataset-pipeline/bin/{task} --simplify-sym-indices --output-module                         --max-memory=40000 --disable-inlining                         --optimize --use-forked-solver                         --use-cex-cache --libc=uclibc --posix-runtime --use-merge                         --only-output-states-covering-new --env-file=klee-test.env --run-in-dir=/data/{user_name}/NeuSE/dataset/merge-record                         --max-sym-array-size=4096 --max-solver-time=1min --max-time=999999min --external-calls=all                         --watchdog --max-static-fork-pct=0.2 --max-static-solve-pct=0.2                         --max-static-cpfork-pct=0.2 --switch-type=internal --max-memory-inhibit=false                         /data/{user_name}/NeuSE/dataset/ir-programl/{bc_name}/{bc_name}.bc --sym-arg {arg} "
    os.chdir(f"/data/{user_name}/NeuSE/dataset/ir-programl/{bc_name}")
    start = time.time()
    # print(f"cmd: {cmd} \n\n start: {start} \n\n")
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = proc.stdout.readlines()
    end = time.time()
    exec_time = round(end-start, 3)
    if task == "klee":
        res_line = ""
        for line in out:
            if b"inference overall time is" in line:
                res_line = line.decode("utf-8")
        infer_time = int(res_line.split("is ")[-1].strip("\n"))/1000
        # running our klee, the output format: coreutils-program name;modified klee/original klee;sym_arg;the time to infer the model; the time to execute the klee
        print(f"{bc_name};{task};{arg};{infer_time};{exec_time}")
    else:
        # running the original klee, the output format: coreutils-program name;modified klee/original klee;sym_arg;0(add the "0" to keep the same format with the above); the time to execute the klee
        print(f"{bc_name};{task};{arg};0;{exec_time}")
# basename;klee-original;5;0;6.014
# basename;klee-original;10;0;94.029
# basename;klee;5;115.188;120.02
# basename;klee;5;85.495;90.025


def exec_threads(bcs, arg_index):
    klee_cmd = ["klee", "klee-original"]
    # args = [5,10,15,20] # the sym_arg;
    # arg_num = len(args)
    task_num = len(klee_cmd)

    # for arg_index in range(arg_num):
    arg_list = []
    # thread_num = task_num * arg_num
    # p = multiprocessing.Pool(processes = thread_num)
    for index in range(len(bcs)):
        for task_index in range(task_num):

            # p.apply_async(exec,args = (klee_cmd[task_index], bcs[index],arg_index),error_callback=print_traceback)
            # exec(klee_cmd[task_index], bcs[index],arg_index)
            arg_list.append((klee_cmd[task_index], bcs[index], arg_index))
    arg_list = tuple(arg_list)
    with multiprocessing.Pool() as pool:
        pool.starmap(exec, arg_list)
        # p.close()
        # p.join()


all_bc = ["basename", "echo", "cat", "chmem", "cfdisk", "link",
          "ping", "diff", "mv"]  # all of the tested coreutils programs

if __name__ == '__main__':

    if len(sys.argv) == 2:
        arg_index = int(sys.argv[1])
        exec_threads(all_bc, arg_index)
    else:
        print("Please input the sym_arg(0(5),1(10),2(15),3(20))")
