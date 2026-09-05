# -*- coding: utf-8 -*-
# This script collects coverage data from KLEE under different QCE configurations
# including baseline cases (no QCE)

import subprocess
from typing import Tuple
import time
import multiprocessing
import argparse
from tqdm import tqdm
import os
import csv
import matplotlib.pyplot as plt
from collections import defaultdict

COREUTILS_PROGRAMS = [
    "base64", "cat", "chcon", "chgrp", "chmod", "chown", "chroot", "cksum", "comm", "cp", "csplit", "cut",
    "date", "dd", "df", "dir", "dircolors", "dirname", "du", "echo", "env", "expand", "expr", "factor",
    "false", "fmt", "fold", "groups", "head", "hostid", "hostname", "id", "join", "kill", "link", "ln",
    "logname", "ls", "md5sum", "mkdir", "mkfifo", "mknod", "mktemp", "mv", "nice", "nl", "nohup", "nproc",
    "numfmt", "od", "paste", "pathchk", "pinky", "pr", "printenv", "printf", "ptx", "pwd", "readlink",
    "realpath", "rm", "rmdir", "runcon", "seq", "sha1sum", "sha224sum", "sha256sum", "sha384sum",
    "sha512sum", "shuf", "sleep", "sort", "split", "stat", "stdbuf", "stty", "sum", "sync", "tac", "tail",
    "tee", "test", "timeout", "touch", "tr", "true", "truncate", "tsort", "tty", "uname", "unexpand",
    "uniq", "unlink", "uptime", "users", "vdir", "wc", "who", "whoami", "yes"
]

DOCKER_EXEC_CMD = "docker exec -it"
DOCKER_IMAGE_NAME = "ccd40c636c9d"
CONCURRENCY = 48
TIMEOUTS = [60 * 60, 4 * 60 * 60, 12 * 60 * 60]
LOW_PRIORITY = "5"


def get_klee_command(program_name: str, timeout: str, use_qce: bool = False) -> str:
    if use_qce:
        klee_cmd = (
            f"/klee-build/Release+Asserts/bin/klee --posix-runtime --allow-external-sym-calls "
            f"--simplify-sym-indices --disable-inlining --optimize --only-output-states-covering-new "
            f"--use-merge=lazy --enable-exec-index=true --enable-qce=true --randomize-fork --use-random-path "
            f"--use-interleaved-covnew-NURS --use-batching-search --batch-instructions 10000 "
            f"--libc=uclibc --max-time {timeout} /coreutils-bc/{program_name}.bc --sym-args 0 1 10 --sym-args 0 2 2 "
            f"--sym-files 1 8 --sym-stdout"
        )
    else:
        klee_cmd = (
            f"/klee-build/Release+Asserts/bin/klee --posix-runtime --allow-external-sym-calls "
            f"--simplify-sym-indices --disable-inlining --optimize --only-output-states-covering-new "
            f"--use-merge=none --enable-exec-index=false --enable-qce=false --randomize-fork --use-random-path "
            f"--use-interleaved-covnew-NURS --use-batching-search --batch-instructions 10000 "
            f"--libc=uclibc --max-time {timeout} /coreutils-bc/{program_name}.bc --sym-args 0 1 10 --sym-args 0 2 2 "
            f"--sym-files 1 8 --sym-stdout"
        )

    return klee_cmd


def parse_coverage_from_stdout(stdout: str) -> Tuple[float, float]:
    total_local_coverage, local_coverage = 0, 0
    global_coverage, total_global_coverage = 0, 0

    for line in stdout.splitlines():
        if "Code coverage is " in line:
            total_local_coverage, local_coverage = line.split(
                " ")[4].split("/")[1], line.split(" ")[4].split("/")[0]
            total_global_coverage, global_coverage = line.split(
                " ")[7].split("/")[1], line.split(" ")[7].split("/")[0]

    if total_local_coverage == 0 or total_global_coverage == 0:
        return 0.0, 0.0
    else:
        return int(local_coverage) / int(total_local_coverage), int(global_coverage) / int(total_global_coverage)


def run_klee_and_get_coverage(program: str, timeout: int, use_qce: bool, output_file: str, lock: multiprocessing.Lock, set_low_priority: bool = False) -> bool:
    curr_time = int(time.time())

    if use_qce:
        docker_name = '_'.join(
            [program, 'klee_w_qce', str(timeout), str(curr_time)])
    else:
        docker_name = '_'.join(
            [program, 'klee_wo_qce', str(timeout), str(curr_time)])

    cmd = [' '.join([DOCKER_EXEC_CMD, docker_name])]
    if set_low_priority:
        cmd.append(f"nice -n {LOW_PRIORITY} " +
                   get_klee_command(program, str(timeout), use_qce=use_qce))
    else:
        cmd.append(get_klee_command(program, str(timeout), use_qce=use_qce))

    docker_run_cmd = f"docker run --name {docker_name} -dt {DOCKER_IMAGE_NAME}"
    docker_stop_cmd = f"docker stop {docker_name}"
    docker_rm_cmd = f"docker rm {docker_name}"

    cmd = " ; ".join([docker_run_cmd, " ".join(cmd),
                      docker_stop_cmd, docker_rm_cmd])
    stdout = subprocess.check_output(cmd, shell=True).decode('utf-8')
    local_coverage_rate, global_coverage_rate = parse_coverage_from_stdout(
        stdout)

    # Acquire the lock before writing to the shared results and file
    with lock:
        with open(output_file, "a+") as file:
            file.write(
                f"{program},{str(timeout)},{str(use_qce)},{local_coverage_rate},{global_coverage_rate}\n")
            file.flush()

    return True


def run_klee_and_get_coverage_wrapper(args_tuple):
    return run_klee_and_get_coverage(*args_tuple)


def update_progress(_):
    progress_bar.update()


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Dump comparison of KLEE with and without QCE")
    parser.add_argument("--task", help="Task to run", type=str)
    parser.add_argument("--input", help="Path to the input file", type=str)
    parser.add_argument(
        "--output", help="Path to the output result file", type=str)
    parser.add_argument(
        "--concurrency", help="Number of concurrent processes to run", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    if args.task == "dump":
        procs = []
        list_of_args = []

        # Create a Manager to share data between processes
        manager = multiprocessing.Manager()

        # Create a lock to synchronize access to the shared resource (text file)
        lock = manager.Lock()

        for program in COREUTILS_PROGRAMS:
            for timeout in TIMEOUTS:
                list_of_args.append(
                    (program, timeout, True, args.output, lock, True))
                list_of_args.append(
                    (program, timeout, False, args.output, lock, True))

        print(f"In total, we have {len(list_of_args)} tasks to run.")

        # Create the output file and write header if it doesn't exist
        if not os.path.exists(args.output):
            with open(args.output, 'w') as outfile:
                outfile.write(
                    "Program,Timeout,UseQCE,LocalCoverageRate,GlobalCoverageRate\n")

        # Create a multiprocessing Pool with a maximum of args.concurrency processes
        with multiprocessing.Pool(processes=args.concurrency) as pool:
            with tqdm(total=len(list_of_args), desc="Processing", unit="task") as progress_bar:
                # Use the imap_unordered function to apply run_klee_and_get_coverage_wrapper to each input arg and update the progress bar
                for _ in pool.imap_unordered(run_klee_and_get_coverage_wrapper, list_of_args, chunksize=1):
                    progress_bar.update(1)
    elif args.task == "plot":
        # Read CSV data from input file
        with open(args.input, 'r') as csvfile:
            data = list(csv.DictReader(csvfile))

        # Prepare data structure to store coverage rates
        coverage_rates = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        # Store coverage rates, filtering out rows with 0.0 values
        for row in data:
            local_coverage_rate = float(row['LocalCoverageRate'])
            global_coverage_rate = float(row['GlobalCoverageRate'])

            if local_coverage_rate == 0.0 or global_coverage_rate == 0.0:
                continue

            program = row['Program']
            timeout = int(row['Timeout'])
            use_qce = row['UseQCE'] == 'True'

            coverage_rates[program][timeout][use_qce] = {'local': local_coverage_rate, 'global': global_coverage_rate}

        # Calculate the ratios
        ratios = defaultdict(dict)
        for program in coverage_rates:
            for timeout in coverage_rates[program]:
                if True in coverage_rates[program][timeout] and False in coverage_rates[program][timeout]:
                    local_ratio = coverage_rates[program][timeout][True]['local'] / coverage_rates[program][timeout][False]['local']
                    global_ratio = coverage_rates[program][timeout][True]['global'] / coverage_rates[program][timeout][False]['global']
                    ratios[(program, timeout)] = {'local': local_ratio, 'global': global_ratio}

        # Set up plotting
        plt.figure(figsize=(10, 6))

        # Bar chart settings
        bar_width = 0.35
        bar_opacity = 0.8

        # X-axis labels
        labels = sorted(ratios.keys())

        for i, (program, timeout) in enumerate(labels):
            local_ratio = ratios[(program, timeout)]['local']
            global_ratio = ratios[(program, timeout)]['global']
            plt.bar(i * 2 + 1, local_ratio, bar_width, alpha=bar_opacity, color='b', label=f'{program} {timeout} Local')
            plt.bar(i * 2 + 2, global_ratio, bar_width, alpha=bar_opacity, color='g', label=f'{program} {timeout} Global')

        plt.xlabel('Program and Timeout Combinations')
        plt.ylabel('Ratio of Coverage Rates (QCE=True / QCE=False)')
        plt.title('Local and Global Coverage Rate Ratios by Program and Timeout')
        plt.xticks(range(1, len(labels) * 2 + 1, 2), [f'{p} {t}' for p, t in labels], rotation=45)
        plt.legend()
        plt.tight_layout()

        # Save the figure to the output file
        plt.savefig(args.output)
    else:
        print("Invalid task. Please choose from dump and plot.")
