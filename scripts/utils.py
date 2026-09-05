import os
import programl as pg
import argparse
from pathlib import Path
import json
import networkx as nx
from enum import Enum
from collections import Counter
from copy import deepcopy
import matplotlib.pyplot as plt
import random
import shutil
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
import matplotlib.pyplot as plt

random.seed(66666)  # fix seed for reproducibility

# for what ratio of all blocks should we traverse from the merge point
# Note the unit here is Basic Block
MAX_NUM_BLOCKS_TO_TRAVERSE_RATIO = 0.000005
MIN_NUM_BLOCKS_TO_TRAVERSE = 15

# Note the unit here is Node
MAX_LENGTH_OF_PATH_RATIO = 0.000005
MIN_LENGTH_OF_PATH = 15

# Filter LLVM nodes in generating MergeGraphs
FILTER_LLVM_NODES = True

# Whether to use unmodified LLVM (if False, use KLEE-generated ones)
USE_UNMODIFIED_LLVM = False

# TODO: Why do we have isolated nodes? There should not be any isolated nodes.
MAX_ISOLATES_RATIO = 0.001

TRAIN_SET_RATIO = 0.80
VAL_SET_RATIO = 0.10
TEST_SET_RATIO = 0.10
# Floating point comparison
assert abs(TRAIN_SET_RATIO + VAL_SET_RATIO + TEST_SET_RATIO - 1.0) < 0.000001, "[Utils][FATAL] Specified ratios for sets are invalid: %f!" % (
    TRAIN_SET_RATIO + VAL_SET_RATIO + TEST_SET_RATIO)

PER_KLEE_RAM_GB = 40

COMPILER_VERSION = {
    "LLVM_6_0": '6',
    "LLVM_3_8": '3.8',
    "LLVM_10_0": '10'
}

VALID_TASKS = {
    "convert-bc-to-ir",
    "convert-ir-to-graph",
    "copy-bitcode-files",
    "copy-assembly-ir-files",
    "test-run-klee-exp",
    "quick-test-run-klee-exp",
    "batch-run-klee-exp",
    "build-klee",
    "build-auto-merge-klee",
    "rebuild-klee",
    "build-programl",
    "kill-klee",
    "clean-up-records",
    "clean-up-dataset",
    "clean-up-klee-files",
    "parse-merge-jsons",
    "gen-programl-graphs",
    "gen-merge-graphs",
    "split-dataset",
    "split-dataset-by-program",
    "split-dataset-by-merge-point",
    "get-dataset-stats",
    "visualize-loss-curve",
    "analyze-merge-records",
    "compile-program-list",
    "gen-vocab",
    "test-auto-merge",
    "visualize-a-merge-graph",
    "test-gen-merge-graph",
}

KLEE_BUILD_DIR = os.path.abspath("../klee-based/build/State_Merge_klee-build/")
KLEE_AUTO_MERGE_BUILD_DIR = os.path.abspath(
    "../klee-based/build/klee_auto_merge/")
KLEE_BENCHMARK_DIR = os.path.abspath("../klee-based/benchmark/")
KLEE_DATASET_DIR = os.path.abspath("../klee-based/dataset/")
DATA_DIR = os.path.abspath("../dataset/")
FIG_DIR = os.path.join(DATA_DIR, "figs")
PROGRAML_DIR = os.path.abspath("../ProGraML/")
COREUTILS_BC_DIR = os.path.abspath(
    "../klee-based/benchmark/coreutils/obj-llvm/src/")
KLEE_BIN_DIR = os.path.join(KLEE_BUILD_DIR, "bin/klee")
KLEE_AUTO_MERGE_BIN_DIR = os.path.join(KLEE_AUTO_MERGE_BUILD_DIR, "bin/klee")
MERGE_RECORD_DIR = os.path.join(DATA_DIR, "merge-record")
MERGE_GRAPH_DIR = os.path.join(DATA_DIR, "merge-graph")
IR_PROGRAML_DIR = os.path.join(DATA_DIR, "ir-programl")
LOG_DIR = os.path.abspath("../log/")

TEMPLATE_RUN_KLEE_CMD = "{klee_fpath} --simplify-sym-indices --output-module \
                        --max-memory=40000 --max-merge-state-search-memory=25000 --disable-inlining \
                        --optimize --use-forked-solver \
                        --use-cex-cache --libc=uclibc --posix-runtime --use-merge \
                        --only-output-states-covering-new --env-file=klee-test.env --run-in-dir={klee_exp_dir} \
                        --max-sym-array-size=4096 --max-solver-time=1min --max-time=720min --external-calls=all \
                        --watchdog --max-static-fork-pct=0.2 --max-static-solve-pct=0.2 \
                        --max-static-cpfork-pct=0.2 --switch-type=internal --max-memory-inhibit=false \
                        --max-static-fork-inst=10 {bc_fpath} {sym_arg_list}"
TEMPLATE_RUN_AUTO_MERGE_KLEE_CMD = "{klee_fpath} --simplify-sym-indices --output-module \
                        --max-memory=40000 --disable-inlining \
                        --optimize --use-forked-solver \
                        --use-cex-cache --libc=uclibc --posix-runtime --use-merge \
                        --only-output-states-covering-new --env-file=klee-test.env --run-in-dir={klee_exp_dir} \
                        --max-sym-array-size=4096 --max-solver-time=1min --max-time=720min --external-calls=all \
                        --watchdog --max-static-fork-pct=0.2 --max-static-solve-pct=0.2 \
                        --max-static-cpfork-pct=0.2 --switch-type=internal --max-memory-inhibit=false \
                        {bc_fpath} {sym_arg_list}"
TEMPLATE_QUICK_RUN_KLEE_CMD = "{klee_fpath} --only-output-states-covering-new --optimize \
                              --libc=uclibc --posix-runtime --use-merge {bc_fpath} {sym_arg_list}"
TEMPLATE_BUILD_KLEE_CMAKE_CMD = 'cmake -DCMAKE_BUILD_TYPE=Debug \
                                -DLLVM_CONFIG_BINARY=../../install/bin/llvm-config \
                                -DENABLE_SOLVER_Z3=ON \
                                -DZ3_INCLUDE_DIRS=../../install/include \
                                -DZ3_LIBRARIES=../../install/lib/libz3.so \
                                -DENABLE_UNIT_TESTS=OFF \
                                -DENABLE_SYSTEM_TESTS=OFF \
                                -DENABLE_TCMALLOC=OFF \
                                -DENABLE_POSIX_RUNTIME=ON \
                                -DENABLE_KLEE_UCLIBC=ON  \
                                -DKLEE_UCLIBC_PATH=../../source/klee-uclibc \
                                -G "CodeBlocks - Unix Makefiles" \
                                ../../source/State_Merge_klee'
TEMPLATE_BUILD_AUTO_MERGE_KLEE_CMAKE_CMD = 'cmake -DCMAKE_BUILD_TYPE=Debug \
                                            -DLLVM_CONFIG_BINARY=../../install/bin/llvm-config \
                                            -DENABLE_SOLVER_Z3=ON \
                                            -DZ3_INCLUDE_DIRS=../../install/include \
                                            -DZ3_LIBRARIES=../../install/lib/libz3.so \
                                            -DENABLE_UNIT_TESTS=OFF \
                                            -DENABLE_SYSTEM_TESTS=OFF \
                                            -DENABLE_POSIX_RUNTIME=ON \
                                            -DENABLE_KLEE_UCLIBC=ON  \
                                            -DKLEE_UCLIBC_PATH=../../source/klee-uclibc \
                                            -DCMAKE_PREFIX_PATH=INSTALL_PATH_OF_GRPC \
                                            -G "CodeBlocks - Unix Makefiles" ../../source/klee-dataset-pipeline/'
TEMPLATE_BUILD_KLEE_MAKE_CMD = "make -j64"
SYM_ARGS_DICT = {
    "default": "--sym-args 0 1 10 --sym-args 0 2 2 --sym-files 1 8 --sym-stdin 8 --sym-stdout",
    "dd": "--sym-args 0 3 10 --sym-files 1 8 --sym-stdin 8 --sym-stdout",
    "dircolors": "--sym-args 0 3 10 --sym-files 2 12 --sym-stdin 12 --sym-stdout",
    "echo": "--sym-args 0 4 300 --sym-files 2 30 --sym-stdin 30 --sym-stdout",
    "expr": "--sym-args 0 1 10 --sym-args 0 3 2 --sym-stdout",
    "mknod": "--sym-args 0 1 10 --sym-args 0 3 2 --sym-files 1 8 --sym-stdin 8 --sym-stdout",
    "od": "--sym-args 0 3 10 --sym-files 2 12 --sym-stdin 12 --sym-stdout",
    "pathchk": "--sym-args 0 1 2 --sym-args 0 1 300 --sym-files 1 8 --sym-stdin 8 --sym-stdout",
    "printf": "--sym-args 0 3 10 --sym-files 2 12 --sym-stdin 12 --sym-stdout",
}
COREUTILS_FILENAMES = [
    "base64", "basename", "cat", "chcon", "chgrp", "chmod", "chown", "chroot", "cksum", "comm", "cp", "csplit", "cut",
    "date", "dd", "df", "dircolors", "dirname", "du", "echo", "env", "expand", "expr", "factor", "false", "fmt", "fold",
    "head", "hostid", "hostname", "id", "ginstall", "join", "kill", "link", "ln", "logname", "ls", "md5sum", "mkdir",
    "mkfifo", "mknod", "mktemp", "mv", "nice", "nl", "nohup", "od", "paste", "pathchk", "pinky", "pr", "printenv", "printf",
    "ptx", "pwd", "readlink", "rm", "rmdir", "runcon", "seq", "setuidgid", "shred", "shuf", "sleep", "sort", "split",
    "stat", "stty", "sum", "sync", "tac", "tail", "tee", "touch", "tr", "tsort", "tty", "uname", "unexpand", "uniq", "unlink",
    "uptime", "users", "wc", "whoami", "who", "yes",
]
ENV_CMD = "env -i /bin/bash -c '(source ../../../scripts/klee-testing-env.sh; env >klee-test.env)'"

MERGE_JSON_FIELDS = {
    "IR_FILENAME": "IR_filename",
    "MERGE_ID": "Merge_ID",
    "MERGE_POINT_ADDR": "Merge_point_addr",
    "MERGED_TIME": "Merged_exploration_time",
    "UNMERGED_TIME": "Unmerged_exploration_time ",
    "MERGED_PATHS": "Merged_paths",
    "MERGED_VARS": "Merged_variables",
    "GLOBAL_MEM_ADDRS": "Global_memories",
    "GLOBAL_VARS": "Global_variables",
    "LOCAL_VARS": "Local_variables",
    "PREV_MERGE_POINT1": "Prev_merge_point_00",
    "PREV_MERGE_POINT2": "Prev_merge_point_01",
}

PROGRAML_ATTRS = {
    "TEXT": "text",
    "FEATURES": "features",
    "BLOCK_ID": "block",
    "INST_ID": "inst_id",
    "OP_INST_ID": "op_inst_id",
    "CONST_INST_ID": "const_inst_id",
    "ARG_INST_ID": "arg_inst_id",
    "BLOCK_ENTRY_EXIT": "block_entry_exit",
}

_GLOBAL_FUTURE_GRAPH_CACHE = {}


class MergeGraphVarInclusion(Enum):
    ONLY_MERGED = 1
    ALL = 2


VAR_INCLUSION_MAP = {
    MergeGraphVarInclusion.ONLY_MERGED: "only-merged",
    MergeGraphVarInclusion.ALL: "all",
}


class ProgramlNodeType(Enum):
    INST = 1
    OP = 2
    ARG = 3
    CONST = 4


class MergeGraphVariant(Enum):
    ALL_VARS = 1
    ONLY_MERGED_VARS = 2


# SPLIT_VAR_INCLUSION = MergeGraphVariant.ONLY_MERGED_VARS
SPLIT_VAR_INCLUSION = MergeGraphVariant.ALL_VARS


class MergeDecision(Enum):
    SHOULD_NOT_MERGE = 0
    SHOULD_MERGE = 1


class TooManyIsolatedNodesError(Exception):
    pass


class NoProGraMLGraphFound(Exception):
    pass


class MergePointNotConnectedError(Exception):
    pass


class MergeRecord(object):
    def __init__(
        self,
        ir_fname,
        merge_id,
        merge_point_addr,
        merged_paths,
        merged_vars,
        merged_time,
        unmerged_time,
        prev_merge_point1,
        prev_merge_point2,
        target_addr=None,
        distance=None,
    ):
        self.ir_fname = ir_fname
        self.merge_id = merge_id
        self.merge_point_addr = merge_point_addr
        self.merged_paths = merged_paths
        self.merged_time = float(merged_time)
        self.unmerged_time = float(unmerged_time)
        self.target_addr = target_addr
        self.distance = distance
        self.prev_merge_point1 = prev_merge_point1
        self.prev_merge_point2 = prev_merge_point2

        self.merged_vars = merged_vars
        if merged_vars is not None:
            self.merged_vars = {}
            self.merged_vars["GLOBAL_MEM_ADDRS"] = merged_vars.get(
                MERGE_JSON_FIELDS["GLOBAL_MEM_ADDRS"])
            self.merged_vars["GLOBAL_VARS"] = merged_vars.get(
                MERGE_JSON_FIELDS["GLOBAL_VARS"])
            self.merged_vars["LOCAL_VARS"] = merged_vars.get(
                MERGE_JSON_FIELDS["LOCAL_VARS"])

    def __repr__(self) -> str:
        fields = ["======= Printing Merge Record ======="]
        fields.append("IR_FILENAME: %s" % self.ir_fname)
        fields.append("MERGE_ID: %s" % self.merge_id)
        fields.append("MERGE_POINT_ADDR: %s" % self.merge_point_addr)
        fields.append("MERGED_TIME: %f" % self.merged_time)
        fields.append("UNMERGED_TIME: %f" % self.unmerged_time)

        fields.append("MERGED_PATHS: ")
        for path_id, path in self.merged_paths.items():
            fields.append("  MERGED_ID %s: %d" % (path_id, len(path)))

        if self.merged_vars is not None:
            fields.append("MERGED_VARS: ")
            for var_type, vars in self.merged_vars.items():
                if vars is None:
                    fields.append("  VAR TYPE %s: %s" % (var_type, vars))
                else:
                    fields.append("  VAR TYPE %s: %d" % (var_type, len(vars)))
        else:
            fields.append("MERGED_VARS: %s" % self.merged_vars)

        fields.append("=====================================")
        repr = '\n'.join(fields)
        return repr


def convert_bc_to_ir(bc_dir, debug=True):
    fnames = os.listdir(bc_dir)
    for fname in fnames:
        if not fname.startswith('.'):
            print("[Utils][INFO] Processing Bitcode file: %s" % fname)
            bc_fpath = os.path.join(bc_dir, fname, fname + '.bc')
            cmd = "llvm-dis-6.0 %s" % bc_fpath
            if debug:
                print("[Utils][INFO] Issuing Shell command: %s" % cmd)
            os.system(cmd)


# Needs to be defined outside of the function (why?)
def func_to_extract_graph(ir_fpath, out_graph_fpath):
    with open(ir_fpath, 'r') as fin:
        ir_str = fin.read()
        graph = pg.from_llvm_ir(
            ir_str, version=COMPILER_VERSION["LLVM_6_0"], timeout=43200)
        pg.save_graphs(path=Path(out_graph_fpath), graphs=[graph])


def convert_ir_to_graph(ir_dir, out_dir):
    fnames = os.listdir(ir_dir)
    pool = multiprocessing.Pool(processes=96)

    for fname in fnames:
        if not fname.startswith('.'):
            if not USE_UNMODIFIED_LLVM:
                ir_fname = fname + '_assembly.ll'
            else:
                ir_fname = fname + '.ll'
            print("[Utils][INFO] Processing IR file: %s" % ir_fname)
            ir_fpath = os.path.join(ir_dir, fname, ir_fname)
            bin_name = fname.replace('.ll', '')
            if USE_UNMODIFIED_LLVM:
                out_graph_fpath = Path(os.path.join(
                    out_dir, fname, bin_name + '.graph.unmodified'))
            else:
                out_graph_fpath = Path(os.path.join(
                    out_dir, fname, bin_name + '.graph'))
            if os.path.exists(out_graph_fpath):
                print("[Utils][INFO] Graph file already exists: %s" %
                      out_graph_fpath)
                continue
            pool.apply_async(func_to_extract_graph,
                             args=(ir_fpath, out_graph_fpath),
                             error_callback=print)

    pool.close()
    pool.join()


def find_and_copy_all_bitcode_files(bc_dir, out_dir, debug=True):
    for root, dirs, fnames in os.walk(bc_dir):
        for fname in fnames:
            if not fname.startswith('.') and fname.endswith('.bc'):
                print("[Utils][INFO] Processing Bitcode file: %s" % fname)
                bin_name = fname.replace('.bc', '')
                bc_fpath = os.path.join(root, fname)
                bc_dir = os.path.join(out_dir, bin_name)
                out_fpath = os.path.join(bc_dir, fname)
                print("[Utils][INFO] Bitcode directory: %s" % bc_dir)
                # First, check if the dir already exists
                if not os.path.exists(bc_dir):
                    os.makedirs(bc_dir)
                    shutil.copy(bc_fpath, out_fpath)


def copy_assembly_ir_files(ir_dir):
    program_names = os.listdir(ir_dir)
    for program_name in program_names:
        ir_fpath = os.path.join(ir_dir, program_name,
                                "klee-last", 'assembly.ll')
        if not os.path.exists(ir_fpath):
            continue
        else:
            print("[Utils][INFO] Processing IR file: %s" % ir_fpath)
            out_fpath = os.path.join(
                ir_dir, program_name, '%s_assembly.ll' % program_name)
            if not os.path.exists(out_fpath):
                shutil.copy(ir_fpath, out_fpath)


def load_graph_file(graph_fpath, convert_to_networtx=True):
    # this would return all graphs (TODO: Find out why there can be multiple graphs from the same file)
    graphs = pg.load_graphs(path=Path(graph_fpath))
    if convert_to_networtx:
        networkx_graphs = []
        for graph in graphs:
            networkx_graph = pg.to_networkx(graph)
            networkx_graphs.append(networkx_graph)
        return networkx_graphs
    return graphs


def parse_merge_json_from_str(json_str, use_dummy_fields=False):
    merge_json = json.loads(json_str)

    # Cannot miss these two fields
    ir_fname = merge_json[MERGE_JSON_FIELDS["IR_FILENAME"]]
    merge_id = merge_json[MERGE_JSON_FIELDS["MERGE_ID"]]

    merge_point_addr = merge_json.get(MERGE_JSON_FIELDS["MERGE_POINT_ADDR"])
    merged_paths = merge_json.get(MERGE_JSON_FIELDS["MERGED_PATHS"])
    merged_vars = merge_json.get(MERGE_JSON_FIELDS["MERGED_VARS"])
    if not use_dummy_fields:
        merged_time = merge_json.get(MERGE_JSON_FIELDS["MERGED_TIME"])
        unmerged_time = merge_json.get(MERGE_JSON_FIELDS["UNMERGED_TIME"])
    else:
        unmerged_time = "0.040674"
        merged_time = "0.040674"
    prev_merge_point1 = merge_json.get(MERGE_JSON_FIELDS["PREV_MERGE_POINT1"])
    prev_merge_point2 = merge_json.get(MERGE_JSON_FIELDS["PREV_MERGE_POINT2"])

    merge_record = MergeRecord(
        ir_fname=ir_fname,
        merge_id=merge_id,
        merge_point_addr=merge_point_addr,
        merged_paths=merged_paths,
        merged_vars=merged_vars,
        merged_time=merged_time,
        unmerged_time=unmerged_time,
        prev_merge_point1=prev_merge_point1,
        prev_merge_point2=prev_merge_point2,
    )

    return merge_record


def parse_merge_json(json_fpath):
    with open(json_fpath, 'r') as fin:
        merge_json = json.load(fin)

    # Cannot miss these two fields
    ir_fname = merge_json[MERGE_JSON_FIELDS["IR_FILENAME"]]
    merge_id = merge_json[MERGE_JSON_FIELDS["MERGE_ID"]]

    merge_point_addr = merge_json.get(MERGE_JSON_FIELDS["MERGE_POINT_ADDR"])
    merged_paths = merge_json.get(MERGE_JSON_FIELDS["MERGED_PATHS"])
    merged_vars = merge_json.get(MERGE_JSON_FIELDS["MERGED_VARS"])
    merged_time = merge_json.get(MERGE_JSON_FIELDS["MERGED_TIME"])
    unmerged_time = merge_json.get(MERGE_JSON_FIELDS["UNMERGED_TIME"])
    prev_merge_point1 = merge_json.get(MERGE_JSON_FIELDS["PREV_MERGE_POINT1"])
    prev_merge_point2 = merge_json.get(MERGE_JSON_FIELDS["PREV_MERGE_POINT2"])

    merge_record = MergeRecord(
        ir_fname=ir_fname,
        merge_id=merge_id,
        merge_point_addr=merge_point_addr,
        merged_paths=merged_paths,
        merged_vars=merged_vars,
        merged_time=merged_time,
        unmerged_time=unmerged_time,
        prev_merge_point1=prev_merge_point1,
        prev_merge_point2=prev_merge_point2,
    )

    return merge_record


def scan_json_dir(json_dir, concurrent=False, test=False):
    all_merges = {}
    json_filenames = sorted(os.listdir(json_dir))
    processed_json_cnt = 0
    for json_fname in json_filenames:
        if not json_fname.startswith("data-"):
            continue
        if concurrent:  # partition JSON files
            merge_id = json_fname.replace("data-", '').replace(".json", '')
            # merge_id is the unique hash value for each merge JSON file
            if int(merge_id) % args.num_instances != args.instance_id:
                print("[Utils][INFO] Skipping merge ID: %s" % merge_id)
                continue
        json_fpath = os.path.join(json_dir, json_fname)
        merge_record = parse_merge_json(json_fpath)
        print("[Utils][INFO] Loaded MergeRecord ID %s for program %s." %
              (merge_record.merge_id, merge_record.ir_fname))
        # DO NOT load the merge record JSON yet; save the file path
        # for now and load it later on. We need to avoid RAM explosion.
        if merge_record.ir_fname not in all_merges:
            all_merges[merge_record.ir_fname] = {
                merge_record.merge_id: json_fpath}
        else:
            all_merges[merge_record.ir_fname][merge_record.merge_id] = json_fpath
        processed_json_cnt += 1
        if test and processed_json_cnt > 200:
            break
    return all_merges


def scan_programl_graph_dir(log_dir):
    all_graphs = {}
    program_names = os.listdir(log_dir)
    for program_name in program_names:
        if USE_UNMODIFIED_LLVM:
            graph_fpath = os.path.join(
                log_dir, program_name, "%s.graph.unmodified" % program_name)
        else:
            graph_fpath = os.path.join(
                log_dir, program_name, "%s.graph" % program_name)
        if not os.path.isfile(graph_fpath):
            print("[Utils][ERROR] ProGraML file does not exist: %s!" %
                  graph_fpath)
            continue
        print("[Utils][INFO] Loaded ProGraML graph %s." % graph_fpath)
        # We can load ProGraML graphs now since there are that many of them
        # (so that this would not trigger RAM explosion issues).
        G = pg.load_graphs(path=Path(graph_fpath))
        all_graphs[program_name] = G
    return all_graphs


def prepare_coreutils_klee_env(coreutils_name, copy_bitcode=False):
    cmds = []
    new_out_dir = os.path.join(IR_PROGRAML_DIR + '/' + coreutils_name)
    if copy_bitcode:
        cmds.append("mkdir -p %s" % new_out_dir)
        cmds.append("cd %s" % new_out_dir)

        cmds.append("cp %s ./" %
                    os.path.join(COREUTILS_BC_DIR, coreutils_name + '.bc'))
    else:
        cmds.append("cd %s" % new_out_dir)
    cmds.append(ENV_CMD)
    final_cmd = ' && '.join(cmds)
    print("Preparing env -- full command: %s" % final_cmd)
    os.system(final_cmd)


def compile_program_list(ir_dir, out_path):
    programs = []

    program_dirs = os.listdir(ir_dir)
    for program in program_dirs:
        if USE_UNMODIFIED_LLVM:
            ll_fpath = os.path.join(ir_dir, program, "%s.ll" % program)
        else:
            ll_fpath = os.path.join(
                ir_dir, program, "%s_assembly.ll" % program)
        if not os.path.isfile(ll_fpath):
            print("[Utils][INFO] LL file not found: %s" % ll_fpath)
            continue
        if USE_UNMODIFIED_LLVM:
            graph_fpath = os.path.join(
                ir_dir, program, "%s.graph.unmodified" % program)
        else:
            graph_fpath = os.path.join(ir_dir, program, "%s.graph" % program)
        if not os.path.isfile(graph_fpath):
            print("[Utils][ERROR] ProGraML file does not exist: %s!" %
                  graph_fpath)
            continue
        programs.append(program)

    print("[Utils][INFO] Found %d programs." % len(programs))

    with open(out_path, 'w') as fout:
        for program in programs:
            fout.write("%s\n" % program)


def load_program_list(in_path):
    programs = []
    with open(in_path, 'r') as fin:
        for line in fin:
            program = line.strip()
            programs.append(program)
    return programs


def run_klee_experiment(
    base_cmd,
    coreutils_name,
    klee_args,
    use_nohup=False,
):
    print("[Utils][INFO] Running klee on %s" % klee_args["bc_fname"])
    prepare_coreutils_klee_env(coreutils_name)

    cmds = []
    new_out_dir = os.path.join(IR_PROGRAML_DIR + '/' + coreutils_name)
    cmds.append("cd %s" % new_out_dir)
    klee_test_env_fpath = os.path.join(
        new_out_dir, klee_args["klee_test_env_fname"])

    bc_fpath = os.path.join(new_out_dir, klee_args["bc_fname"])

    klee_cmd = base_cmd.format(
        klee_fpath=klee_args["klee_fpath"],
        klee_exp_dir=klee_args["klee_exp_dir"],
        bc_fpath=bc_fpath,
        klee_test_env_fpath=klee_test_env_fpath,
        sym_arg_list=klee_args["sym_arg_list"],
    )
    if use_nohup:
        klee_cmd = "nohup " + klee_cmd + "> run_klee.log 2>&1 &"
    else:
        klee_cmd = klee_cmd
    cmds.append(klee_cmd)
    final_cmd = ' && '.join(cmds)
    print("Klee exeuction full command: %s" % final_cmd)
    os.system(final_cmd)


def build_klee(rebuild=False, build_auto_merge=False):
    cmds = []
    if build_auto_merge:
        cmds.append("cd %s" % KLEE_AUTO_MERGE_BUILD_DIR)
    else:
        cmds.append("cd %s" % KLEE_BUILD_DIR)
    if rebuild:
        cmds.append("rm -rf *")
    if build_auto_merge:
        cmds.append(TEMPLATE_BUILD_AUTO_MERGE_KLEE_CMAKE_CMD)
    else:
        cmds.append(TEMPLATE_BUILD_KLEE_CMAKE_CMD)
    cmds.append(TEMPLATE_BUILD_KLEE_MAKE_CMD)
    final_cmd = ' && '.join(cmds)
    print("Full command: %s" % final_cmd)
    os.system(final_cmd)


def build_programl():
    cmds = []
    cmds.append("cd %s" % PROGRAML_DIR)
    cmds.append("make install")
    final_cmd = ' && '.join(cmds)
    os.system(final_cmd)


def kill_klee():
    os.system("killall -9 klee")


def clean_up_records():
    os.system("rm -rf %s/*" % MERGE_RECORD_DIR)
    os.system("cd %s && rm -rf sandbox && wget https://www.doc.ic.ac.uk/~cristic/klee/sandbox.tgz && tar xzfv sandbox.tgz" % MERGE_RECORD_DIR)


def clean_up_dataset():
    os.system("rm -r %s" % os.path.join(MERGE_GRAPH_DIR, "train"))
    os.system("mkdir %s" % os.path.join(MERGE_GRAPH_DIR, "train"))
    os.system("rm -r %s" % os.path.join(MERGE_GRAPH_DIR, "val"))
    os.system("mkdir %s" % os.path.join(MERGE_GRAPH_DIR, "val"))
    os.system("rm -r %s" % os.path.join(MERGE_GRAPH_DIR, "test"))
    os.system("mkdir %s" % os.path.join(MERGE_GRAPH_DIR, "test"))
    os.system("rm %s/*" % os.path.join(MERGE_GRAPH_DIR, "processed"))


def clean_up_klee_files(delete_records=False):
    if delete_records:
        os.system("rm -rf %s" % MERGE_RECORD_DIR)
        os.system("mkdir %s" % MERGE_RECORD_DIR)
        os.system("cd %s && rm -rf sandbox && wget https://www.doc.ic.ac.uk/~cristic/klee/sandbox.tgz && tar xzfv sandbox.tgz" % MERGE_RECORD_DIR)
    os.system("rm -rf %s/*/klee-out-*" % IR_PROGRAML_DIR)
    os.system("rm -rf %s/*/klee-last" % IR_PROGRAML_DIR)
    os.system("rm -rf %s/*/run_klee.log" % IR_PROGRAML_DIR)


def gen_programl_graphs(log_dir, concurrent=False):
    program_names = sorted(os.listdir(log_dir))
    for i in range(len(program_names)):
        if concurrent:
            if i % args.num_instances != args.instance_id:
                continue
        program_name = program_names[i]
        if USE_UNMODIFIED_LLVM:
            ir_fpath = os.path.join(log_dir, program_name,
                                    "%s.ll" % program_name)
        else:
            ir_fpath = os.path.join(log_dir, program_name,
                                    "%s_assembly.ll" % program_name)
        if os.path.isfile(ir_fpath):
            with open(ir_fpath, 'r') as fin:
                G = pg.from_llvm_ir(
                    fin.read(), version=COMPILER_VERSION["LLVM_6_0"], timeout=7200)
                networkx_graph = pg.to_networkx(G)
                print("[Utils][INFO] Program name: %s | # nodes: %d | # edges: %d" % (
                    program_name, len(networkx_graph.nodes()), len(networkx_graph.edges())))
                if USE_UNMODIFIED_LLVM:
                    graph_fpath = os.path.join(
                        log_dir, program_name, "%s.graph.unmodified" % program_name)
                else:
                    graph_fpath = os.path.join(
                        log_dir, program_name, "%s.graph" % program_name)
                print("[Utils][INFO] Saving to %s..." % graph_fpath)
                pg.save_graphs(path=Path(graph_fpath), graphs=[G])
        else:
            print("[Utils][ERROR] IR file does not exist: %s!" % ir_fpath)


def gen_merge_graph_in_programl_format(
    node_ids_to_keep,
    programl_graph,
    isolates,
    old_merge_point_node_id,
    old_merge_point_block_nodes,
):
    merge_graph = deepcopy(programl_graph)

    function_ids_to_keep = set()
    nodes = []
    node_ids_mapping = {}
    for i in range(len(merge_graph.node)):
        if i in node_ids_to_keep and i not in set(isolates):
            if i in old_merge_point_block_nodes:
                # Insert special token to indicate the merge point
                merge_graph.node[i].features.feature["full_text"].bytes_list.value[0] = b'<MERGE_POINT_BLOCK> ' + \
                    merge_graph.node[i].features.feature["full_text"].bytes_list.value[0]
            nodes.append(merge_graph.node[i])
            node_ids_mapping[i] = len(nodes) - 1
            function_ids_to_keep.add(merge_graph.node[i].function)
    del merge_graph.node[:]
    merge_graph.node.extend(nodes)

    edges = []
    for edge in merge_graph.edge:
        if edge.source in node_ids_mapping and edge.target in node_ids_mapping:
            edge.source = node_ids_mapping[edge.source]
            edge.target = node_ids_mapping[edge.target]
            edges.append(edge)
    del merge_graph.edge[:]
    merge_graph.edge.extend(edges)

    module_ids_to_keep = set()
    functions = []
    for i in range(len(merge_graph.function)):
        if i in function_ids_to_keep:
            functions.append(merge_graph.function[i])
            module_ids_to_keep.add(merge_graph.function[i].module)
    del merge_graph.function[:]
    merge_graph.function.extend(functions)

    modules = []
    for i in range(len(merge_graph.module)):
        if i in module_ids_to_keep:
            modules.append(merge_graph.module[i])
    del merge_graph.module[:]
    merge_graph.module.extend(modules)

    print("[Utils][INFO] Stats for MergeGraph (in ProGraML): # nodes %d | # edges %d." % (
        len(merge_graph.node), len(merge_graph.edge)))

    return merge_graph, node_ids_mapping[old_merge_point_node_id]


def is_module_function_entry_exit_node(node):
    return node[PROGRAML_ATTRS["TEXT"]] == "[external]" or node[PROGRAML_ATTRS["TEXT"]] == "; undefined function"


def is_llvm_debug_info_node(node):
    if PROGRAML_ATTRS["FEATURES"] not in node:
        return False
    features = node[PROGRAML_ATTRS["FEATURES"]]
    # "full_text" should be the only element in the list
    return features["full_text"][0].startswith("call void @llvm")


def gen_merge_graph(
    merge_record,
    programl_graph,
    var_inclusion,
    disable_cache=False,
    sanity_check=False,
):
    def parse_inst_id(inst_id):
        inst_id = os.path.basename(inst_id)
        llvm_block_id, inst_idx = inst_id.rsplit("-", 1)
        return llvm_block_id, int(inst_idx)

    networkx_graph = pg.to_networkx(programl_graph)
    # deepcopy as the initial MergeGraph; will be trimmed based on merge record later
    merge_graph = deepcopy(networkx_graph)

    if sanity_check:
        components = list(nx.weakly_connected_components(merge_graph))
        print("[Utils][INFO] # unconnected components (before removing LLVM nodes): %d" % (
            len(components) - 1))

        nodes_to_remove = set()
        for node in merge_graph.nodes():
            attrs = merge_graph.nodes[node]
            block_id = attrs[PROGRAML_ATTRS["BLOCK_ID"]]

            if is_llvm_debug_info_node(attrs):
                # Skip LLVM nodes
                nodes_to_remove.add(node)

        merge_graph.remove_nodes_from(list(nodes_to_remove))

        components = list(nx.weakly_connected_components(merge_graph))
        print("[Utils][INFO] # unconnected components (after removing LLVM nodes): %d" % (
            len(components) - 1))

        exit()

    stats = []
    print("[Utils][INFO] Merge point addr: %s | Merge ID: %s" %
          (merge_record.merge_point_addr, merge_record.merge_id))

    min_length_of_merged_path = min(
        list(map(lambda x: len(x), list(merge_record.merged_paths.values())))
    )
    max_length_of_path = max(
        min_length_of_merged_path * MAX_LENGTH_OF_PATH_RATIO,
        MIN_LENGTH_OF_PATH
    )
    print("[Utils][INFO] Max length of path: %d." % max_length_of_path)

    merge_point_llvm_block_id, _ = parse_inst_id(merge_record.merge_point_addr)
    relevant_inst_ids = set(merge_record.merge_point_addr)
    llvm_block_ids_to_keep = set(merge_point_llvm_block_id)
    for _, merged_path in merge_record.merged_paths.items():
        # From closest to furthest w.r.t. the merge point

        for _, inst_id in reversed(sorted(merged_path.items())):
            relevant_inst_ids.add(inst_id)
            llvm_block_id, _ = parse_inst_id(inst_id)
            llvm_block_ids_to_keep.add(llvm_block_id)
            if len(llvm_block_ids_to_keep) > max_length_of_path:
                break
    print("[Utils][INFO] # relevant inst nodes: %d | # blocks: %d." %
          (len(relevant_inst_ids), len(llvm_block_ids_to_keep)))

    merged_vars_inst_ids = set()
    if merge_record.merged_vars is not None:
        local_vars = merge_record.merged_vars["LOCAL_VARS"]
        merged_vars_inst_ids = local_vars
    print("[Utils][INFO] # merged var nodes: %d" % len(merged_vars_inst_ids))

    # Simple way: record in 1st pass and remove unwanted nodes in a later pass
    # 1st pass -- get keep list for instruction nodes in merged paths
    past_inst_nodes_to_keep = set()
    past_var_nodes_to_keep = set()
    block_ids_to_keep = set()
    merge_point = None
    curr_merge_point_node_inst_id = 999999999
    for node in merge_graph.nodes():
        attrs = merge_graph.nodes[node]
        block_id = attrs[PROGRAML_ATTRS["BLOCK_ID"]]

        if FILTER_LLVM_NODES and is_llvm_debug_info_node(attrs):
            # Skip LLVM nodes
            continue
        if PROGRAML_ATTRS["FEATURES"] in attrs:
            features = attrs[PROGRAML_ATTRS["FEATURES"]]
            if PROGRAML_ATTRS["INST_ID"] in features:
                inst_id = features[PROGRAML_ATTRS["INST_ID"]][0]
                llvm_block_id, inst_idx = parse_inst_id(inst_id)
                if llvm_block_id == merge_point_llvm_block_id:
                    if inst_idx < curr_merge_point_node_inst_id:
                        curr_merge_point_node_inst_id = inst_idx
                        merge_point = node
                        merge_point_block_id = block_id
                if llvm_block_id in llvm_block_ids_to_keep:
                    past_inst_nodes_to_keep.add(node)
                    block_ids_to_keep.add(block_id)
                    node_type = ProgramlNodeType.INST
                    stats.append(node_type)

    assert merge_point is not None, "[Utils][FATAL] Merge point not found!"
    print("[Utils][INFO] Found merge point: %d | %s!" %
          (merge_point, merge_graph.nodes[merge_point]))

    # Handle the block of merge point
    # TODO: Find a smarter to do this!
    merge_point_nodes = set()
    merge_point_block_inst_idx = 9999999999  # big number
    for node in merge_graph.nodes():
        attrs = merge_graph.nodes[node]
        block_id = attrs[PROGRAML_ATTRS["BLOCK_ID"]]

        if FILTER_LLVM_NODES and is_llvm_debug_info_node(attrs):
            # Skip LLVM nodes
            continue
        if block_id == merge_point_block_id:
            # For both inst, var and entry/exit nodes
            merge_point_nodes.add(node)
        if block_id in block_ids_to_keep and PROGRAML_ATTRS["INST_ID"] not in attrs:
            # Add block entry/exit nodes
            past_inst_nodes_to_keep.add(node)
        if PROGRAML_ATTRS["FEATURES"] in attrs:
            features = attrs[PROGRAML_ATTRS["FEATURES"]]
            if block_id == merge_point_block_id:
                if PROGRAML_ATTRS["INST_ID"] in features:
                    inst_id = features[PROGRAML_ATTRS["INST_ID"]][0]
                    _, inst_idx = parse_inst_id(inst_id)
                    if inst_idx < merge_point_block_inst_idx:
                        merge_point_block_inst_idx = inst_idx
                        merge_point = node  # first inst node in the block
                    node_type = ProgramlNodeType.INST
                    stats.append(node_type)

    past_inst_nodes_to_keep = past_inst_nodes_to_keep.union(merge_point_nodes)

    all_block_ids = set()
    for node in merge_graph.nodes():
        attrs = merge_graph.nodes[node]
        block_id = attrs[PROGRAML_ATTRS["BLOCK_ID"]]
        all_block_ids.add(block_id)

        if FILTER_LLVM_NODES and is_llvm_debug_info_node(attrs):
            # Skip LLVM nodes
            continue

        if PROGRAML_ATTRS["FEATURES"] in attrs:
            features = attrs[PROGRAML_ATTRS["FEATURES"]]
            if PROGRAML_ATTRS["OP_INST_ID"] in features:
                if block_id in block_ids_to_keep:
                    inst_id = features[PROGRAML_ATTRS["OP_INST_ID"]][0]
                    node_type = ProgramlNodeType.OP
                    if var_inclusion == MergeGraphVarInclusion.ONLY_MERGED:
                        if inst_id in merged_vars_inst_ids:
                            past_var_nodes_to_keep.add(node)
                            stats.append(node_type)
                    elif var_inclusion == MergeGraphVarInclusion.ALL:
                        if inst_id in merged_vars_inst_ids or inst_id in relevant_inst_ids:
                            past_var_nodes_to_keep.add(node)
                            stats.append(node_type)
            elif PROGRAML_ATTRS["ARG_INST_ID"] in features:
                if block_id in block_ids_to_keep:
                    inst_id = features[PROGRAML_ATTRS["ARG_INST_ID"]][0]
                    node_type = ProgramlNodeType.ARG
                    if var_inclusion == MergeGraphVarInclusion.ONLY_MERGED:
                        if inst_id in merged_vars_inst_ids:
                            past_var_nodes_to_keep.add(node)
                            stats.append(node_type)
                    elif var_inclusion == MergeGraphVarInclusion.ALL:
                        if inst_id in merged_vars_inst_ids or inst_id in relevant_inst_ids:
                            past_var_nodes_to_keep.add(node)
                            stats.append(node_type)
            elif PROGRAML_ATTRS["CONST_INST_ID"] in features:
                if block_id in block_ids_to_keep:
                    inst_id = features[PROGRAML_ATTRS["CONST_INST_ID"]][0]
                    node_type = ProgramlNodeType.CONST
                    if var_inclusion == MergeGraphVarInclusion.ONLY_MERGED:
                        if inst_id in merged_vars_inst_ids:
                            past_var_nodes_to_keep.add(node)
                            stats.append(node_type)
                    elif var_inclusion == MergeGraphVarInclusion.ALL:
                        if inst_id in merged_vars_inst_ids or inst_id in relevant_inst_ids:
                            past_var_nodes_to_keep.add(node)
                            stats.append(node_type)

    print("[Utils][INFO] Stats for ProGraML graph (merged paths): %s | # inst nodes %d | # var nodes %d | # total nodes: %d | # total edges: %d | # blocks: %d" % (
        str(Counter(stats)),
        len(past_inst_nodes_to_keep),
        len(past_var_nodes_to_keep),
        len(networkx_graph.nodes()),
        len(networkx_graph.edges()),
        len(all_block_ids)
    ))

    found_future_cache = False
    if not disable_cache:
        if merge_record.ir_fname in _GLOBAL_FUTURE_GRAPH_CACHE:
            if merge_record.merge_point_addr in _GLOBAL_FUTURE_GRAPH_CACHE[merge_record.ir_fname]:
                print("[Utils][INFO] Found future graph from cache!")
                future_nodes_to_keep = _GLOBAL_FUTURE_GRAPH_CACHE[
                    merge_record.ir_fname][merge_record.merge_point_addr]
                found_future_cache = True
    if disable_cache or not found_future_cache:
        # 1.5th pass -- start from merge point, get keep list for subsequent blocks
        # (# of blocks limited by hyperparameter)
        # this flag will be set to True when all blocks are traversed
        max_num_blocks_to_traverse = max(
            int(len(all_block_ids) * MAX_NUM_BLOCKS_TO_TRAVERSE_RATIO),
            MIN_NUM_BLOCKS_TO_TRAVERSE
        )
        print("[Utils][INFO] Max # blocks: %d." %
              max_num_blocks_to_traverse)
        future_nodes_to_keep = set()  # no need to differentiate inst and var nodes
        # Merge point block is included in both past and future
        future_nodes_to_keep = future_nodes_to_keep.union(merge_point_nodes)
        traversed_block_ids = set()

        # Traverse from merge point (BFS) bounded by MAX_NUM_BLOCKS_TO_TRAVERSE
        # queue = list(merge_point_nodes)
        queue = [merge_point]
        visited = set()

        while len(queue) > 0:
            curr_node = queue.pop(0)
            if FILTER_LLVM_NODES:
                if not is_llvm_debug_info_node(merge_graph.nodes[curr_node]):
                    # Skip LLVM nodes
                    future_nodes_to_keep.add(curr_node)
            else:
                future_nodes_to_keep.add(curr_node)

            for succesor in merge_graph.successors(curr_node):
                attrs = merge_graph.nodes[succesor]
                block_id = attrs[PROGRAML_ATTRS["BLOCK_ID"]]
                if succesor not in visited:
                    traversed_block_ids.add(block_id)
                    visited.add(succesor)
                    if len(traversed_block_ids) <= max_num_blocks_to_traverse:
                        if is_module_function_entry_exit_node(attrs):
                            # Skip module function entry/exit nodes
                            continue
                        queue.append(succesor)
        if not disable_cache:
            if merge_record.ir_fname not in _GLOBAL_FUTURE_GRAPH_CACHE:
                _GLOBAL_FUTURE_GRAPH_CACHE[merge_record.ir_fname] = {
                    merge_record.merge_point_addr: future_nodes_to_keep}
            else:
                _GLOBAL_FUTURE_GRAPH_CACHE[merge_record.ir_fname][merge_record.merge_point_addr] = future_nodes_to_keep

    print("[Utils][INFO] # nodes of before merge point: %d | after: %d" %
          (len(past_inst_nodes_to_keep), len(future_nodes_to_keep)))

    # 2nd pass -- get removal list
    nodes_to_keep, nodes_to_remove = set(), set()
    for node in merge_graph.nodes():
        if node not in past_inst_nodes_to_keep and node not in past_var_nodes_to_keep and node not in future_nodes_to_keep:
            nodes_to_remove.add(node)
        else:
            nodes_to_keep.add(node)
    print("[Utils][INFO] Kept %d nodes and removed %d nodes." %
          (len(nodes_to_keep), len(nodes_to_remove)))

    # Check for isolated nodes
    merge_graph.remove_nodes_from(nodes_to_remove)

    # Find unconnected components
    unconnected_nodes = []
    components = list(nx.weakly_connected_components(merge_graph))
    if len(components) > 1:
        print("[Utils][INFO] Found %d unconnected components." %
              len(components))
    largest_component = max(components, key=len)
    for component in components:
        if component != largest_component:
            unconnected_nodes.extend(component)

    print("[Utils][INFO] Found %d unconnected nodes" % len(unconnected_nodes))
    if len(unconnected_nodes) > max(int(MAX_ISOLATES_RATIO * len(merge_graph.nodes())), 1000):
        print("[Utils][ERROR] Too many unconnected nodes: %d." %
              len(unconnected_nodes))
        raise TooManyIsolatedNodesError

    if merge_point in set(unconnected_nodes):
        print("[Utils][FATAL] Merge point %d should not be in unconnected nodes: %s" % (
            merge_point, str(unconnected_nodes)))
        raise MergePointNotConnectedError

    # Now trim the original ProGraML graph based on node ids
    # (this is a bit tricky, since we need to preserve the order of nodes)
    merge_graph_programl, merge_point_node_id = gen_merge_graph_in_programl_format(
        nodes_to_keep,
        programl_graph,
        unconnected_nodes,
        merge_point,
        merge_point_nodes,
    )

    print("[Utils][INFO] Merge merge node ID: %d" % merge_point_node_id)

    merge_graph_networkx = pg.to_networkx(merge_graph_programl)
    # Check for unconnected nodes
    components = list(
        nx.weakly_connected_components(merge_graph_networkx))
    assert len(components) == 1, "[Utils][FATAL] There should not be any unconnected components now: %d." % len(
        components)
    print("[Utils][INFO] Stats for MergeGraph (in NetworkX): # nodes %d | # edges %d." %
          (len(merge_graph_networkx.nodes()), len(merge_graph_networkx.edges())))

    return merge_graph_programl


def gen_merge_graphs(merge_records, programl_graphs):
    merge_graphs = {}

    for ir_fname, merges in merge_records.items():
        program_name = Path(ir_fname).name.replace('.bc', '')
        if program_name not in programl_graphs:
            print(
                "[Utils][ERROR] %s not in loaded ProGraML graphs! Skipping..." % program_name)
            continue
        programl_graph = programl_graphs[program_name]

        for merge_id, merge_fpath in merges.items():
            # First, check if MergeGraph already exists
            merge_graph_fname1 = "MergeGraph-%s-%s.pb" % (
                merge_id, VAR_INCLUSION_MAP[MergeGraphVarInclusion.ONLY_MERGED])
            merge_graph_fname2 = "MergeGraph-%s-%s.pb" % (
                merge_id, VAR_INCLUSION_MAP[MergeGraphVarInclusion.ALL])
            merge_graph_fpath1 = os.path.join(
                MERGE_GRAPH_DIR, "raw", merge_graph_fname1)
            merge_graph_fpath2 = os.path.join(
                MERGE_GRAPH_DIR, "raw", merge_graph_fname2)

            if os.path.isfile(merge_graph_fpath1) and os.path.isfile(merge_graph_fpath2):
                print(
                    "[Utils][ERROR] MergeGraph w/ ID %s already generated. Skipping..." % merge_id)
                continue

            print("=========== [Generation Begins] ==========")
            merge = parse_merge_json(merge_fpath)

            # Filter merges without any variables -- no variable means no merge
            if merge.merged_vars is None:
                print("[Utils][ERROR] Merge record #%s has no variables. Skipping..." %
                      merge.merge_id)
                print("=========== [Generation Done] ==========")
                continue
            else:
                # Make sure merge records are well-formed
                assert merge.merged_vars[
                    "LOCAL_VARS"] is not None, "[Utils][FATAL] No local variable(s) found: %s" % merge.merge_id
                assert len(
                    programl_graph) == 1, "[Utils][FATAL] Merge record #%s has > 1 graph!" % merge.merge_id

                if abs(merge.merged_time - merge.unmerged_time) / min(merge.merged_time, merge.unmerged_time) < 0.05:
                    print("[Utils][WARNING] Merge record #%s has very small time difference. Skipping..." %
                          merge.merge_id)
                    print("=========== [Generation Done] ==========")
                    continue
                if merge.merged_time <= merge.unmerged_time:
                    label = '1'
                else:
                    label = '0'

                for var_inlusion in [MergeGraphVarInclusion.ONLY_MERGED, MergeGraphVarInclusion.ALL]:
                    merge_graph_fname = "MergeGraph-%s-%s-%s.pb" % (
                        merge_id, VAR_INCLUSION_MAP[var_inlusion], label)
                    merge_graph_fpath = os.path.join(
                        MERGE_GRAPH_DIR, "raw", merge_graph_fname)

                    if programl_graph[0] is None:
                        continue
                    try:
                        merge_graph = gen_merge_graph(
                            merge, programl_graph[0], var_inlusion)
                    except TooManyIsolatedNodesError:
                        continue
                    except MergePointNotConnectedError:
                        continue

                    pg.save_graphs(path=Path(merge_graph_fpath),
                                   graphs=[merge_graph])
                    if ir_fname not in merge_graphs:
                        merge_graphs[ir_fname] = {}
                        if merge_id not in merge_graphs[ir_fname]:
                            merge_graphs[ir_fname][merge_id] = [
                                merge_graph_fname]
                        else:
                            merge_graphs[ir_fname][merge_id].append(
                                merge_graph_fname)
                    print(
                        "[Utils][INFO] MergePoint for %s is generated and saved (%s variables)." % (merge_id, VAR_INCLUSION_MAP[var_inlusion]))
                    print("=========== [Generation Done] ==========")

    return merge_graphs


def test_gen_merge_graph(merge_json_fpath, program_name, merge_graph_fpath):
    if USE_UNMODIFIED_LLVM:
        graph_fpath = os.path.join(
            IR_PROGRAML_DIR, program_name, "%s.graph.unmodified" % program_name)
    else:
        graph_fpath = os.path.join(
            IR_PROGRAML_DIR, program_name, "%s.graph" % program_name)
    G = pg.load_graphs(path=Path(graph_fpath))[0]
    merge = parse_merge_json(merge_json_fpath)

    print("[Utils][INFO] ====== Test-generating MergeGraph ======")
    G = pg.load_graphs(path=Path(graph_fpath))[0]
    merge_graph_both = gen_merge_graph(
        merge, G, MergeGraphVarInclusion.ALL, disable_cache=True)
    pg.save_graphs(Path(merge_graph_fpath + '-both.pb'),
                   [merge_graph_both])
    visualize_a_merge_graph(merge_graph_fpath + '-both.pb',
                            os.path.join(FIG_DIR, 'both.dot'))


def get_merge_label(merge_record):
    merged_time = merge_record[MERGE_JSON_FIELDS["MERGED_TIME"]]
    unmerged_time = merge_record[MERGE_JSON_FIELDS["UNMERGED_TIME"]]
    if merged_time <= unmerged_time:
        return MergeDecision.SHOULD_MERGE
    else:
        return MergeDecision.SHOULD_NOT_MERGE


def get_program_name(merge_record):
    ir_fname = merge_record[MERGE_JSON_FIELDS["IR_FILENAME"]]
    program_name = os.path.basename(ir_fname).replace(".bc", '')
    return program_name


def get_merge_point_fname(merge_record):
    merge_point_addr = merge_record[MERGE_JSON_FIELDS["MERGE_POINT_ADDR"]]
    merge_point_fname = os.path.basename(merge_point_addr).split('-')[0]
    return merge_point_fname


def get_merge_point(merge_record):
    return merge_record[MERGE_JSON_FIELDS["MERGE_POINT_ADDR"]]


def get_graph_size(graph_fname):
    graph_fpath = os.path.join(MERGE_GRAPH_DIR, "raw", graph_fname)
    if os.path.getsize(graph_fpath) == 0:
        return None, None
    graph = pg.load_graphs(graph_fpath)[0]
    num_nodes = len(graph.node)
    num_edges = len(graph.edge)
    return num_nodes, num_edges


def get_nesting(merge_record):
    prev_label1 = merge_record[MERGE_JSON_FIELDS["PREV_MERGE_POINT1"]]
    prev_label2 = merge_record[MERGE_JSON_FIELDS["PREV_MERGE_POINT2"]]
    return prev_label1, prev_label2


def get_dataset_stats(
    merge_graph_variant=MergeGraphVariant.ALL_VARS,
):
    def process_graph(graph_fname):
        if not graph_fname.endswith(".pb"):
            return None
        if merge_graph_variant == MergeGraphVariant.ALL_VARS:
            if "only-merged" in graph_fname:
                return None
        elif merge_graph_variant == MergeGraphVariant.ONLY_MERGED_VARS:
            if "all" in graph_fname:
                return None

        merge_id = graph_fname.split("-")[1]
        merge_record_fpath = os.path.join(
            MERGE_RECORD_DIR, "data-%s.json" % merge_id)
        with open(merge_record_fpath, 'r') as fin:
            merge_record = json.load(fin)

        labal = get_merge_label(merge_record)
        merge_point = get_merge_point(merge_record)
        num_nodes, num_edges = get_graph_size(graph_fname)
        nesting = {"prev1": [], "prev2": []}
        nesting1, nesting2 = get_nesting(merge_record)
        nesting["prev1"].append(nesting1)
        nesting["prev2"].append(nesting2)

        return labal, merge_point, num_nodes, num_edges, nesting

    labal_stats = []
    merge_point_stats = []
    nodes_stats = []
    edges_stats = []
    nesting_stats = []
    per_merge_point_stats = {}
    graph_fnames = os.listdir(os.path.join(MERGE_GRAPH_DIR, "raw"))

    with ThreadPoolExecutor(max_workers=os.cpu_count()+4) as threads:
        t_res = list(
            tqdm(threads.map(process_graph, graph_fnames), total=len(graph_fnames)))

    for res in t_res:
        if res is not None:
            labal_stats.append(res[0])
            merge_point_stats.append(res[1])
            nodes_stats.append(res[2])
            edges_stats.append(res[3])

            if res[1] not in per_merge_point_stats:
                per_merge_point_stats[res[1]] = {
                    MergeDecision.SHOULD_MERGE: 0,
                    MergeDecision.SHOULD_NOT_MERGE: 0
                }
            per_merge_point_stats[res[1]][res[0]] += 1

            nesting_stats.append(
                str(res[4]["prev1"]) + "," + str(res[4]["prev2"]))

    print("[Utils][INFO] Label stats: %s | %s" %
          (str(Counter(labal_stats)), str(per_merge_point_stats)))
    print("[Utils][INFO] Merge point stats: %s" %
          str(len(Counter(merge_point_stats))))
    print("[Utils][INFO] Mean # nodes: %f | # edges: %f" % (sum(
        nodes_stats) / len(nodes_stats), sum(edges_stats) / len(edges_stats)))
    print("[Utils][INFO] Nesting stats: %s" % str(Counter(nesting_stats)))


def visualize_loss_curve(json_fpath):
    with open(json_fpath, 'r') as fin:
        data = json.load(fin)
    all_epochs = data
    epoch_losses = []
    epoch_accs = []
    for epoch in all_epochs:
        epoch_loss = epoch["train_results"][0]
        epoch_losses.append(epoch_loss)
        epoch_acc = epoch["test_results"][1]
        epoch_accs.append(epoch_acc)

    epochs = [i+1 for i in range(len(all_epochs))]
    plt.plot(epochs, epoch_losses, label="Loss")
    plt.plot(epochs, epoch_accs, label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss / Testing Accuracy")
    plt.legend()
    run_name = os.path.basename(json_fpath).split(".")[0]
    fig_fpath = os.path.join(
        os.path.dirname('/'.join(json_fpath.split('/'))[:-1]), "%s_loss_curve.png" % run_name)
    plt.savefig(fig_fpath)
    plt.close()


def analyze_merge_records():
    merge_record_fnames = os.listdir(MERGE_RECORD_DIR)

    merge_point_addrs = []
    ir_fnames = []
    label_stats = []

    for merge_record_fname in merge_record_fnames:
        if not merge_record_fname.startswith("data-") or not merge_record_fname.endswith(".json"):
            continue
        merge_record_fpath = os.path.join(MERGE_RECORD_DIR, merge_record_fname)
        merge_record = parse_merge_json(merge_record_fpath)
        if merge_record.merged_vars is None:
            continue
        elif merge_record.merged_vars["LOCAL_VARS"] is None:
            continue
        merge_point_addrs.append(merge_record.merge_point_addr)
        ir_fnames.append(merge_record.ir_fname)
        if merge_record.merged_time < merge_record.unmerged_time:
            label_stats.append("SHOULD_MERGE")
        else:
            label_stats.append("SHOULD_NOT_MERGE")

    print("[Utils][INFO] # distinct merge point addrs: %d" %
          len(Counter(merge_point_addrs)))
    print("[Utils][INFO] # distinct IR filenames: %d" %
          len(Counter(ir_fnames)))
    print("[Utils][INFO] Label stats: %s" %
          str(Counter(label_stats)))
    print("[Utils][INFO] %s" % Counter(merge_point_addrs))
    print("[Utils][INFO] %s" % Counter(ir_fnames))


def split_dataset_by_name(
    balance=False,
    merge_graph_variant=MergeGraphVariant.ALL_VARS,
    by_merge_point_file=False,
    debug=False,
):
    graph_fnames = []
    print("[Utils][INFO] MergeGraph variant: %s" % merge_graph_variant)
    for graph_fname in os.listdir(os.path.join(MERGE_GRAPH_DIR, "raw")):
        if not graph_fname.endswith(".pb"):
            continue
        if merge_graph_variant == MergeGraphVariant.ALL_VARS:
            if "only-merged" in graph_fname:
                continue
        elif merge_graph_variant == MergeGraphVariant.ONLY_MERGED_VARS:
            if "all" in graph_fname:
                continue
        graph_fnames.append(graph_fname)

    if debug:
        graph_fnames = graph_fnames[:1000]

    print("[Utils][INFO] Total # collected graphs: %d" %
          len(set(graph_fnames)))

    random.shuffle(graph_fnames)

    if balance:
        val_count = {}
        test_count = {}

    names = set()
    stats = {}
    for i in range(len(graph_fnames)):
        graph_fname = graph_fnames[i]
        merge_id = graph_fname.split("-")[1]
        merge_record_fpath = os.path.join(
            MERGE_RECORD_DIR, "data-%s.json" % merge_id)
        with open(merge_record_fpath, 'r') as fin:
            merge_record = json.load(fin)
        if not by_merge_point_file:
            name = get_program_name(merge_record)
            names.add(name)
        else:
            name = get_merge_point_fname(merge_record)
            names.add(name)
        if name not in stats:
            stats[name] = []
        stats[name].append(get_merge_label(merge_record))

    for name, labels in stats.items():
        stats[name] = Counter(labels)

    if debug:
        print("[Utils][INFO] Number of names: %d" % len(names))
        print("[Utils][INFO] Stats of names: %s" % stats)

    train_names = set()
    val_names = set()
    test_names = set()

    names = list(names)
    random.shuffle(names)

    for i in range(len(names)):
        name = names[i]
        if float(i) / len(names) <= TRAIN_SET_RATIO:
            train_names.add(name)
        if TRAIN_SET_RATIO < float(i) / len(names) <= TRAIN_SET_RATIO + VAL_SET_RATIO:
            val_names.add(name)
        if TRAIN_SET_RATIO + VAL_SET_RATIO <= float(i) / len(names):
            test_names.add(name)
    print("[Utils][INFO] Splitted dataset details:")
    print("-- TRAIN: %s" % str(train_names))
    print("-- VAL: %s" % str(val_names))
    print("-- TEST: %s" % str(test_names))

    train_lables = []
    val_labels = []
    test_labels = []

    for i in range(len(graph_fnames)):
        graph_fname = graph_fnames[i]
        merge_id = graph_fname.split("-")[1]
        merge_record_fpath = os.path.join(
            MERGE_RECORD_DIR, "data-%s.json" % merge_id)
        with open(merge_record_fpath, 'r') as fin:
            merge_record = json.load(fin)
        if not by_merge_point_file:
            name = get_program_name(merge_record)
        else:
            name = get_merge_point_fname(merge_record)

        if name in train_names:
            if balance:
                merge_id = graph_fname.split("-")[1]
                merge_record_fpath = os.path.join(
                    MERGE_RECORD_DIR, "data-%s.json" % merge_id)
                with open(merge_record_fpath, 'r') as fin:
                    merge_record = json.load(fin)
                label = get_merge_label(merge_record)
                train_lables.append(label)
            shutil.copy(os.path.join(MERGE_GRAPH_DIR, "raw", graph_fname),
                        os.path.join(MERGE_GRAPH_DIR, "train", graph_fname))
        elif name in val_names:
            if balance:
                if name not in val_count:
                    val_count[name] = {
                        MergeDecision.SHOULD_MERGE: 0,
                        MergeDecision.SHOULD_NOT_MERGE: 0,
                    }
                merge_id = graph_fname.split("-")[1]
                merge_record_fpath = os.path.join(
                    MERGE_RECORD_DIR, "data-%s.json" % merge_id)
                with open(merge_record_fpath, 'r') as fin:
                    merge_record = json.load(fin)
                label = get_merge_label(merge_record)
                val_count[name][label] += 1
                if len(stats[name].values()) == 1:
                    continue
                if val_count[name][label] > min(stats[name].values()):
                    continue
                val_labels.append(label)
            shutil.copy(os.path.join(MERGE_GRAPH_DIR, "raw", graph_fname),
                        os.path.join(MERGE_GRAPH_DIR, "val", graph_fname))
        elif name in test_names:
            if balance:
                if name not in test_count:
                    test_count[name] = {
                        MergeDecision.SHOULD_MERGE: 0,
                        MergeDecision.SHOULD_NOT_MERGE: 0,
                    }
                merge_id = graph_fname.split("-")[1]
                merge_record_fpath = os.path.join(
                    MERGE_RECORD_DIR, "data-%s.json" % merge_id)
                with open(merge_record_fpath, 'r') as fin:
                    merge_record = json.load(fin)
                label = get_merge_label(merge_record)
                test_count[name][label] += 1
                if len(stats[name].values()) == 1:
                    continue
                if test_count[name][label] > min(stats[name].values()):
                    continue
                test_labels.append(label)
            shutil.copy(os.path.join(MERGE_GRAPH_DIR, "raw", graph_fname),
                        os.path.join(MERGE_GRAPH_DIR, "test", graph_fname))

    print("[Utils][INFO] Label stats:")
    print("-- TRAIN: %s" % str(Counter(train_lables)))
    print("-- VAL: %s" % str(Counter(val_labels)))
    print("-- TEST: %s" % str(Counter(test_labels)))

    if debug:
        input("Press Enter to continue...")


def split_dataset(
    balance=False,
    merge_graph_variant=MergeGraphVariant.ALL_VARS,
):
    graph_fnames = []
    for graph_fname in os.listdir(os.path.join(MERGE_GRAPH_DIR, "raw")):
        if not graph_fname.endswith(".pb"):
            continue
        if merge_graph_variant == MergeGraphVariant.ALL_VARS:
            if "only-merged" in graph_fname:
                continue
        elif merge_graph_variant == MergeGraphVariant.ONLY_MERGED_VARS:
            if "all" in graph_fname:
                continue
        graph_fnames.append(graph_fname)

    print("[Utils][INFO] Total # collected graphs: %d" %
          len(set(graph_fnames)))

    random.shuffle(graph_fnames)

    if balance:
        val_count = {
            MergeDecision.SHOULD_MERGE: 0,
            MergeDecision.SHOULD_NOT_MERGE: 0,
        }
        test_count = {
            MergeDecision.SHOULD_MERGE: 0,
            MergeDecision.SHOULD_NOT_MERGE: 0,
        }

    stats = []
    for i in range(len(graph_fnames)):
        graph_fname = graph_fnames[i]
        merge_id = graph_fname.split("-")[1]
        merge_record_fpath = os.path.join(
            MERGE_RECORD_DIR, "data-%s.json" % merge_id)
        with open(merge_record_fpath, 'r') as fin:
            merge_record = json.load(fin)
        stats.append(get_merge_label(merge_record))
        stats_counter = Counter(stats)
        smaller_ratio = min(stats_counter.values()) / \
            sum(stats_counter.values())

    for i in range(len(graph_fnames)):
        graph_fname = graph_fnames[i]

        if float(i) / len(graph_fnames) <= TRAIN_SET_RATIO:
            shutil.copy(os.path.join(MERGE_GRAPH_DIR, "raw", graph_fname),
                        os.path.join(MERGE_GRAPH_DIR, "train", graph_fname))
        if TRAIN_SET_RATIO < float(i) / len(graph_fnames) <= TRAIN_SET_RATIO + VAL_SET_RATIO:
            if balance:
                label = get_merge_label(graph_fname)
                val_count[label] += 1
                if val_count[label] > int(smaller_ratio * VAL_SET_RATIO * len(graph_fnames)):
                    shutil.copy(os.path.join(MERGE_GRAPH_DIR, "raw", graph_fname),
                                os.path.join(MERGE_GRAPH_DIR, "train", graph_fname))
                    continue
            shutil.copy(os.path.join(MERGE_GRAPH_DIR, "raw", graph_fname),
                        os.path.join(MERGE_GRAPH_DIR, "val", graph_fname))
        if TRAIN_SET_RATIO + VAL_SET_RATIO <= float(i) / len(graph_fnames):
            if balance:
                label = get_merge_label(graph_fname)
                test_count[label] += 1
                if test_count[label] > int(smaller_ratio * TEST_SET_RATIO * len(graph_fnames)):
                    shutil.copy(os.path.join(MERGE_GRAPH_DIR, "raw", graph_fname),
                                os.path.join(MERGE_GRAPH_DIR, "train", graph_fname))
                    continue
            shutil.copy(os.path.join(MERGE_GRAPH_DIR, "raw", graph_fname),
                        os.path.join(MERGE_GRAPH_DIR, "test", graph_fname))


def gen_vocab(in_dir, out_file, debug=False):
    def get_tokens_from_graph(graph_fname):
        tokens = []
        graph_fpath = os.path.join(in_dir, graph_fname)
        G = pg.load_graphs(graph_fpath)[0]
        for node in G.node:
            tokens.append(node.text)
        return tokens

    tokens = []
    fnames = []
    for fname in os.listdir(in_dir):
        if not fname.endswith(".pb"):
            continue
        if not "-all-" in fname:
            continue
        fnames.append(fname)

    if debug:
        fnames = fnames[:300]

    with ThreadPoolExecutor(max_workers=os.cpu_count()+4) as threads:
        token_list = list(
            tqdm(threads.map(get_tokens_from_graph, fnames), total=len(fnames)))

    tokens = []
    for token in token_list:
        tokens.extend(token)

    total_num_tokens = len(tokens)
    stats = Counter(tokens)

    with open(out_file, 'w') as fout:
        # header
        fout.write(
            "cumulative_frequency\tcumulative_node_frequency\tcount\ttext\n")
        cum_count = 0
        for token, count in stats.most_common():
            cum_count += count
            cum_freq = cum_count / total_num_tokens
            cum_node_freq = cum_count / total_num_tokens
            fout.write("%f\t%f\t%d\t%s\n" %
                       (cum_freq, cum_node_freq, stats[token], token))

    return tokens


def test_auto_merge():
    klee_auto_merge_args = {
        "klee_fpath": KLEE_AUTO_MERGE_BIN_DIR,
        "klee_exp_dir": MERGE_RECORD_DIR,
        "bc_fname": "echo.bc",
        "klee_test_env_fname": "klee-test.env",
        "sym_arg_list": SYM_ARGS_DICT["echo"],
    }
    print("[Utils][INFO] Testing automatic merging on echo...")
    run_klee_experiment(
        base_cmd=TEMPLATE_RUN_AUTO_MERGE_KLEE_CMD,
        coreutils_name="echo",
        klee_args=klee_auto_merge_args,
        use_nohup=False,
    )


def visualize_a_merge_graph(merge_graph_fpath, save_fig_fpath):
    G = pg.load_graphs(merge_graph_fpath)[0]
    print("[Utils][INFO] # nodes: %d | # edges: %d." %
          (len(G.node), len(G.edge)))
    nx_graph = pg.to_networkx(G)
    nx.drawing.nx_pydot.write_dot(nx_graph, save_fig_fpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This script can be used for various ultility purposes.')
    parser.add_argument('--task', type=str, help='specify your task.')
    parser.add_argument('--subtask', type=str,
                        help='specify your sub-task.', default=None)
    parser.add_argument('--instance-id', type=int,
                        help='instance id for concurrence.')
    parser.add_argument('--graph-size', type=int, default=-1,
                        help='vesion (variant) of MergeGraph to use and load.')
    parser.add_argument('--num-instances', type=int,
                        help='total number instances id for concurrence.')
    parser.add_argument('--in-dir', type=str, help='directory to work on.')
    parser.add_argument('--in-file', type=str, help='file to work on.')
    parser.add_argument('--add-in-arg', type=str, help='additional arg.')
    parser.add_argument('--out-file', type=str, help='file to dump to.')
    parser.add_argument('--out-dir', type=str,
                        help='dir to dump converted results.')
    parser.add_argument('--max-merges', type=int,
                        help='max number of merges to process.')
    parser.add_argument('--load-id', type=int, default=-1,
                        help='Start ID for running which batch of Coreutils programs.')
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug mode for the specified task.")
    args = parser.parse_args()

    assert args.task in VALID_TASKS, "[Utils][ERROR] Task %s not supported!" % args.task

    print("[Utils][INFO] All args used in this run: %s" % str(args))

    if args.task == "convert-bc-to-ir":
        print("[Utils][INFO] Converting LLVM bitcode files to IR files...")
        convert_bc_to_ir(args.in_dir)
    if args.task == "convert-ir-to-graph":
        print("[Utils][INFO] Converting LLVM IR files to ProGraML graphs...")
        convert_ir_to_graph(args.in_dir, args.out_dir)
    if args.task == "copy-bitcode-files":
        print("[Utils][INFO] Copying LLVM Bitcode files to working directory...")
        find_and_copy_all_bitcode_files(args.in_dir, args.out_dir)
    if args.task == "copy-assembly-ir-files":
        print("[Utils][INFO] Copying LLVM assembly IR files to working directory...")
        copy_assembly_ir_files(args.in_dir)
    if args.task == "test-run-klee-exp":
        print("[Utils][INFO] Test running echo.bc in KLEE...")
        klee_args = {
            "klee_fpath": KLEE_BIN_DIR,
            "klee_exp_dir": MERGE_RECORD_DIR,
            "bc_fname": "echo.bc",
            "klee_test_env_fname": "klee-test.env",
            "sym_arg_list": SYM_ARGS_DICT["echo"],
        }
        run_klee_experiment(
            base_cmd=TEMPLATE_RUN_KLEE_CMD,
            coreutils_name="echo",
            klee_args=klee_args,
            use_nohup=False,
        )
    if args.task == "quick-test-run-klee-exp":
        print("[Utils][INFO] Quick test running echo.bc in KLEE...")
        klee_args = {
            "klee_fpath": KLEE_BIN_DIR,
            "klee_exp_dir": MERGE_RECORD_DIR,
            "bc_fname": "echo.bc",
            "klee_test_env_fname": "klee-test.env",
            "sym_arg_list": "",  # no args means no input (for quick test only)
        }
        run_klee_experiment(
            base_cmd=TEMPLATE_QUICK_RUN_KLEE_CMD,
            coreutils_name="echo",
            klee_args=klee_args,
            use_nohup=False,
        )
    if args.task == "batch-run-klee-exp":
        print("[Utils][INFO] Batch running Coreutils IRs in KLEE...")
        print("[Utils][INFO] Loading compiled program list...")
        program_names = sorted(load_program_list(args.in_file))
        if args.load_id != -1:
            print(
                "[Utils][INFO] Reading memory usage | current per-KLEE-instance memory limit: %dGB." % PER_KLEE_RAM_GB)
            #ava_mem = psutil.virtual_memory().available / 1024 / 1024 / 1024
            ava_mem = 1700
            num_klee_instances = int(ava_mem / PER_KLEE_RAM_GB)
            if (args.load_id + 1) * num_klee_instances > len(program_names):
                print("[Utils][INFO] Last batch of programs.")
                coreutils_programs_to_run = program_names[args.load_id *
                                                          num_klee_instances:]
            else:
                coreutils_programs_to_run = program_names[
                    args.load_id * num_klee_instances:
                    (args.load_id + 1) * num_klee_instances
                ]
            print("[Utils][INFO] Running %d KLEE instances: %s..." %
                  (len(coreutils_programs_to_run), coreutils_programs_to_run))
        else:
            coreutils_programs_to_run = program_names
        if args.debug:
            print("[Utils][INFO] Running KLEE in debug mode...")
            coreutils_programs_to_run = ["cat"]
        for coreutils_name in coreutils_programs_to_run:
            if coreutils_name not in SYM_ARGS_DICT:
                sym_arg_list = SYM_ARGS_DICT["default"]
            else:
                sym_arg_list = SYM_ARGS_DICT[coreutils_name]
            klee_args = {
                "klee_fpath": KLEE_BIN_DIR,
                "klee_exp_dir": MERGE_RECORD_DIR,
                "bc_fname": "%s.bc" % coreutils_name,
                "klee_test_env_fname": "klee-test.env",
                "sym_arg_list": sym_arg_list,
            }
            run_klee_experiment(
                base_cmd=TEMPLATE_RUN_KLEE_CMD,
                coreutils_name=coreutils_name,
                klee_args=klee_args,
                use_nohup=True,
            )
    if args.task == "build-klee":
        print("[Utils][INFO] Building KLEE...")
        build_klee()
    if args.task == "rebuild-klee":
        print("[Utils][INFO] Rebuilding KLEE...")
        build_klee(rebuild=True)
    if args.task == "build-auto-merge-klee":
        print("[Utils][INFO] Building auto-merge KLEE...")
        build_klee(build_auto_merge=True)
    if args.task == "build-programl":
        print("[Utils][INFO] Building ProGraML...")
        build_programl()
    if args.task == "clean-up-records":
        print("[Utils][INFO] Cleaning up log files...")
        clean_up_records()
    if args.task == "clean-up-dataset":
        print("[Utils][INFO] Cleaning up dataset files...")
        clean_up_dataset()
    if args.task == "clean-up-klee-files":
        print("[Utils][INFO] Cleaning KLEE runtime files...")
        clean_up_klee_files()
    if args.task == "kill-klee":
        print("[Utils][INFO] Killing KLEE...")
        kill_klee()
    if args.task == "parse-merge-jsons":
        print("[Utils][INFO] Parsing JSON files...")
        scan_json_dir(MERGE_RECORD_DIR, test=args.debug)
    if args.task == "gen-programl-graphs":
        print("[Utils][INFO] Generating ProGraML graphs...")
        gen_programl_graphs(IR_PROGRAML_DIR, concurrent=True)
    if args.task == "gen-merge-graphs":
        if args.graph_size == -1:
            raise argparse.ArgumentError(
                "[Utils][ERROR] Graph size not specified.")
        MERGE_GRAPH_DIR += "-%d" % args.graph_size

        print("[Utils][INFO] Generating MergeGraphs...")
        print("[Utils][INFO] Step #1: Loading all MergeRecords...")
        merge_records = scan_json_dir(
            MERGE_RECORD_DIR, concurrent=True, test=args.debug)
        print("[Utils][INFO] Step #2: Loading all ProGraML graphs...")
        programl_graphs = scan_programl_graph_dir(IR_PROGRAML_DIR)
        num_merges = 0
        for _, merge_ids in merge_records.items():
            num_merges += len(merge_ids)
        print("[Utils][INFO] # IR files: %d | # merges: %d | # ProGraML graphs: %d..." %
              (len(merge_records), num_merges, len(programl_graphs)))
        print("[Utils][INFO] Step #3: Generating MergeGraphs...")
        # Let us cache things to reduce overhead
        # _GLOBAL_FUTURE_GRAPH_CACHE = {}  # moved to the top of the file
        # All filenames are returned instead of MergeGraph objects (for avoiding RAM issues)
        merge_graphs = gen_merge_graphs(merge_records, programl_graphs)
    if args.task == "split-dataset":
        if args.graph_size == -1:
            raise argparse.ArgumentError(
                "[Utils][ERROR] Graph size not specified.")
        MERGE_GRAPH_DIR += "-%d" % args.graph_size

        print("[Utils][INFO] Splitting dataset into train/val/test sets...")
        split_dataset(balance=True, merge_graph_variant=SPLIT_VAR_INCLUSION)
    if args.task == "split-dataset-by-program":
        if args.graph_size == -1:
            raise argparse.ArgumentError(
                "[Utils][ERROR] Graph size not specified.")
        MERGE_GRAPH_DIR += "-%d" % args.graph_size

        print(
            "[Utils][INFO] Splitting dataset into train/val/test sets by program names...")
        split_dataset_by_name(
            balance=True, merge_graph_variant=SPLIT_VAR_INCLUSION)
    if args.task == "split-dataset-by-merge-point":
        if args.graph_size == -1:
            raise argparse.ArgumentError(
                "[Utils][ERROR] Graph size not specified.")
        MERGE_GRAPH_DIR += "-%d" % args.graph_size

        print(
            "[Utils][INFO] Splitting dataset into train/val/test sets by merge point files...")
        split_dataset_by_name(
            balance=True, merge_graph_variant=SPLIT_VAR_INCLUSION, by_merge_point_file=True)
    if args.task == "get-dataset-stats":
        if args.graph_size == -1:
            raise argparse.ArgumentError(
                "[Utils][ERROR] Graph size not specified.")
        MERGE_GRAPH_DIR += "-%d" % args.graph_size

        print("[Utils][INFO] Getting stats for dataset...")
        get_dataset_stats()
    if args.task == "visualize-loss-curve":
        print("[Utils][INFO] Visualizing loss curve...")
        visualize_loss_curve(args.in_file)
    if args.task == "analyze-merge-records":
        print("[Utils][INFO] Analyzing generated merge records...")
        analyze_merge_records()
    if args.task == "compile-program-list":
        print("[Utils][INFO] Compiling program list...")
        compile_program_list(args.in_dir, args.out_file)
    if args.task == "gen-vocab":
        if args.graph_size == -1:
            raise argparse.ArgumentError(
                "[Utils][ERROR] Graph size not specified.")
        MERGE_GRAPH_DIR += "-%d" % args.graph_size

        print("[Utils][INFO] Generating vocabulary...")
        gen_vocab(args.in_dir, args.out_file)
    if args.task == "test-auto-merge":
        print("[Utils][INFO] Testing auto-merge KLEE...")
        test_auto_merge()
    if args.task == "visualize-a-merge-graph":
        print("[Utils][INFO] Visualizing the given merge graph...")
        visualize_a_merge_graph(args.in_file, args.out_file)
    if args.task == "test-gen-merge-graph":
        print("[Utils][INFO] Testing gen_merge_graph()...")
        test_gen_merge_graph(args.in_file, args.add_in_arg, args.out_file)
