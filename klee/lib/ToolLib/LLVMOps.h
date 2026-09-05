

#ifndef INC_LLVM_RELATED_H
#define INC_LLVM_RELATED_H

#include "BasicHeaders.h"

llvm::BasicBlock *get_real_basic_block(llvm::BasicBlock *b);

llvm::BasicBlock *get_final_basic_block(llvm::BasicBlock *b);

std::string get_file_name(const llvm::Function *f);

std::string get_real_function_name(const llvm::Function *f);

std::string get_structure_name(std::string name);

void dump_inst(llvm::Instruction *inst);

std::string dump_inst_booltin(llvm::Instruction *inst);

std::string real_inst_str(std::string str);

/// Compute the true target of a function call, resolving LLVM aliases
/// and bitcasts.
llvm::Function *get_target_function(llvm::Value *calledVal);

#define llvm_print(type, empty, out, print, str) \
    if ((type) >= DEBUG_LEVEL)                        \
    {                                                 \
        if ((empty) == 1)                             \
        {                                             \
            (str) = "";                               \
        }                                             \
        llvm::raw_string_ostream dump(str);           \
        print(dump);                                  \
        if ((out) == 1)                               \
        {                                             \
            log(type, str);                      \
        }                                             \
    }

#define add(print, str) llvm_print(4, 0, 0, print, str)
#define print(print, str) llvm_print(4, 1, 0, print, str)
#define dump(type, print, str) llvm_print(type, 1, 1, print, str)
#define dump_debug(print, str) dump(0, print, str)
#define dump_add(type, print, str) llvm_print(type, 0, 1, print, str)
#define dump_add_debug(print, str) dump_add(0, print, str)

// strID: Path-NameFunction-NoBB-NoInst
std::string function_to_strID(const llvm::Function *f);

std::string basicblock_to_strID(const llvm::BasicBlock *b);

std::string inst_to_strID(const llvm::Instruction *inst);

llvm::Function *strID_to_function(llvm::Module *m, const std::string &str);

llvm::BasicBlock *strID_to_basicblock(llvm::Module *m, const std::string &str);

llvm::Instruction *strID_to_inst(llvm::Module *m, const std::string &str);

// Some constants for blocking/allowing keywords
const static std::vector<std::string> MERGE_POINT_FILENAME_KEYWORD_ALLOWLIST = {};
const static std::vector<std::string> MERGE_POINT_DIRNAME_KEYWORD_ALLOWLIST = {};

// Maximum number of merges to process
const long unsigned int MAX_NUM_MERGES = 100000;

#endif //INC_LLVM_RELATED_H
