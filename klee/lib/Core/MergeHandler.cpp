//===-- MergeHandler.cpp --------------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "klee/MergeHandler.h"
#include "klee/Interpreter.h"
#include "CoreStats.h"
#include "Executor.h"
#include "Searcher.h"
#include "klee/ExecutionState.h"
#include "klee/Internal/Support/ErrorHandling.h"
#include "../Inference/inference_client.cc"
#include "../ToolLib/LLVMOps.h"
#include "StatsTracker.h"
namespace klee {

/*** Test generation options ***/

llvm::cl::OptionCategory MergeCat("Path merging options",
                                  "These options control path merging.");

llvm::cl::opt<bool> UseMerge(
    "use-merge", llvm::cl::init(false),
    llvm::cl::desc("Enable support for path merging via klee_open_merge() and "
                   "klee_close_merge() (default=false)"),
    llvm::cl::cat(klee::MergeCat));

llvm::cl::opt<bool> DebugLogMerge(
    "debug-log-merge", llvm::cl::init(false),
    llvm::cl::desc("Debug information for path merging (default=false)"),
    llvm::cl::cat(klee::MergeCat));

llvm::cl::opt<bool> UseIncompleteMerge(
    "use-incomplete-merge", llvm::cl::init(false),
    llvm::cl::desc("Heuristic-based path merging (default=false)"),
    llvm::cl::cat(klee::MergeCat));

llvm::cl::opt<bool> DebugLogIncompleteMerge(
    "debug-log-incomplete-merge", llvm::cl::init(false),
    llvm::cl::desc("Debug information for incomplete path merging (default=false)"),
    llvm::cl::cat(klee::MergeCat));

llvm::cl::opt<int> TorchServerPort( // add port flag support
    "port", llvm::cl::init(50051),
    llvm::cl::desc("Specify the port number for the torch server (default=50051)"),
    llvm::cl::cat(klee::MergeCat));

llvm::cl::opt<std::string> MergeDecisionMode(
        "merge-decision-mode", 
        llvm::cl::init("query"), 
        llvm::cl::desc("Specifies the merge decision mode: random, query, merge, unmerge"),
        llvm::cl::cat(MergeCat));

double MergeHandler::getMean() {
  if (closedStateCount == 0)
    return 0;

  return closedMean;
}

unsigned MergeHandler::getInstructionDistance(ExecutionState *es){
  return es->steppedInstructions - openInstruction;
}

ExecutionState *MergeHandler::getPrioritizeState(){
  for (ExecutionState *cur_state : openStates) {
    bool stateIsClosed =
        (executor->mergingSearcher->inCloseMerge.find(cur_state) !=
         executor->mergingSearcher->inCloseMerge.end());

    if (!stateIsClosed && (getInstructionDistance(cur_state) < 2 * getMean())) {
      return cur_state;
    }
  }
  return 0;
}


void MergeHandler::addOpenState(ExecutionState *es){
  openStates.push_back(es);
}

void MergeHandler::removeOpenState(ExecutionState *es){
  auto it = std::find(openStates.begin(), openStates.end(), es);
  assert(it != openStates.end());
  std::swap(*it, openStates.back());
  openStates.pop_back();
}
std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        std::cout << "popen() failed!";
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}
void MergeHandler::addClosedState(ExecutionState *es,
                                         llvm::Instruction *mp) {
  // Update stats
  ++closedStateCount; // mp is the merge point
  // Convert instruction to string using LLVM's raw_string_ostream

  // std::string instrStr;
  // llvm::raw_string_ostream rso(instrStr);
  // mp->print(rso);
  // klee_message("state %p reaches close point at %p, the instruction is %s", es, mp, rso.str().c_str());

  // Incremental update of mean (travelling mean)
  // https://math.stackexchange.com/a/1836447
  closedMean += (static_cast<double>(getInstructionDistance(es)) - closedMean) /
               closedStateCount;

  // Remove from openStates
  removeOpenState(es);

  auto closePoint = reachedCloseMerge.find(mp);

  // If no other state has yet encountered this klee_close_merge instruction,
  // add a new element to the map
  if (closePoint == reachedCloseMerge.end()) {
    reachedCloseMerge[mp].push_back(es);
    executor->mergingSearcher->pauseState(*es);
    // klee_message("state %p is paused and added to reachedCloseMerge", es);
  } else {
    // Otherwise try to merge with any state in the map element for this
    // instruction
    auto &cpv = closePoint->second;
    bool mergedSuccessful = false;

    // X: do not merge by default
    cpv.push_back(es);
    executor->mergingSearcher->pauseState(*es);
    klee_message("state %p is paused", es);
    return;

    for (auto& mState: cpv) {
      if (mState->merge(*es)) {
        executor->terminateState(*es);
        executor->mergingSearcher->inCloseMerge.erase(es);
        mergedSuccessful = true;
        break;
      }
    }
    if (!mergedSuccessful) {
      cpv.push_back(es);
      executor->mergingSearcher->pauseState(*es);
    }
  }
}

// void printCallStack() {
//     void *array[10];
//     size_t size;
//     char **strings;
//     size_t i;

//     // Get void*'s for all entries on the stack
//     size = backtrace(array, 10);

//     // Print out all the frames to stderr
//     strings = backtrace_symbols(array, size);

//     std::cout << "Call stack:" << std::endl;

//     for (i = 0; i < size; i++) {
//         std::cout << strings[i] << std::endl;
//     }

//     free(strings);
// }

static std::string print_stack_trace() {
  std::string str;
  llvm::raw_string_ostream dump(str);
  llvm::sys::PrintStackTrace(dump);
  return str;
}


void printInstructionWithKleeMessage(klee::ExecutionState *mState) {
    if (mState->pc->inst) {
        std::string instStr;
        llvm::raw_string_ostream rso(instStr);
        mState->pc->inst->print(rso);

        // Now use klee_message to print the instruction information
        klee_message("merge successful at instruction: %s", rso.str().c_str());
    } else {
        klee_message("No instruction available to print.");
    }
}

unsigned countImmediatePredecessors(llvm::BasicBlock *block) {
    return std::distance(pred_begin(block), pred_end(block));
}

void MergeHandler::releaseStates() {
  std::string stackTrace = print_stack_trace();
  // klee_message("stacktrace of releaseStates(), Stack Trace: %s", stackTrace.c_str());// create print, this
  // std::cout << "MergeHandler is deleted " <<  this << std::endl;
  for (auto& curMergeGroup: reachedCloseMerge) {
    llvm::Instruction* closePointInst = curMergeGroup.first;
    std::string instStr;
    llvm::raw_string_ostream rso(instStr);
    closePointInst->print(rso);

    // Start by logging the instruction of the close point
    // klee_message("Executor starts releasing state at close point: %s", rso.str().c_str());

    for (auto curState: curMergeGroup.second) {
      klee_message("Releasing state: %p at merge point instruction: %s", curState, rso.str().c_str());
      executor->mergingSearcher->continueState(*curState);
      executor->mergingSearcher->inCloseMerge.erase(curState); // states in inCloseMerge are states paused and scheduled for merge
    }
  }

  // X: try to merge here
  for (auto &curMergeGroup : reachedCloseMerge) {
    // move states from curMergeGroup to merge_grounp
    std::set<ExecutionState *> merge_grounp;
    merge_grounp.insert(curMergeGroup.second.begin(),
                        curMergeGroup.second.end());
    curMergeGroup.second.clear();
    if (merge_grounp.size() > 1) {
      klee_message("merge_grounp begin with number of states: %ld, begin with state %p", merge_grounp.size(), *merge_grounp.begin());
      // for (auto state : merge_grounp) {
      //   std::string instrStr;
      //   llvm::raw_string_ostream rso(instrStr);
      //   state->pc->inst->print(rso);
      //   llvm::BasicBlock *cur_block = state->pc->inst->getParent();
      //   unsigned numPredecessors = countImmediatePredecessors(cur_block);
      //   klee_message("Current instruction at state %p, the number of blocks point to current block %u, Instruction: %s, Source location %s", state, cur_block, rso.str().c_str(), state->pc->getSourceLocation().c_str());
      //   // std::cout << "Current instruction: " << *(state->pc->inst) << "Source location: " << state->pc->getSourceLocation();
      // }
    }

    // for all states, try to merge them in different group
    std::vector<std::vector<ExecutionState *> *> all_merged_grounp;
    bool merge_flag = false;
    while (!merge_grounp.empty()) {
      // find states that can merge in merge_grounp
      auto mState = *merge_grounp.begin();
      merge_grounp.erase(mState);

      std::vector<ExecutionState *> temp_merge_group; // can be m_state in combination_state?
      nlohmann::json json;
      for (auto temp_state : merge_grounp) {
        if (mState->canMerge(*temp_state)) { // check for some KLEE conditions for merge
          // std::cout << "at instruction" << *(mState->pc->inst) << " it can merge";
          printInstructionWithKleeMessage(mState);
          mState->preMerge(*temp_state, json);
          temp_merge_group.push_back(temp_state);
          // klee_message("with base state %p, state %p can be merged", mState, temp_state);
        }
        // else{
        //   std::cout << "at instruction" << *(mState->pc->inst) << " it can't merge";
        // }
      }

      // remove states that can merge in merge_grounp
      for (auto temp_state : temp_merge_group) {
        merge_grounp.erase(temp_state);
      }

      // whether mrege or not
      auto temp_merged_group = new std::vector<ExecutionState *>;
      temp_merged_group->push_back(mState);
      if (temp_merge_group.empty()) {
        // not try to merge if no possible state
        // klee_message("there is no state to be merged");
      } else {
        klee_message("number of states can be merged: %ld", temp_merge_group.size());
        // Get the current instruction
        llvm::Instruction* curInst = mState->pc->inst;

        // Reference to the cache in the executor
        auto& cache = executor->mergeDecisionCache;

        bool decision;

        // Check if the decision is cached
        if (cache.find(curInst) != cache.end()) {
          // Use cached decision
          decision = cache[curInst];
          klee_message("Using cached decision for instruction %p: %s", curInst, decision ? "merge" : "don't merge");
        } else {
          // Call merge_or_not and cache the result
          decision = merge_or_not(mState, json);
          cache[curInst] = decision;
          klee_message("Caching decision for instruction %p: %s", curInst, decision ? "merge" : "don't merge");
        }
        if (decision) { // check whether mState is merge or not merge state
          // std::cout << "one merge is happening"<<std::endl;
          // std::cout << "check code change"<<std::endl;
          for (auto &es : temp_merge_group) {
            // Log each state before attempting to merge
            klee_message("Attempting to merge state %p with %p", mState, es);
            if (mState->merge(*es)) {
              // revoke pauseState() in addClosedState()
              // always first add and then remvoe
              // things for merge
              merge_flag = true;
              klee_message("merge done. Successfully merged into mState %p. Terminate merged states: %p", mState, es);
              executor->terminateState(*es);
              break;
            } else {
              klee_message("Merge failed. Keeping state %p for further attempts", es);
              temp_merged_group->push_back(es);
            }
          }
          // if (merge_flag){
            // klee_message("Merge happened in this merge point");
            // executor->statsTracker->performStatsLineWrite();
            // executor->statsTracker->performIStatsWrite();
            // const char* outputfolder = globalOutputDirectory.c_str();
            // std::string command = "klee-stats " + std::string(outputfolder);
            // std::string output = exec(command.c_str()); 
            // std::ofstream outFile(std::string(outputfolder) + "/merge_gap.txt", std::ios::app); // the folder is generated
            // if (outFile.is_open()) {
            //   outFile << output;
            //   klee_message("current wall time(before query) added to query_gap.txt");
            //   outFile.close();
            // } else {
            //   std::cerr << "Unable to open file for writing coverage information.\n";
            // }
          // }
        } else {
          temp_merged_group->insert(temp_merged_group->end(),
                                    temp_merge_group.begin(),
                                    temp_merge_group.end());
        }
      }
      all_merged_grounp.push_back(temp_merged_group);
    }
    // if (merge_flag){
    //   klee_message("Merge happened in this merge point");
    //   executor->statsTracker->performStatsLineWrite();
    //   executor->statsTracker->performIStatsWrite();
    //   const char* outputfolder = globalOutputDirectory.c_str();
    //   std::string command = "klee-stats " + std::string(outputfolder);
    //   std::string output = exec(command.c_str()); 
    //   std::ofstream outFile(std::string(outputfolder) + "/merge_gap.txt", std::ios::app); // the folder is generated
    //   if (outFile.is_open()) {
    //     outFile << output;
    //     klee_message("current wall time(before query) added to query_gap.txt");
    //     outFile.close();
    //   } else {
    //     std::cerr << "Unable to open file for writing coverage information.\n";
    //   }
    // }
    // else{
    //   klee_message("Merge didn't happen in this merge point");
    //   executor->statsTracker->performStatsLineWrite();
    //   executor->statsTracker->performIStatsWrite();
    //   const char* outputfolder = globalOutputDirectory.c_str();
    //   std::string command = "klee-stats " + std::string(outputfolder);
    //   std::string output = exec(command.c_str()); 
    //   std::ofstream outFile(std::string(outputfolder) + "/merge_gap.txt", std::ios::app); // the folder is generated
    //   if (outFile.is_open()) {
    //     outFile << output;
    //     klee_message("current wall time(before query) added to query_gap.txt");
    //     outFile.close();
    //   } else {
    //     std::cerr << "Unable to open file for writing coverage information.\n";
    //   }
    // }

    // add merged states back to curMergeGroup
    for (auto temp_merge_vector : all_merged_grounp) {
      for (auto es : *temp_merge_vector) {
        merge_grounp.insert(es);
      }
    }
    for (auto temp_merge_vector : all_merged_grounp) {
      delete temp_merge_vector;
    }
    if (merge_grounp.size() > 1) {
      klee_message("merge_grounp end with number of states: %ld", merge_grounp.size());
    }
    curMergeGroup.second.insert(curMergeGroup.second.begin(),
                                merge_grounp.begin(), merge_grounp.end());
  }

  reachedCloseMerge.clear();
}

bool MergeHandler::hasMergedStates() {
  return (!reachedCloseMerge.empty());
}

// X: TODO: @X update this function
bool MergeHandler::merge_or_not(ExecutionState * es, nlohmann::json &json) {
  // std::this_thread::sleep_for(std::chrono::seconds(12));

  // executor->statsTracker->performStatsLineWrite();
  // executor->statsTracker->performIStatsWrite();
  // const char* outputfolder = globalOutputDirectory.c_str();
  // std::string command = "klee-stats " + std::string(outputfolder);
  // std::string before_output = exec(command.c_str()); 
  // std::ofstream before_outFile(std::string(outputfolder) + "/query_gap.txt", std::ios::app); // the folder is generated
  // if (before_outFile.is_open()) {
  //   before_outFile << before_output;
  //   klee_message("current wall time(before query) added to query_gap.txt");
  //   before_outFile.close();
  // } else {
  //   std::cerr << "Unable to open file for writing coverage information.\n";
  // }

  if (klee::MergeDecisionMode == "random") {
    //klee_message("use random strategy");
        return (rand() % 2) == 0;  // Randomly return true or false
    } else if (klee::MergeDecisionMode == "merge") {
      //klee_message("use all merge strategy");
        return true;  // Always merge
    } else if (klee::MergeDecisionMode == "unmerge") {
      //klee_message("use all not merge strategy");
        return false;  // Never merge
    } else if (klee::MergeDecisionMode == "query") {
  json["Merge_ID"] = "dsfasd";
  json["Prev_merge_point_00"] = "dsfasd";
  json["Prev_merge_point_01"] = "dsfasd";
  json["IR_filename"] = es->inputfile;
  std::cout << json["IR_filename"] <<std::endl;
  json["Merge_point_addr"] = inst_to_strID(es->pc->inst);
  json["Merged_exploration_time"] = "0.040674";
  json["Unmerged_exploration_time"] = "0.054674";
  json["const_size"] = es->constraints.size();
  json["expl_depth"] = es->depth;
  json["instruction_count"] = es->steppedInstructions;
  json["instsSinceCovNew"] = es->instsSinceCovNew;
  json["number_of_symbolics"] = es->symbolics.size();
  json["stack_depth"] = es->stack.size();
  // json["coveredNew"] = es->coveredNew;
  if (es->coveredNew){
    json["coveredNew"] = 1;
  }
  else{
    json["coveredNew"] = 0;
  }
  // klee_message(json.dump().c_str());
  klee_message("[INFO] Firing client now...");

  // InferenceClient client(
  //   grpc::CreateChannel("localhost:7777",
  //                       grpc::InsecureChannelCredentials()));
  // bool is_merge = client.infer(json);

  char name[40];
  tmpnam(name);
  std::string filename = "/data/X/NeuSE/temp_jsons";
  filename += name;
  filename += ".json";
  std::cout << filename<<std::endl;
  std::ofstream file(filename);
  file << json.dump();



  // std::string cmd ="curl http://localhost:50051/predictions/mergegraph0429 -T " + filename;
  // executor->statsTracker->performStatsLineWrite();
  // executor->statsTracker->performIStatsWrite();
  // const char* outputfolder = globalOutputDirectory.c_str();
  // std::string command = "klee-stats " + std::string(outputfolder);
  // std::string before_output = exec(command.c_str()); 
  // std::ofstream before_outFile(std::string(outputfolder) + "/query_gap.txt", std::ios::app); // the folder is generated
  // if (before_outFile.is_open()) {
  //   before_outFile << before_output;
  //   klee_message("current wall time(before query) added to query_gap.txt");
  //   before_outFile.close();
  // } else {
  //   std::cerr << "Unable to open file for writing coverage information.\n";
  // }
  std::string cmd = "curl http://localhost:" + std::to_string(TorchServerPort.getValue()) + "/predictions/mergegraph0429 -T " + filename; // updated command to include port through flag
  auto start = std::chrono::high_resolution_clock::now();
  std::string res = exec(cmd.c_str());
  auto end = std::chrono::high_resolution_clock::now();
  auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  // auto ms_int = duration_cast<std::chrono::milliseconds>(end - start);
  infer_exec_duration += ms_int.count();
  std::cout << "the current inference time is " << ms_int.count()<<std::endl;//change it to klee
  std::cout << "the current inference overall time is " << infer_exec_duration<<std::endl;//change it to klee
  // std::string after_output = exec(command.c_str());
  // std::ofstream after_outFile(std::string(outputfolder) + "/query_gap.txt", std::ios::app); // the folder is generated
  // if (after_outFile.is_open()) {
  //   after_outFile << after_output;
  //   klee_message("current wall time(after query) added to query_gap.txt");
  //   after_outFile.close();
  // } else {
  //   std::cerr << "Unable to open file for writing coverage information.\n";
  // }
  std::cout << "merge result is " << res<<std::endl;
  klee_message("Start printing command");
  klee_message(cmd.c_str());
  // std::string jsonStr = json.dump();
  //klee_message(jsonStr.c_str());
  //sleep(5000);
  //exit(0);

   


  cmd = "rm " + filename;
  exec(cmd.c_str());
  std::cout<<"one possible merge"<<std::endl;
  if(res.find("0") == std::string::npos){ // 0 not found in res
    // std::cout<<"0 not found in res. server returns unmerge decision"<<std::endl;
    return false;
  } else {
    // std::cout<<"0 found in res. server returns merge decision"<<std::endl;
    return true;
  }}
}

// static std::string print_stack_trace() {
//   std::string str;
//   llvm::raw_string_ostream dump(str);
//   llvm::sys::PrintStackTrace(dump);
//   return str;
// }

MergeHandler::MergeHandler(Executor *_executor, ExecutionState *es)
    : executor(_executor), openInstruction(es->steppedInstructions),
      closedMean(0), closedStateCount(0) {
    executor->mergingSearcher->mergeGroups.push_back(this);
  // std::string stackTrace = print_stack_trace();
  // klee_message("merge handler created, push into merge groups, Stack ID: %p, Stack Trace: %s", this, stackTrace.c_str());// create print, this
  // Log the state after removal for completeness
    // klee_message("Constructor: (after push this into mergegroups)");
    // for (auto& handler : executor->mergingSearcher->mergeGroups) {
    //     klee_message("Stack ID %p - Handler at %p", this, handler);
    // }
  addOpenState(es);
}


MergeHandler::~MergeHandler() {
  // std::cout<<"in ~MergeHandler"<<std::endl;
  // If no such element is found, it returns the end iterator of the range which is executor->mergingSearcher->mergeGroups.end()
  // klee_message("Entering destructor of MergeHandler %p", this);
  auto it = std::find(executor->mergingSearcher->mergeGroups.begin(),
                      executor->mergingSearcher->mergeGroups.end(), this);
  // Use the 'this' pointer address as part of the ID
  // std::string stackTrace = print_stack_trace();
  // klee_message("Stack ID: %p, Stack Trace: %s", this, stackTrace.c_str());
  // Log the state after removal for completeness
  // klee_message("Destructor: (before pop out of mergegroups)");
  // for (auto& handler : executor->mergingSearcher->mergeGroups) {
  //     klee_message("Stack ID %p - Handler at %p", this, handler);
  // }
  assert(it != executor->mergingSearcher->mergeGroups.end() &&
         "All MergeHandlers should be registered in mergeGroups");
  std::swap(*it, executor->mergingSearcher->mergeGroups.back());
  executor->mergingSearcher->mergeGroups.pop_back();

  releaseStates();
}
}
