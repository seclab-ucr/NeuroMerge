klee_add_component(ToolLib
        BasicHeaders.h
        LLVMOps.h
        Logging.h 
        LLVMOps.cpp
        )

set(LLVM_COMPONENTS
        core
        executionengine
        mcjit
        native
        support
        )

klee_get_llvm_libs(LLVM_LIBS ${LLVM_COMPONENTS})
target_link_libraries(ToolLib PUBLIC ${LLVM_LIBS})