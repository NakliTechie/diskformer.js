# diskformer × llama.cpp driver. LLAMA_DIR: a checkout of NakliTechie/llama.cpp, branch webgpu-disk-tier.
#   native:  CMAKE_PREFIX_PATH=<dawn> cmake -B build -DLLAMA_DIR=... && cmake --build build --target df-chat
#   browser: emcmake cmake -B build-wasm -DLLAMA_DIR=... -DEMDAWNWEBGPU_DIR=<pkg> && cmake --build build-wasm --target df-chat
cmake_minimum_required(VERSION 3.20)
project(diskformer-ggml CXX C)

set(LLAMA_DIR "" CACHE PATH "llama.cpp checkout (branch webgpu-disk-tier)")
if(NOT LLAMA_DIR)
    message(FATAL_ERROR "set -DLLAMA_DIR")
endif()

set(GGML_WEBGPU ON CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
if(EMSCRIPTEN)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
endif()
add_subdirectory(${LLAMA_DIR} llama EXCLUDE_FROM_ALL)

add_executable(df-chat driver.cpp)
target_link_libraries(df-chat PRIVATE llama-common)
target_compile_features(df-chat PRIVATE cxx_std_17)

if(EMSCRIPTEN)
    # llama.cpp builds its wasm as memory64 (LLAMA_WASM_MEM64), so the driver must too
    set_target_properties(df-chat PROPERTIES SUFFIX ".mjs" LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/df_gpu.js)
    target_compile_options(df-chat PRIVATE -sMEMORY64=1)
    target_link_options(df-chat PRIVATE
        -sMEMORY64=1 -sMODULARIZE -sEXPORT_ES6 -sENVIRONMENT=worker
        -sALLOW_MEMORY_GROWTH -sMAXIMUM_MEMORY=8GB -sSTACK_SIZE=8MB
        -lworkerfs.js --js-library ${CMAKE_CURRENT_SOURCE_DIR}/df_gpu.js
        "-sEXPORTED_FUNCTIONS=['_df_load','_df_chat','_df_complete','_df_stats','_malloc','_free']"
        "-sJSPI_EXPORTS=['df_load','df_chat','df_complete']"
        "-sEXPORTED_RUNTIME_METHODS=['FS','WORKERFS','UTF8ToString','stringToNewUTF8']")
endif()
