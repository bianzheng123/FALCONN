cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_COMPILER "g++")
#set(CMAKE_C_COMPILER "gcc")

project(FALCONN LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

string(ASCII 27 Esc)
set(ColourReset "${Esc}[m")
set(Green "${Esc}[32m")
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message("${Green}Release mode${ColourReset}")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message("${Green}Clang${ColourReset}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Ofast -DNDEBUG -openmp -march=native -fpic -ftree-vectorize ${OpenMP_CXX_FLAGS}")
        add_definitions(-O3 -DNDEBUG -openmp -march=native -fpic -ftree-vectorize ${OpenMP_CXX_FLAGS})
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message("${Green}GNU${ColourReset}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Ofast -lrt -DNDEBUG -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 ${OpenMP_CXX_FLAGS}")
        add_definitions(-O3 -lrt -DNDEBUG -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 ${OpenMP_CXX_FLAGS})
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message("${Green}MSVC${ColourReset}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Ofast -lrt -DNDEBUG -openmp -march=native -fpic -w -fopenmp -ftree-vectorize ${OpenMP_CXX_FLAGS}")
        add_definitions(-O3 -lrt -DNDEBUG -openmp -march=native -fpic -w -fopenmp -ftree-vectorize ${OpenMP_CXX_FLAGS})
    else ()
        message("${Green}Unknown compiler${ColourReset}")
    endif ()
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    #    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -march=native -fsanitize=address -lasan ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -march=native ${OpenMP_CXX_FLAGS}")
    #    add_definitions(-DNDEBUG -march=native)
    message("${Green}Debug mode${ColourReset}")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native ${OpenMP_CXX_FLAGS}")
    message("${Green}Unknown mode${ColourReset}")
endif ()

#find_package(Python3 COMPONENTS Interpreter Development)

# Header only dependencies
include_directories(external/eigen/Eigen)
#add_subdirectory(external/googletest)
#include_directories(external/googletest)
include_directories(external/nlohmann/nlohmann)
add_subdirectory(external/pybind11)
#include_directories(external/pybind11)
include_directories(external/simple-serializer)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

#pybind11_add_module(GreedyProbeEnd2End src/app/end2end/GreedyProbeEnd2End.cpp)
#target_link_libraries(GreedyProbeEnd2End PRIVATE OpenMP::OpernMP_CXX Eigen3::Eigen spdlog::spdlog ${OpenBLAS_LIB})
#target_include_directories(GreedyProbeEnd2End PRIVATE ${OpenBLAS_INCLUDE_DIR})

#add_executable(ipnsw src/app/IP/ipnsw_cpp.cpp)
#target_link_libraries(ipnsw PRIVATE OpenMP::OpenMP_CXX spdlog::spdlog)

add_executable(bench src/benchmark/random_benchmark.cc)
target_include_directories(bench PRIVATE src/include external/eigen)
#target_link_libraries(bench GTest::gtest_main)

enable_testing()

add_executable(lsh_table_test src/test/lsh_table_test.cc)
target_include_directories(lsh_table_test PRIVATE src/include external/eigen)
target_link_libraries(lsh_table_test GTest::gtest_main)

add_executable(nn_query_test src/test/nn_query_test.cc)
target_include_directories(nn_query_test PRIVATE src/include external/eigen)
target_link_libraries(nn_query_test GTest::gtest_main)

add_executable(glove src/examples/glove/glove.cc)
target_include_directories(glove PRIVATE src/include external/eigen)
target_link_libraries(glove GTest::gtest_main)

add_executable(main src/examples/self/main.cc)
target_include_directories(main PRIVATE src/include external/eigen)
target_link_libraries(main GTest::gtest_main)

#include(GoogleTest)
#gtest_discover_tests(lsh_table_test)

#add_executable(t1 test.cpp)
#target_link_libraries(t1 PRIVATE spdlog::spdlog)

