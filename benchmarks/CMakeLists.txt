# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

set(NSPARSE_BENCHMARK_SRC
    index_search_benchmark.cpp
)
# Fetch Google Benchmark
include(FetchContent)
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

# google-benchmark's internal libpfm perf counters need libpfm4-dev and are not
# required by the benchmark protocol (it uses external `perf stat`). Leave OFF so
# the build stays portable on a fresh benchmark host without extra packages.
set(BENCHMARK_ENABLE_LIBPFM OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googlebenchmark)

if(NSPARSE_BENCHMARK_SRC)
    find_package(OpenMP REQUIRED)

    add_executable(nsparse_benchmark ${NSPARSE_BENCHMARK_SRC})

    target_include_directories(nsparse_benchmark PRIVATE ${PROJECT_SOURCE_DIR})

    target_link_libraries(nsparse_benchmark PRIVATE
        nsparse
        OpenMP::OpenMP_CXX
        benchmark::benchmark
        benchmark::benchmark_main
        absl::flat_hash_set
        absl::flat_hash_map
    )
endif()

# Latency / residency / recall driver for a single serialized index. Not a
# google-benchmark target: it reports the per-query tail, the RssAnon/RssFile
# split and recall, none of which that harness can express.
add_executable(sq_residency_bench sq_residency_bench.cpp)
target_include_directories(sq_residency_bench PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(sq_residency_bench PRIVATE
    nsparse
    OpenMP::OpenMP_CXX
    absl::flat_hash_set
    absl::flat_hash_map
)

# Build-time benchmark: SeismicIndex build() on CPU vs GPU (cuSPARSE). The GPU
# case is compiled in only when the nsparse target carries CUDA support (the
# NSPARSE_WITH_GPU define propagates from the nsparse target).
add_executable(nsparse_build_benchmark index_build_benchmark.cpp)
target_include_directories(nsparse_build_benchmark PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(nsparse_build_benchmark PRIVATE
    nsparse
    OpenMP::OpenMP_CXX
    benchmark::benchmark
    absl::flat_hash_set
    absl::flat_hash_map
)
