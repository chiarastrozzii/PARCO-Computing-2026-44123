cmake_minimum_required(VERSION 3.16)
project(SpMV_MPI LANGUAGES C)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(spmv
    SpMV_MPI.c
    communications/comm.c
    config/config.c
    multiplication/mul.c
)

target_link_libraries(spmv
    MPI::MPI_C
    OpenMP::OpenMP_C
)