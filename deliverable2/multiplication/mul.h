#ifndef MUL_H
#define MUL_H

#include "../config/config.h"
#include <mpi.h>


void spmv(const Sparse_CSR* sparse_csr, const double* vec, double* res, int parallel);
int block_size(int coord, int n, int p);
double *gather_res_1D(double *local_result, int actual_local_rows, int local_nnz, int *row_local, int n_rows, int rank, int size);
double *gather_res_2D(double* local_result,int n_rows, int p, int q, int pr, int pc, MPI_Comm grid_comm);

#endif // MUL_H 