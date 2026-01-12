#ifndef MUL_H
#define MUL_H

#include "../config/config.h"
#include <mpi.h>
#include <stdbool.h>


void spmv(const Sparse_CSR* sparse_csr, const double* vec, double* res, int parallel);
int block_size(int coord, int n, int p);
static int cmp_int(const void* a, const void *b);
double *prepare_x_1D(const Sparse_CSR *csr, const double *x_owned, int x_owned_len, int rank, int size, int **col_map_out, int *local_x_size, int *tot_send, int *tot_recv);
static int col_owner(int col, int size);
static int cmp_int(const void* a, const void *b);
void remapping_columns(Sparse_CSR *csr, int *col_map, int local_x_size, int rank);
double *gather_res_1D(double *local_result, int actual_local_rows, int local_nnz, int *row_local, int n_rows, int rank, int size);
double *gather_res_2D(double* local_result,int n_rows, int p, int q, int pr, int pc, MPI_Comm grid_comm);

#endif // MUL_H 