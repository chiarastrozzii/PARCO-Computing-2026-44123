#ifndef COMM_H
#define COMM_H

#include <stddef.h>
#include <mpi.h>

void scatter_entries(    
    int rank,
    int size,
    int n_rows,
    int n_nz,
    const int *row_indices,
    const int *col_indices,
    const double *values,
    const int *nnz_rank,
    int local_nnz,
    int *row_local,
    int *col_local,
    double *val_local
);

void scatter_entries_2D(
    int rank,
    int p, int q,
    MPI_Comm grid_comm,
    int n_rows,
    int n_cols,
    int n_nz,
    const int *row_indices,
    const int *col_indices,
    const double *values,
    const int *nnz_rank,
    int local_nnz,
    int *row_local,
    int *col_local,
    double *val_local
);

int block_start(int coord, int n, int p);

#endif // COMM_H