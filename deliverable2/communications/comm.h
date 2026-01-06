#ifndef COMM_H
#define COMM_H

#include <stddef.h>

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

#endif // COMM_H