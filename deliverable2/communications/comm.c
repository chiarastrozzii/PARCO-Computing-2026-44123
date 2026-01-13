#include "comm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <mpi.h>

void scatter_entries(    
    int rank,
    int size,
    int n_nz,
    const int *row_indices,
    const int *col_indices,
    const double *values,
    const int *nnz_rank,
    int local_nnz,
    int *row_local,
    int *col_local,
    double *val_local
){
    int *row_buf = NULL;
    int *col_buf = NULL;
    double *val_buf = NULL;
    int *displs = NULL;

    if (rank == 0) {
        row_buf = malloc(n_nz * sizeof(int));
        col_buf = malloc(n_nz * sizeof(int));
        val_buf = malloc(n_nz * sizeof(double));
        displs  = malloc(size * sizeof(int));

        displs[0] = 0; //displacement is the starting index for each process, it states where data starts for each process in the buffer
        for (int i=1; i<size; ++i)
            displs[i] = displs[i - 1] + nnz_rank[i - 1];

        int *cursor = calloc(size, sizeof(int)); //how many entries have been assigned to each process so far

        for (size_t i = 0; i < n_nz; ++i) {
            int owner = row_indices[i]%size; //determine which process owns the row
            int pos   = displs[owner] + cursor[owner]; //put the entry in the position of the next free slot for that process

            //global row → local row
            row_buf[pos] = row_indices[i]/size; //only row indeces are remapped
            col_buf[pos] = col_indices[i]; //column indices remain the same as they're needed for vector access, needed since we're changing the order
            val_buf[pos] = values[i];
            cursor[owner]++;
        }

        free(cursor);
    }

    const int *sendcounts_i = (rank == 0) ? nnz_rank : NULL;
    const int *displs_i     = (rank == 0) ? displs   : NULL;

    const int *row_buf_i    = (rank == 0) ? row_buf  : NULL;
    const int *col_buf_i    = (rank == 0) ? col_buf  : NULL;
    const double *val_buf_i = (rank == 0) ? val_buf  : NULL;

    MPI_Scatterv(row_buf_i, sendcounts_i, displs_i, MPI_INT, row_local, local_nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(col_buf_i, sendcounts_i, displs_i, MPI_INT, col_local, local_nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(val_buf_i, sendcounts_i, displs_i, MPI_DOUBLE, val_local, local_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(row_buf);
        free(col_buf);
        free(val_buf);
        free(displs);
    }
}

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
){
    int *row_buf = NULL;
    int *col_buf = NULL;
    double *val_buf = NULL;
    int *displs = NULL;

    if (rank == 0) {
        int size = p * q;

        row_buf = malloc(n_nz * sizeof(int));
        col_buf = malloc(n_nz * sizeof(int));
        val_buf = malloc(n_nz * sizeof(double));
        displs  = malloc(size * sizeof(int));

        displs[0] = 0;
        for (int i = 1; i < size; i++)
            displs[i] = displs[i - 1] + nnz_rank[i - 1];

        int *cursor = calloc(size, sizeof(int));

        for (int i = 0; i < n_nz; i++) {
            int pr = owner_block(row_indices[i], n_rows, p);
            int pc = owner_block(col_indices[i], n_cols, q);

            if (pr < 0 || pc < 0) {
                fprintf(stderr, "owner_block failed for (%d,%d)\n", row_indices[i], col_indices[i]);
                exit(1);
            }
        
            int coords[2] = {pr, pc};
            int owner;
            MPI_Cart_rank(grid_comm, coords, &owner);
        
            int pos = displs[owner] + cursor[owner]++;
        
            int row_start = block_start(pr, n_rows, p);
            int col_start = block_start(pc, n_cols, q);
        
            row_buf[pos] = row_indices[i] - row_start;
            col_buf[pos] = col_indices[i] - col_start;
            val_buf[pos] = values[i];
        }

        free(cursor);
    }

    const int *sendcounts_i = (rank == 0) ? nnz_rank : NULL;
    const int *displs_i     = (rank == 0) ? displs   : NULL;

    const int *row_buf_i    = (rank == 0) ? row_buf  : NULL;
    const int *col_buf_i    = (rank == 0) ? col_buf  : NULL;
    const double *val_buf_i = (rank == 0) ? val_buf  : NULL;

    MPI_Scatterv(row_buf_i, sendcounts_i, displs_i, MPI_INT,
                 row_local, local_nnz, MPI_INT, 0, grid_comm);

    MPI_Scatterv(col_buf_i, sendcounts_i, displs_i, MPI_INT,
                 col_local, local_nnz, MPI_INT, 0, grid_comm);

    MPI_Scatterv(val_buf_i, sendcounts_i, displs_i, MPI_DOUBLE,
                 val_local, local_nnz, MPI_DOUBLE, 0, grid_comm);

    if (rank == 0) {
        free(row_buf);
        free(col_buf);
        free(val_buf);
        free(displs);
    }
}

int block_start(int coord, int n, int p){
    int base = n / p;
    int rem  = n % p;

    if (coord < rem)
        return coord * (base + 1);
    else
        return coord * base + rem;
}

int block_size(int coord, int n, int p){
    int start = block_start(coord, n, p);
    int end = block_start(coord + 1, n, p);

    return end-start;
}

int owner_block(int idx, int n, int P) {
    int lo = 0, hi = P - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int s = block_start(mid, n, P);
        int e = block_start(mid + 1, n, P);
        if (idx < s) hi = mid - 1;
        else if (idx >= e) lo = mid + 1;
        else return mid;
    }
    return -1; // should never happen if idx in [0,n)
}