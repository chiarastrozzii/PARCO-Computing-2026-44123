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

    MPI_Scatterv(row_buf, nnz_rank, displs, MPI_INT, row_local, local_nnz, MPI_INT, 0, MPI_COMM_WORLD); //works correctly since row_buf is grouped by rank
    MPI_Scatterv(col_buf, nnz_rank, displs, MPI_INT, col_local, local_nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(val_buf, nnz_rank, displs, MPI_DOUBLE, val_local, local_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(row_buf);
        free(col_buf);
        free(val_buf);
        free(displs);
    }
}