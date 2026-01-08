#include "mul.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <mpi.h>

void spmv(const Sparse_CSR* sparse_csr, const double* vec, double* res, int parallel){ //if everything is inside the same project we can maybe create a common module
    if (parallel == 1){
        #pragma omp parallel for schedule(runtime)
        for (size_t i=0; i<sparse_csr->n_rows; ++i){
            res[i] = 0;
            size_t nz_start = sparse_csr->row_ptrs[i];
            size_t nz_end = sparse_csr->row_ptrs[i+1];
        
            for (size_t j = nz_start; j < nz_end; ++j) {
                size_t col = sparse_csr->col_indices[j];
                double val = sparse_csr->values[j];
                res[i] += val * vec[col];
            }
        }
    }else if(parallel == 2){
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < sparse_csr->n_rows; ++i) {
            double sum = 0.0;
            size_t nz_start = sparse_csr->row_ptrs[i];
            size_t nz_end   = sparse_csr->row_ptrs[i+1];

            #pragma omp simd reduction(+:sum)
            for (size_t j = nz_start; j < nz_end; ++j) {
                size_t col = sparse_csr->col_indices[j];
                sum += sparse_csr->values[j] * vec[col];
            }

            res[i] = sum;
        }

    }else{
        for (size_t i=0; i<sparse_csr->n_rows; ++i){
            res[i] = 0;
            size_t nz_start = sparse_csr->row_ptrs[i];
            size_t nz_end = sparse_csr->row_ptrs[i+1];
        
            for (size_t j = nz_start; j < nz_end; ++j) {
                size_t col = sparse_csr->col_indices[j];
                double val = sparse_csr->values[j];
                res[i] += val * vec[col];
            }
        }
    }
    

}

int block_size(int coord, int n, int p) {
    int base = n / p;
    int rem  = n % p;
    return (coord < rem) ? base + 1 : base;
}

double *gather_res_1D(double *local_result, int actual_local_rows, int local_nnz, int *row_local, int n_rows, int rank, int size){
    //gather number of local rows for each rank
    for (int i = 0; i < local_nnz; i++) {
        if (row_local[i] + 1 > actual_local_rows)
            actual_local_rows = row_local[i] + 1; // +1 because local rows are 0-based
    }

    //gather counts from all ranks
    int *receiver_counts = NULL;
    if (rank == 0){
        receiver_counts = malloc(size * sizeof(int));
    }

    MPI_Gather(&actual_local_rows, 1, MPI_INT, receiver_counts, 1, MPI_INT, 0, MPI_COMM_WORLD); //many -> one (receiver_counts)

    //compute displacements for Gatherv
    int *displs = NULL;
    if (rank == 0){
        displs = malloc(size * sizeof(int));
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + receiver_counts[i - 1];
        }
    }

    //allocate gathered buffer
    double *gathered_results = NULL;
    if (rank == 0){
        gathered_results = malloc(n_rows * sizeof(double)); // safe max size
    }

    //gather all local results into gathered_results
    MPI_Gatherv(local_result, actual_local_rows, MPI_DOUBLE, gathered_results, receiver_counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //reconstruct global vector y
    double *y_global = NULL;
    if (rank == 0){
        y_global = calloc(n_rows, sizeof(double)); //rank 0 reconstructs the global result vector

        for (int r = 0; r < size; r++) {
            for (int i = 0; i < receiver_counts[r]; i++) {
                int global_row = r + i * size;
                if(global_row < n_rows)
                    y_global[global_row] = gathered_results[displs[r] + i];
            }
        }
    }

    //clean up
    if (rank == 0){
        free(receiver_counts);
        free(displs);
        free(gathered_results);
    }

    return y_global;
}

double *gather_res_2D(double* local_result, int n_rows, int p, int q, int pr, int pc, MPI_Comm grid_comm) {
    //compute local block size
    int row_start = (pr * n_rows) / p;
    int row_end = ((pr + 1) * n_rows) / p;
    int local_n_rows= row_end - row_start;
    int sendcount = local_n_rows; //to be sent from each process, needed so that the receiver count matches the sender count


    MPI_Comm row_comm;
    MPI_Comm_split(grid_comm, pr, pc, &row_comm);

    //reduce across columns
    double *row_result = calloc(local_n_rows, sizeof(double));
    MPI_Reduce(local_result, row_result, local_n_rows, MPI_DOUBLE, MPI_SUM, 0, row_comm);


    //gather only from pc==0 ranks
    MPI_Comm col0_comm; //communicator for column 0
    MPI_Comm_split(MPI_COMM_WORLD, pc == 0 ? 0 : MPI_UNDEFINED, pr, &col0_comm); //split communicator, if the process is in column 0, it gets color 0, else MPI_UNDEFINED (does not join communicator)
    //processes with color 0 are ranked in ascending order of pr

    int col0_rank = -1;
    if (pc == 0) {
        MPI_Comm_rank(col0_comm, &col0_rank);
    }

    int *recvcounts = NULL, *displs = NULL;
    double *y_global = NULL;
    if (pc == 0 && col0_rank == 0){
        recvcounts = malloc(p * sizeof(int));
    }

    if (pc == 0){
        MPI_Gather(&sendcount, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, col0_comm); //root gathers the sizes from each pc==0 rank
    }

    if (pc == 0 && col0_rank == 0){ //root compute displacements
        displs = malloc(p * sizeof(int));
        displs[0] = 0;
        for (int i = 1; i < p; i++) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
        y_global = calloc(n_rows, sizeof(double));
    }

    if (pc == 0) {
        if (col0_rank == 0) { //in gatherv, only the root can use recvcounts, displs and receive buffer, all other must pass NULL
            MPI_Gatherv(row_result, local_n_rows, MPI_DOUBLE,
                        y_global, recvcounts, displs,
                        MPI_DOUBLE, 0, col0_comm);
        } else {
            MPI_Gatherv(row_result, local_n_rows, MPI_DOUBLE,
                        NULL, NULL, NULL,
                        MPI_DOUBLE, 0, col0_comm);
        }
    }

    //clean up
    free(row_result);
    MPI_Comm_free(&row_comm);
    if (pc == 0) {
        MPI_Comm_free(&col0_comm);
    }
    if (pc == 0 && col0_rank == 0) {
        free(recvcounts);
        free(displs);
    }
    return y_global;
}