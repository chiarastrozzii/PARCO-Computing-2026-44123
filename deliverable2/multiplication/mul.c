#include "mul.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <mpi.h>

void spmv(const Sparse_CSR* sparse_csr, const double* vec, double* res, int parallel){
    if(parallel == 1){
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

static int cmp_int(const void* a, const void *b){
    int x = *(int*)a;
    int y = *(int*)b;
    return (x > y) - (x < y);
}

static int col_owner(int col, int size) {
     return col % size; 
}



double *prepare_x_1D(const Sparse_CSR *csr, const double *x_owned, int x_owned_len, int rank, int size, int **col_map_out, int *local_x_size_out, int *tot_send, int *tot_recv){
    int n_nz = csr->n_nz;

    // Step 1: Collect all unique columns in local rows
    int *uniq_cols = malloc(n_nz * sizeof(int));
    int n_unique = 0;

    for (int i = 0; i < n_nz; i++){
        int col = csr->col_indices[i];
        bool found = false;
        for (int k = 0; k < n_unique; k++){
            if (uniq_cols[k] == col){
                found = true;
                break;
            }
        }
        if (!found) uniq_cols[n_unique++] = col;
    }
    qsort(uniq_cols, n_unique, sizeof(int), cmp_int);

    //split owned vs ghost entries
    int *owned_indices = malloc(n_unique * sizeof(int));
    int *ghost_indices = malloc(n_unique * sizeof(int));
    int n_owned=0, n_ghosts=0;

    for (int i=0; i<n_unique; i++){
        int col = uniq_cols[i];
        if(col_owner(col, size) == rank){
            owned_indices[n_owned++] = col;
        }else{
            ghost_indices[n_ghosts++] = col;
        }
    }
    free(uniq_cols);

    //communication of ghost values

    int *recv_counts = calloc(size, sizeof(int));
    for (int i = 0; i < n_ghosts; i++){
        int owner = col_owner(ghost_indices[i], size);
        recv_counts[owner]++;
    }

    // Step 4: Prepare recv displacements
    int *recv_displs = malloc(size * sizeof(int));
    recv_displs[0] = 0;
    for (int i = 1; i < size; i++){
        recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
    }
        
    int total_recv = recv_displs[size-1] + recv_counts[size-1];

    int *recv_cols = malloc(total_recv * sizeof(int));
    int *pos = calloc(size, sizeof(int));
    for (int i = 0; i < n_ghosts; i++){
        int owner = col_owner(ghost_indices[i], size);
        int idx = recv_displs[owner] + pos[owner]++;
        recv_cols[idx] = ghost_indices[i];
    }
    free(pos);

    //exchange requests with owners
    int *send_counts = calloc(size, sizeof(int));
    MPI_Alltoall(recv_counts, 1, MPI_INT, send_counts, 1, MPI_INT, MPI_COMM_WORLD);

    int *send_displs = malloc(size * sizeof(int));
    send_displs[0] = 0;
    for (int i = 1; i < size; i++)
        send_displs[i] = send_displs[i - 1] + send_counts[i - 1];

    int total_send = send_displs[size - 1] + send_counts[size - 1];

    int *send_cols = malloc(total_send * sizeof(int));
    MPI_Alltoallv(recv_cols, recv_counts, recv_displs, MPI_INT, send_cols, send_counts, send_displs, MPI_INT, MPI_COMM_WORLD);

    //owners then prepare the actual values to share
    double *send_vals = malloc(total_send * sizeof(double));
    for (int i = 0; i < total_send; i++) {
        int col = send_cols[i];
        // sanity: I should only be asked for columns I own
        if (col_owner(col, size) != rank){
            fprintf(stderr, "[Rank %d] ERROR: got request for col %d but owner is %d\n",
                    rank, col, col_owner(col, size));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int li = col/size;
        if (li < 0 || li >= x_owned_len){
            fprintf(stderr, "[Rank %d] ERROR: local index %d out of range for col %d (x_owned_len=%d)\n",
                    rank, li, col, x_owned_len);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        send_vals[i] = x_owned[li];
    }

    //receive ghost values
    double *recv_vals = malloc(total_recv * sizeof(double));
    MPI_Alltoallv(send_vals, send_counts, send_displs, MPI_DOUBLE, recv_vals, recv_counts, recv_displs, MPI_DOUBLE, MPI_COMM_WORLD);

    //now that we have all entries, build the local vector
    int local_x_size = n_owned + n_ghosts;
    double *x_local = malloc(local_x_size * sizeof(double));
    //global -> local column map
    int *col_map = malloc(local_x_size * sizeof(int));

    for (int i = 0; i < n_owned; i++){
        int col = owned_indices[i];
        int li = col/size;
        if (li < 0 || li >= x_owned_len){
            fprintf(stderr, "[Rank %d] ERROR: owned col %d local index %d out of range (x_owned_len=%d)\n",
                    rank, col, li, x_owned_len);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        x_local[i] = x_owned[li];
        col_map[i] = col;
    }

    for (int i = 0; i < n_ghosts; i++){
        int gcol = ghost_indices[i];
        double val = 0.0;
        bool ok = false;
        for (int k = 0; k < total_recv; k++){
            if (recv_cols[k] == gcol){
                val = recv_vals[k];
                ok = true;
                break;
            }
        }
        if (!ok){
            fprintf(stderr, "[Rank %d] ERROR: did not receive ghost value for col %d\n", rank, gcol);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        x_local[n_owned + i] = val;
        col_map[n_owned + i] = gcol;
    }

    *col_map_out = col_map;
    *local_x_size_out = local_x_size;
    *tot_send = total_send;
    *tot_recv = total_recv;
        
    //cleanup
    free(owned_indices);
    free(ghost_indices);
    free(recv_counts);
    free(recv_displs);
    free(send_counts);
    free(send_displs);
    free(recv_cols);
    free(send_cols);
    free(send_vals);
    free(recv_vals);

    return x_local;     

}

//double *prepare_x_2D(const Sparse_CSR *csr, const double *vec, MPI_Comm col_comm, int **col_map_out, int *local_x_size){

    //each column communicator owns a block of x
    //broadcast that block to all ranks in the column
    //no ghost detection needed
    //build 
//}


void remapping_columns(Sparse_CSR *csr, int *col_map, int local_x_size, int rank){
    int nnz = csr->n_nz;

    int *global_to_local = malloc(csr->n_cols * sizeof(int));
    for (int i = 0; i < csr->n_cols; i++) global_to_local[i] = -1;

    for (int local_idx = 0; local_idx < local_x_size; local_idx++) {
        int global_col = col_map[local_idx];
        global_to_local[global_col] = local_idx;
    }

    for (int i = 0; i < nnz; i++) {
        int g = csr->col_indices[i];
        int l = global_to_local[g];
        if (l < 0) {
            fprintf(stderr, "[Rank %d] ERROR: missing mapping for global col %d\n", rank, g);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        csr->col_indices[i] = l;
    }

    free(global_to_local);
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