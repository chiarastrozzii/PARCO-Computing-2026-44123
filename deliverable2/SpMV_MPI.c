#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <mpi.h>
#include <math.h>

#include "config/config.h"
#include "communications/comm.h"
#include "multiplication/mul.h"


#define N_RUNS 100

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //addressing all processes using MPI_COMM_WORLD, we're giving each process a unique id
    MPI_Comm_size(MPI_COMM_WORLD, &size); //total number of processes

    if (argc < 2){
        if (rank == 0){
            fprintf(stderr, "Usage: %s <matrix_file> <2D/1D>\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    bool is_2D = false;
    if (argc >= 3){
        if (strcmp(argv[2], "2D") == 0){
            is_2D = true;
        }
    }

    const char* matrix_file = argv[1];
    int n_rows, n_cols, n_nz;
    int* row_indices = NULL;
    int* col_indices = NULL;
    double* values = NULL;

    if (rank == 0){
        read_matrix_market_file(matrix_file, &n_rows, &n_cols, &n_nz, &row_indices, &col_indices, &values); //only rank 0 reads the file
    }
    
    //broadcast matrix dimensions to all processes
    MPI_Bcast(&n_rows, 1, MPI_INT, 0, MPI_COMM_WORLD); 
    MPI_Bcast(&n_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_nz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_nnz = 0;
    int *row_local = NULL;
    int *col_local = NULL;
    double *val_local = NULL;

    int p, q;


    if (!is_2D){
        //compute ownership of rows [cyclic distribution]
        int *nnz_rank = NULL;

        if (rank == 0) {
            nnz_rank = calloc(size, sizeof(int)); //allocate and initialize to zero using calloc
            for (size_t i = 0; i < n_nz; ++i) {
                int owner = row_indices[i]%size;
                nnz_rank[owner]++;
            }
        }

        MPI_Scatter(nnz_rank, 1, MPI_INT, &local_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD); //root process, 1 element sent to each process, type, receiver , number of elements, type, root, communicator

        //each process allocates memory for local buffers
        row_local = malloc(local_nnz * sizeof(int));
        col_local = malloc(local_nnz * sizeof(int));
        val_local = malloc(local_nnz * sizeof(double));

        //re-order data on rank 0 and scatter all entries to respective processes
        scatter_entries(rank, size, n_rows, n_nz, row_indices, col_indices, values, nnz_rank, local_nnz, row_local, col_local, val_local);

        if (rank == 0) {
            free(nnz_rank);
        }

    }else{
        p = (int)sqrt(size);
        while (size % p != 0) p--;
        q = size / p;

        MPI_Comm grid_comm;
        int dims[2] = {p, q};
        int periods[2] = {0, 0};
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);

        int *nnz_rank = NULL;

        if (rank == 0) {
            nnz_rank = calloc(size, sizeof(int));

            for (size_t i = 0; i < n_nz; ++i) {
                int pr = row_indices[i] * p / n_rows;
                int pc = col_indices[i] * q / n_cols;

                int coords[2] = {pr, pc};
                int owner;
                MPI_Cart_rank(grid_comm, coords, &owner);
                nnz_rank[owner]++;
            } 

        }

        MPI_Scatter(nnz_rank, 1, MPI_INT,
                    &local_nnz, 1, MPI_INT,
                    0, grid_comm);

        row_local = malloc(local_nnz * sizeof(int));
        col_local = malloc(local_nnz * sizeof(int));
        val_local = malloc(local_nnz * sizeof(double));

        scatter_entries_2D(rank, p, q, grid_comm,
                           n_rows, n_cols, n_nz,
                           row_indices, col_indices, values,
                           nnz_rank, local_nnz,
                           row_local, col_local, val_local);

        if (rank == 0) free(nnz_rank);
    }


    //statistics about nnz distribution
    int min_nnz, max_nnz, sum_nnz;
    MPI_Reduce(&local_nnz, &min_nnz, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_nnz, &max_nnz, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_nnz, &sum_nnz, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nMatrix dimensions: %d x %d with %d non-zero entries\n", n_rows, n_cols, n_nz);
        printf("Non-zero entries distribution among %d processes:\n", size);
        printf("Min nnz: %d\n", min_nnz);
        printf("Max nnz: %d\n", max_nnz);
        
        double avg_nnz = (double)sum_nnz / size;
        printf("Avg nnz: %.2f\n", avg_nnz);
        
    }

    //each process creates its local CSR matrix
    Sparse_CSR local_csr;
    int local_n_rows;

    if (!is_2D){
        local_n_rows = (n_rows + size - 1 - rank) / size; //ceil division to account for uneven distribution
        for (int i = 0; i < local_nnz; i++) {
            if (row_local[i] < 0 || row_local[i] >= local_n_rows) {
                printf("[Rank %d] INVALID ROW %d (local_n_rows=%d)\n",
                    rank, row_local[i], local_n_rows);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        create_sparse_csr(local_n_rows, n_cols, local_nnz, row_local, col_local, val_local, &local_csr);
    }else{
        // Find how many distinct rows this rank owns
        local_n_rows = 0;
        for (int i = 0; i < local_nnz; i++) {
            if (row_local[i] + 1 > local_n_rows)
                local_n_rows = row_local[i] + 1;
        }

        create_sparse_csr(local_n_rows, n_cols, local_nnz, row_local, col_local, val_local, &local_csr);
    }


    //serialized print to check the matrix
    MPI_Barrier(MPI_COMM_WORLD); //synchronize all processes
    for (int r = 0; r < size; r++) {
        if (rank == r) {
            printf("\n[Rank %d] Local CSR:\n", rank);
            print_sparse_csr(&local_csr);
            fflush(stdout);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //rank 0 creates the random vector, which will be broadcasted to all processes
    double *vec = malloc(n_cols * sizeof(double));
    if (rank == 0){
        random_vector(vec, n_cols);
        //printf("\nInput vector x:\n");
        //for (size_t i = 0; i < n_cols; ++i) {
        //    printf("x[%zu] = %.2f\n", i, vec[i]);
        //}   
    }

    MPI_Bcast(vec, n_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD); //simple, memory heavy approach
    long long comm_bytes = 0;

    if (rank == 0) {
       comm_bytes = (long long)(size - 1) * n_cols * sizeof(double);
    } else {
       comm_bytes = (long long)n_cols * sizeof(double);
    }

    long long min_comm, max_comm, sum_comm; //sum_comm is total communication volume

    MPI_Reduce(&comm_bytes, &min_comm, 1, MPI_LONG_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_bytes, &max_comm, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_bytes, &sum_comm, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double avg_comm = (double)sum_comm / size;
        printf("\ncommunication volume per rank (bytes):\n");
        printf("min per rank: %lld\n", min_comm);
        printf("max per rank: %lld\n", max_comm);
        printf("avg communication volume: %.2f\n", avg_comm);
    }

    double *local_result = calloc(local_csr.n_rows, sizeof(double)); //initialize to zero

    spmv(&local_csr, vec, local_result, 0); //warm-up run (warm caches, avoids first-run overheads)
    double local_times[N_RUNS];

    for (int run = 0; run < N_RUNS; run++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        spmv(&local_csr, vec, local_result, 1); //using parallel version with OpenMP

        MPI_Barrier(MPI_COMM_WORLD);
        double end = MPI_Wtime();

        local_times[run] = end - start;
    }

    double max_times[N_RUNS];
    for (int i=0; i<N_RUNS; i++){
        MPI_Reduce(&local_times[i], &max_times[i], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }


    double avg_time = 0.0;
    if (rank == 0) {
        double min_time = max_times[0];
        double max_time = max_times[0];

        for (int i = 0; i < N_RUNS; i++) {
            avg_time += max_times[i];
            if (max_times[i] < min_time) min_time = max_times[i];
            if (max_times[i] > max_time) max_time = max_times[i];
        }

        avg_time /= N_RUNS;

        printf("\nSpMV Time over %d iterations:\n", N_RUNS);
        printf("Min: %.6f s\n", min_time);
        printf("Max: %.6f s\n", max_time);
        printf("Avg: %.6f s\n", avg_time);
    }

    //flops + gflops calculation
    if (rank == 0) {
        long long flops = 2LL * n_nz;  
        double gflops = flops / (avg_time * 1e9);

        printf("Total FLOPs per SpMV: %lld\n", flops);
        printf("Performance: %.3f GFLOP/s\n", gflops);
    }

    //GATHER RESULTS TO RANK 0
    //compute actual number of local rows for each rank based on row_local
    int actual_local_rows = 0;
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

        if(!is_2D){
            //cyclic distribution
            for (int r = 0; r < size; r++) {
                for (int i = 0; i < receiver_counts[r]; i++) {
                    int global_row = r + i * size;
                    if(global_row < n_rows)
                        y_global[global_row] = gathered_results[displs[r] + i];
                }
            }
        }else{
            //2D block distribution
            for (int r = 0; r < size; r++) {
                int owner_pr = r / q;
                int row_start = block_start(owner_pr, n_rows, p);

                for (int i = 0; i < receiver_counts[r]; i++) {
                    int global_row = row_start + i;
                    if (global_row < n_rows)
                        y_global[global_row] += gathered_results[displs[r] + i]; //each rank contributes partial sums for its rows
                }
            }
        }

        //print result vector
        printf("\nResult vector y:\n");
        for (size_t i = 0; i < n_rows; ++i) {
            printf("y[%zu] = %.2f\n", i, y_global[i]);
        }

        free(receiver_counts);
        free(displs);
        free(gathered_results);
        free(y_global);
    }

    if (rank == 0){
        free(row_indices);
        free(col_indices);
        free(values);
    }

    free(row_local);
    free(col_local);
    free(val_local);
    free(vec);
    free_sparse_csr(&local_csr);

    MPI_Finalize();
    return 0;
}

