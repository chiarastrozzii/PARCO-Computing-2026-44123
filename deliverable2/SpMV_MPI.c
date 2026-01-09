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

    //test print of the matrix read from file
    //for (int r = 0; r < size; r++) {
    //    if (rank == r && rank == 0) {
    //        printf("Matrix read from file %s:\n", matrix_file);
    //        for (size_t i = 0; i < n_nz; ++i) {
    //            printf("(%d, %d) -> %.2f\n", row_indices[i], col_indices[i], values[i]);
    //        }
    //        fflush(stdout);
    //    }
    //    MPI_Barrier(MPI_COMM_WORLD);
    //}
    
    //broadcast matrix dimensions to all processes
    MPI_Bcast(&n_rows, 1, MPI_INT, 0, MPI_COMM_WORLD); 
    MPI_Bcast(&n_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_nz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_nnz = 0;
    int *row_local = NULL;
    int *col_local = NULL;
    double *val_local = NULL;

    int p, q;
    int pc, pr;
    MPI_Comm grid_comm;


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
        scatter_entries(rank, size, n_nz, row_indices, col_indices, values, nnz_rank, local_nnz, row_local, col_local, val_local);

        //for (int i = 0; i < local_nnz; i++) {
        //    int local_r = row_local[i];
        //    if (local_r < 0) {
        //        fprintf(stderr, "[Rank %d] NEGATIVE local row %d\n", rank, local_r);
        //        MPI_Abort(MPI_COMM_WORLD, 1);
        //    }
        //}

       // MPI_Barrier(MPI_COMM_WORLD);
       // for (int r = 0; r < size; r++) {
       //     MPI_Barrier(MPI_COMM_WORLD);
       //     if (rank == r) {
       //         printf("\n[Rank %d] SCATTERED COO (local_row, global_col, val):\n", rank);
       //         for (int i = 0; i < local_nnz; i++) {
       //             printf("(%d, %d) -> %.2f\n", row_local[i], col_local[i], val_local[i]);
       //         }
       //         fflush(stdout);
       //     }
       // }
       // MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            free(nnz_rank);
        }

    }else{
        p = (int)sqrt(size);
        while (size % p != 0) p--;
        q = size / p;

        int dims[2] = {p, q};
        int periods[2] = {0, 0};
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);

        int coords[2];
        MPI_Cart_coords(grid_comm, rank, 2, coords);
        pr = coords[0];
        pc = coords[1];

        int *nnz_rank = NULL;

        if (rank == 0) {
            nnz_rank = calloc(size, sizeof(int));

            for (size_t i = 0; i < n_nz; ++i) {
                int owner_pr = row_indices[i] * p / n_rows;
                int owner_pc = col_indices[i] * q / n_cols;

                int coords[2] = {owner_pr, owner_pc};
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

    // 1) CSR sizes must match what we requested
    if ((int)local_csr.n_rows != local_n_rows) {
        fprintf(stderr, "[Rank %d] CSR n_rows mismatch: local_n_rows=%d csr.n_rows=%zu\n",
                rank, local_n_rows, local_csr.n_rows);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if ((int)local_csr.n_nz != local_nnz) {
        fprintf(stderr, "[Rank %d] CSR n_nz mismatch: local_nnz=%d csr.n_nz=%zu\n",
                rank, local_nnz, local_csr.n_nz);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 2) CSR row_ptrs must be nondecreasing and end at n_nz
    for (int r = 0; r < local_n_rows; r++) {
        if (local_csr.row_ptrs[r] > local_csr.row_ptrs[r+1]) {
            fprintf(stderr, "[Rank %d] CSR row_ptrs decrease at r=%d\n", rank, r);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    if (local_csr.row_ptrs[local_n_rows] != (size_t)local_nnz) {
        fprintf(stderr, "[Rank %d] CSR row_ptrs end mismatch: row_ptrs[last]=%zu local_nnz=%d\n",
                rank, local_csr.row_ptrs[local_n_rows], local_nnz);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }



    for (int r = 0; r < size; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
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

    int x_owned_n = (n_cols + size - 1 - rank) / size;   // count of cols owned by this rank
    double *x_owned = malloc(x_owned_n * sizeof(double));

    if (rank == 0) {
        // send owned pieces to everyone (including self)
        for (int r = 0; r < size; r++) {
            int cnt = (n_cols + size - 1 - r) / size;
            if (r == 0) {
                for (int k = 0; k < cnt; k++) x_owned[k] = vec[r + k*size];
            } else {
                double *tmp = malloc(cnt * sizeof(double));
                for (int k = 0; k < cnt; k++) tmp[k] = vec[r + k*size];
                MPI_Send(tmp, cnt, MPI_DOUBLE, r, 123, MPI_COMM_WORLD);
                free(tmp);
            }
        }
    } else {
        MPI_Recv(x_owned, x_owned_n, MPI_DOUBLE, 0, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


    //MPI_Bcast(vec, n_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD); //simple, memory heavy approach

    double *x_local;
    int *col_map;
    int local_x_size; 

    int x_owned_len = (n_cols + size - 1 - rank) / size;    
    int tot_send = 0;
    int tot_recv = 0;

    if (!is_2D){
        x_local = prepare_x_1D(&local_csr, x_owned, x_owned_len, rank, size, &col_map, &local_x_size, &tot_send, &tot_recv);
    }else{
        //x_local = prepare_x_2D();
    }

    //test to see the x_local vec
    //printf("[Rank %d] x_local: ", rank);
    //for (int i = 0; i < local_x_size; i++)
    //    printf("%f ", x_local[i]);
    //printf("\n");

    remapping_columns(&local_csr, col_map, local_x_size, rank);
    //test to see if remapping is successfull
    for (int i = 0; i < local_csr.n_nz; i++) {
        if (local_csr.col_indices[i] < 0 || local_csr.col_indices[i] >= local_x_size) {
            fprintf(stderr, "[Rank %d] BAD LOCAL COL %zu (local_x_size=%d)\n",
                    rank, local_csr.col_indices[i], local_x_size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    long long comm_bytes =
    (long long)(tot_send + tot_recv) * sizeof(int) +
    (long long)(tot_send + tot_recv) * sizeof(double);

    // Optionally track sent/recv separately:
    long long sent_bytes =
        (long long)tot_send * sizeof(int) +
        (long long)tot_send * sizeof(double);

    long long recv_bytes =
        (long long)tot_recv * sizeof(int) +
        (long long)tot_recv * sizeof(double);


    long long min_comm, max_comm, sum_comm;

    MPI_Reduce(&comm_bytes, &min_comm, 1, MPI_LONG_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_bytes, &max_comm, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_bytes, &sum_comm, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double avg_comm = (double)sum_comm / size;
        printf("\nGhost exchange communication per rank (bytes):\n");
        printf("min per rank: %lld\n", min_comm);
        printf("max per rank: %lld\n", max_comm);
        printf("avg per rank: %.2f\n", avg_comm);
    }


    double *local_result = calloc(local_csr.n_rows, sizeof(double)); //initialize to zero

    spmv(&local_csr, x_local, local_result, 0); //warm-up run (warm caches, avoids first-run overheads)
    double local_times[N_RUNS];

    for (int run = 0; run < N_RUNS; run++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        spmv(&local_csr, x_local, local_result, 1); //using parallel version with OpenMP

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

    //GATHER RESULTS
    int actual_local_rows = 0;
    if (!is_2D){
        double *y_global = gather_res_1D(local_result, actual_local_rows, local_nnz, row_local, n_rows, rank, size);
        if (rank == 0){
            //print result vector
            printf("\nResult vector y [1D CASE]:\n");
            for (size_t i = 0; i < n_rows; ++i) {
                printf("y[%zu] = %.2f\n", i, y_global[i]);
            }
            free(y_global);
        }
    }else{
        double *y_global = gather_res_2D(local_result, n_rows, p, q, pr, pc, grid_comm);
        if (pr == 0 && pc == 0){ //only rank (0,0) has the full result
            //print result vector
            printf("\nResult vector y [2D CASE]:\n");
            for (size_t i = 0; i < n_rows; ++i) {
                printf("y[%zu] = %.2f\n", i, y_global[i]);
            }
            free(y_global);
        }
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

