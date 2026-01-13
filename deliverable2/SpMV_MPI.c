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

static int cmp_double(const void* a, const void* b){
    double x = *(const double*)a;
    double y = *(const double*)b;
    return (x > y) - (x < y);
}

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //addressing all processes using MPI_COMM_WORLD, we're giving each process a unique id
    MPI_Comm_size(MPI_COMM_WORLD, &size); //total number of processes

    if (argc < 4){
        if (rank == 0){
            fprintf(stderr, "Usage: %s <matrix_file> <2D/1D> <SEQ/PAR> \n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    bool is_2D = false;
    bool parallel = false;

    if (strcmp(argv[2], "2D") == 0){
        is_2D = true;
    }

    if (strcmp(argv[3], "PAR") == 0){
        parallel = true;
    }

    if (rank == 0) {
        printf("decomposition: %s | mode: %s\n", is_2D ? "2D" : "1D", parallel ? "OpenMP" : "Sequential");
    }



    const char* matrix_file = argv[1];
    int n_rows, n_cols, n_nz;
    int* row_indices = NULL;
    int* col_indices = NULL;
    double* values = NULL;

    double comm_time = 0.0;

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

    MPI_Comm row_comm, col_comm;


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

        if (rank == 0) {
            free(nnz_rank);
        }

    }else{
        p = (int)sqrt(size);
        while (size % p != 0) p--;
        q = size / p;

        int dims[2] = {p, q};
        int periods[2] = {0, 0};
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);

        int coords[2];
        MPI_Cart_coords(grid_comm, rank, 2, coords);
        pr = coords[0];
        pc = coords[1];

        int *nnz_rank = NULL;

        if (rank == 0) {
            nnz_rank = calloc(size, sizeof(int));

            for (size_t i = 0; i < n_nz; ++i) {
                int pr = owner_block(row_indices[i], n_rows, p);
                int pc = owner_block(col_indices[i], n_cols, q);

                int coords[2] = {pr, pc};
                int owner;
                MPI_Cart_rank(grid_comm, coords, &owner);
                nnz_rank[owner]++;
            } 

            int sum = 0;
            for (int r = 0; r < size; r++) sum += nnz_rank[r];
            if (sum != n_nz) {
                fprintf(stderr, "nnz_rank sum mismatch: sum=%d n_nz=%d\n", sum, n_nz);
                MPI_Abort(MPI_COMM_WORLD, 1);
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
        // ind how many distinct rows this rank owns
        local_n_rows = block_start(pr+1, n_rows, p) - block_start(pr, n_rows, p);
        int local_n_cols = block_start(pc+1, n_cols,q) - block_start(pc, n_cols, q);
        
        for (int i = 0; i < local_nnz; i++) {
            if (row_local[i] < 0 || row_local[i] >= local_n_rows ||
                col_local[i] < 0 || col_local[i] >= local_n_cols) {
                fprintf(stderr, "[Rank %d] BAD 2D LOCAL IDX row=%d col=%d (rows=%d cols=%d)\n",
                        rank, row_local[i], col_local[i], local_n_rows, local_n_cols);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }   

        create_sparse_csr(local_n_rows, local_n_cols, local_nnz, row_local, col_local, val_local, &local_csr);
    }

    //CSR sizes must match what is requested
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

    //CSR row_ptrs must be nondecreasing and end at n_nz
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



    //for (int r = 0; r < size; r++) {
    //    MPI_Barrier(MPI_COMM_WORLD);
    //    if (rank == r) {
    //        printf("\n[Rank %d] Local CSR:\n", rank);
    //        print_sparse_csr(&local_csr);
    //        fflush(stdout);
    //    }
    //}
    //MPI_Barrier(MPI_COMM_WORLD);


    //rank 0 creates the random vector, which will be broadcasted to all processes
    double *vec = malloc(n_cols * sizeof(double));
    if (rank == 0){
        random_vector(vec, n_cols);
        //printf("\nInput vector x:\n");
        //for (size_t i = 0; i < n_cols; ++i) {
        //    printf("x[%zu] = %.2f\n", i, vec[i]);
        //}   
    }

    double *y_ref = NULL;
    if (rank == 0) {
        y_ref = calloc(n_rows, sizeof(double));
        for (int k = 0; k < n_nz; k++) {
            int r = row_indices[k];
            int c = col_indices[k];
            y_ref[r] += values[k] * vec[c];
        }
    }

    //MPI_Bcast(vec, n_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD); //simple, memory heavy approach

    double *x_local;
    int *col_map = NULL;
    int local_x_size; 
   
    int tot_send = 0;
    int tot_recv = 0;

    if (!is_2D){
        int x_owned_len = (n_cols + size - 1 - rank) / size; 
        double *x_owned = malloc(x_owned_len * sizeof(double));

        double t0 = MPI_Wtime();

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

        }else {
            MPI_Recv(x_owned, x_owned_len, MPI_DOUBLE, 0, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        x_local = prepare_x_1D(&local_csr, x_owned, x_owned_len, rank, size, &col_map, &local_x_size, &tot_send, &tot_recv);

        double t1 = MPI_Wtime();
        comm_time += (t1 - t0); 

        long long comm_bytes =
        (long long)(tot_send + tot_recv) * (sizeof(int) + sizeof(double));

        long long min_comm, max_comm, sum_comm;
        MPI_Reduce(&comm_bytes, &min_comm, 1, MPI_LONG_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&comm_bytes, &max_comm, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&comm_bytes, &sum_comm, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("\n1D ghost exchange payload per rank (bytes):\n");
            printf("min per rank: %lld\n", min_comm);
            printf("max per rank: %lld\n", max_comm);
            printf("avg per rank: %.2f\n", (double)sum_comm / size);
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

        free(x_owned);

    }else{
        MPI_Comm_split(grid_comm, pr, pc, &row_comm);
        MPI_Comm_split(grid_comm, pc, pr, &col_comm);

        int x_block_len = block_size(pc, n_cols, q);
        int x_start = block_start(pc, n_cols, q);

        double *x_block = malloc(x_block_len * sizeof(double));

        double t0 = MPI_Wtime();

        if (pr == 0){
            if (rank==0){
                //send each block to (o, pc) -> first column
                for (int dest_pc = 0; dest_pc < q; dest_pc++) {
                    int len = block_size(dest_pc, n_cols, q);
                    int start = block_start(dest_pc, n_cols, q);

                    if (dest_pc == 0) {
                        memcpy(x_block, vec + start, len * sizeof(double));
                    } else {
                        int coords2[2] = {0, dest_pc};
                        int dest_rank;
                        MPI_Cart_rank(grid_comm, coords2, &dest_rank);
                        MPI_Send(vec + start, len, MPI_DOUBLE, dest_rank, 111, grid_comm); //pointer to the starting element of data to send, num of elements to send, data type of each element, rank of dest process, message tag
                    }
                }
            }else{
                int coords0[2] = {0, 0};
                int root_rank;
                MPI_Cart_rank(grid_comm, coords0, &root_rank);
                MPI_Recv(x_block, x_block_len, MPI_DOUBLE, root_rank, 111, grid_comm, MPI_STATUS_IGNORE);
            }
        }

        MPI_Bcast(x_block, x_block_len, MPI_DOUBLE, 0, col_comm);

        double t1 = MPI_Wtime(); 
        comm_time += (t1 - t0); 

        //communication volume per rank in the case of 2D partitioning
        long long sent = 0, recv = 0;
        if (pr == 0) {
            if (pc == 0) {
                for (int dest_pc = 1; dest_pc < q; dest_pc++) {
                    int len = block_size(dest_pc, n_cols, q);
                    sent += (long long)len * sizeof(double);
                }
            } else {
                recv += (long long)x_block_len * sizeof(double);
            }
        }

        if (pr == 0) {
            sent += (long long)(p - 1) * x_block_len * sizeof(double);
        } else {
            recv += (long long)x_block_len * sizeof(double);
        }

        long long comm_bytes = sent + recv;

        long long min_comm, max_comm, sum_comm;
        MPI_Reduce(&comm_bytes, &min_comm, 1, MPI_LONG_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&comm_bytes, &max_comm, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&comm_bytes, &sum_comm, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("\nCommunication volume per rank (2D x-distribution payload bytes):\n");
            printf("min per rank: %lld\n", min_comm);
            printf("max per rank: %lld\n", max_comm);
            printf("avg per rank: %.2f\n", (double)sum_comm / size);
        }

        x_local = x_block;
        local_x_size = x_block_len;

        MPI_Comm_free(&row_comm);
        MPI_Comm_free(&col_comm);
    }
    
    


    double *local_result = calloc(local_csr.n_rows, sizeof(double)); //initialize to zero

    int openmp = parallel ? 1 : 0;

    spmv(&local_csr, x_local, local_result, openmp); //warm-up run (warm caches, avoids first-run overheads)
    double local_times[N_RUNS];

    for (int run = 0; run < N_RUNS; run++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        spmv(&local_csr, x_local, local_result, openmp); //using parallel version with OpenMP

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

        double sorted_times[N_RUNS];
        memcpy(sorted_times, max_times, N_RUNS * sizeof(double));
        qsort(sorted_times, N_RUNS, sizeof(double), cmp_double);

        int p90_index = (int)ceil(0.9 * N_RUNS) - 1;
        if (p90_index < 0) p90_index = 0;
        if (p90_index >= N_RUNS) p90_index = N_RUNS - 1;

        double p90_time = sorted_times[p90_index];

        printf("\nSpMV Time over %d iterations:\n", N_RUNS);
        printf("Min: %.6f s\n", min_time);
        printf("Max: %.6f s\n", max_time);
        printf("Avg: %.6f s\n", avg_time);
        printf("P90: %.6f s\n", p90_time);
    }

    double comm_max = 0.0;
    MPI_Reduce(&comm_time, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nset up comm vs SpMV compute breakdown:\n");
        printf("set up communication time (max rank): %.6f s\n", comm_max);
        printf("SpMV avg time per iteration (max over ranks): %.6f s\n", avg_time);
        printf("setup_comm/spmv_time ratio: %.3f\n", comm_max / avg_time);
    }


    //flops + gflops calculation
    if (rank == 0) {
        long long flops = 2LL * n_nz;  
        double gflops = flops / (avg_time * 1e9);

        printf("Total FLOPs per SpMV: %lld\n", flops);
        printf("Performance: %.3f GFLOP/s\n", gflops);
    }

    //memory footprint per rank
    long long mem_bytes = 0;

    //CSR
    mem_bytes += (long long)(local_csr.n_rows + 1) * sizeof(size_t);
    mem_bytes += (long long)local_csr.n_nz * sizeof(size_t);
    mem_bytes += (long long)local_csr.n_nz * sizeof(double);

    //x_local
    mem_bytes += (long long)local_x_size * sizeof(double);

    // col_map only for 1D
    if (!is_2D && col_map) {
        mem_bytes += (long long)local_x_size * sizeof(int);
    }

    //vec
    if (rank == 0){
        mem_bytes += (long long)n_cols * sizeof(double);
    }

    //rank-0 global matrix storage (only on root)
    if (rank == 0) {
        mem_bytes += (long long)n_nz * sizeof(int);      // row_indices
        mem_bytes += (long long)n_nz * sizeof(int);      // col_indices
        mem_bytes += (long long)n_nz * sizeof(double);   // values
    }

    long long min_mem, max_mem, sum_mem;
    MPI_Reduce(&mem_bytes, &min_mem, 1, MPI_LONG_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mem_bytes, &max_mem, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mem_bytes, &sum_mem, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nMemory footprint per rank (bytes):\n");
        printf("min per rank: %lld\n", min_mem);
        printf("max per rank: %lld\n", max_mem);
        printf("avg per rank: %.2f\n", (double)sum_mem / size);
    }



    //GATHER RESULTS
    if (!is_2D){
        double *y_global = gather_res_1D(local_result, local_n_rows, local_nnz, row_local, n_rows, rank, size);
        if (rank == 0){
            //print result vector
            //printf("\nResult vector y [1D CASE]:\n");
            //for (size_t i = 0; i < n_rows; ++i) {
            //    printf("y[%zu] = %.2f\n", i, y_global[i]);
            //}
            
            //JUST TO TEST
            bool ok = true;
            for (int i = 0; i < n_rows; i++) {
                if (fabs(y_global[i] - y_ref[i]) > 1e-9) {
                    printf("❌ MISMATCH at row %d: parallel=%.6f  ref=%.6f\n", i, y_global[i], y_ref[i]);
                    ok=false;
                    break;
                }
            }
            if (ok) printf("serial check passed!");

            free(y_ref);
            free(y_global);
        }
    }else{
        double *y_global = gather_res_2D(local_result, n_rows, p, q, pr, pc, grid_comm);
        if (pr == 0 && pc == 0){ //only rank (0,0) has the full result
            //print result vector
            //printf("\nResult vector y [2D CASE]:\n");
            //for (size_t i = 0; i < n_rows; ++i) {
            //    printf("y[%zu] = %.2f\n", i, y_global[i]);
            //}

            //JUST TO TEST
            bool ok = true;
            for (int i = 0; i < n_rows; i++) {
                if (fabs(y_global[i] - y_ref[i]) > 1e-9) {
                    printf("❌ MISMATCH at row %d: parallel=%.6f  ref=%.6f\n", i, y_global[i], y_ref[i]);
                    ok = false;
                    break;
                }
            }

            if (ok) printf("serial check passed");

            free(y_ref);
            free(y_global);
        }
    }

    free(row_local);
    free(col_local);
    free(val_local);

    if (rank == 0){
        free(row_indices);
        free(col_indices);
        free(values);
    }

    free(vec);
    free_sparse_csr(&local_csr);
    free(x_local);
    if(col_map){ //only free it if not NULL
        free(col_map);
    }
    

    MPI_Finalize();
    return 0;
}

