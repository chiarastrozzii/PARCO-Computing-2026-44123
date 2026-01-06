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