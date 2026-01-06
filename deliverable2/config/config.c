#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>


void create_sparse_csr(
    size_t n_rows,
    size_t n_cols,
    size_t n_nz,

    const int *row,
    const int* col,
    const double* val,
    Sparse_CSR* output_csr
){
    output_csr->n_rows = n_rows;
    output_csr->n_cols = n_cols;
    output_csr->n_nz = n_nz;

    output_csr->row_ptrs = calloc(n_rows + 1, sizeof(size_t)); //calloc-> initializes memory and also fill it with zero, useful for initializing the pointer
    output_csr->col_indices = calloc(n_nz, sizeof(size_t));
    output_csr->values = calloc(n_nz, sizeof(double));

    //count non-zero per rows
    for (size_t i = 0; i < n_nz; ++i) {
        output_csr->row_ptrs[row[i] + 1]++; //if we don't remap rows -> out of bounds
    }

    for (size_t i = 0; i<n_rows; ++i){
        output_csr->row_ptrs[i + 1] += output_csr->row_ptrs[i];
    }

    //we fill the indeces of the columns and the values
    size_t *row_offset = calloc(n_rows, sizeof(size_t)); // track current position in each row
    for (size_t i = 0; i < n_nz; ++i) {
        int r = row[i];
        size_t dest = output_csr->row_ptrs[r] + row_offset[r];
        output_csr->col_indices[dest] = col[i];
        output_csr->values[dest] = val[i];
        row_offset[r]++;
    }

    free(row_offset);

}


void print_sparse_csr(Sparse_CSR* sparse_csr){
    printf("\n");
    printf("row\tcol\tval\n");
    printf("---\n");
    for (size_t i=0; i<sparse_csr->n_rows; ++i){
        size_t nz_start = sparse_csr->row_ptrs[i];
        size_t nz_end = sparse_csr->row_ptrs[i+1];
        
        for (size_t j = nz_start; j < nz_end; ++j) {
            size_t col = sparse_csr->col_indices[j];
            double val = sparse_csr->values[j];
            printf("%zu\t%zu\t%.2f\n", i, col, val);
        }
    }
}


void free_sparse_csr(Sparse_CSR* sparse_csr){
    free(sparse_csr->row_ptrs);
    free(sparse_csr-> col_indices);
    free(sparse_csr-> values);
}


void read_matrix_market_file(
    const char* filename,
    int* n_rows,
    int* n_cols,
    int* n_nz,
    int** row_indices,
    int** col_indices,
    double** values
){
    FILE* f = fopen(filename, "r");
    if (f == NULL){
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    char line[256];
    do{
        if(!fgets(line, sizeof(line), f)) {
            fprintf(stderr, "Unexpected end of file\n");
            exit(EXIT_FAILURE);
        }
    }while (line[0] == '%');

    sscanf(line, "%d %d %d", n_rows, n_cols, n_nz);
    //allocates memory without overriding
    *row_indices = malloc((*n_nz) * sizeof(int));
    *col_indices = malloc((*n_nz) * sizeof(int));
    *values = malloc((*n_nz) * sizeof(double));


    for (size_t i = 0; i < *n_nz; ++i) {
        int r, c;
        double v;
        fscanf(f, "%d %d %lf", &r, &c, &v);
        (*row_indices)[i] = r - 1; // convert to 0-based index
        (*col_indices)[i] = c - 1; // convert to 0-based index
        (*values)[i] = v;
    }

    fclose(f);
}

//creates the random vector for SpMV
void random_vector(double* vec, size_t size){
   static bool seeded = false;
    if (!seeded) { //sees run one, so that multiple calls for benchmar not reset the seed
        srand(time(NULL));
        seeded = true;
    }

    for(size_t i = 0; i < size; ++i){
        vec[i] = rand() % 10; 
    }
}