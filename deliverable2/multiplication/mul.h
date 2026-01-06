#ifndef MUL_H
#define MUL_H

#include "../config/config.h"


void spmv(const Sparse_CSR* sparse_csr, const double* vec, double* res, int parallel);

#endif // MUL_H 