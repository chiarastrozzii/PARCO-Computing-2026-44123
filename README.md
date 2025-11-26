# Parallel Sparse Matrix-Vector Multiplication with OpenMP

## 📌 Overview
This project evaluates the performance of **Sparse Matrix-Vector Multiplication (SpMV)** implemented using the **Compressed Sparse Row (CSR)** format and parallelized with **OpenMP**.

The project includes:

- **Sequential** SpMV implementation  
- **Parallel** SpMV implementation with: OpenMP scheduling policies (`static`, `dynamic`, `guided`, `auto`, `runtime`), different thread count and different chuncks sizes
- **SIMD-optimized** inner loop version  
- **Block-based CSB (Compressed Sparse Blocks)** variant  
- Automated scripts to run experiments, collect timings, generate CSV output and generate graphs.

Performance was tested on multiple matrices with different sparsity patterns (which can be found in the 'matrix_mkt' folder) on the institutional HPC cluster, based on 64-core x86-64 CPU supporting OpenMP 4.5+ 

---

## ⚙️ Requirements

### Local Machine
- GCC 15 with OpenMP support  
- Python 3 (optional for scripts)  

### Cluster
- GCC (default module)  
- Python 3 (optional for scripts)  

---

## Compilation (GCC):
to compile the CSR format -> 
```bash
gcc -fopenmp csr/matrix_vector_csr.c -o sparse_seq
```

to compile the CSB format -> 
```bash
gcc -fopenmp csb/matrix_CSB.c -o sparse_seq
```

## How To run

### Configurable parameters
```bash
./sparse_seq <matrix_name> <SEQ/PAR> <schedule_type> <thread_count> <chunk_size>
```
•	matrix_file: Path to the input matrix in Matrix Market format
•	schedule_type: OpenMP scheduling type (static, dynamic, guided, auto)
•	num_threads: Number of threads to use (e.g., 4, 16, 32, 64)
•	num_chunks: Chunk size for scheduling (0, 1, 10, 50, 100)

### Benchmark run
- **CSR format**
  ```bash
  chmod u+x csr_script.sh #if the user doesn't have permissions
  ./csr_script.sh matrix_vector_csr.c
  ```
- **CSB format**
  ```bash
  chmod u+x csb_script.sh #if the user doesn't have permissions
  ./csb_script.sh matrix_CSB.c
  ```
  




