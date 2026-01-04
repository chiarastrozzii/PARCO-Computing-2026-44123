import os
import random
import sys

def generate_matrix(n, nnz, filename):
    with open(filename, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n") #same exact format as the SuiteSparse matrices
        f.write("% Random synthetic matrix\n")
        f.write(f"{n} {n} {nnz}\n")

        for _ in range(nnz):
            i = random.randint(1, n)
            j = random.randint(1, n)
            v = random.uniform(0.1, 10.0)
            f.write(f"{i} {j} {v:.6f}\n")

def main():
    os.makedirs("random_matrices", exist_ok=True)

    configs = [
        (1000, 100_000),
        (2000, 200_000),
        (4000, 400_000),
        (8000, 800_000),
        (16000, 1_600_000),
    ]

    for n, nnz in configs:
        fname = f"random_matrices/random_{n}_nnz{nnz}.mtx"
        print(f"Generating {fname}")
        generate_matrix(n, nnz, fname)

if __name__ == "__main__":
    main()