import os
import random

def generate_matrix(n, nnz, filename):
    with open(filename, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write("% Random synthetic matrix\n")
        f.write(f"{n} {n} {nnz}\n")

        for _ in range(nnz):
            i = random.randint(1, n)
            j = random.randint(1, n)
            v = random.uniform(0.1, 10.0)
            f.write(f"{i} {j} {v:.6f}\n")

def main():
    random.seed(42)

    # Path of this script: deliverable2/scripts/
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Parent directory: deliverable2/
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    # Target: deliverable2/random_matrices/
    outdir = os.path.join(project_root, "random_matrices")
    os.makedirs(outdir, exist_ok=True)

    BASE_N = 1000
    BASE_NNZ = 100_000
    PROCESS_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128]

    for p in PROCESS_COUNTS:
        n = BASE_N * p
        nnz = BASE_NNZ * p
        fname = os.path.join(outdir, f"random_{n}_nnz{nnz}.mtx")
        print(f"Generating {fname}")
        generate_matrix(n, nnz, fname)

if __name__ == "__main__":
    main()