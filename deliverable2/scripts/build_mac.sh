#!/usr/bin/env bash
set -e

echo "🔧 Building SpMV_MPI on macOS (MPI + OpenMP)"

# Ensure Homebrew OpenMP is installed
if [ ! -f /opt/homebrew/opt/libomp/lib/libomp.dylib ]; then
  echo "❌ libomp not found. Install it with:"
  echo "   brew install libomp"
  exit 1
fi

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)

mkdir -p "$PROJECT_ROOT/build"
cd "$PROJECT_ROOT/build"

cmake \
  -DCMAKE_C_COMPILER=mpicc \
  -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib \
  ..

make -j

echo "✅ Build completed successfully"