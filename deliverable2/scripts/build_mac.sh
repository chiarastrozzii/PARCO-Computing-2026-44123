#!/usr/bin/env bash
set -e

echo "🔧 Building SpMV_MPI on macOS (MPI + OpenMP)"

# Ensure Homebrew OpenMP is installed
if [ ! -f /opt/homebrew/opt/libomp/lib/libomp.dylib ]; then
  echo "❌ libomp not found. Install it with:"
  echo "   brew install libomp"
  exit 1
fi

# Ensure MPI is installed
if ! command -v mpicc >/dev/null 2>&1; then
  echo "❌ mpicc not found. Install MPI with:"
  echo "   brew install open-mpi"
  exit 1
fi

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)

rm -rf "$PROJECT_ROOT/build"
mkdir -p "$PROJECT_ROOT/build"
cd "$PROJECT_ROOT/build"

cmake .. \
  -DCMAKE_C_COMPILER=clang \
  -DMPI_C_COMPILER="$(command -v mpicc)" \
  -DCMAKE_PREFIX_PATH="/opt/homebrew/opt/open-mpi;/opt/homebrew/opt/libomp" \

make -j

echo "✅ Build completed successfully"
