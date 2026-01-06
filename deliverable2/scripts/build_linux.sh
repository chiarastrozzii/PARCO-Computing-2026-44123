#!/usr/bin/env bash
set -e

echo "🔧 Building SpMV_MPI on Linux (MPI + OpenMP)"

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)

mkdir -p "$PROJECT_ROOT/build"
cd "$PROJECT_ROOT/build"

cmake ..
make -j

echo "✅ Build completed successfully"