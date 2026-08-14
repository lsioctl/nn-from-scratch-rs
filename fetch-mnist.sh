#!/usr/bin/env bash
# Fetch the MNIST handwritten digit dataset into ./data
#
# The files are gzipped, and this project has no dependencies — including no
# gzip decoder — so we decompress here rather than in Rust. What Rust reads is
# the raw IDX format, which is simple enough to parse by hand (see src/mnist.rs).
#
# yann.lecun.com now blocks scripted downloads, so we use the mirror that
# PyTorch's own test suite uses.
set -euo pipefail

MIRROR="https://ossci-datasets.s3.amazonaws.com/mnist"
FILES=(
    train-images-idx3-ubyte
    train-labels-idx1-ubyte
    t10k-images-idx3-ubyte
    t10k-labels-idx1-ubyte
)

mkdir -p data
cd data

for f in "${FILES[@]}"; do
    if [[ -f "$f" ]]; then
        echo "have  $f"
        continue
    fi
    echo "get   $f"
    curl -sS --fail --max-time 300 -O "$MIRROR/$f.gz"
    gunzip -f "$f.gz"
done

echo
echo "MNIST ready in ./data"
ls -la
