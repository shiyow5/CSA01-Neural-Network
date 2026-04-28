#!/bin/bash

SRC="hopfield_neural_network.c"
RESULTS_DIR="results"

mkdir -p "$RESULTS_DIR"

for noise in 0.00 0.10 0.15; do
    echo "=== noise_rate = $noise ==="

    sed "s/#define noise_rate.*/#define noise_rate $noise/" "$SRC" > /tmp/hopfield_tmp.c
    gcc -O2 -o /tmp/hopfield_exp /tmp/hopfield_tmp.c -lm
    if [ $? -ne 0 ]; then
        echo "Compile failed (noise=$noise)"
        continue
    fi

    OUT="$RESULTS_DIR/noise_$(echo "$noise" | tr '.' '_').txt"
    yes "" | /tmp/hopfield_exp > "$OUT" 2>&1
    echo "  -> $OUT"
done

echo "Done."
