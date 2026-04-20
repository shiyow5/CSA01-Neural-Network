#!/bin/bash
###############################################################
# 8-bit parity experiment
# Tests hidden neuron counts: 8, 16, 24, 32
# (J = 9, 17, 25, 33 including bias)
###############################################################
cd "$(dirname "$0")"

TRIALS=5
MAX_EPOCH=100000
RESULT_FILE="parity8_results.txt"

echo "=============================================" > "$RESULT_FILE"
echo "  BP 8-bit Parity Check Experiment Results"   >> "$RESULT_FILE"
echo "=============================================" >> "$RESULT_FILE"
echo "eta = 0.5, lambda = 1.0, desired_error = 0.01" >> "$RESULT_FILE"
echo "Trials per config: $TRIALS"                    >> "$RESULT_FILE"
echo "Max epochs: $MAX_EPOCH"                        >> "$RESULT_FILE"
echo ""                                              >> "$RESULT_FILE"

for HIDDEN in 8 16 24 32; do
    J=$((HIDDEN + 1))

    echo "--- Compiling with $HIDDEN hidden neurons (J=$J) ---"
    gcc parity8_experiment.c -o parity8_exp.out -lm \
        -DJ_VAL=$J -DMAX_EPOCH=$MAX_EPOCH
    if [ $? -ne 0 ]; then
        echo "Compilation failed for J=$J"
        exit 1
    fi

    echo "Running $TRIALS trials for $HIDDEN hidden neurons..."
    echo "---------------------------------------------" >> "$RESULT_FILE"
    echo "Hidden neurons: $HIDDEN (J=$J including bias)" >> "$RESULT_FILE"
    echo "---------------------------------------------" >> "$RESULT_FILE"

    converged=0
    total_epochs=0
    min_epoch=999999
    max_epoch=0
    results=""

    for trial in $(seq 1 $TRIALS); do
        echo "  Trial $trial/$TRIALS..."
        output=$(./parity8_exp.out $trial)
        epoch=$(echo "$output" | grep "EPOCHS:" | awk '{print $2}')
        error=$(echo "$output" | grep "FINAL_ERROR:" | awk '{print $2}')
        correct=$(echo "$output" | grep "CORRECT:" | awk '{print $2}')
        status=$(echo "$output" | grep "STATUS:" | awk '{print $2}')

        results="${results}  Trial $trial: epochs=$epoch, error=$error, correct=$correct/256, status=$status\n"

        if [ "$status" = "CONVERGED" ]; then
            converged=$((converged + 1))
            total_epochs=$((total_epochs + epoch))
            if [ "$epoch" -lt "$min_epoch" ]; then min_epoch=$epoch; fi
            if [ "$epoch" -gt "$max_epoch" ]; then max_epoch=$epoch; fi
        fi
    done

    echo -e "$results" >> "$RESULT_FILE"

    if [ $converged -gt 0 ]; then
        avg_epoch=$((total_epochs / converged))
        echo "Summary: converged=$converged/$TRIALS, avg=$avg_epoch, min=$min_epoch, max=$max_epoch" >> "$RESULT_FILE"
    else
        echo "Summary: converged=0/$TRIALS (none converged within $MAX_EPOCH epochs)" >> "$RESULT_FILE"
    fi
    echo "" >> "$RESULT_FILE"

    echo "  => $HIDDEN hidden: $converged/$TRIALS converged"
done

echo "=============================================" >> "$RESULT_FILE"
echo "Results saved to $RESULT_FILE"

rm -f parity8_exp.out
