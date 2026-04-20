#!/bin/bash
###############################################################
# Run BP experiment for 4-bit parity with different hidden sizes
# Hidden neurons: 4, 6, 8, 10 → J (with bias): 5, 7, 9, 11
#
# Usage: ./run_experiment.sh
###############################################################

TRIALS=10
MAX_EPOCH=50000
RESULT_FILE="experiment_results.txt"

echo "=============================================" > "$RESULT_FILE"
echo "  BP 4-bit Parity Check Experiment Results"   >> "$RESULT_FILE"
echo "=============================================" >> "$RESULT_FILE"
echo "eta = 0.5, lambda = 1.0, desired_error = 0.001" >> "$RESULT_FILE"
echo "Trials per config: $TRIALS"                    >> "$RESULT_FILE"
echo "Max epochs: $MAX_EPOCH"                        >> "$RESULT_FILE"
echo ""                                              >> "$RESULT_FILE"

for HIDDEN in 4 6 8 10; do
    J=$((HIDDEN + 1))  # +1 for bias neuron

    echo "--- Compiling with $HIDDEN hidden neurons (J=$J) ---"
    gcc multilayer_perceptron_exp.c -o bp_exp.out -lm \
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
        output=$(./bp_exp.out $trial)
        epoch=$(echo "$output" | grep "EPOCHS:" | awk '{print $2}')
        error=$(echo "$output" | grep "FINAL_ERROR:" | awk '{print $2}')
        correct=$(echo "$output" | grep "CORRECT:" | awk '{print $2}')
        status=$(echo "$output" | grep "STATUS:" | awk '{print $2}')

        results="${results}  Trial $trial: epochs=$epoch, error=$error, correct=$correct/16, status=$status\n"

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

rm -f bp_exp.out
