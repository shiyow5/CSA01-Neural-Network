#!/bin/bash
###############################################################
# Momentum experiment on 4-bit parity
# Tests alpha = 0.0, 0.3, 0.5, 0.7, 0.9
# with J = 5 (4 hidden + 1 bias)
###############################################################
cd "$(dirname "$0")"

TRIALS=10
MAX_EPOCH=50000
J_VAL=5
RESULT_FILE="momentum_results.txt"

echo "=============================================" > "$RESULT_FILE"
echo "  Momentum Experiment on 4-bit Parity"        >> "$RESULT_FILE"
echo "=============================================" >> "$RESULT_FILE"
echo "eta = 0.5, lambda = 1.0, desired_error = 0.001" >> "$RESULT_FILE"
echo "Hidden neurons: $((J_VAL - 1)) (J=$J_VAL)"    >> "$RESULT_FILE"
echo "Trials per config: $TRIALS"                    >> "$RESULT_FILE"
echo "Max epochs: $MAX_EPOCH"                        >> "$RESULT_FILE"
echo ""                                              >> "$RESULT_FILE"

# Generate error curve CSVs for seed=1
echo "Generating error curve CSVs..."
for ALPHA_INT in 0 50 90; do
    gcc momentum_experiment.c -o momentum_exp.out -lm \
        -DJ_VAL=$J_VAL -DMAX_EPOCH=$MAX_EPOCH -DALPHA_VAL=$ALPHA_INT -DDUMP_CSV
    ./momentum_exp.out 1 > /dev/null
done

for ALPHA_INT in 0 30 50 70 90; do
    ALPHA_DISP=$(echo "scale=1; $ALPHA_INT / 100" | bc)

    echo "--- Compiling with alpha=$ALPHA_DISP ---"
    gcc momentum_experiment.c -o momentum_exp.out -lm \
        -DJ_VAL=$J_VAL -DMAX_EPOCH=$MAX_EPOCH -DALPHA_VAL=$ALPHA_INT
    if [ $? -ne 0 ]; then
        echo "Compilation failed for alpha=$ALPHA_DISP"
        exit 1
    fi

    echo "Running $TRIALS trials for alpha=$ALPHA_DISP..."
    echo "---------------------------------------------" >> "$RESULT_FILE"
    echo "Momentum alpha=$ALPHA_DISP"                    >> "$RESULT_FILE"
    echo "---------------------------------------------" >> "$RESULT_FILE"

    converged=0
    total_epochs=0
    min_epoch=999999
    max_epoch=0
    results=""

    for trial in $(seq 1 $TRIALS); do
        output=$(./momentum_exp.out $trial)
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

    echo "  => alpha=$ALPHA_DISP: $converged/$TRIALS converged"
done

echo "=============================================" >> "$RESULT_FILE"
echo "Results saved to $RESULT_FILE"

rm -f momentum_exp.out
