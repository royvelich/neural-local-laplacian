#!/bin/bash

# Check if sweep-id argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <sweep-id>"
    echo "Example: $0 my-sweep-123"
    exit 1
fi

SWEEP_ID="$1"

echo "Starting infinite training loop with sweep-id: $SWEEP_ID"
echo "Press Ctrl+C to stop the loop"

# Function to check if any GPU processes are running
check_gpu_processes() {
    local gpu_processes
    gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)

    if [ -n "$gpu_processes" ]; then
        return 1  # GPU processes found
    else
        return 0  # No GPU processes
    fi
}

# Function to wait until GPU is free
wait_for_gpu_free() {
    echo "Checking GPU status..."

    while ! check_gpu_processes; do
        echo "GPU processes still running. Waiting 5 seconds..."
        nvidia-smi --query-compute-apps=pid,process_name --format=csv
        sleep 5
    done

    echo "GPU is now free. Proceeding to next iteration."
}

# Infinite loop
iteration=1
while true; do
    echo ""
    echo "========================================="
    echo "Starting iteration $iteration"
    echo "Time: $(date)"
    echo "========================================="

    # Run the training script
    echo "Running: python ./train_fmaps.py --sweep-id $SWEEP_ID"
    python ./train_fmaps.py --sweep-id "$SWEEP_ID"

    echo "Training iteration $iteration completed with exit code: $?"

    # Wait for GPU to be free before starting next iteration
    wait_for_gpu_free

    iteration=$((iteration + 1))

    echo "Waiting 2 seconds before next iteration..."
    sleep 2
done