#!/bin/bash

# Check if sweep-id argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <sweep-id> [threads-per-rank]"
    echo "Example: $0 my-sweep-123"
    echo "Example: $0 my-sweep-123 15"
    exit 1
fi

SWEEP_ID="$1"

# Auto-detect or use provided thread count
if [ -n "$2" ]; then
    THREADS="$2"
else
    N_CPUS=$(nproc)
    N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    N_GPUS=${N_GPUS:-1}
    THREADS=$(( N_CPUS / (N_GPUS > 0 ? N_GPUS : 1) ))
    THREADS=$(( THREADS > 0 ? THREADS : 1 ))
fi

export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS

# Create log directory
LOG_DIR="sweep_logs/$(date +%Y%m%d_%H%M%S)_${SWEEP_ID##*/}"
mkdir -p "$LOG_DIR"

echo "Starting infinite training loop with sweep-id: $SWEEP_ID"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS  MKL_NUM_THREADS=$MKL_NUM_THREADS  (CPUs=$(nproc), GPUs=${N_GPUS:-manual})"
echo "Logs: $LOG_DIR/"
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
    LOG_FILE="$LOG_DIR/run_$(printf '%03d' $iteration).log"

    echo ""
    echo "========================================="
    echo "Starting iteration $iteration"
    echo "Time: $(date)"
    echo "Log: $LOG_FILE"
    echo "========================================="

    # Run the training script, logging stdout+stderr to file and terminal
    python ./train_fmaps.py --sweep-id "$SWEEP_ID" 2>&1 | tee "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}

    echo "" >> "$LOG_FILE"
    echo "=========================================" >> "$LOG_FILE"
    echo "Exit code: $EXIT_CODE" >> "$LOG_FILE"
    echo "Time: $(date)" >> "$LOG_FILE"

    echo "Training iteration $iteration completed with exit code: $EXIT_CODE"

    # Check for OOM kills
    if [ $EXIT_CODE -ne 0 ]; then
        echo "--- Recent OOM events ---" | tee -a "$LOG_FILE"
        dmesg | grep -i "oom\|killed" | tail -5 | tee -a "$LOG_FILE"
        echo "-------------------------" | tee -a "$LOG_FILE"
    fi

    # Wait for GPU to be free before starting next iteration
    wait_for_gpu_free

    iteration=$((iteration + 1))

    echo "Waiting 2 seconds before next iteration..."
    sleep 2
done