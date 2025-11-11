#!/bin/bash
# Test script for train_mappo_microgrids.py
# This script runs a quick training test with minimal iterations

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test tracking
TESTS_PASSED=0
TESTS_FAILED=0
TEST_RESULTS=()

echo "=========================================="
echo "Testing MAPPO Training Script"
echo "=========================================="

# Check if training script exists
if [ ! -f "examples/train_mappo_microgrids.py" ]; then
    echo -e "${RED}Error: examples/train_mappo_microgrids.py not found${NC}"
    exit 1
fi

# Activate virtual environment if needed
if [ ! -z "${VIRTUAL_ENV:-}" ]; then
    echo "Using virtual environment: $VIRTUAL_ENV"
else
    echo -e "${YELLOW}Warning: No virtual environment detected${NC}"
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check required packages
echo "Checking required packages..."
python -c "import ray; import torch; import gymnasium" 2>/dev/null || {
    echo -e "${RED}Error: Required packages not installed (ray, torch, gymnasium)${NC}"
    exit 1
}
echo -e "${GREEN}Required packages found${NC}"

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up test checkpoints..."
    rm -rf ./checkpoints/test_mappo_shared 2>/dev/null || true
    rm -rf ./checkpoints/test_ippo_independent 2>/dev/null || true
    rm -rf ./checkpoints/test_mappo_no_shared_reward 2>/dev/null || true
    rm -rf ./checkpoints/test_mappo_custom_params 2>/dev/null || true
    echo "Cleanup complete"
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Function to run test and track results
run_test() {
    local test_name="$1"
    local test_description="$2"
    shift 2
    local test_cmd=("$@")

    echo ""
    echo "=========================================="
    echo "Test: $test_name"
    echo "Description: $test_description"
    echo "=========================================="

    # Run the test with timeout (5 minutes)
    if timeout 300 "${test_cmd[@]}" > /tmp/test_output_$$.log 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        TEST_RESULTS+=("${GREEN}✓${NC} $test_name: PASSED")
    else
        EXIT_CODE=$?
        echo -e "${RED}✗ FAILED (exit code: $EXIT_CODE)${NC}"
        echo "Last 20 lines of output:"
        tail -20 /tmp/test_output_$$.log
        TESTS_FAILED=$((TESTS_FAILED + 1))
        TEST_RESULTS+=("${RED}✗${NC} $test_name: FAILED")
        rm -f /tmp/test_output_$$.log
        return 1
    fi
    rm -f /tmp/test_output_$$.log
    return 0
}

# Pre-cleanup
cleanup

# Test 1: Basic MAPPO training (shared policy)
run_test \
    "MAPPO Shared Policy" \
    "Basic MAPPO training with shared policy (5 iterations)" \
    python examples/train_mappo_microgrids.py \
        --iterations 5 \
        --num-workers 2 \
        --train-batch-size 1000 \
        --checkpoint-freq 5 \
        --experiment-name test_mappo_shared \
        --no-cuda

# Verify checkpoint was created
if [ ! -d "./checkpoints/test_mappo_shared" ]; then
    echo -e "${RED}Error: Checkpoint directory not created${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
else
    echo -e "${GREEN}Checkpoint directory verified${NC}"
fi

# Test 2: IPPO training (independent policies)
run_test \
    "IPPO Independent Policies" \
    "IPPO training with independent policies (5 iterations)" \
    python examples/train_mappo_microgrids.py \
        --iterations 5 \
        --independent-policies \
        --num-workers 2 \
        --train-batch-size 1000 \
        --checkpoint-freq 5 \
        --experiment-name test_ippo_independent \
        --no-cuda

# Verify checkpoint was created
if [ ! -d "./checkpoints/test_ippo_independent" ]; then
    echo -e "${RED}Error: Checkpoint directory not created${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
else
    echo -e "${GREEN}Checkpoint directory verified${NC}"
fi

# Test 3: Different environment configurations
run_test \
    "MAPPO No Shared Reward" \
    "MAPPO with no shared reward configuration (5 iterations)" \
    python examples/train_mappo_microgrids.py \
        --iterations 5 \
        --no-share-reward \
        --penalty 20 \
        --num-workers 2 \
        --train-batch-size 1000 \
        --checkpoint-freq 5 \
        --experiment-name test_mappo_no_shared_reward \
        --no-cuda

# Verify checkpoint was created
if [ ! -d "./checkpoints/test_mappo_no_shared_reward" ]; then
    echo -e "${RED}Error: Checkpoint directory not created${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
else
    echo -e "${GREEN}Checkpoint directory verified${NC}"
fi

# Test 4: Custom hyperparameters
run_test \
    "MAPPO Custom Hyperparameters" \
    "MAPPO with custom hyperparameters (3 iterations)" \
    python examples/train_mappo_microgrids.py \
        --iterations 3 \
        --lr 1e-4 \
        --gamma 0.95 \
        --lambda 0.9 \
        --hidden-dim 128 \
        --num-workers 2 \
        --train-batch-size 1000 \
        --sgd-minibatch-size 64 \
        --num-sgd-iter 5 \
        --checkpoint-freq 3 \
        --experiment-name test_mappo_custom_params \
        --no-cuda

# Verify checkpoint was created
if [ ! -d "./checkpoints/test_mappo_custom_params" ]; then
    echo -e "${RED}Error: Checkpoint directory not created${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
else
    echo -e "${GREEN}Checkpoint directory verified${NC}"
fi

# Test 5: Error handling - Invalid parameters
run_test \
    "Error Handling" \
    "Test script handles invalid parameters gracefully" \
    bash -c "python examples/train_mappo_microgrids.py --iterations 0 --no-cuda 2>&1 | grep -q 'error\|Error\|invalid' && exit 0 || exit 1" || true

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "Total Tests: $((TESTS_PASSED + TESTS_FAILED))"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo ""
echo "Test Results:"
for result in "${TEST_RESULTS[@]}"; do
    echo -e "  $result"
done
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}=========================================="
    echo "All tests completed successfully!"
    echo "==========================================${NC}"
    exit 0
else
    echo -e "${RED}=========================================="
    echo "Some tests failed!"
    echo "==========================================${NC}"
    exit 1
fi
