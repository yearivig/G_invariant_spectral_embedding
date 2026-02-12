#!/bin/bash
#
# Quick-start script for running SO(3)/SO(2) experiments
# Usage: ./run_experiment.sh
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}  SO(3)/SO(2) Journal Experiment Runner  ${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "converges_so3_so2_exp_journal.py" ]; then
    echo -e "${YELLOW}Warning: converges_so3_so2_exp_journal.py not found${NC}"
    echo "Changing to the correct directory..."
    cd /a/home/cc/students/math/vigderyeari/Documents/G_invariant_kernel/exp3/successful_exp3
fi

# Check Python
echo -e "${GREEN}✓${NC} Checking Python installation..."
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        PYTHON=python3
    else
        echo -e "${YELLOW}Error: Python not found${NC}"
        exit 1
    fi
else
    PYTHON=python
fi

echo -e "${GREEN}✓${NC} Using: $($PYTHON --version)"
echo ""

# Check dependencies
echo -e "${GREEN}✓${NC} Checking dependencies..."
$PYTHON -c "import numpy, matplotlib, tqdm" 2>/dev/null || {
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -q -r requirements.txt
}

echo -e "${GREEN}✓${NC} All dependencies satisfied"
echo ""

# Create plots directory
mkdir -p plots

# Run the experiment
echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}  Starting Experiment                    ${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""
echo "This will run 3 kernels with the following parameters:"
echo "  - ell = 1"
echo "  - num_points = 10,000 (300,000 for Euclidean)"
echo "  - num_trials = 500"
echo "  - epsilon range: [exp(-6), exp(-1)]"
echo ""
echo -e "${YELLOW}Note: This may take 30-60 minutes to complete.${NC}"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Run the main script
$PYTHON converges_so3_so2_exp_journal.py

# Check if completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${BLUE}==========================================${NC}"
    echo -e "${GREEN}✓ Experiment completed successfully!${NC}"
    echo -e "${BLUE}==========================================${NC}"
    echo ""
    echo "Output files:"
    echo "  📊 Plots: plots/*.pdf"
    echo "  💾 Data:  so3_so2_P_1_00_results.pkl"
    echo ""
    echo "To regenerate plots with custom styling:"
    echo "  $PYTHON replot_from_pickle.py"
    echo ""
else
    echo -e "${YELLOW}Error: Experiment failed${NC}"
    exit 1
fi

