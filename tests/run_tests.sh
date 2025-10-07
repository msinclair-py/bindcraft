#!/bin/bash
# Test runner script for BindCraft

set -e  # Exit on error

echo "================================="
echo "BindCraft Test Suite"
echo "================================="
echo

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed${NC}"
    echo "Install with: pip install pytest pytest-cov"
    exit 1
fi

# Parse command line arguments
COVERAGE=false
SLOW=false
VERBOSE=false
SPECIFIC_TEST=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage|-c)
            COVERAGE=true
            shift
            ;;
        --slow|-s)
            SLOW=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --test|-t)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  -c, --coverage    Run with coverage report"
            echo "  -s, --slow        Include slow tests"
            echo "  -v, --verbose     Verbose output"
            echo "  -t, --test FILE   Run specific test file"
            echo "  -h, --help        Show this help message"
            echo
            echo "Examples:"
            echo "  $0                    # Run all fast tests"
            echo "  $0 --coverage         # Run with coverage"
            echo "  $0 --slow             # Include slow tests"
            echo "  $0 -t test_qc.py      # Run specific test file"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest"

# Add verbosity
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -vv"
fi

# Add coverage
if [ "$COVERAGE" = true ]; then
    if ! command -v pytest-cov &> /dev/null; then
        echo -e "${YELLOW}Warning: pytest-cov not installed, skipping coverage${NC}"
    else
        PYTEST_CMD="$PYTEST_CMD --cov=. --cov-report=html --cov-report=term-missing"
    fi
fi

# Handle slow tests
if [ "$SLOW" = false ]; then
    PYTEST_CMD="$PYTEST_CMD -m 'not slow'"
    echo -e "${YELLOW}Note: Skipping slow tests (use --slow to include)${NC}"
    echo
fi

# Add specific test file
if [ -n "$SPECIFIC_TEST" ]; then
    if [ ! -f "$SPECIFIC_TEST" ]; then
        echo -e "${RED}Error: Test file not found: $SPECIFIC_TEST${NC}"
        exit 1
    fi
    PYTEST_CMD="$PYTEST_CMD $SPECIFIC_TEST"
    echo "Running specific test: $SPECIFIC_TEST"
else
    echo "Running all tests..."
fi

echo
echo "Command: $PYTEST_CMD"
echo "================================="
echo

# Run tests
if $PYTEST_CMD; then
    echo
    echo "================================="
    echo -e "${GREEN}All tests passed!${NC}"
    echo "================================="
    
    if [ "$COVERAGE" = true ]; then
        echo
        echo "Coverage report generated in htmlcov/index.html"
    fi
    
    exit 0
else
    echo
    echo "================================="
    echo -e "${RED}Some tests failed!${NC}"
    echo "================================="
    exit 1
fi
