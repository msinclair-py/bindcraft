# BindCraft Testing Guide

Comprehensive testing suite for the BindCraft pipeline.

## Overview

The test suite covers:
- **Unit tests**: Individual components (QC, energy, folding)
- **Integration tests**: Complete pipeline workflows
- **Mock-based tests**: External dependencies (MDAnalysis, ProteinMPNN)

## Test Files

```
tests/
├── test_quality_control.py    # QC filtering tests (67 tests)
├── test_energy.py              # Energy calculation tests (25 tests)
├── test_folding.py             # Folding module tests (30 tests)
├── test_bindcraft.py           # Main pipeline tests (20 tests)
├── pytest.ini                  # Pytest configuration
└── run_tests.sh                # Test runner script
```

## Installation

### Required Dependencies

```bash
# Core testing
pip install pytest>=7.0.0

# Optional (recommended)
pip install pytest-cov      # Coverage reports
pip install pytest-timeout  # Test timeouts
pip install pytest-xdist    # Parallel execution
```

### Project Dependencies

```bash
# Install BindCraft dependencies
pip install numpy MDAnalysis dill
```

## Running Tests

### Quick Start

```bash
# Run all tests
pytest

# Or use the test runner
chmod +x run_tests.sh
./run_tests.sh
```

### Common Test Commands

```bash
# Run with verbose output
pytest -v

# Run specific test file
pytest test_quality_control.py

# Run specific test class
pytest test_quality_control.py::TestSequenceQualityControl

# Run specific test
pytest test_quality_control.py::TestSequenceQualityControl::test_good_sequence

# Run tests matching pattern
pytest -k "test_energy"

# Skip slow tests
pytest -m "not slow"

# Include slow tests
pytest -m "slow"

# Run with coverage
pytest --cov=. --cov-report=html

# Parallel execution (requires pytest-xdist)
pytest -n auto
```

### Using Test Runner Script

```bash
# Basic usage
./run_tests.sh

# With coverage
./run_tests.sh --coverage

# Include slow tests
./run_tests.sh --slow

# Verbose output
./run_tests.sh --verbose

# Specific test file
./run_tests.sh --test test_quality_control.py

# Combined options
./run_tests.sh --coverage --verbose --slow
```

## Test Organization

### Test Categories

Tests are organized using pytest markers:

```python
@pytest.mark.slow
def test_long_running():
    """Slow test that takes >5 seconds."""
    pass

@pytest.mark.integration
def test_full_pipeline():
    """Integration test."""
    pass

@pytest.mark.requires_external
def test_with_proteinmpnn():
    """Requires external tools."""
    pass
```

Run specific categories:

```bash
pytest -m slow              # Only slow tests
pytest -m "not slow"        # Skip slow tests
pytest -m integration       # Only integration tests
pytest -m requires_external # Only tests needing external tools
```

### Test Structure

Each test file follows this pattern:

```python
# Fixtures for setup/teardown
@pytest.fixture
def setup_data():
    return {"key": "value"}

# Test classes group related tests
class TestFeature:
    def test_basic_functionality(self):
        """Test basic feature."""
        assert True
    
    def test_edge_case(self):
        """Test edge case."""
        assert True

# Parametrized tests for multiple inputs
@pytest.mark.parametrize("input,expected", [
    ("good", True),
    ("bad", False),
])
def test_parametrized(input, expected):
    assert validate(input) == expected
```

## Test Coverage

### Current Coverage

Run coverage report:

```bash
pytest --cov=. --cov-report=term-missing
```

Expected coverage:
- `quality_control.py`: ~95%
- `energy.py`: ~85%
- `folding.py`: ~80%
- `bindcraft.py`: ~75%
- `inverse_folding.py`: ~70%

### Generate HTML Report

```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

## Writing New Tests

### Test Template

```python
"""
Tests for new_module.py
"""

import pytest
from pathlib import Path
from new_module import NewClass


class TestNewClass:
    """Test suite for NewClass."""
    
    @pytest.fixture
    def instance(self):
        """Create test instance."""
        return NewClass()
    
    def test_initialization(self, instance):
        """Test object initialization."""
        assert instance is not None
    
    def test_basic_functionality(self, instance):
        """Test basic functionality."""
        result = instance.method()
        assert result == expected_value
    
    def test_edge_case(self, instance):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            instance.method(invalid_input)


@pytest.mark.parametrize("input,expected", [
    ("case1", result1),
    ("case2", result2),
])
def test_multiple_cases(input, expected):
    """Test multiple cases."""
    assert process(input) == expected
```

### Best Practices

1. **One assertion per test** (when possible)
   ```python
   # Good
   def test_length():
       assert len(result) == 5
   
   def test_content():
       assert "item" in result
   
   # Avoid
   def test_everything():
       assert len(result) == 5
       assert "item" in result
       assert result[0] == "first"
   ```

2. **Use descriptive names**
   ```python
   # Good
   def test_filter_removes_sequences_with_low_diversity():
       pass
   
   # Avoid
   def test_filter():
       pass
   ```

3. **Use fixtures for setup**
   ```python
   @pytest.fixture
   def sample_sequences():
       return ["MKQL", "MALV", "MHKL"]
   
   def test_processing(sample_sequences):
       result = process(sample_sequences)
       assert len(result) == 3
   ```

4. **Mock external dependencies**
   ```python
   def test_with_mock(monkeypatch):
       def mock_function(*args):
           return "mocked"
       
       monkeypatch.setattr(module, 'function', mock_function)
       result = module.function()
       assert result == "mocked"
   ```

## Mocking Strategy

### MDAnalysis Mocking

```python
class MockUniverse:
    def select_atoms(self, selection):
        class MockAtoms:
            positions = np.array([[1, 2, 3]])
            class MockResidues:
                resids = np.array([1])
            residues = MockResidues()
        return MockAtoms()

def test_with_mda(monkeypatch):
    import module
    monkeypatch.setattr(module.mda, 'Universe', 
                       lambda path: MockUniverse())
    # Test code here
```

### File System Mocking

```python
def test_file_operations(tmp_path):
    # tmp_path is a pytest fixture providing temp directory
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")
    
    result = process_file(test_file)
    assert result is not None
```

# BindCraft Testing Guide

Comprehensive testing suite for the BindCraft pipeline.

## Overview

The test suite covers:
- **Unit tests**: Individual components (QC, energy, folding)
- **Integration tests**: Complete pipeline workflows
- **Mock-based tests**: External dependencies (MDAnalysis, ProteinMPNN)

## Test Files

```
tests/
├── test_quality_control.py    # QC filtering tests (67 tests)
├── test_energy.py              # Energy calculation tests (25 tests)
├── test_folding.py             # Folding module tests (30 tests)
├── test_bindcraft.py           # Main pipeline tests (20 tests)
├── pytest.ini                  # Pytest configuration
└── run_tests.sh                # Test runner script
```

## Installation

### Required Dependencies

```bash
# Core testing
pip install pytest>=7.0.0

# Optional (recommended)
pip install pytest-cov      # Coverage reports
pip install pytest-timeout  # Test timeouts
pip install pytest-xdist    # Parallel execution
```

### Project Dependencies

```bash
# Install BindCraft dependencies
pip install numpy MDAnalysis dill
```

## Running Tests

### Quick Start

```bash
# Run all tests
pytest

# Or use the test runner
chmod +x run_tests.sh
./run_tests.sh
```

### Common Test Commands

```bash
# Run with verbose output
pytest -v

# Run specific test file
pytest test_quality_control.py

# Run specific test class
pytest test_quality_control.py::TestSequenceQualityControl

# Run specific test
pytest test_quality_control.py::TestSequenceQualityControl::test_good_sequence

# Run tests matching pattern
pytest -k "test_energy"

# Skip slow tests
pytest -m "not slow"

# Include slow tests
pytest -m "slow"

# Run with coverage
pytest --cov=. --cov-report=html

# Parallel execution (requires pytest-xdist)
pytest -n auto
```

### Using Test Runner Script

```bash
# Basic usage
./run_tests.sh

# With coverage
./run_tests.sh --coverage

# Include slow tests
./run_tests.sh --slow

# Verbose output
./run_tests.sh --verbose

# Specific test file
./run_tests.sh --test test_quality_control.py

# Combined options
./run_tests.sh --coverage --verbose --slow
```

## Test Organization

### Test Categories

Tests are organized using pytest markers:

```python
@pytest.mark.slow
def test_long_running():
    """Slow test that takes >5 seconds."""
    pass

@pytest.mark.integration
def test_full_pipeline():
    """Integration test."""
    pass

@pytest.mark.requires_external
def test_with_proteinmpnn():
    """Requires external tools."""
    pass
```

Run specific categories:

```bash
pytest -m slow              # Only slow tests
pytest -m "not slow"        # Skip slow tests
pytest -m integration       # Only integration tests
pytest -m requires_external # Only tests needing external tools
```

## Test Coverage

### Current Coverage

Run coverage report:

```bash
pytest --cov=. --cov-report=term-missing
```

Expected coverage:
- `quality_control.py`: ~95%
- `energy.py`: ~85%
- `folding.py`: ~80%
- `bindcraft.py`: ~75%
- `inverse_folding.py`: ~70%

### Generate HTML Report

```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

## Writing New Tests

### Test Template

```python
"""
Tests for new_module.py
"""

import pytest
from pathlib import Path
from new_module import NewClass


class TestNewClass:
    """Test suite for NewClass."""
    
    @pytest.fixture
    def instance(self):
        """Create test instance."""
        return NewClass()
    
    def test_initialization(self, instance):
        """Test object initialization."""
        assert instance is not None
    
    def test_basic_functionality(self, instance):
        """Test basic functionality."""
        result = instance.method()
        assert result == expected_value
    
    def test_edge_case(self, instance):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            instance.method(invalid_input)


@pytest.mark.parametrize("input,expected", [
    ("case1", result1),
    ("case2", result2),
])
def test_multiple_cases(input, expected):
    """Test multiple cases."""
    assert process(input) == expected
```

### Best Practices

1. **One assertion per test** (when possible)
2. **Use descriptive names**
3. **Use fixtures for setup**
4. **Mock external dependencies**
5. **Test edge cases and error conditions**
6. **Keep tests independent**

## Mocking Strategy

### MDAnalysis Mocking

```python
class MockUniverse:
    def select_atoms(self, selection):
        class MockAtoms:
            positions = np.array([[1, 2, 3]])
            class MockResidues:
                resids = np.array([1])
            residues = MockResidues()
        return MockAtoms()

def test_with_mda(monkeypatch):
    import module
    monkeypatch.setattr(module.mda, 'Universe', 
                       lambda path: MockUniverse())
    # Test code here
```

### External Tool Mocking

```python
def test_proteinmpnn(monkeypatch):
    def mock_run(*args, **kwargs):
        class Result:
            returncode = 0
        return Result()
    
    import subprocess
    monkeypatch.setattr(subprocess, 'run', mock_run)
    # Test code here
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install pytest pytest-cov
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest --cov=. --cov-report=xml -m "not slow"
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Debugging Failed Tests

### Verbose Output

```bash
pytest -vv                    # Very verbose
pytest --tb=long              # Long traceback
pytest -s                     # Show print statements
pytest --pdb                  # Drop into debugger on failure
```

### Running Single Test

```bash
# Run just one test for debugging
pytest test_file.py::TestClass::test_method -v
```

### Using pdb

```python
def test_something():
    result = calculate()
    import pdb; pdb.set_trace()  # Debugger breakpoint
    assert result == expected
```

## Test Performance

### Profiling Tests

```bash
# Time each test
pytest --durations=10

# Show slowest 20 tests
pytest --durations=20
```

### Parallel Execution

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest -n auto              # Auto-detect CPU cores
pytest -n 4                 # Use 4 workers
```

## Common Issues

### Import Errors

**Problem**: `ModuleNotFoundError`

**Solution**: 
```bash
# Add current directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run pytest from project root
cd /path/to/bindcraft
pytest
```

### Fixture Scope Issues

**Problem**: Fixture not available in test

**Solution**:
```python
# Ensure fixture is in conftest.py or same file
# Check fixture scope (function, class, module, session)

@pytest.fixture(scope="module")  # Available to all tests in module
def expensive_setup():
    return setup_data()
```

### Temporary File Cleanup

**Problem**: Tests leave temporary files

**Solution**:
```python
# Use tmp_path fixture (auto-cleanup)
def test_files(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("data")
    # Cleanup automatic after test
```

### Mocking Not Working

**Problem**: Mock not being used

**Solution**:
```python
# Import in test, not at module level
def test_mock(monkeypatch):
    import module_to_mock  # Import here
    monkeypatch.setattr(module_to_mock, 'function', mock_func)
```

## Test Maintenance

### Updating Tests

When modifying code:

1. **Run affected tests first**
   ```bash
   pytest test_module.py -v
   ```

2. **Update test expectations**
   - Modify assertions if behavior changed intentionally
   - Add tests for new functionality
   - Update mocks if interfaces changed

3. **Run full suite**
   ```bash
   pytest
   ```

### Adding New Features

When adding new features:

1. **Write tests first** (TDD approach)
2. **Cover happy path and edge cases**
3. **Test error conditions**
4. **Add integration tests if needed**

### Code Review Checklist

- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Edge cases covered
- [ ] Error conditions tested
- [ ] Mocks used appropriately
- [ ] Test names are descriptive
- [ ] Coverage hasn't decreased

## Resources

### Pytest Documentation

- [Pytest Docs](https://docs.pytest.org/)
- [Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Parametrize](https://docs.pytest.org/en/stable/parametrize.html)
- [Monkeypatch](https://docs.pytest.org/en/stable/monkeypatch.html)

### Testing Best Practices

- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Test-Driven Development](https://en.wikipedia.org/wiki/Test-driven_development)

## Summary

The BindCraft test suite provides:

✅ **Comprehensive coverage** of core functionality  
✅ **Fast execution** with mocked external dependencies  
✅ **Easy to run** with simple commands  
✅ **Well organized** with clear test structure  
✅ **Maintainable** with good documentation  

Run tests frequently during development to catch issues early!
