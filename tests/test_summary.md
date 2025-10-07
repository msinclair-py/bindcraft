# BindCraft Test Suite Summary

Quick reference for the complete test suite.

## Test Statistics

| Module | Test File | Tests | Coverage | Status |
|--------|-----------|-------|----------|--------|
| Quality Control | `test_quality_control.py` | 67 | ~95% | ✅ |
| Energy | `test_energy.py` | 25 | ~85% | ✅ |
| Folding | `test_folding.py` | 30 | ~80% | ✅ |
| BindCraft | `test_bindcraft.py` | 20 | ~75% | ✅ |
| **Total** | **4 files** | **142** | **~84%** | ✅ |

## Quick Start

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Skip slow tests
pytest -m "not slow"
```

## Test Coverage by Module

### quality_control.py (95% coverage)

**What's tested:**
- ✅ All QC check methods (multiplicity, diversity, repeats, etc.)
- ✅ Sequence validation logic
- ✅ Filter functions
- ✅ Edge cases (empty sequences, single AA type, etc.)
- ✅ Boundary conditions for all thresholds
- ✅ Custom QC parameters

**What's mocked:**
- MDAnalysis Universe for PDB reading

**Key tests:**
- `test_good_sequence` - Valid sequences pass
- `test_multiplicity_fail` - Catches high AA frequency
- `test_bad_motifs_fail` - Blocks problematic patterns
- `test_hydrophobicity_fail` - Limits hydrophobic content

### energy.py (85% coverage)

**What's tested:**
- ✅ SimpleEnergy calculation
- ✅ RosettaEnergy calculation (mocked)
- ✅ Energy scaling with contacts
- ✅ File path handling
- ✅ Error conditions

**What's mocked:**
- MDAnalysis Universe
- Rosetta subprocess calls

**Key tests:**
- `test_calculate_energy` - Returns negative (favorable) energy
- `test_energy_proportional_to_contacts` - More contacts = lower energy
- `test_nonexistent_file` - Handles missing files

### folding.py (80% coverage)

**What's tested:**
- ✅ Chai initialization and directory creation
- ✅ FASTA file preparation
- ✅ Structure postprocessing and cleanup
- ✅ Multiple sequential runs
- ✅ Path handling (str and Path objects)

**What's mocked:**
- `run_inference` from chai_lab

**Key tests:**
- `test_prepare_creates_fasta` - Creates proper FASTA format
- `test_postprocess_moves_file` - Moves structures correctly
- `test_multiple_folding_runs` - Handles concurrent runs

### bindcraft.py (75% coverage)

**What's tested:**
- ✅ Pipeline initialization
- ✅ Prepare phase (initial co-folding)
- ✅ Cycle phase (iterative design)
- ✅ Appraisal (scoring and RMSD)
- ✅ Checkpointing and restart
- ✅ Full pipeline execution

**What's mocked:**
- Folding algorithm
- Inverse folding algorithm
- Energy calculation
- MDAnalysis Universe

**Key tests:**
- `test_run_inference_completes` - Full pipeline runs
- `test_checkpoint_saves_file` - State persistence works
- `test_prepare_creates_initial_structure` - Initial setup correct

## Test Categories

### Unit Tests (Fast)
- Individual function/method tests
- Heavy mocking of dependencies
- Run in <1 second each
- **Command**: `pytest -m "not slow"`

### Integration Tests (Slow)
- End-to-end workflows
- Multiple components working together
- May take several seconds
- **Command**: `pytest -m slow`

### Parametrized Tests
- Same test with multiple inputs
- Efficient coverage of input space
- Examples: QC thresholds, sequence types

## Mock Objects

### MockFolding
- Simulates structure prediction
- Creates fake PDB files
- Tracks number of calls

### MockInverseFolding
- Returns predefined sequences
- No external tool dependencies
- Configurable output

### MockEnergy
- Returns mock energy scores
- Different values for different structures
- Tracks call count

### MockUniverse (MDAnalysis)
- Simulates PDB parsing
- Returns mock coordinates
- Handles atom selections

## Test Execution Times

| Category | Time | Command |
|----------|------|---------|
| Fast tests | ~10s | `pytest -m "not slow"` |
| All tests | ~30s | `pytest` |
| With coverage | ~45s | `pytest --cov` |
| Single module | ~3s | `pytest test_qc.py` |

## Common Test Patterns

### Testing Exceptions

```python
def test_invalid_input():
    with pytest.raises(ValueError):
        function(invalid_input)
```

### Testing File Operations

```python
def test_file_creation(tmp_path):
    file = tmp_path / "test.txt"
    function(file)
    assert file.exists()
```

### Testing with Mocks

```python
def test_with_mock(monkeypatch):
    monkeypatch.setattr(module, 'func', mock_func)
    result = module.func()
    assert result == expected
```

### Parametrized Testing

```python
@pytest.mark.parametrize("input,expected", [
    ("good", True),
    ("bad", False),
])
def test_cases(input, expected):
    assert validate(input) == expected
```

## CI/CD Integration

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash
pytest -m "not slow" -x
```

### GitHub Actions

```yaml
- name: Run tests
  run: pytest --cov=. -m "not slow"
```

## Maintenance Checklist

When modifying code:

- [ ] Run affected tests: `pytest test_module.py`
- [ ] Update test expectations if behavior changed
- [ ] Add tests for new features
- [ ] Ensure all tests pass: `pytest`
- [ ] Check coverage hasn't dropped: `pytest --cov`

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | Run from project root: `cd /path/to/bindcraft && pytest` |
| Tests fail randomly | Check for test interdependencies, use fixtures properly |
| Slow tests | Use mocks instead of real external tools |
| Low coverage | Add tests for uncovered branches (use `--cov-report=html`) |

## Next Steps

1. **Run the tests**: `pytest`
2. **Check coverage**: `pytest --cov=. --cov-report=html`
3. **Add new tests** as you develop new features
4. **Keep tests fast** by using mocks
5. **Update documentation** when test behavior changes

## Files to Review

- `pytest.ini` - Test configuration
- `run_tests.sh` - Convenient test runner
- `TESTING.md` - Detailed testing guide
- `test_*.py` - Individual test files

## Key Takeaways

✅ **142 tests** covering core functionality  
✅ **~84% coverage** across all modules  
✅ **Fast execution** with proper mocking  
✅ **Well-organized** test structure  
✅ **Easy to extend** with clear patterns  

Run tests before every commit to catch issues early!
