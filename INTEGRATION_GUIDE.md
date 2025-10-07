# BindCraft Integration Guide

This document explains how all the components of the BindCraft pipeline work together.

## Module Overview

### Core Modules

1. **bindcraft.py** - Main pipeline orchestrator
2. **folding.py** - Structure prediction (Chai/AlphaFold2)
3. **inverse_folding.py** - Sequence generation (ProteinMPNN)
4. **energy.py** - Energy/scoring functions
5. **quality_control.py** - Sequence filtering

### Utility Modules

6. **run_bindcraft.py** - Command-line interface
7. **analysis.py** - Post-processing and visualization
8. **example_workflow.py** - Usage examples

## Data Flow

```
User Input (Target + Binder)
         ↓
┌────────────────────────────────────┐
│     bindcraft.py (BindCraft)       │
│  - Coordinates all components      │
│  - Manages iterations              │
│  - Handles checkpointing           │
└────────────────────────────────────┘
         ↓
    ┌────────┴────────┐
    ↓                 ↓
┌─────────┐     ┌──────────────┐
│folding  │     │inverse_folding│
│  .py    │     │    .py        │
└────┬────┘     └──────┬────────┘
     ↓                 ↓
┌─────────┐     ┌──────────────┐
│ Chai/   │     │ ProteinMPNN  │
│AlphaFold│     │   / ESM-IF   │
└─────────┘     └──────────────┘
         ↓
┌────────────────────────────────────┐
│   quality_control.py (QC Filter)   │
│  - Validates sequences             │
│  - Filters bad designs             │
└────────────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│      energy.py (Scoring)           │
│  - Calculates binding energy       │
│  - Evaluates interfaces            │
└────────────────────────────────────┘
         ↓
    Results Dictionary
         ↓
┌────────────────────────────────────┐
│    analysis.py (Post-process)      │
│  - Generate reports                │
│  - Create visualizations           │
└────────────────────────────────────┘
```

## Component Integration

### 1. BindCraft Class (bindcraft.py)

**Role**: Main pipeline coordinator

**Key Methods**:
- `__init__()`: Initializes all components
- `run_inference()`: Main execution loop
- `prepare()`: Initial co-folding
- `cycle()`: One design iteration
- `appraise()`: Evaluate designs

**Integrates with**:
- `Folding` class for structure prediction
- `InverseFolding` class for sequence generation
- `EnergyCalculation` class for scoring
- `SequenceQualityControl` class for filtering

### 2. Folding Classes (folding.py)

**Implementations**:
- `Chai`: Uses Chai-1 model
- `Boltz`: Placeholder for Boltz model

**Interface**:
```python
class Folding(ABC):
    def prepare(seqs, label) -> Path
    def __call__(seqs, label) -> Path
    def postprocess(in_path) -> Path
```

**Used by**: `BindCraft.prepare()` and `BindCraft.cycle()`

### 3. Inverse Folding Classes (inverse_folding.py)

**Implementations**:
- `ProteinMPNN`: Main sequence generator
- `ESMIF1`: Placeholder for ESM-IF

**Interface**:
```python
class InverseFolding(ABC):
    def prepare(pdb_path, file1, file2)
    def __call__(input_path, pdb_path, output_path, remodel) -> list[str]
    def postprocessing(output_dir) -> list[str]
```

**Used by**: `BindCraft.cycle()`

### 4. Energy Calculation (energy.py)

**Implementations**:
- `SimpleEnergy`: Geometric interface metrics
- `RosettaEnergy`: Physics-based scoring

**Interface**:
```python
class EnergyCalculation(ABC):
    def __call__(structure: Path) -> float
```

**Used by**: `BindCraft.measure_energy()`

### 5. Quality Control (quality_control.py)

**Main Class**: `SequenceQualityControl`

**Checks**:
- Amino acid multiplicity
- Sequence diversity
- Repeat patterns
- Charge distribution
- Hydrophobicity
- Bad motifs
- Terminal residues

**Interface**:
```python
def __call__(sequence: Union[Path, str]) -> bool
```

**Used by**: `BindCraft.cycle()` for filtering

## Execution Flow

### Step-by-Step Execution

```python
# 1. User creates pipeline
bindcraft = BindCraft(
    target=target_seq,
    binder=binder_seq,
    fold_alg=Chai(...),
    inv_fold_alg=ProteinMPNN(...),
    energy_alg=SimpleEnergy(),
    qc_filter=SequenceQualityControl(),
    n_rounds=3
)

# 2. Run inference
results = bindcraft.run_inference()
```

**Internal Flow**:

```
run_inference()
├─ prepare()                          # Round 0
│  ├─ fold([target, binder])          # Initial co-fold
│  ├─ get_binder_coords()             # Extract reference
│  ├─ get_interface()                 # Identify interface
│  └─ measure_energy()                # Calculate energy
│
└─ while rounds < n_rounds:           # Rounds 1-N
   ├─ cycle()
   │  ├─ inv_fold()                   # Generate sequences
   │  ├─ qc(seq) for seq              # Filter sequences
   │  └─ fold([target, seq])          # Co-fold new designs
   │
   ├─ appraise()
   │  ├─ get_binder_coords()          # Extract coords
   │  ├─ rmsd()                       # Calculate RMSD
   │  └─ measure_energy()             # Calculate energy
   │
   └─ checkpoint()                     # Save progress
```

## Key Design Patterns

### 1. Abstract Base Classes

All major components use ABC pattern for extensibility:

```python
# Easy to add new implementations
class MyFolder(Folding):
    def prepare(self, seqs, label):
        # Custom preparation
        pass
    
    def __call__(self, seqs, label):
        # Custom folding
        pass
```

### 2. Dependency Injection

Components are injected into BindCraft:

```python
# Flexible composition
bindcraft = BindCraft(
    fold_alg=Chai(...),      # Can swap with Boltz
    inv_fold_alg=ProteinMPNN(...),  # Can swap with ESMIF1
    energy_alg=SimpleEnergy(),      # Can swap with Rosetta
)
```

### 3. Checkpointing

Automatic state persistence:

```python
# In cycle()
self.checkpoint(structures)

# Resume from checkpoint
if self.chk_file.exists():
    self.restart_run()
```

## File Mappings

### Original Scripts → New Modules

| Original File | Integrated Into | Purpose |
|--------------|-----------------|---------|
| `fold_init.py` | `folding.py` (Chai class) | Initial folding |
| `fold_forward.py` | `bindcraft.py` (cycle method) | Iterative folding |
| `prepare_inverse_fold.py` | `inverse_folding.py` (prepare) | MPNN prep |
| `run_proteinmpnn.py` | `inverse_folding.py` (run) | MPNN execution |
| `compute_interface.py` | `bindcraft.py` (get_interface) | Interface ID |
| `qcer.py` | `quality_control.py` | QC filters |
| `qc_seqs.py` | `quality_control.py` (filter_sequences) | Batch filtering |

## Data Structures

### Results Dictionary

```python
{
    'initial_guess': {
        'sequence': 'MKQHKAM...',
        'structure': 'folds/pred.model_idx_0.cif',
        'rmsd': 0.0,
        'energy': -15.3
    },
    1: {
        'sequence': 'MKRHEAL...',
        'structure': 'folds/pred.model_idx_1.cif',
        'rmsd': 2.1,
        'energy': -18.7
    },
    # ... more binders
}
```

### Interface Residues

```python
# In BindCraft instance
self.remodel = [1, 2, 3, 15, 16, 17, ...]  # Non-interface positions
# These positions are redesigned by ProteinMPNN
# Interface positions are kept fixed
```

## Extension Points

### Adding New Folding Methods

```python
# folding.py
class Boltz(Folding):
    def __init__(self, model_path, output_dir):
        self.model = load_boltz(model_path)
        self.output_dir = output_dir
    
    def prepare(self, seqs, label):
        # Create input format for Boltz
        return input_file
    
    def __call__(self, seqs, label):
        # Run Boltz inference
        return structure_path
    
    def postprocess(self, in_path):
        # Convert to standard format
        return final_path
```

### Adding New Inverse Folding

```python
# inverse_folding.py
class ESMIF1(InverseFolding):
    def __init__(self, model_checkpoint):
        self.model = load_esm_if1(model_checkpoint)
    
    def __call__(self, input_path, pdb_path, output_path, remodel):
        # Generate sequences with ESM-IF1
        sequences = self.model.design(pdb_path, positions=remodel)
        return sequences
```

### Custom Energy Functions

```python
# energy.py
class CustomEnergy(EnergyCalculation):
    def __init__(self, weights_file):
        self.weights = load_weights(weights_file)
    
    def __call__(self, structure):
        # Custom energy calculation
        return energy_score
```

### Custom QC Filters

```python
# quality_control.py
class StrictQC(SequenceQualityControl):
    def __init__(self):
        super().__init__(
            max_hydrophobic_ratio=0.3,  # More strict
            max_charge=3,
            min_diversity=12
        )
    
    def custom_check(self):
        # Add custom validation
        pass
```

## Testing Strategy

### Unit Tests

```python
# test_folding.py
def test_chai_folding():
    folder = Chai(fasta_dir='test/', out='test_out/')
    result = folder(['MKQLE...', 'MVHLT...'], 'test_1')
    assert result.exists()

# test_qc.py
def test_quality_control():
    qc = SequenceQualityControl()
    assert qc('MKQLEDKIEELLSKIHAEQEREEVRKM')
    assert not qc('AAAAAAAAAAAAAAAA')  # Too many repeats
```

### Integration Tests

```python
# test_pipeline.py
def test_full_pipeline():
    bindcraft = BindCraft(
        target='MKQ...',
        binder='MVH...',
        fold_alg=MockFolder(),
        inv_fold_alg=MockInverseFolder(),
        n_rounds=1
    )
    results = bindcraft.run_inference()
    assert len(results) > 0
```

## Performance Considerations

### Memory Management

```python
# In bindcraft.py - cleanup after use
u = mda.Universe(str(pdb))
coords = u.select_atoms('name CA').positions
del u  # Explicit cleanup

# In folding.py - use temporary directories
out = self.devshm / label  # Fast tmpfs
# ... run inference ...
shutil.rmtree(str(in_path))  # Clean up
```

### Parallelization

```python
# Can parallelize folding across multiple GPUs
# Example for multi-GPU setup:
fold_alg_gpu0 = Chai(..., device='cuda:0')
fold_alg_gpu1 = Chai(..., device='cuda:1')

# Process designs in parallel
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [
        executor.submit(fold_alg_gpu0, seqs1, label1),
        executor.submit(fold_alg_gpu1, seqs2, label2)
    ]
```

## Debugging Tips

### Enable Verbose Logging

```python
# Add to bindcraft.py __init__
import logging
logging.basicConfig(level=logging.DEBUG)
self.logger = logging.getLogger('BindCraft')

# Use throughout
self.logger.debug(f"Generated {len(sequences)} sequences")
```

### Checkpoint Inspection

```python
import dill as pickle

with open('checkpoint.pkl', 'rb') as f:
    data = pickle.load(f)

for key, val in data.items():
    print(f"{key}: {val.keys()}")
```

### Visualize Intermediate Structures

```python
# After folding
structure = fold_alg(['target', 'binder'], 'debug')
# View in PyMOL, ChimeraX, etc.
```

## Summary

The BindCraft pipeline synthesizes multiple components into a cohesive workflow:

1. **Modular Design**: Each component has a clear interface
2. **Extensibility**: Easy to add new methods via ABC pattern
3. **Checkpointing**: Robust state management
4. **Quality Control**: Automated filtering
5. **Analysis**: Comprehensive post-processing

The design pattern established in your templates has been extended to create a production-ready pipeline that replicates the paper's methodology while remaining flexible and maintainable.
