"""
Tests for bindcraft.py main pipeline.
"""

import pytest
from pathlib import Path
import numpy as np
import dill as pickle
from bindcraft import BindCraft
from folding import Folding
from inverse_folding import InverseFolding
from energy import EnergyCalculation
from quality_control import SequenceQualityControl


class MockFolding(Folding):
    """Mock folding class for testing."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.call_count = 0
    
    def prepare(self, seqs, label):
        return Path("mock.fa")
    
    def __call__(self, seqs, label):
        self.call_count += 1
        # Create mock structure file
        structure = self.output_dir / f"structure_{label}.pdb"
        structure.write_text(f"MOCK PDB\nSEQ1: {seqs[0]}\nSEQ2: {seqs[1]}")
        return structure
    
    def postprocess(self, in_path):
        return in_path


class MockInverseFolding(InverseFolding):
    """Mock inverse folding class for testing."""
    
    def __init__(self, sequences_to_return=None):
        self.sequences = sequences_to_return or [
            "MKQLEDKIEAFRSTV",
            "MALKVIEDREASTV",
            "MHKLERDIKAFRSWV"
        ]
        self.call_count = 0
    
    def prepare(self, pdb_path, file1, file2):
        pass
    
    def __call__(self, input_path, pdb_path, output_path, remodel):
        self.call_count += 1
        return self.sequences
    
    def postprocessing(self, output_dir):
        return self.sequences


class MockEnergy(EnergyCalculation):
    """Mock energy calculation."""
    
    def __init__(self):
        self.call_count = 0
    
    def __call__(self, structure):
        self.call_count += 1
        # Return different energies for different structures
        if "initial" in str(structure):
            return -10.0
        return -15.0 - np.random.random() * 5.0


class TestBindCraftInitialization:
    """Test BindCraft initialization."""
    
    @pytest.fixture
    def mock_components(self, tmp_path):
        """Create mock components."""
        fold_alg = MockFolding(tmp_path / "folds")
        inv_fold_alg = MockInverseFolding()
        energy_alg = MockEnergy()
        qc_filter = SequenceQualityControl()
        return fold_alg, inv_fold_alg, energy_alg, qc_filter
    
    def test_basic_initialization(self, tmp_path, mock_components):
        """Test basic BindCraft initialization."""
        fold_alg, inv_fold_alg, energy_alg, qc_filter = mock_components
        
        bindcraft = BindCraft(
            target="MKQLEDKIEELLSKYH",
            binder="MALKVIEDRKA",
            fold_alg=fold_alg,
            inv_fold_alg=inv_fold_alg,
            energy_alg=energy_alg,
            qc_filter=qc_filter,
            chk_file=str(tmp_path / "checkpoint.pkl"),
            n_rounds=3
        )
        
        assert bindcraft.target == "MKQLEDKIEELLSKYH"
        assert bindcraft.binder == "MALKVIEDRKA"
        assert bindcraft.n_rounds == 3
        assert bindcraft.trial == 0
        assert bindcraft.bnum == 1
    
    def test_initialization_with_defaults(self, tmp_path, mock_components):
        """Test initialization uses defaults when not provided."""
        fold_alg, inv_fold_alg, _, _ = mock_components
        
        bindcraft = BindCraft(
            target="MKQL",
            binder="MALV",
            fold_alg=fold_alg,
            inv_fold_alg=inv_fold_alg,
            chk_file=str(tmp_path / "chk.pkl")
        )
        
        # Should have default energy and QC
        assert bindcraft.energy is not None
        assert bindcraft.qc is not None
        assert bindcraft.cutoff == 5.0
    
    def test_custom_parameters(self, tmp_path, mock_components):
        """Test custom parameters are stored."""
        fold_alg, inv_fold_alg, energy_alg, qc_filter = mock_components
        
        bindcraft = BindCraft(
            target="MKQL",
            binder="MALV",
            fold_alg=fold_alg,
            inv_fold_alg=inv_fold_alg,
            energy_alg=energy_alg,
            qc_filter=qc_filter,
            chk_file=str(tmp_path / "chk.pkl"),
            n_rounds=5,
            cutoff=6.0,
            custom_param="custom_value"
        )
        
        assert bindcraft.n_rounds == 5
        assert bindcraft.cutoff == 6.0
        assert bindcraft.custom_param == "custom_value"


class TestBindCraftPrepare:
    """Test the prepare method."""
    
    @pytest.fixture
    def bindcraft_instance(self, tmp_path, monkeypatch):
        """Create BindCraft instance with mocked MDAnalysis."""
        fold_alg = MockFolding(tmp_path / "folds")
        inv_fold_alg = MockInverseFolding()
        
        bindcraft = BindCraft(
            target="MKQLEDKIEELLSKYH",
            binder="MALKVIEDRKA",
            fold_alg=fold_alg,
            inv_fold_alg=inv_fold_alg,
            energy_alg=MockEnergy(),
            chk_file=str(tmp_path / "chk.pkl")
        )
        
        # Mock MDAnalysis Universe
        class MockUniverse:
            def select_atoms(self, sel):
                class MockAtoms:
                    positions = np.random.rand(11, 3)  # 11 residues
                    class MockResidues:
                        resids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
                    residues = MockResidues()
                return MockAtoms()
        
        def mock_universe(path):
            return MockUniverse()
        
        import bindcraft as bc_module
        monkeypatch.setattr(bc_module.mda, 'Universe', mock_universe)
        
        return bindcraft
    
    def test_prepare_creates_initial_structure(self, bindcraft_instance):
        """Test prepare creates initial structure."""
        results = bindcraft_instance.prepare()
        
        assert 'initial_guess' in results
        assert 'sequence' in results['initial_guess']
        assert 'structure' in results['initial_guess']
        assert 'rmsd' in results['initial_guess']
        assert 'energy' in results['initial_guess']
        
        assert results['initial_guess']['sequence'] == bindcraft_instance.binder
        assert results['initial_guess']['rmsd'] == 0.0
    
    def test_prepare_sets_reference(self, bindcraft_instance):
        """Test prepare sets reference coordinates."""
        bindcraft_instance.prepare()
        
        assert bindcraft_instance.reference is not None
        assert isinstance(bindcraft_instance.reference, np.ndarray)
    
    def test_prepare_identifies_interface(self, bindcraft_instance):
        """Test prepare identifies interface residues."""
        bindcraft_instance.prepare()
        
        assert bindcraft_instance.remodel is not None
        assert isinstance(bindcraft_instance.remodel, list)


class TestBindCraftCycle:
    """Test the cycle method."""
    
    @pytest.fixture
    def prepared_bindcraft(self, tmp_path, monkeypatch):
        """Create prepared BindCraft instance."""
        fold_alg = MockFolding(tmp_path / "folds")
        inv_fold_alg = MockInverseFolding()
        
        bindcraft = BindCraft(
            target="MKQLEDKIEELLSKYH",
            binder="MALKVIEDRKA",
            fold_alg=fold_alg,
            inv_fold_alg=inv_fold_alg,
            energy_alg=MockEnergy(),
            qc_filter=SequenceQualityControl(max_hydrophobic_ratio=0.6),  # Relaxed
            chk_file=str(tmp_path / "chk.pkl")
        )
        
        # Mock MDAnalysis
        class MockUniverse:
            def select_atoms(self, sel):
                class MockAtoms:
                    positions = np.random.rand(11, 3)
                    class MockResidues:
                        resids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
                    residues = MockResidues()
                return MockAtoms()
        
        def mock_universe(path):
            return MockUniverse()
        
        import bindcraft as bc_module
        monkeypatch.setattr(bc_module.mda, 'Universe', mock_universe)
        
        # Run prepare
        bindcraft.prepare()
        
        return bindcraft
    
    def test_cycle_generates_sequences(self, prepared_bindcraft):
        """Test cycle generates new sequences."""
        new_structures = prepared_bindcraft.cycle()
        
        assert len(new_structures) > 0
        assert all('sequence' in v for v in new_structures.values())
        assert all('structure' in v for v in new_structures.values())
    
    def test_cycle_increments_bnum(self, prepared_bindcraft):
        """Test cycle increments binder number."""
        initial_bnum = prepared_bindcraft.bnum
        new_structures = prepared_bindcraft.cycle()
        
        assert prepared_bindcraft.bnum > initial_bnum
        assert prepared_bindcraft.bnum == initial_bnum + len(new_structures)
    
    def test_cycle_applies_qc(self, tmp_path, monkeypatch):
        """Test cycle applies quality control."""
        fold_alg = MockFolding(tmp_path / "folds")
        
        # Mock inverse folding to return bad sequences
        bad_sequences = ["AAAAAAAAAAAAAAAA", "KKKKKKKKKKKKKKKK"]
        inv_fold_alg = MockInverseFolding(bad_sequences)
        
        bindcraft = BindCraft(
            target="MKQL",
            binder="MALV",
            fold_alg=fold_alg,
            inv_fold_alg=inv_fold_alg,
            energy_alg=MockEnergy(),
            qc_filter=SequenceQualityControl(),  # Strict QC
            chk_file=str(tmp_path / "chk.pkl")
        )
        
        # Mock MDAnalysis
        class MockUniverse:
            def select_atoms(self, sel):
                class MockAtoms:
                    positions = np.random.rand(4, 3)
                    class MockResidues:
                        resids = np.array([1, 2, 3, 4])
                    residues = MockResidues()
                return MockAtoms()
        
        def mock_universe(path):
            return MockUniverse()
        
        import bindcraft as bc_module
        monkeypatch.setattr(bc_module.mda, 'Universe', mock_universe)
        
        bindcraft.prepare()
        new_structures = bindcraft.cycle()
        
        # Bad sequences should be filtered out
        assert len(new_structures) == 0


class TestBindCraftAppraise:
    """Test the appraise method."""
    
    @pytest.fixture
    def bindcraft_with_structures(self, tmp_path, monkeypatch):
        """Create BindCraft with mock structures."""
        fold_alg = MockFolding(tmp_path / "folds")
        
        bindcraft = BindCraft(
            target="MKQL",
            binder="MALV",
            fold_alg=fold_alg,
            inv_fold_alg=MockInverseFolding(),
            energy_alg=MockEnergy(),
            chk_file=str(tmp_path / "chk.pkl")
        )
        
        # Mock MDAnalysis
        class MockUniverse:
            def select_atoms(self, sel):
                class MockAtoms:
                    positions = np.random.rand(4, 3)
                    class MockResidues:
                        resids = np.array([1, 2, 3, 4])
                    residues = MockResidues()
                return MockAtoms()
        
        def mock_universe(path):
            return MockUniverse()
        
"""
Tests for bindcraft.py main pipeline.
"""

import pytest
from pathlib import Path
import numpy as np
import dill as pickle
from bindcraft import BindCraft
from folding import Folding
from inverse_folding import InverseFolding
from energy import EnergyCalculation
from quality_control import SequenceQualityControl


class MockFolding(Folding):
    """Mock folding class for testing."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.call_count = 0
    
    def prepare(self, seqs, label):
        return Path("mock.fa")
    
    def __call__(self, seqs, label):
        self.call_count += 1
        # Create mock structure file
        structure = self.output_dir / f"structure_{label}.pdb"
        structure.write_text(f"MOCK PDB\nSEQ1: {seqs[0]}\nSEQ2: {seqs[1]}")
        return structure
    
    def postprocess(self, in_path):
        return in_path


class MockInverseFolding(InverseFolding):
    """Mock inverse folding class for testing."""
    
    def __init__(self, sequences_to_return=None):
        self.sequences = sequences_to_return or [
            "MKQLEDKIEAFRSTV",
            "MALKVIEDREASTV",
            "MHKLERDIKAFRSWV"
        ]
        self.call_count = 0
    
    def prepare(self, pdb_path, file1, file2):
        pass
    
    def __call__(self, input_path, pdb_path, output_path, remodel):
        self.call_count += 1
        return self.sequences
    
    def postprocessing(self, output_dir):
        return self.sequences


class MockEnergy(EnergyCalculation):
    """Mock energy calculation."""
    
    def __init__(self):
        self.call_count = 0
    
    def __call__(self, structure):
        self.call_count += 1
        # Return different energies for different structures
        if "initial" in str(structure):
            return -10.0
        return -15.0 - np.random.random() * 5.0


class TestBindCraftInitialization:
    """Test BindCraft initialization."""
    
    @pytest.fixture
    def mock_components(self, tmp_path):
        """Create mock components."""
        fold_alg = MockFolding(tmp_path / "folds")
        inv_fold_alg = MockInverseFolding()
        energy_alg = MockEnergy()
        qc_filter = SequenceQualityControl()
        return fold_alg, inv_fold_alg, energy_alg, qc_filter
    
    def test_basic_initialization(self, tmp_path, mock_components):
        """Test basic BindCraft initialization."""
        fold_alg, inv_fold_alg, energy_alg, qc_filter = mock_components
        
        bindcraft = BindCraft(
            target="MKQLEDKIEELLSKYH",
            binder="MALKVIEDRKA",
            fold_alg=fold_alg,
            inv_fold_alg=inv_fold_alg,
            energy_alg=energy_alg,
            qc_filter=qc_filter,
            chk_file=str(tmp_path / "checkpoint.pkl"),
            n_rounds=3
        )
        
        assert bindcraft.target == "MKQLEDKIEELLSKYH"
        assert bindcraft.binder == "MALKVIEDRKA"
        assert bindcraft.n_rounds == 3
        assert bindcraft.trial == 0
        assert bindcraft.bnum == 1
    
    def test_initialization_with_defaults(self, tmp_path, mock_components):
        """Test initialization uses defaults when not provided."""
        fold_alg, inv_fold_alg, _, _ = mock_components
        
        bindcraft = BindCraft(
            target="MKQL",
            binder="MALV",
            fold_alg=fold_alg,
            inv_fold_alg=inv_fold_alg,
            chk_file=str(tmp_path / "chk.pkl")
        )
        
        # Should have default energy and QC
        assert bindcraft.energy is not None
        assert bindcraft.qc is not None
        assert bindcraft.cutoff == 5.0


class TestBindCraftPrepare:
    """Test the prepare method."""
    
    @pytest.fixture
    def bindcraft_instance(self, tmp_path, monkeypatch):
        """Create BindCraft instance with mocked MDAnalysis."""
        fold_alg = MockFolding(tmp_path / "folds")
        inv_fold_alg = MockInverseFolding()
        
        bindcraft = BindCraft(
            target="MKQLEDKIEELLSKYH",
            binder="MALKVIEDRKA",
            fold_alg=fold_alg,
            inv_fold_alg=inv_fold_alg,
            energy_alg=MockEnergy(),
            chk_file=str(tmp_path / "chk.pkl")
        )
        
        # Mock MDAnalysis Universe
        class MockUniverse:
            def select_atoms(self, sel):
                class MockAtoms:
                    positions = np.random.rand(11, 3)
                    class MockResidues:
                        resids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
                    residues = MockResidues()
                return MockAtoms()
        
        def mock_universe(path):
            return MockUniverse()
        
        import bindcraft as bc_module
        monkeypatch.setattr(bc_module.mda, 'Universe', mock_universe)
        
        return bindcraft
    
    def test_prepare_creates_initial_structure(self, bindcraft_instance):
        """Test prepare creates initial structure."""
        results = bindcraft_instance.prepare()
        
        assert 'initial_guess' in results
        assert 'sequence' in results['initial_guess']
        assert 'structure' in results['initial_guess']
        assert 'rmsd' in results['initial_guess']
        assert 'energy' in results['initial_guess']
        
        assert results['initial_guess']['sequence'] == bindcraft_instance.binder
        assert results['initial_guess']['rmsd'] == 0.0


class TestBindCraftCheckpointing:
    """Test checkpointing functionality."""
    
    def test_checkpoint_saves_file(self, tmp_path):
        """Test checkpoint creates file."""
        fold_alg = MockFolding(tmp_path / "folds")
        
        bindcraft = BindCraft(
            target="MKQL",
            binder="MALV",
            fold_alg=fold_alg,
            inv_fold_alg=MockInverseFolding(),
            chk_file=str(tmp_path / "test_checkpoint.pkl")
        )
        
        structures = {
            'test_1': {'sequence': 'MKQL', 'energy': -10.0}
        }
        
        bindcraft.checkpoint(structures)
        
        assert bindcraft.chk_file.exists()
    
    def test_checkpoint_content(self, tmp_path):
        """Test checkpoint saves correct content."""
        fold_alg = MockFolding(tmp_path / "folds")
        
        bindcraft = BindCraft(
            target="MKQL",
            binder="MALV",
            fold_alg=fold_alg,
            inv_fold_alg=MockInverseFolding(),
            chk_file=str(tmp_path / "test_checkpoint.pkl")
        )
        
        structures = {
            1: {'sequence': 'MKQL', 'energy': -10.0},
            2: {'sequence': 'MALV', 'energy': -12.0}
        }
        
        bindcraft.checkpoint(structures)
        
        # Load and verify
        with open(bindcraft.chk_file, 'rb') as f:
            loaded = pickle.load(f)
        
        assert loaded == structures
    
    def test_restart_from_checkpoint(self, tmp_path):
        """Test restarting from checkpoint."""
        chk_file = tmp_path / "restart_test.pkl"
        
        # Create checkpoint
        structures = {
            'initial_guess': {'sequence': 'MKQL'},
            1: {'structure': 'folds/trial_0_seq_1.pdb'},
            2: {'structure': 'folds/trial_1_seq_2.pdb'}
        }
        
        with open(chk_file, 'wb') as f:
            pickle.dump(structures, f)
        
        # Create BindCraft instance (should load checkpoint)
        fold_alg = MockFolding(tmp_path / "folds")
        
        bindcraft = BindCraft(
            target="MKQL",
            binder="MALV",
            fold_alg=fold_alg,
            inv_fold_alg=MockInverseFolding(),
            chk_file=str(chk_file)
        )
        
        # Should have restored state
        assert bindcraft.trial >= 1
        assert bindcraft.bnum > 1


class TestBindCraftFullPipeline:
    """Test complete pipeline execution."""
    
    @pytest.fixture
    def full_bindcraft(self, tmp_path, monkeypatch):
        """Create fully configured BindCraft."""
        fold_alg = MockFolding(tmp_path / "folds")
        inv_fold_alg = MockInverseFolding()
        
        bindcraft = BindCraft(
            target="MKQLEDKIEELLSKYH",
            binder="MALKVIEDRKA",
            fold_alg=fold_alg,
            inv_fold_alg=inv_fold_alg,
            energy_alg=MockEnergy(),
            qc_filter=SequenceQualityControl(max_hydrophobic_ratio=0.6),
            chk_file=str(tmp_path / "pipeline_test.pkl"),
            n_rounds=2
        )
        
        # Mock MDAnalysis
        class MockUniverse:
            def select_atoms(self, sel):
                class MockAtoms:
                    positions = np.random.rand(11, 3)
                    class MockResidues:
                        resids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
                    residues = MockResidues()
                return MockAtoms()
        
        def mock_universe(path):
            return MockUniverse()
        
        import bindcraft as bc_module
        monkeypatch.setattr(bc_module.mda, 'Universe', mock_universe)
        
        return bindcraft
    
    def test_run_inference_completes(self, full_bindcraft):
        """Test full pipeline runs to completion."""
        results = full_bindcraft.run_inference()
        
        assert len(results) > 0
        assert 'initial_guess' in results
    
    def test_run_inference_returns_results(self, full_bindcraft):
        """Test pipeline returns valid results."""
        results = full_bindcraft.run_inference()
        
        for key, value in results.items():
            assert 'sequence' in value
            assert 'structure' in value
            assert 'rmsd' in value or value['rmsd'] == 0.0
            assert 'energy' in value
    
    def test_multiple_rounds_execute(self, full_bindcraft):
        """Test multiple rounds execute correctly."""
        full_bindcraft.n_rounds = 3
        results = full_bindcraft.run_inference()
        
        # Should have initial + designs from 3 rounds
        assert len(results) >= 1


class TestBindCraftHelperMethods:
    """Test helper methods."""
    
    @pytest.fixture
    def bindcraft(self, tmp_path, monkeypatch):
        """Create BindCraft instance."""
        fold_alg = MockFolding(tmp_path / "folds")
        
        bc = BindCraft(
            target="MKQL",
            binder="MALV",
            fold_alg=fold_alg,
            inv_fold_alg=MockInverseFolding(),
            chk_file=str(tmp_path / "test.pkl")
        )
        
        # Mock MDAnalysis
        class MockUniverse:
            def select_atoms(self, sel):
                class MockAtoms:
                    positions = np.array([[1.0, 2.0, 3.0],
                                         [4.0, 5.0, 6.0],
                                         [7.0, 8.0, 9.0],
                                         [10.0, 11.0, 12.0]])
                    class MockResidues:
                        resids = np.array([1, 2, 3, 4])
                    residues = MockResidues()
                return MockAtoms()
        
        def mock_universe(path):
            return MockUniverse()
        
        import bindcraft as bc_module
        monkeypatch.setattr(bc_module.mda, 'Universe', mock_universe)
        
        return bc
    
    def test_get_binder_coords_single(self, bindcraft, tmp_path):
        """Test getting coordinates from single PDB."""
        pdb = tmp_path / "test.pdb"
        pdb.write_text("MOCK PDB")
        
        coords = bindcraft.get_binder_coords(pdb)
        
        assert isinstance(coords, np.ndarray)
        assert coords.shape == (4, 3)
    
    def test_get_binder_coords_list(self, bindcraft, tmp_path):
        """Test getting coordinates from list of PDBs."""
        pdbs = [tmp_path / f"test{i}.pdb" for i in range(3)]
        for pdb in pdbs:
            pdb.write_text("MOCK PDB")
        
        coords = bindcraft.get_binder_coords(pdbs)
        
        assert isinstance(coords, np.ndarray)
        assert coords.shape == (3, 4, 3)
    
    def test_get_interface(self, bindcraft, tmp_path):
        """Test interface identification."""
        pdb = tmp_path / "test.pdb"
        pdb.write_text("MOCK PDB")
        
        bindcraft.get_interface(pdb)
        
        assert bindcraft.remodel is not None
        assert isinstance(bindcraft.remodel, list)


@pytest.mark.slow
class TestBindCraftIntegration:
    """Integration tests (marked as slow)."""
    
    def test_end_to_end_small(self, tmp_path, monkeypatch):
        """Test end-to-end with minimal rounds."""
        fold_alg = MockFolding(tmp_path / "folds")
        inv_fold_alg = MockInverseFolding([
            "MKQLEDKIEAFRSTV",  # Should pass QC
        ])
        
        bindcraft = BindCraft(
            target="MKQL",
            binder="MALV",
            fold_alg=fold_alg,
            inv_fold_alg=inv_fold_alg,
            energy_alg=MockEnergy(),
            qc_filter=SequenceQualityControl(max_hydrophobic_ratio=0.6),
            chk_file=str(tmp_path / "e2e.pkl"),
            n_rounds=1
        )
        
        # Mock MDAnalysis
        class MockUniverse:
            def select_atoms(self, sel):
                class MockAtoms:
                    positions = np.random.rand(4, 3)
                    class MockResidues:
                        resids = np.array([1, 2, 3, 4])
                    residues = MockResidues()
                return MockAtoms()
        
        def mock_universe(path):
            return MockUniverse()
        
        import bindcraft as bc_module
        monkeypatch.setattr(bc_module.mda, 'Universe', mock_universe)
        
        results = bindcraft.run_inference()
        
        # Verify results structure
        assert len(results) >= 1
        assert bindcraft.chk_file.exists()
        
        # Verify all results have required fields
        for data in results.values():
            assert 'sequence' in data
            assert 'structure' in data
            assert 'rmsd' in data
            assert 'energy' in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
