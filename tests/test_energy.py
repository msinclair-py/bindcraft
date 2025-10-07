"""
Tests for energy.py module.
"""

import pytest
from pathlib import Path
import tempfile
import numpy as np
from energy import SimpleEnergy, RosettaEnergy, EnergyCalculation


class MockUniverse:
    """Mock MDAnalysis Universe for testing."""
    
    def __init__(self, n_contacts=10):
        self.n_contacts = n_contacts
    
    def select_atoms(self, selection):
        """Mock select_atoms method."""
        if 'chainID A' in selection and 'around' in selection:
            return MockAtomGroup(5)  # Interface A
        elif 'chainID B' in selection and 'around' in selection:
            return MockAtomGroup(5)  # Interface B
        elif 'chainID A' in selection:
            return MockAtomGroup(50)  # All chain A
        elif 'chainID B' in selection:
            return MockAtomGroup(50)  # All chain B
        return MockAtomGroup(0)


class MockAtomGroup:
    """Mock MDAnalysis AtomGroup."""
    
    def __init__(self, n_residues):
        self.n_res = n_residues
    
    @property
    def residues(self):
        return self
    
    def __len__(self):
        return self.n_res


class TestSimpleEnergy:
    """Test suite for SimpleEnergy class."""
    
    @pytest.fixture
    def energy_calc(self):
        """Create SimpleEnergy instance."""
        return SimpleEnergy()
    
    @pytest.fixture
    def mock_pdb(self, tmp_path, monkeypatch):
        """Create a mock PDB file and patch MDAnalysis."""
        pdb_file = tmp_path / "test_structure.pdb"
        pdb_file.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000")
        
        # Patch MDAnalysis Universe
        def mock_universe(path):
            return MockUniverse(n_contacts=10)
        
        import energy
        monkeypatch.setattr(energy.mda, 'Universe', mock_universe)
        
        return pdb_file
    
    def test_initialization(self, energy_calc):
        """Test SimpleEnergy initialization."""
        assert isinstance(energy_calc, SimpleEnergy)
        assert isinstance(energy_calc, EnergyCalculation)
    
    def test_calculate_energy(self, energy_calc, mock_pdb):
        """Test energy calculation returns negative value."""
        energy = energy_calc(mock_pdb)
        assert isinstance(energy, (int, float))
        assert energy < 0  # Favorable energy should be negative
    
    def test_energy_proportional_to_contacts(self, tmp_path, monkeypatch):
        """Test that more contacts give more favorable energy."""
        energy_calc = SimpleEnergy()
        
        # Mock for few contacts
        def mock_universe_few(path):
            return MockUniverse(n_contacts=5)
        
        # Mock for many contacts
        def mock_universe_many(path):
            return MockUniverse(n_contacts=15)
        
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text("ATOM")
        
        import energy
        
        # Test with few contacts
        monkeypatch.setattr(energy.mda, 'Universe', mock_universe_few)
        energy_few = energy_calc(pdb_file)
        
        # Test with many contacts
        monkeypatch.setattr(energy.mda, 'Universe', mock_universe_many)
        energy_many = energy_calc(pdb_file)
        
        # More contacts should give more negative (favorable) energy
        assert energy_many < energy_few
    
    def test_nonexistent_file(self, energy_calc):
        """Test handling of nonexistent PDB file."""
        fake_path = Path("/nonexistent/file.pdb")
        with pytest.raises(Exception):
            energy_calc(fake_path)
    
    def test_energy_output_type(self, energy_calc, mock_pdb):
        """Test that energy output is numeric."""
        energy = energy_calc(mock_pdb)
        assert isinstance(energy, (int, float, np.number))
        assert not np.isnan(energy)


class TestRosettaEnergy:
    """Test suite for RosettaEnergy class."""
    
    @pytest.fixture
    def rosetta_path(self, tmp_path):
        """Create mock Rosetta installation."""
        rosetta_dir = tmp_path / "rosetta"
        rosetta_dir.mkdir()
        
        # Create mock score executable
        score_app = rosetta_dir / "score_jd2"
        score_app.write_text("#!/bin/bash\necho 'SCORE:     1.234'")
        score_app.chmod(0o755)
        
        return rosetta_dir
    
    def test_initialization(self, rosetta_path):
        """Test RosettaEnergy initialization."""
        energy_calc = RosettaEnergy(
            rosetta_path=rosetta_path,
            score_function='ref2015'
        )
        assert energy_calc.rosetta_path == rosetta_path
        assert energy_calc.score_function == 'ref2015'
    
    def test_calculate_energy_mock(self, tmp_path, rosetta_path, monkeypatch):
        """Test energy calculation with mocked Rosetta."""
        # Create mock PDB
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text("ATOM      1  CA  ALA A   1")
        
        # Create mock score file
        score_file = tmp_path / "score.sc"
        score_file.write_text(
            "SCORE: total_score description\n"
            "SCORE:     -123.45 test_structure\n"
        )
        
        # Mock subprocess.run
        def mock_run(*args, **kwargs):
            class Result:
                returncode = 0
                stdout = ""
                stderr = ""
            return Result()
        
        import energy
        monkeypatch.setattr(energy.subprocess, 'run', mock_run)
        
        energy_calc = RosettaEnergy(rosetta_path)
        energy = energy_calc(pdb_file)
        
        assert isinstance(energy, float)
        assert energy == -123.45
    
    def test_missing_rosetta(self, tmp_path):
        """Test error when Rosetta not found."""
        fake_path = tmp_path / "nonexistent_rosetta"
        energy_calc = RosettaEnergy(fake_path)
        
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text("ATOM")
        
        # Should handle missing Rosetta gracefully
        energy = energy_calc(pdb_file)
        assert np.isnan(energy) or energy is None or isinstance(energy, float)


class TestEnergyCalculation:
    """Test the abstract base class interface."""
    
    def test_abc_cannot_instantiate(self):
        """Test that EnergyCalculation cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EnergyCalculation()
    
    def test_custom_implementation(self, tmp_path):
        """Test creating custom energy implementation."""
        
        class CustomEnergy(EnergyCalculation):
            def __call__(self, structure):
                return -42.0
        
        custom = CustomEnergy()
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM")
        
        energy = custom(pdb)
        assert energy == -42.0
    
    def test_interface_compliance(self):
        """Test that implementations comply with interface."""
        assert hasattr(SimpleEnergy, '__call__')
        assert hasattr(RosettaEnergy, '__call__')
        
        # Test signature
        import inspect
        sig_simple = inspect.signature(SimpleEnergy().__call__)
        sig_rosetta = inspect.signature(RosettaEnergy(Path('.')).__call__)
        
        assert len(sig_simple.parameters) == 1
        assert len(sig_rosetta.parameters) == 1


class TestEnergyComparison:
    """Test energy calculations produce reasonable relative values."""
    
    @pytest.fixture
    def mock_structures(self, tmp_path, monkeypatch):
        """Create mock structures with different numbers of contacts."""
        
        def create_mock_universe(n_contacts):
            def mock_universe(path):
                return MockUniverse(n_contacts=n_contacts)
            return mock_universe
        
        structures = {}
        for name, contacts in [('good', 20), ('medium', 10), ('poor', 5)]:
            pdb = tmp_path / f"{name}.pdb"
            pdb.write_text("ATOM")
            structures[name] = (pdb, contacts, create_mock_universe(contacts))
        
        return structures
    
    def test_relative_energies(self, mock_structures, monkeypatch):
        """Test that better binding gives more negative energy."""
        import energy as energy_module
        energy_calc = SimpleEnergy()
        
        energies = {}
        for name, (pdb, contacts, mock_func) in mock_structures.items():
            monkeypatch.setattr(energy_module.mda, 'Universe', mock_func)
            energies[name] = energy_calc(pdb)
        
        # More contacts should give more negative energy
        assert energies['good'] < energies['medium'] < energies['poor']
        
        # All should be negative (favorable)
        assert all(e < 0 for e in energies.values())


class TestEnergyEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_structure(self, tmp_path, monkeypatch):
        """Test handling of structure with no contacts."""
        
        class EmptyUniverse:
            def select_atoms(self, sel):
                return MockAtomGroup(0)
        
        def mock_universe(path):
            return EmptyUniverse()
        
        import energy
        monkeypatch.setattr(energy.mda, 'Universe', mock_universe)
        
        pdb = tmp_path / "empty.pdb"
        pdb.write_text("ATOM")
        
        calc = SimpleEnergy()
        result = calc(pdb)
        assert result == 0  # No contacts = zero energy
    
    def test_path_as_string(self, tmp_path, monkeypatch):
        """Test that Path and string inputs both work."""
        
        def mock_universe(path):
            return MockUniverse(n_contacts=10)
        
        import energy
        monkeypatch.setattr(energy.mda, 'Universe', mock_universe)
        
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM")
        
        calc = SimpleEnergy()
        
        # Test with Path object
        energy_path = calc(pdb)
        
        # Test with string
        energy_str = calc(str(pdb))
        
        assert energy_path == energy_str
    
    def test_large_complex(self, tmp_path, monkeypatch):
        """Test with large protein complex."""
        
        class LargeUniverse:
            def select_atoms(self, sel):
                if 'around' in sel:
                    return MockAtomGroup(100)  # Large interface
                return MockAtomGroup(500)
        
        def mock_universe(path):
            return LargeUniverse()
        
        import energy
        monkeypatch.setattr(energy.mda, 'Universe', mock_universe)
        
        pdb = tmp_path / "large.pdb"
        pdb.write_text("ATOM")
        
        calc = SimpleEnergy()
        energy = calc(pdb)
        
        # Should handle large complexes
        assert isinstance(energy, (int, float))
        assert energy < -100  # Large interface = very negative


@pytest.mark.parametrize("n_contacts,expected_sign", [
    (0, 0),      # No contacts
    (5, -1),     # Few contacts = negative
    (20, -1),    # Many contacts = negative
    (100, -1),   # Very many = still negative
])
def test_parametrized_contacts(n_contacts, expected_sign, tmp_path, monkeypatch):
    """Parametrized test for different contact numbers."""
    
    def mock_universe(path):
        return MockUniverse(n_contacts=n_contacts)
    
    import energy
    monkeypatch.setattr(energy.mda, 'Universe', mock_universe)
    
    pdb = tmp_path / "test.pdb"
    pdb.write_text("ATOM")
    
    calc = SimpleEnergy()
    result = calc(pdb)
    
    if expected_sign == 0:
        assert result == 0
    elif expected_sign == -1:
        assert result < 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
