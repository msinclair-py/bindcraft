"""
Tests for folding.py module.
"""

import pytest
from pathlib import Path
import shutil
from folding import Folding, Chai, Boltz


class TestFoldingABC:
    """Test the abstract Folding class."""
    
    def test_cannot_instantiate_abstract(self):
        """Test that Folding ABC cannot be instantiated."""
        with pytest.raises(TypeError):
            Folding()
    
    def test_custom_implementation(self, tmp_path):
        """Test creating a custom folding implementation."""
        
        class MockFolder(Folding):
            def __init__(self):
                pass
            
            def prepare(self, seqs, label):
                return Path("mock.fa")
            
            def __call__(self, seqs, label):
                return Path("mock.pdb")
            
            def postprocess(self, in_path):
                return Path("final.pdb")
        
        folder = MockFolder()
        assert isinstance(folder, Folding)


class TestChai:
    """Test suite for Chai folding class."""
    
    @pytest.fixture
    def chai_setup(self, tmp_path):
        """Setup Chai instance with temporary directories."""
        fasta_dir = tmp_path / "fastas"
        out_dir = tmp_path / "folds"
        
        chai = Chai(
            fasta_dir=fasta_dir,
            out=out_dir
        )
        
        return chai, fasta_dir, out_dir
    
    def test_initialization(self, chai_setup):
        """Test Chai initialization."""
        chai, fasta_dir, out_dir = chai_setup
        
        assert chai.fasta_dir == fasta_dir
        assert chai.out == out_dir
        assert chai.devshm == Path('/dev/shm')
        assert fasta_dir.exists()
        assert out_dir.exists()
    
    def test_prepare_creates_fasta(self, chai_setup):
        """Test that prepare creates fasta file."""
        chai, fasta_dir, out_dir = chai_setup
        
        target_seq = "MKQLEDKIEELLSKYH"
        binder_seq = "MALKVIEDRKA"
        label = "test_run_1"
        
        fasta_path = chai.prepare([target_seq, binder_seq], label)
        
        assert fasta_path.exists()
        assert fasta_path.parent == fasta_dir
        assert fasta_path.suffix == ".fa"
        assert label in fasta_path.name
        
        # Check content
        content = fasta_path.read_text()
        assert ">protein|target" in content
        assert ">protein|binder" in content
        assert target_seq in content
        assert binder_seq in content
    
    def test_prepare_fasta_format(self, chai_setup):
        """Test fasta file has correct format."""
        chai, fasta_dir, out_dir = chai_setup
        
        target = "MKQL"
        binder = "MALV"
        
        fasta_path = chai.prepare([target, binder], "test")
        content = fasta_path.read_text()
        
        lines = content.strip().split('\n')
        assert len(lines) == 4
        assert lines[0] == ">protein|target"
        assert lines[1] == target
        assert lines[2] == ">protein|binder"
        assert lines[3] == binder
    
    def test_prepare_overwrites_existing(self, chai_setup):
        """Test that prepare overwrites existing fasta."""
        chai, fasta_dir, out_dir = chai_setup
        
        label = "same_label"
        
        # Create first fasta
        fasta1 = chai.prepare(["SEQ1", "SEQ2"], label)
        content1 = fasta1.read_text()
        
        # Create second with same label
        fasta2 = chai.prepare(["SEQ3", "SEQ4"], label)
        content2 = fasta2.read_text()
        
        assert fasta1 == fasta2
        assert "SEQ3" in content2
        assert "SEQ1" not in content2
    
    def test_postprocess_moves_file(self, chai_setup, tmp_path):
        """Test postprocess moves and cleans up."""
        chai, fasta_dir, out_dir = chai_setup
        
        # Create mock input directory with predicted structure
        in_dir = tmp_path / "mock_prediction"
        in_dir.mkdir()
        mock_structure = in_dir / "pred.model_idx_0.cif"
        mock_structure.write_text("MOCK CIF CONTENT")
        
        # Run postprocess
        result_path = chai.postprocess(in_dir)
        
        # Check file was moved to output directory
        assert result_path.parent == out_dir
        assert result_path.name == "pred.model_idx_0.cif"
        assert result_path.exists()
        assert result_path.read_text() == "MOCK CIF CONTENT"
        
        # Check input directory was deleted
        assert not in_dir.exists()
    
    def test_postprocess_cleanup(self, chai_setup, tmp_path):
        """Test that postprocess cleans up temporary files."""
        chai, fasta_dir, out_dir = chai_setup
        
        # Create mock directory with multiple files
        in_dir = tmp_path / "cleanup_test"
        in_dir.mkdir()
        
        (in_dir / "pred.model_idx_0.cif").write_text("STRUCTURE")
        (in_dir / "pred.model_idx_1.cif").write_text("STRUCTURE2")
        (in_dir / "other_file.txt").write_text("OTHER")
        
        chai.postprocess(in_dir)
        
        # All should be cleaned up
        assert not in_dir.exists()
        assert not (in_dir / "other_file.txt").exists()
    
    def test_call_integration_mock(self, chai_setup, monkeypatch, tmp_path):
        """Test __call__ method with mocked run_inference."""
        chai, fasta_dir, out_dir = chai_setup
        
        # Mock run_inference
        def mock_run_inference(**kwargs):
            # Create mock output
            output_dir = Path(kwargs['output_dir'])
            output_dir.mkdir(exist_ok=True, parents=True)
            mock_cif = output_dir / "pred.model_idx_0.cif"
            mock_cif.write_text("MOCK STRUCTURE")
        
        import folding
        monkeypatch.setattr(folding, 'run_inference', mock_run_inference)
        
        # Run folding
        result = chai(["MKQLED", "MALVIE"], "test_fold")
        
        assert result.exists()
        assert result.parent == out_dir
        assert "pred.model_idx_0.cif" in result.name
    
    def test_multiple_folding_runs(self, chai_setup, monkeypatch):
        """Test multiple folding runs don't interfere."""
        chai, fasta_dir, out_dir = chai_setup
        
        def mock_run_inference(**kwargs):
            output_dir = Path(kwargs['output_dir'])
            output_dir.mkdir(exist_ok=True, parents=True)
            mock_cif = output_dir / "pred.model_idx_0.cif"
            mock_cif.write_text(f"STRUCTURE for {kwargs['fasta_file']}")
        
        import folding
        monkeypatch.setattr(folding, 'run_inference', mock_run_inference)
        
        # Run multiple times
        result1 = chai(["SEQ1", "SEQ2"], "run1")
        result2 = chai(["SEQ3", "SEQ4"], "run2")
        result3 = chai(["SEQ5", "SEQ6"], "run3")
        
        assert result1 != result2 != result3
        assert all(r.exists() for r in [result1, result2, result3])
    
    def test_fasta_label_uniqueness(self, chai_setup):
        """Test that different labels create different fasta files."""
        chai, fasta_dir, out_dir = chai_setup
        
        fasta1 = chai.prepare(["SEQ1", "SEQ2"], "label1")
        fasta2 = chai.prepare(["SEQ1", "SEQ2"], "label2")
        
        assert fasta1 != fasta2
        assert fasta1.exists()
        assert fasta2.exists()
    
    def test_empty_sequences(self, chai_setup):
        """Test handling of empty sequences."""
        chai, fasta_dir, out_dir = chai_setup
        
        # Should still create fasta, even if empty
        fasta = chai.prepare(["", ""], "empty_test")
        assert fasta.exists()
        
        content = fasta.read_text()
        assert ">protein|target" in content
        assert ">protein|binder" in content


class TestChaiEdgeCases:
    """Test edge cases for Chai class."""
    
    def test_very_long_sequences(self, tmp_path):
        """Test with very long protein sequences."""
        chai = Chai(tmp_path / "fastas", tmp_path / "folds")
        
        long_seq = "A" * 10000
        fasta = chai.prepare([long_seq, long_seq], "long_test")
        
        assert fasta.exists()
        content = fasta.read_text()
        assert long_seq in content
    
    def test_special_characters_in_label(self, tmp_path):
        """Test labels with special characters."""
        chai = Chai(tmp_path / "fastas", tmp_path / "folds")
        
        # Some special chars should work
        label = "test_run-1.2"
        fasta = chai.prepare(["SEQ", "SEQ"], label)
        assert fasta.exists()
        assert label in fasta.name
    
    def test_path_types(self, tmp_path):
        """Test that both Path and string work for directories."""
        # String paths
        chai1 = Chai(str(tmp_path / "fastas1"), str(tmp_path / "folds1"))
        assert isinstance(chai1.fasta_dir, Path)
        assert isinstance(chai1.out, Path)
        
        # Path objects
        chai2 = Chai(tmp_path / "fastas2", tmp_path / "folds2")
        assert isinstance(chai2.fasta_dir, Path)
        assert isinstance(chai2.out, Path)
    
    def test_nonexistent_structure_in_postprocess(self, tmp_path):
        """Test postprocess with missing structure file."""
        chai = Chai(tmp_path / "fastas", tmp_path / "folds")
        
        in_dir = tmp_path / "empty_dir"
        in_dir.mkdir()
        
        with pytest.raises(Exception):
            chai.postprocess(in_dir)


class TestBoltz:
    """Test suite for Boltz folding class (placeholder)."""
    
    def test_initialization(self):
        """Test Boltz initialization."""
        boltz = Boltz()
        assert isinstance(boltz, Folding)
    
    def test_methods_exist(self):
        """Test that Boltz has required methods."""
        boltz = Boltz()
        assert hasattr(boltz, 'prepare')
        assert hasattr(boltz, '__call__')
        assert hasattr(boltz, 'postprocess')


class TestFoldingInterface:
    """Test that all folding classes implement the interface correctly."""
    
    def test_chai_implements_interface(self, tmp_path):
        """Test Chai implements all required methods."""
        chai = Chai(tmp_path / "fastas", tmp_path / "folds")
        
        assert hasattr(chai, 'prepare')
        assert hasattr(chai, '__call__')
        assert hasattr(chai, 'postprocess')
        
        assert callable(chai.prepare)
        assert callable(chai)
        assert callable(chai.postprocess)
    
    def test_method_signatures(self, tmp_path):
        """Test method signatures match interface."""
        import inspect
        
        chai = Chai(tmp_path / "fastas", tmp_path / "folds")
        
        # Check prepare signature
        sig_prepare = inspect.signature(chai.prepare)
        assert 'seqs' in sig_prepare.parameters
        assert 'label' in sig_prepare.parameters
        
        # Check __call__ signature
        sig_call = inspect.signature(chai.__call__)
        assert 'seqs' in sig_call.parameters
        assert 'label' in sig_call.parameters
        
        # Check postprocess signature
        sig_post = inspect.signature(chai.postprocess)
        assert 'in_path' in sig_post.parameters


class TestFoldingIntegration:
    """Integration tests combining multiple methods."""
    
    def test_full_workflow_mock(self, tmp_path, monkeypatch):
        """Test complete workflow from prepare to postprocess."""
        chai = Chai(tmp_path / "fastas", tmp_path / "folds")
        
        # Mock run_inference
        def mock_run_inference(**kwargs):
            output_dir = Path(kwargs['output_dir'])
            output_dir.mkdir(exist_ok=True, parents=True)
            (output_dir / "pred.model_idx_0.cif").write_text("STRUCTURE")
        
        import folding
        monkeypatch.setattr(folding, 'run_inference', mock_run_inference)
        
        # Full workflow
        target = "MKQLEDKIEELLSKYH"
        binder = "MALKVIEDRKA"
        label = "integration_test"
        
        # Prepare
        fasta = chai.prepare([target, binder], label)
        assert fasta.exists()
        
        # Fold (mocked)
        structure = chai([target, binder], label)
        assert structure.exists()
        assert structure.parent == chai.out
        
        # Verify fasta still exists
        assert fasta.exists()
    
    def test_sequential_runs(self, tmp_path, monkeypatch):
        """Test multiple sequential folding runs."""
        chai = Chai(tmp_path / "fastas", tmp_path / "folds")
        
        def mock_run_inference(**kwargs):
            output_dir = Path(kwargs['output_dir'])
            output_dir.mkdir(exist_ok=True, parents=True)
            (output_dir / "pred.model_idx_0.cif").write_text("STRUCTURE")
        
        import folding
        monkeypatch.setattr(folding, 'run_inference', mock_run_inference)
        
        results = []
        for i in range(5):
            result = chai([f"TARGET{i}", f"BINDER{i}"], f"run_{i}")
            results.append(result)
        
        # All should exist and be unique
        assert len(results) == 5
        assert len(set(results)) == 5
        assert all(r.exists() for r in results)


@pytest.mark.parametrize("target,binder,label", [
    ("MKQLE", "MALVI", "test1"),
    ("A" * 100, "K" * 50, "long_seq"),
    ("ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY", "all_aa"),
])
def test_parametrized_folding_prepare(tmp_path, target, binder, label):
    """Parametrized test for different sequence types."""
    chai = Chai(tmp_path / "fastas", tmp_path / "folds")
    
    fasta = chai.prepare([target, binder], label)
    
    assert fasta.exists()
    content = fasta.read_text()
    assert target in content
    assert binder in content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
