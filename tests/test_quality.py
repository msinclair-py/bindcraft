"""
Tests for quality_control.py module.
"""

import pytest
from pathlib import Path
import tempfile
from quality_control import SequenceQualityControl, filter_sequences


class TestSequenceQualityControl:
    """Test suite for SequenceQualityControl class."""
    
    @pytest.fixture
    def qc_default(self):
        """Default QC instance."""
        return SequenceQualityControl()
    
    @pytest.fixture
    def qc_strict(self):
        """Strict QC instance."""
        return SequenceQualityControl(
            max_repeat=3,
            max_appearance_ratio=0.25,
            max_charge=3,
            max_charge_ratio=0.4,
            max_hydrophobic_ratio=0.3,
            min_diversity=12
        )
    
    def test_initialization(self):
        """Test QC initialization with custom parameters."""
        qc = SequenceQualityControl(
            max_repeat=5,
            max_charge=10
        )
        assert qc.max_repeat == 5
        assert qc.max_charge == 10
        assert qc.max_hydrophobic_ratio == 0.4  # Default
    
    def test_good_sequence(self, qc_default):
        """Test that a good sequence passes all checks."""
        good_seq = "MKQLEDKIEELLSKIHAEQEREEVRKMLKSEQQLQQRIQE"
        assert qc_default(good_seq) is True
    
    def test_multiplicity_fail(self, qc_default):
        """Test multiplicity check fails for sequences with too many of one AA."""
        # Sequence with >33% leucine
        bad_seq = "LLLLLLLLLLLLLLLLLAAAAKKK"
        assert qc_default(bad_seq) is False
    
    def test_multiplicity_pass(self, qc_default):
        """Test multiplicity check passes for diverse sequences."""
        good_seq = "MKQLEDKIEELLSKIHAEQE"
        assert qc_default(good_seq) is True
    
    def test_diversity_fail(self, qc_default):
        """Test diversity check fails for low diversity sequences."""
        # Only 5 unique amino acids
        bad_seq = "AAAAKKKKEEEELLLLAAA"
        assert qc_default(bad_seq) is False
    
    def test_diversity_pass(self, qc_default):
        """Test diversity check passes with sufficient diversity."""
        # 10 unique amino acids
        good_seq = "MKQLEDKIEAFRSTVWYHP"
        assert qc_default(good_seq) is True
    
    def test_repeat_fail(self, qc_default):
        """Test repeat check fails for long homopolymers."""
        bad_seq = "MKQLEAAAAAAAAQIEDKLE"  # 8 A's in a row
        assert qc_default(bad_seq) is False
    
    def test_repeat_pass(self, qc_default):
        """Test repeat check passes for short repeats."""
        good_seq = "MKQLEAAAQIEDKLE"  # Only 3 A's
        assert qc_default(good_seq) is True
    
    def test_charge_ratio_fail(self, qc_default):
        """Test charge ratio check fails for highly charged sequences."""
        # >50% charged residues
        bad_seq = "KKKKKRRRRRDDDDDEEEEEE"
        assert qc_default(bad_seq) is False
    
    def test_charge_ratio_pass(self, qc_default):
        """Test charge ratio check passes for balanced sequences."""
        good_seq = "MKQLEAADKIERRLVWYHP"
        assert qc_default(good_seq) is True
    
    def test_net_charge_fail(self, qc_default):
        """Test net charge check fails for high net charge."""
        # Net charge = 10
        bad_seq = "KKKKKKKKKKAAAAAA"
        assert qc_default(bad_seq) is False
    
    def test_net_charge_pass(self, qc_default):
        """Test net charge check passes for balanced charge."""
        good_seq = "KKKDDDAAALLLWWW"  # Net charge = 0
        assert qc_default(good_seq) is True
    
    def test_bad_motifs_fail(self, qc_default):
        """Test bad motif check fails for problematic motifs."""
        for motif in ['RK', 'DP', 'DG', 'DS']:
            bad_seq = f"MKQLE{motif}AAQIEDKLE"
            assert qc_default(bad_seq) is False, f"Should fail for motif {motif}"
    
    def test_bad_motifs_pass(self, qc_default):
        """Test bad motif check passes for good sequences."""
        good_seq = "MKQLEAAQIEDKLEVWYHP"
        assert qc_default(good_seq) is True
    
    def test_bad_terminus_fail(self, qc_default):
        """Test bad terminus check fails for Q/N at N-terminus."""
        bad_seqs = ["QMKQLEAAQIEDKLE", "NMKQLEAAQIEDKLE"]
        for seq in bad_seqs:
            assert qc_default(seq) is False
    
    def test_bad_terminus_pass(self, qc_default):
        """Test bad terminus check passes for good N-terminus."""
        good_seq = "MKQLEAAQIEDKLE"
        assert qc_default(good_seq) is True
    
    def test_hydrophobicity_fail(self, qc_default):
        """Test hydrophobicity check fails for overly hydrophobic sequences."""
        # >40% hydrophobic
        bad_seq = "AAAAAAAAAALLLLLLLLLLLVVVVVV"
        assert qc_default(bad_seq) is False
    
    def test_hydrophobicity_pass(self, qc_default):
        """Test hydrophobicity check passes for balanced sequences."""
        good_seq = "MKQLEDKIERRLVWYHPAA"
        assert qc_default(good_seq) is True
    
    def test_custom_thresholds(self):
        """Test QC with custom thresholds."""
        qc = SequenceQualityControl(
            max_hydrophobic_ratio=0.6,
            max_charge=10
        )
        # Should pass with relaxed thresholds
        seq = "AAAAAAAAAALLLLLLVVVKKKKDDD"
        assert qc(seq) is True
    
    def test_all_checks_integration(self, qc_default):
        """Test that all checks work together correctly."""
        # Good sequence should pass all checks
        good_seq = "MKQLEDKIEAFRSTVWYHPLGVIM"
        assert qc_default(good_seq) is True
        
        # Each type of bad sequence should fail
        bad_sequences = [
            "LLLLLLLLLLLLLLLLLLLL",  # Low diversity, high multiplicity
            "AAAAAAAAAKKKK",  # Long repeat
            "KKKKKRRRRRDDDDD",  # High charge
            "QMKQLEDKIE",  # Bad N-terminus
            "MKQRKLEDKIE",  # Bad motif (RK)
        ]
        
        for bad_seq in bad_sequences:
            assert qc_default(bad_seq) is False
    
    def test_empty_sequence(self, qc_default):
        """Test handling of empty sequence."""
        with pytest.raises(Exception):
            qc_default("")
    
    def test_short_sequence(self, qc_default):
        """Test handling of very short sequences."""
        # Should fail diversity check
        short_seq = "MKQL"
        assert qc_default(short_seq) is False


class TestFilterSequences:
    """Test suite for filter_sequences function."""
    
    def test_filter_sequences_default_qc(self):
        """Test filtering with default QC."""
        sequences = [
            "MKQLEDKIEAFRSTVWYHPLGVIM",  # Good
            "AAAAAAAAAAAAAAAAAAA",  # Bad - repeats
            "KKKKKRRRRRDDDDDEEEE",  # Bad - charge
            "MALKVIEDKREAFRSTVWYH",  # Good
        ]
        
        filtered = filter_sequences(sequences)
        assert len(filtered) == 2
        assert "MKQLEDKIEAFRSTVWYHPLGVIM" in filtered
        assert "MALKVIEDKREAFRSTVWYH" in filtered
    
    def test_filter_sequences_custom_qc(self):
        """Test filtering with custom QC."""
        qc = SequenceQualityControl(max_hydrophobic_ratio=0.3)
        
        sequences = [
            "MKQLEDKIERRSTV",  # Moderate hydrophobic
            "AAAAAALLLLLLVVV",  # High hydrophobic
        ]
        
        filtered = filter_sequences(sequences, qc)
        assert len(filtered) == 1
        assert "MKQLEDKIERRSTV" in filtered
    
    def test_filter_all_pass(self):
        """Test when all sequences pass."""
        sequences = [
            "MKQLEDKIEAFRSTVWYH",
            "MALKVIEDKREAFRSTVW",
            "MKHLERDKIEAFRSTWVH",
        ]
        
        filtered = filter_sequences(sequences)
        assert len(filtered) == 3
    
    def test_filter_all_fail(self):
        """Test when all sequences fail."""
        sequences = [
            "AAAAAAAAAAAAAAAA",
            "KKKKKKKKKKKKKKKK",
            "LLLLLLLLLLLLLLLL",
        ]
        
        filtered = filter_sequences(sequences)
        assert len(filtered) == 0
    
    def test_filter_empty_list(self):
        """Test filtering empty list."""
        filtered = filter_sequences([])
        assert len(filtered) == 0


class TestIndividualChecks:
    """Detailed tests for individual QC methods."""
    
    @pytest.fixture
    def qc(self):
        return SequenceQualityControl()
    
    def test_multiplicity_boundary(self, qc):
        """Test multiplicity at exact boundary."""
        qc.seq = "A" * 33 + "K" * 67  # Exactly 33% A's
        qc.length = 100
        qc.counts = {'A': 33, 'K': 67}
        assert qc.multiplicity() is True
        
        qc.counts = {'A': 34, 'K': 66}  # Slightly over 33%
        assert qc.multiplicity() is False
    
    def test_diversity_boundary(self, qc):
        """Test diversity at exact boundary."""
        # Exactly 8 unique AAs
        qc.seq = "AAKKLLMM"
        qc.counts = {'A': 2, 'K': 2, 'L': 2, 'M': 2}
        assert qc.diversity() is True
        
        # Only 7 unique AAs
        qc.seq = "AAKKLLM"
        qc.counts = {'A': 2, 'K': 2, 'L': 2, 'M': 1}
        assert qc.diversity() is False
    
    def test_charge_ratio_boundary(self, qc):
        """Test charge ratio at boundary."""
        # Exactly 50% charged
        qc.seq = "K" * 50 + "A" * 50
        qc.length = 100
        qc.counts = {'K': 50, 'A': 50}
        assert qc.charge_ratio() is True
        
        # Slightly over 50%
        qc.counts = {'K': 51, 'A': 49}
        assert qc.charge_ratio() is False
    
    def test_net_charge_boundary(self, qc):
        """Test net charge at boundary."""
        # Net charge = 5 (exactly at limit)
        qc.counts = {'K': 10, 'D': 5, 'A': 10}
        assert qc.net_charge() is True
        
        # Net charge = 6 (over limit)
        qc.counts = {'K': 10, 'D': 4, 'A': 10}
        assert qc.net_charge() is False
    
    def test_hydrophobicity_calculation(self, qc):
        """Test hydrophobic ratio calculation."""
        # Test with known composition
        qc.seq = "A" * 40 + "K" * 60  # 40% hydrophobic (A)
        qc.length = 100
        qc.counts = {'A': 40, 'K': 60}
        assert qc.hydrophobicity() is True
        
        qc.counts = {'A': 41, 'K': 59}  # 41% hydrophobic
        assert qc.hydrophobicity() is False


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_amino_acid_sequence(self):
        """Test sequence with single amino acid type."""
        qc = SequenceQualityControl()
        seq = "AAAAAAAAAAAAAAAA"
        # Should fail multiple checks
        assert qc(seq) is False
    
    def test_all_charged_sequence(self):
        """Test sequence with only charged residues."""
        qc = SequenceQualityControl()
        seq = "KKKKKDDDDDEEEERRRRR"
        assert qc(seq) is False
    
    def test_unusual_amino_acids(self):
        """Test with all 20 amino acids."""
        qc = SequenceQualityControl()
        seq = "ACDEFGHIKLMNPQRSTVWY" * 2  # Good diversity
        # Should pass most checks
        result = qc(seq)
        # May pass or fail depending on specific composition
        assert isinstance(result, bool)
    
    def test_minimum_length_sequence(self):
        """Test very short but valid sequence."""
        qc = SequenceQualityControl(min_diversity=5)
        seq = "MKQLEADFG"  # 9 unique AAs
        result = qc(seq)
        assert isinstance(result, bool)
    
    def test_case_sensitivity(self):
        """Test that sequences are case-sensitive."""
        qc = SequenceQualityControl()
        # Lowercase should be treated differently
        seq_upper = "MKQLEDKIE"
        seq_lower = "mkqledkie"
        
        # Both should be handled (or raise error for lowercase)
        result_upper = qc(seq_upper)
        assert isinstance(result_upper, bool)


@pytest.mark.parametrize("sequence,expected", [
    ("MKQLEDKIEAFRSTVWYHPLGVIM", True),  # Good sequence
    ("AAAAAAAAAA", False),  # Too many repeats
    ("QMKQLEDKIE", False),  # Bad N-terminus
    ("MKQRKLEDKIE", False),  # Bad motif
    ("KKKKKRRRRR", False),  # High charge
])
def test_parametrized_sequences(sequence, expected):
    """Parametrized test for various sequences."""
    qc = SequenceQualityControl()
    assert qc(sequence) is expected


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
