"""
Analysis utilities for BindCraft results.
"""

import dill as pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import MDAnalysis as mda
from collections import Counter

from .energy import SimpleEnergy


def load_checkpoint(checkpoint_file: Path) -> Dict[str, Any]:
    """Load results from checkpoint file."""
    with open(checkpoint_file, 'rb') as f:
        return pickle.load(f)


def analyze_sequence_diversity(structures: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze sequence diversity in designed binders.
    
    Args:
        structures: Dictionary of structure data
        
    Returns:
        Dictionary with diversity metrics
    """
    sequences = [data['sequence'] for data in structures.values()]
    
    # Calculate pairwise identity
    n_seqs = len(sequences)
    identities = []
    
    for i in range(n_seqs):
        for j in range(i+1, n_seqs):
            seq1, seq2 = sequences[i], sequences[j]
            if len(seq1) == len(seq2):
                identity = sum(a == b for a, b in zip(seq1, seq2)) / len(seq1)
                identities.append(identity)
    
    # Amino acid composition
    all_aas = ''.join(sequences)
    aa_counts = Counter(all_aas)
    
    return {
        'n_sequences': n_seqs,
        'mean_identity': np.mean(identities) if identities else 0,
        'std_identity': np.std(identities) if identities else 0,
        'aa_composition': dict(aa_counts),
        'unique_sequences': len(set(sequences))
    }


def analyze_energies(structures: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze energy distribution of designs.
    
    Args:
        structures: Dictionary of structure data
        
    Returns:
        Dictionary with energy statistics
    """
    energies = [data['energy'] for data in structures.values() 
               if not np.isnan(data['energy'])]
    
    if not energies:
        return {'error': 'No valid energies found'}
    
    return {
        'min_energy': np.min(energies),
        'max_energy': np.max(energies),
        'mean_energy': np.mean(energies),
        'median_energy': np.median(energies),
        'std_energy': np.std(energies)
    }


def analyze_rmsd(structures: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze RMSD distribution of designs.
    
    Args:
        structures: Dictionary of structure data
        
    Returns:
        Dictionary with RMSD statistics
    """
    rmsds = [data['rmsd'] for data in structures.values() 
            if not np.isnan(data['rmsd'])]
    
    if not rmsds:
        return {'error': 'No valid RMSDs found'}
    
    return {
        'min_rmsd': np.min(rmsds),
        'max_rmsd': np.max(rmsds),
        'mean_rmsd': np.mean(rmsds),
        'median_rmsd': np.median(rmsds),
        'std_rmsd': np.std(rmsds)
    }


def plot_energy_vs_rmsd(structures: Dict[str, Any], 
                        output_file: Path = None):
    """
    Create scatter plot of energy vs RMSD.
    
    Args:
        structures: Dictionary of structure data
        output_file: Path to save plot (optional)
    """
    energies = []
    rmsds = []
    
    for data in structures.values():
        if not np.isnan(data['energy']) and not np.isnan(data['rmsd']):
            energies.append(data['energy'])
            rmsds.append(data['rmsd'])
    
    plt.figure(figsize=(8, 6))
    plt.scatter(rmsds, energies, alpha=0.6)
    plt.xlabel('RMSD (Å)', fontsize=12)
    plt.ylabel('Energy', fontsize=12)
    plt.title('Energy vs RMSD', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_sequence_logo(sequences: list[str], 
                       output_file: Path = None):
    """
    Create sequence logo from multiple alignments.
    
    Args:
        sequences: List of sequences (must be same length)
        output_file: Path to save plot (optional)
    """
    if not sequences or len(set(len(s) for s in sequences)) > 1:
        print("Error: All sequences must have the same length")
        return
    
    seq_len = len(sequences[0])
    n_seqs = len(sequences)
    
    # Calculate position-specific amino acid frequencies
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    freq_matrix = np.zeros((len(aa_list), seq_len))
    
    for pos in range(seq_len):
        counts = Counter(seq[pos] for seq in sequences)
        for i, aa in enumerate(aa_list):
            freq_matrix[i, pos] = counts.get(aa, 0) / n_seqs
    
    # Plot
    fig, ax = plt.subplots(figsize=(max(12, seq_len * 0.3), 4))
    
    for pos in range(seq_len):
        sorted_idx = np.argsort(freq_matrix[:, pos])
        bottom = 0
        for idx in sorted_idx:
            if freq_matrix[idx, pos] > 0:
                ax.bar(pos, freq_matrix[idx, pos], bottom=bottom, 
                      width=0.8, label=aa_list[idx] if pos == 0 else "")
                bottom += freq_matrix[idx, pos]
    
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Sequence Conservation', fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, seq_len, max(1, seq_len // 20)))
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    
    plt.close()


def generate_report(checkpoint_file: Path, 
                   output_dir: Path = None):
    """
    Generate comprehensive analysis report.
    
    Args:
        checkpoint_file: Path to checkpoint file
        output_dir: Directory to save outputs
    """
    if output_dir is None:
        output_dir = checkpoint_file.parent / 'analysis'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print("Loading checkpoint...")
    structures = load_checkpoint(checkpoint_file)
    
    # Analyze
    print("Analyzing diversity...")
    diversity = analyze_sequence_diversity(structures)
    
    print("Analyzing energies...")
    energy_stats = analyze_energies(structures)
    
    print("Analyzing RMSD...")
    rmsd_stats = analyze_rmsd(structures)
    
    # Generate plots
    print("Generating plots...")
    plot_energy_vs_rmsd(structures, output_dir / 'energy_vs_rmsd.png')
    
    sequences = [data['sequence'] for data in structures.values()]
    if sequences and len(set(len(s) for s in sequences)) == 1:
        plot_sequence_logo(sequences, output_dir / 'sequence_logo.png')
    
    # Write report
    report_file = output_dir / 'analysis_report.txt'
    with open(report_file, 'w') as f:
        f.write("BindCraft Analysis Report\n")
        f.write("="*60 + "\n\n")
        
        f.write("Sequence Diversity:\n")
        f.write("-"*40 + "\n")
        for key, value in diversity.items():
            if key != 'aa_composition':
                f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("Energy Statistics:\n")
        f.write("-"*40 + "\n")
        for key, value in energy_stats.items():
            f.write(f"  {key}: {value:.3f}\n")
        f.write("\n")
        
        f.write("RMSD Statistics:\n")
        f.write("-"*40 + "\n")
        for key, value in rmsd_stats.items():
            f.write(f"  {key}: {value:.3f}\n")
        f.write("\n")
    
    print(f"\nReport saved to {report_file}")
    print("Analysis complete!")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analysis.py <checkpoint_file>")
        sys.exit(1)
    
    checkpoint = Path(sys.argv[1])
    if not checkpoint.exists():
        print(f"Error: Checkpoint file not found: {checkpoint}")
        sys.exit(1)
    
    generate_report(checkpoint)
