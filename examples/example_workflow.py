#!/usr/bin/env python3
"""
Example workflow demonstrating BindCraft pipeline usage.

This script shows how to:
1. Set up the pipeline components
2. Run binder design
3. Analyze results
4. Extract top candidates
"""

from pathlib import Path
from bindcraft import BindCraft
from folding import Chai
from inverse_folding import ProteinMPNN
from energy import SimpleEnergy
from quality_control import SequenceQualityControl
import analysis


def design_pd1_binder():
    """
    Example: Design a binder against PD-1 immune checkpoint receptor.
    """
    print("="*70)
    print("Example: Designing binder against PD-1")
    print("="*70)
    
    # Target and initial binder sequences
    # PD-1 extracellular domain (residues 25-170)
    target_seq = (
        "MQIPQAPWPVVWAVLQLGWRPGWFLDSPDRPWNPPTFSPALLVVTEGDNATFTCSFSN"
        "TSESFVLNWYRMSPSNQTDKLAAFPEDRSQPGQDCRFRVTQLPNGRDFHMSVVRARR"
        "NDSGTYLCGAISLAPKAQIKESLRAELRVTERRAEVPTAHPSPSPRPAGQFQTLV"
    )
    
    # Initial binder - small helical scaffold
    binder_seq = (
        "MKQLEDKIEELLSKIHAEQEREEVRKMLKSEQQLQQRIQE"
        "LYEQLSELTQEKNEEKLSQLQKEIQELEAEQQQKQ"
    )
    
    # Setup directories
    work_dir = Path('./pd1_binder_design')
    work_dir.mkdir(exist_ok=True)
    
    # Initialize folding (using Chai-1)
    print("\n1. Initializing folding algorithm...")
    fold_alg = Chai(
        fasta_dir=work_dir / 'fastas',
        out=work_dir / 'folds'
    )
    
    # Initialize inverse folding (ProteinMPNN)
    print("2. Initializing inverse folding algorithm...")
    mpnn_path = Path('/path/to/ProteinMPNN')  # Update this path!
    
    if not mpnn_path.exists():
        print(f"\nERROR: ProteinMPNN not found at {mpnn_path}")
        print("Please update the mpnn_path variable with correct installation path")
        return
    
    inv_fold_alg = ProteinMPNN(
        proteinmpnn_path=mpnn_path,
        num_seq=50,  # Generate 50 sequences per round
        sampling_temp='0.1',
        batch_size=250
    )
    
    # Initialize energy calculation
    print("3. Initializing energy calculation...")
    energy_alg = SimpleEnergy()
    
    # Setup quality control with relaxed parameters for this target
    print("4. Setting up quality control filters...")
    qc_filter = SequenceQualityControl(
        max_repeat=4,
        max_appearance_ratio=0.35,
        max_charge=6,
        max_charge_ratio=0.5,
        max_hydrophobic_ratio=0.45,
        min_diversity=10,
        bad_motifs=['RK', 'DP', 'DG', 'DS'],
        bad_n_termini=['Q', 'N']
    )
    
    # Create BindCraft pipeline
    print("5. Creating BindCraft pipeline...")
    bindcraft = BindCraft(
        target=target_seq,
        binder=binder_seq,
        fold_alg=fold_alg,
        inv_fold_alg=inv_fold_alg,
        energy_alg=energy_alg,
        qc_filter=qc_filter,
        chk_file=str(work_dir / 'checkpoint.pkl'),
        n_rounds=3,
        cutoff=5.0  # Interface distance cutoff in Angstroms
    )
    
    print("\n" + "="*70)
    print("Starting binder design...")
    print("="*70)
    print(f"Target length: {len(target_seq)} residues")
    print(f"Initial binder length: {len(binder_seq)} residues")
    print(f"Number of rounds: 3")
    print(f"Sequences per round: 50")
    print(f"Working directory: {work_dir}")
    print("="*70 + "\n")
    
    # Run the pipeline
    try:
        results = bindcraft.run_inference()
        
        print("\n" + "="*70)
        print("Design completed successfully!")
        print("="*70)
        print(f"Total binders generated: {len(results)}")
        
        # Sort by energy
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].get('energy', float('inf'))
        )
        
        # Display top 5 designs
        print("\nTop 5 binder designs:")
        print("-"*70)
        
        for i, (binder_id, data) in enumerate(sorted_results[:5], 1):
            print(f"\n{i}. Binder ID: {binder_id}")
            print(f"   Sequence: {data['sequence']}")
            print(f"   Length: {len(data['sequence'])} residues")
            print(f"   RMSD: {data['rmsd']:.2f} Å")
            print(f"   Energy: {data['energy']:.2f}")
            print(f"   Structure: {data['structure']}")
        
        # Save detailed results
        results_file = work_dir / 'detailed_results.txt'
        with open(results_file, 'w') as f:
            f.write("PD-1 Binder Design - Detailed Results\n")
            f.write("="*70 + "\n\n")
            
            for binder_id, data in sorted_results:
                f.write(f"Binder ID: {binder_id}\n")
                f.write(f"Sequence: {data['sequence']}\n")
                f.write(f"Length: {len(data['sequence'])} residues\n")
                f.write(f"RMSD: {data['rmsd']:.2f} Å\n")
                f.write(f"Energy: {data['energy']:.2f}\n")
                f.write(f"Structure: {data['structure']}\n")
                f.write("-"*70 + "\n\n")
        
        print(f"\nDetailed results saved to: {results_file}")
        
        # Run analysis
        print("\nGenerating analysis report...")
        analysis.generate_report(
            work_dir / 'checkpoint.pkl',
            work_dir / 'analysis'
        )
        
        print("\n" + "="*70)
        print("Workflow completed successfully!")
        print("="*70)
        print(f"\nOutput files in: {work_dir}")
        print("- checkpoint.pkl: Full results")
        print("- detailed_results.txt: All binder sequences and metrics")
        print("- analysis/: Plots and statistical analysis")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()


def design_custom_binder(target_seq: str, 
                        binder_seq: str,
                        output_dir: Path,
                        mpnn_path: Path,
                        n_rounds: int = 3,
                        n_seqs: int = 100):
    """
    Generic function to design binders for any target.
    
    Args:
        target_seq: Target protein sequence
        binder_seq: Initial binder sequence
        output_dir: Output directory
        mpnn_path: Path to ProteinMPNN installation
        n_rounds: Number of design rounds
        n_seqs: Number of sequences per round
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Setup components
    fold_alg = Chai(
        fasta_dir=output_dir / 'fastas',
        out=output_dir / 'folds'
    )
    
    inv_fold_alg = ProteinMPNN(
        proteinmpnn_path=mpnn_path,
        num_seq=n_seqs,
        sampling_temp='0.1',
        batch_size=250
    )
    
    # Create pipeline
    bindcraft = BindCraft(
        target=target_seq,
        binder=binder_seq,
        fold_alg=fold_alg,
        inv_fold_alg=inv_fold_alg,
        energy_alg=SimpleEnergy(),
        qc_filter=SequenceQualityControl(),
        chk_file=str(output_dir / 'checkpoint.pkl'),
        n_rounds=n_rounds,
        cutoff=5.0
    )
    
    # Run design
    results = bindcraft.run_inference()
    
    # Generate report
    analysis.generate_report(
        output_dir / 'checkpoint.pkl',
        output_dir / 'analysis'
    )
    
    return results


def quick_analysis_example():
    """
    Example of analyzing existing results.
    """
    print("\nExample: Quick analysis of existing results")
    print("="*70)
    
    checkpoint_file = Path('./pd1_binder_design/checkpoint.pkl')
    
    if not checkpoint_file.exists():
        print(f"Checkpoint file not found: {checkpoint_file}")
        print("Run design_pd1_binder() first to generate results.")
        return
    
    # Load results
    structures = analysis.load_checkpoint(checkpoint_file)
    
    # Analyze
    diversity = analysis.analyze_sequence_diversity(structures)
    energy_stats = analysis.analyze_energies(structures)
    rmsd_stats = analysis.analyze_rmsd(structures)
    
    print(f"\nSequence Diversity:")
    print(f"  Unique sequences: {diversity['unique_sequences']}/{diversity['n_sequences']}")
    print(f"  Mean identity: {diversity['mean_identity']:.1%}")
    
    print(f"\nEnergy Statistics:")
    print(f"  Best energy: {energy_stats['min_energy']:.2f}")
    print(f"  Mean energy: {energy_stats['mean_energy']:.2f}")
    
    print(f"\nRMSD Statistics:")
    print(f"  Min RMSD: {rmsd_stats['min_rmsd']:.2f} Å")
    print(f"  Mean RMSD: {rmsd_stats['mean_rmsd']:.2f} Å")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        # Run analysis only
        quick_analysis_example()
    else:
        # Run full design workflow
        design_pd1_binder()
        
    print("\n" + "="*70)
    print("Example workflow complete!")
    print("="*70)
