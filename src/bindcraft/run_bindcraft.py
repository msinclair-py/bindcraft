#!/usr/bin/env python3
"""
Main execution script for BindCraft pipeline.
"""

from pathlib import Path
import argparse
from .core import BindCraft, Chai, ProteinMPNN
from .analysis import SimpleEnergy
from .util import SequenceQualityControl


def setup_directories(base_dir: Path):
    """Create necessary directory structure."""
    dirs = {
        'fastas': base_dir / 'fastas',
        'folds': base_dir / 'folds',
        'inverse_folds': base_dir / 'inverse_folds',
        'analysis': base_dir / 'analysis',
    }
    
    for d in dirs.values():
        d.mkdir(exist_ok=True, parents=True)
    
    return dirs


def main():
    parser = argparse.ArgumentParser(
        description='BindCraft: One-shot design of functional protein binders'
    )
    
    # Required arguments
    parser.add_argument('--target', type=str, required=True,
                       help='Target protein sequence')
    parser.add_argument('--binder', type=str, required=True,
                       help='Initial binder sequence')
    
    # Directory arguments
    parser.add_argument('--work_dir', type=Path, default=Path('.'),
                       help='Working directory (default: current directory)')
    parser.add_argument('--proteinmpnn_path', type=Path, required=True,
                       help='Path to ProteinMPNN installation')
    
    # Pipeline parameters
    parser.add_argument('--n_rounds', type=int, default=3,
                       help='Number of design rounds (default: 3)')
    parser.add_argument('--n_seqs', type=int, default=1000,
                        help='Number of sequences per round (default: 1000)')
    parser.add_argument('--temp', type=str, default='0.1',
                       help='ProteinMPNN sampling temperature (default: 0.1)')
    parser.add_argument('--batch_size', type=int, default=250,
                        help='ProteinMPNN batch size (default: 250)')
    parser.add_argument('--dist_cutoff', type=float, default=4.0,
                       help='Interface distance cutoff in Å (default: 4.0)')
    parser.add_argument('--rmsd_cutoff', type=float, default=5.0,
                       help='Binder RMSD cutoff in Å (default: 5.0)')
    parser.add_argument('--energy_cutoff', type=float, default=-50.0,
                        help='Cutoff for binder to target interaction energy in kcal/mol (default: -50.0)')
    
    # Quality control parameters
    parser.add_argument('--max_hydrophobic', type=float, default=0.6,
                       help='Maximum hydrophobic ratio (default: 0.6)')
    parser.add_argument('--max_charge', type=int, default=5,
                       help='Maximum net charge (default: 5)')
    
    # Optional arguments
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pkl',
                       help='Checkpoint file name (default: checkpoint.pkl)')
    parser.add_argument('--restart', action='store_true',
                       help='Restart from checkpoint')
    parser.add_argument('--mpi', action='store_true',
                        help='Whether or not we are parallelizing with MPI')
    
    args = parser.parse_args()
    
    # Setup directories
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        work_dir = Path(args.work_dir) / f'rank{rank}'
    else:
        work_dir = Path(args.work_dir)

    dirs = setup_directories(work_dir)
    
    # Initialize folding algorithm
    fold_alg = Chai(
        fasta_dir=dirs['fastas'],
        out=dirs['folds']
    )
    
    # Initialize inverse folding algorithm
    inv_fold_alg = ProteinMPNN(
        proteinmpnn_path=args.proteinmpnn_path,
        num_seq=args.n_seqs,
        sampling_temp=args.temp,
        batch_size=args.batch_size
    )
    
    # Initialize energy calculation
    energy_alg = SimpleEnergy()
    
    # Initialize quality control
    qc_filter = SequenceQualityControl(
        max_hydrophobic_ratio=args.max_hydrophobic,
        max_charge=args.max_charge
    )
    
    # Initialize BindCraft
    bindcraft = BindCraft(
        cwd=work_dir,
        target=args.target,
        binder=args.binder,
        fold_alg=fold_alg,
        inv_fold_alg=inv_fold_alg,
        energy_alg=energy_alg,
        qc_filter=qc_filter,
        chk_file=str(args.work_dir / args.checkpoint),
        n_rounds=args.n_rounds,
        dist_cutoff=args.dist_cutoff,
        rmsd_cutoff=args.rmsd_cutoff,
        energy_cutoff=args.energy_cutoff,
    )
    
    print("="*60)
    print("BindCraft: One-shot protein binder design")
    print("="*60)
    print(f"Target sequence: {args.target[:50]}...")
    print(f"Binder sequence: {args.binder[:50]}...")
    print(f"Number of rounds: {args.n_rounds}")
    print(f"Sequences per round: {args.n_seqs}")
    print(f"Working directory: {args.work_dir}")
    print("="*60)
    
    # Run the pipeline
    try:
        results = bindcraft.run_inference()
        
        print("\n" + "="*60)
        print("Design completed successfully!")
        print(f"Total structures generated: {len(results)}")
        print("="*60)
        
        # Print top 5 binders by energy
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].get('energy', float('inf'))
        )
        
        print("\nTop 5 designs by energy:")
        print("-"*60)
        for i, (binder_id, data) in enumerate(sorted_results[:5], 1):
            print(f"{i}. Binder {binder_id}")
            print(f"   Sequence: {data['sequence'][:50]}...")
            print(f"   RMSD: {data['rmsd']:.2f} Å")
            print(f"   Energy: {data['energy']:.2f}")
            print(f"   Structure: {data['structure']}")
            print()
        
        # Save summary
        summary_file = args.work_dir / 'analysis' / 'design_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("BindCraft Design Summary\n")
            f.write("="*60 + "\n")
            f.write(f"Target: {args.target}\n")
            f.write(f"Initial binder: {args.binder}\n")
            f.write(f"Rounds: {args.n_rounds}\n")
            f.write(f"Total designs: {len(results)}\n\n")
            
            for binder_id, data in sorted_results:
                f.write(f"Binder {binder_id}\n")
                f.write(f"  Sequence: {data['sequence']}\n")
                f.write(f"  RMSD: {data['rmsd']:.2f}\n")
                f.write(f"  Energy: {data['energy']:.2f}\n")
                f.write(f"  Structure: {data['structure']}\n\n")
        
        print(f"Summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        raise
    

if __name__ == '__main__':
    main()
