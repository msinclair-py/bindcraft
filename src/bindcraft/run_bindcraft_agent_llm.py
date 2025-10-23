"""
Example script for running the peptide design workflow with LLM decision-making.

This script demonstrates how to use the agentic workflow with Argo LLM integration
for intelligent workflow decisions (continue, restart, or branch).

Environment variables required:
- ARGO_URL: Argo API endpoint (e.g., https://argo.alcf.anl.gov/api/v1/inference)
- ARGO_USER: Your ALCF username
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from academy.manager import Manager
from academy.exchange import LocalExchangeFactory
from academy.logging import init_logging

from bindcraft.core.agentic_llm import (
    ForwardFoldingAgent,
    InverseFoldingAgent,
    QualityControlAgent,
    AnalysisAgent,
    LLMDecisionAgent,
    PeptideDesignCoordinator,
)
from bindcraft.core.folding import Chai
from bindcraft.core.inverse_folding import ProteinMPNN
from bindcraft.analysis.energy import SimpleEnergy
from bindcraft.util.quality_control import SequenceQualityControl


async def main():
    """Run the peptide design workflow with LLM decision-making."""
    init_logging('INFO')

    # Get Argo credentials from environment
    argo_url = os.getenv("ARGO_URL")
    argo_user = os.getenv("ARGO_USER")

    if not argo_url or not argo_user:
        raise ValueError(
            "ARGO_URL and ARGO_USER environment variables must be set. "
            "Example: export ARGO_URL='https://argo.alcf.anl.gov/api/v1/inference' "
            "export ARGO_USER='your_username'"
        )

    # Set up working directories
    cwd = Path.cwd() / "bindcraft_run_llm"
    cwd.mkdir(exist_ok=True)
    fasta_dir = cwd / "fastas"
    folds_dir = cwd / "folds"
    fasta_dir.mkdir(exist_ok=True)
    folds_dir.mkdir(exist_ok=True)

    print(f"Working directory: {cwd}")
    print(f"FASTA directory: {fasta_dir}")
    print(f"Folds directory: {folds_dir}")

    # Initialize algorithm instances with required parameters
    chai = Chai(
        fasta_dir=fasta_dir,
        out=folds_dir,
        diffusion_steps=100,
        device='xpu:0'  # Change to 'cpu' if GPU not available
    )

    proteinmpnn = ProteinMPNN(
        proteinmpnn_path=Path("/path/to/ProteinMPNN"),  # Update with actual path
        num_seq=1,
        max_retries=5,
        sampling_temp='0.1',
        batch_size=250,
        model_name='v_48_020',
        model_weights='soluble_model_weights',
        device='xpu:0'  # Change to 'cpu' if GPU not available
    )

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(),
    ) as manager:
        print("Launching agents...")

        # Launch individual agents
        forward_folder = await manager.launch(
            ForwardFoldingAgent,
            args=(chai,)
        )
        print("✓ ForwardFoldingAgent launched")

        inverse_folder = await manager.launch(
            InverseFoldingAgent,
            args=(proteinmpnn,)
        )
        print("✓ InverseFoldingAgent launched")

        qc_agent = await manager.launch(
            QualityControlAgent,
            args=(SequenceQualityControl(),)
        )
        print("✓ QualityControlAgent launched")

        analyzer = await manager.launch(
            AnalysisAgent,
            args=(SimpleEnergy(),)
        )
        print("✓ AnalysisAgent launched")

        # Launch LLM decision agent with Argo credentials
        llm_decision = await manager.launch(
            LLMDecisionAgent,
            args=(argo_url, argo_user, "gpt4o")  # Adjust model as needed
        )
        print("✓ LLMDecisionAgent launched")

        # Launch coordinator with handles to all agents
        coordinator = await manager.launch(
            PeptideDesignCoordinator,
            args=(forward_folder, inverse_folder, qc_agent, analyzer, llm_decision)
        )
        print("✓ PeptideDesignCoordinator launched")

        # Define sequences for design
        target_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
        binder_sequence = "MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF"

        print("\n" + "="*60)
        print("Starting peptide design workflow with LLM decision-making")
        print("="*60)
        print(f"Target sequence: {target_sequence[:30]}...")
        print(f"Binder sequence: {binder_sequence[:30]}...")
        print(f"LLM model: gpt4o (via Argo)")
        print(f"LLM check interval: every 3 cycles")
        print("="*60 + "\n")

        # Run the workflow
        results = await coordinator.run_full_workflow(
            target_sequence=target_sequence,
            binder_sequence=binder_sequence,
            fasta_base_path=fasta_dir,
            pdb_base_path=folds_dir,
            remodel_indices=[],  # Interface indices to redesign (empty = all)
            n_rounds=9,  # Run up to 9 rounds
            llm_check_interval=3,  # Check with LLM every 3 cycles
        )

        print("\n" + "="*60)
        print("Workflow Results")
        print("="*60)
        print(f"Success: {results['success']}")
        print(f"Rounds completed: {results['rounds_completed']}")
        print(f"Total sequences generated: {results['total_sequences_generated']}")
        print(f"Total sequences filtered: {results['total_sequences_filtered']}")
        print(f"Best energy achieved: {results['best_energy']:.2f}")

        if results.get('llm_decisions'):
            print(f"\nLLM Decisions ({len(results['llm_decisions'])} checkpoints):")
            for decision in results['llm_decisions']:
                print(f"  Trial {decision['trial']}: {decision['action'].upper()}")
                print(f"    Reasoning: {decision['reasoning'][:100]}...")

        if results.get('recommendation'):
            print(f"\nFinal LLM Recommendation: {results['recommendation'].upper()}")

        if results.get('error_message'):
            print(f"\nError: {results['error_message']}")

        print("="*60)
        print(f"Results saved to: {cwd}")
        print("="*60)


if __name__ == '__main__':
    asyncio.run(main())

