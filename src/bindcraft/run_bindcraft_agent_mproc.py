from __future__ import annotations

import asyncio
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from academy.manager import Manager
from academy.exchange import LocalExchangeFactory
from academy.logging import init_logging
from academy.exchange.cloud import spawn_http_exchange

from bindcraft.core.agentic import (
    ForwardFoldingAgent,
    InverseFoldingAgent,
    QualityControlAgent,
    AnalysisAgent,
    PeptideDesignCoordinator,
)

from bindcraft.core.folding import Chai
from bindcraft.core.inverse_folding import ProteinMPNN
from bindcraft.analysis.energy import SimpleEnergy
from bindcraft.util.quality_control import SequenceQualityControl

# Global counter for assigning GPUs in round-robin fashion
_folding_worker_counter = 0
_folding_worker_lock = multiprocessing.Lock()

def set_gpu_for_folding():
    """Initializer for folding processes - assigns GPUs 0-3 in round-robin."""
    import os
    global _folding_worker_counter

    # Assign GPU based on worker count (round-robin across 0-3)
    with _folding_worker_lock:
        gpu_id = _folding_worker_counter % 4
        _folding_worker_counter += 1

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"Set GPU for folding process to GPU {gpu_id}")
    init_logging('INFO')

def set_gpu_for_other_tasks():
    """Initializer for inverse folding/analysis/QC processes - uses GPU 4."""
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    print(f"Set GPU for other tasks process to GPU 4")
    init_logging('INFO')


EXCHANGE_PORT = 5346
async def main():
    init_logging('INFO')

    # Set up working directories
    cwd = Path.cwd() / "bindcraft_run"
    cwd.mkdir(exist_ok=True)
    fasta_dir = cwd / "fastas"
    folds_dir = cwd / "folds"
    fasta_dir.mkdir(exist_ok=True)
    folds_dir.mkdir(exist_ok=True)

    # these need to be somehow passed into the call
    nseqs = 25
    batch_size = 250
    retries = 5
    temp = '0.1'
    mpnn_model = 'v_48_020'
    mpnn_weights = 'soluble_model_weights'
    device = 'cuda'

    qc_kwargs = {
        'max_repeat': 4,
        'max_appearance_ratio': 0.33,
        'max_charge': 5,
        'max_charge_ratio': 0.5,
        'max_hydrophobic_ratio': 0.8,
        'min_diversity': 8,
        'bad_motifs': [],
        'bad_n_termini': None # use defaults
    }

    # Initialize algorithm instances with required parameters
    chai = Chai(
        fasta_dir=fasta_dir,
        out=folds_dir,
        diffusion_steps=100,
        device=device  # or 'cpu' if GPU not available
    )

    proteinmpnn = ProteinMPNN(
        proteinmpnn_path=Path("/eagle/FoundEpidem/avasan/Softwares/ProteinMPNN"),  # Update with actual path
        num_seq=nseqs,
        max_retries=retries,
        sampling_temp=temp,
        batch_size=batch_size,
        model_name=mpnn_model,
        model_weights=mpnn_weights,
        device=device  # or 'cpu' if GPU not available
    )
    with spawn_http_exchange('localhost', EXCHANGE_PORT) as factory:
        # Get the spawn context (spawn_http_exchange already sets the start method)
        try:
            mp_context = multiprocessing.get_context('spawn')
        except RuntimeError:
            # If context is already set, just use the default
            mp_context = None

        # Create separate executors for different task types
        # Folding uses 4 GPUs (GPUs 0-3) with round-robin assignment
        folding_executor = ProcessPoolExecutor(
            max_workers=4,
            initializer=set_gpu_for_folding,
            mp_context=mp_context
        )

        # Inverse folding, QC, and analysis use 1 GPU (GPU 4)
        other_tasks_executor = ProcessPoolExecutor(
            max_workers=5,
            initializer=set_gpu_for_other_tasks,
            mp_context=mp_context
        )

        async with await Manager.from_exchange_factory(
            factory=factory,
            executors={
                'folding': folding_executor,
                'other': other_tasks_executor,
            },
        ) as manager:
            # Launch individual agents
            forward_folder = await manager.launch(
                ForwardFoldingAgent,
                args=(chai,),
                executor='folding'
            )
            inverse_folder = await manager.launch(
                InverseFoldingAgent,
                args=(proteinmpnn,),
                executor='other'
            )
            qc_agent = await manager.launch(
                QualityControlAgent,
                args=(SequenceQualityControl(**qc_kwargs),),
                executor='other'
            )
            analyzer = await manager.launch(
                AnalysisAgent,
                args=(SimpleEnergy(),),
                executor='other'
            )

            # Launch coordinator with handles to other agents
            coordinator = await manager.launch(
                PeptideDesignCoordinator,
                args=(forward_folder, inverse_folder, qc_agent, analyzer, nseqs, retries),
                executor='other'
            )

            # Define sequences for design
            target_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
            binder_sequence = "MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF"

            # Run the workflow
            results = await coordinator.run_full_workflow(
                target_sequence=target_sequence,
                binder_sequence=binder_sequence,
                fasta_base_path=fasta_dir,
                pdb_base_path=folds_dir,
                remodel_indices=[],  # Interface indices to redesign
                n_rounds=3
            )

            print(f"Workflow completed: {results}")

if __name__ == '__main__':
    asyncio.run(main())
