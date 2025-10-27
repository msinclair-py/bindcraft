import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from academy.manager import Manager
from academy.exchange import LocalExchangeFactory
from academy.logging import init_logging

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
    device = 'cuda:0'

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

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(),
    ) as manager:
        # Launch individual agents
        forward_folder = await manager.launch(
            ForwardFoldingAgent,
            args=(chai,)
        )
        inverse_folder = await manager.launch(
            InverseFoldingAgent,
            args=(proteinmpnn,)
        )
        qc_agent = await manager.launch(
            QualityControlAgent,
            args=(SequenceQualityControl(**qc_kwargs),)
        )
        analyzer = await manager.launch(
            AnalysisAgent,
            args=(SimpleEnergy(),)
        )

        # Launch coordinator with handles to other agents
        coordinator = await manager.launch(
            PeptideDesignCoordinator,
            args=(forward_folder, inverse_folder, qc_agent, analyzer, nseqs, retries)
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
