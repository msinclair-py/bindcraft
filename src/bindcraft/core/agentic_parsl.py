"""
Agentic workflow for peptide design using Academy agents.

This module implements the BindCraft peptide design pipeline as an agentic workflow
using Academy agents, enabling dynamic decision-making and adaptive optimization.

Workflow steps:
1. Forward Folding: Initial structure prediction
2. Inverse Folding: Sequence generation
3. Quality Control: Filter sequences
4. Refolding: Predict structures for new sequences
5. Analysis & Filtering: Evaluate and select best candidates
"""

from academy.agent import Agent, action
from academy.handle import Handle
import asyncio
import logging
import parsl
from parsl import Config
from parsl import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.launchers import MpiExecLauncher
from pathlib import Path
from typing import Any, Optional

from .folding import Folding
from .inverse_folding import InverseFolding
from ..analysis.energy import EnergyCalculation, SimpleEnergy
from ..util.quality_control import SequenceQualityControl

logger = logging.getLogger(__name__)

@parsl.python_app
def fold_sequence_task(
    fold_alg: Folding,
    sequence: str,
    label: str,
    seq_label: str,
    constraints: Optional[dict]=None,
    glycan_chains: Optional[dict]=None,
    glycan_restraint: str = None
) -> dict:
    """Parsl task for folding a single sequence."""

    try:
        result = fold_alg(sequence, label, seq_label, constraints)
    except:
        result = fold_alg(sequence, label, constraints)
    return result

@parsl.python_app
def inverse_fold_task(
    inv_fold_alg: InverseFolding,
    input_path: Path,
    pdb_path: Path,
    output_path: Path,
    remodel_positions: list[int]
) -> list[str]:

    sequences = inv_fold_alg(
        input_path=input_path,
        pdb_path=pdb_path,
        output_path=output_path,
        remodel_positions=remodel_positions,
    )

    return sequences

@parsl.python_app
def qc_task(
    qc_alg: SequenceQualityControl,
    seqs: list[str],
) -> None:
    for seq in seqs:
        pass # NOTE: finish this

@parsl.python_app
def energy_task(
    energy_alg: EnergyCalculation,
    structure: Path,
) -> float:
    energy = energy_alg(structure)
    return energy

class ForwardFoldingAgent(Agent):
    """
    Agent responsible for all folding tasks.
    """
    def __init__(self,
                 fold_alg: Folding,
                 parsl_config: Config):
        self.fold_alg = fold_alg
        self.config = parsl_config
    
    async def agent_on_startup(self) -> None:
        """Initialize Parsl on agent startup."""
        logger.info(f'Initializing Parsl workers')
        self.dfk = parsl.load(self.config)

    async def agent_on_shutdown(self) -> None:
        """Clean up Parsl on agent shutdown."""
        logger.info('Cleaning up Parsl')
        if self.dfk:
            self.dfk.cleanup()
            self.dfk = None
        parsl.clear()

    @action
    async def fold_sequences(
        self,
        sequences: list[str],
        names: list[str],
        constraints: Optional[list[dict]]=None,
    ) -> str:
        """Perform initial forward folding on target-binder complex."""
        logger.info(f"Folding {len(sequences)} seqs with Chai-1")
        
        if isinstance(sequences, str): # single sequence passed
            sequences = [[sequences]]
        
        futures = []
        for sequence, name, constraint in zip(sequences, names, constraints):
            if isinstance(sequence, str): # single sequence to fold
                sequence = [sequence]

            futures.append(
                asyncio.wrap_future(
                    fold_sequence_task(self.fold_alg, sequence, name, constraint)
                )
            )

        results = await asyncio.gather(*futures)

        return results


class InverseFoldingAgent:
    def __init__(self,
                 inv_fold_alg: InverseFolding,
                 parsl_config: Config):
        self.inv_fold_alg = inv_fold_alg
        self.config = parsl_config

    async def agent_on_startup(self) -> None:
        """Initialize Parsl on agent startup."""
        #max_workers = self.config.executors[0].max_workers
        logger.info(f'Initializing Parsl workers')
        self.dfk = parsl.load(self.config)

    async def agent_on_shutdown(self) -> None:
        """Clean up Parsl on agent shutdown."""
        logger.info('Cleaning up Parsl')
        if self.dfk:
            self.dfk.cleanup()
            self.dfk = None
        parsl.clear()

    @action
    async def generate_sequences(
        self,
        fasta_in: Path,
        pdb_path: Path,
        fasta_out: Path,
        remodel_indices: list[int],
    ) -> list[str]:
        """Generate new sequences via inverse folding."""
        logger.info(f"Inverse folding: Generating sequences")
        
        sequences = await asyncio.wrap_future(inverse_fold_task(
            inv_fold_alg=self.inv_fold_alg,
            input_path=fasta_in,
            pdb_path=pdb_path,
            output_path=fasta_out,
            remodel_positions=remodel_indices
        ))

        logger.info(f"Generated {len(sequences)} sequences")
        return sequences


class FoldingAgent(Agent):
    """
    Agent responsible for all folding tasks.
    """
    def __init__(self,
                 fold_alg: Folding,
                 inv_fold_alg: InverseFolding,
                 parsl_config: Config):
        self.fold_alg = fold_alg
        self.inv_fold_alg = inv_fold_alg
        self.config = parsl_config
    
    async def agent_on_startup(self) -> None:
        """Initialize Parsl on agent startup."""
        #max_workers = self.config.executors[0].max_workers
        logger.info(f'Initializing Parsl workers')
        self.dfk = parsl.load(self.config)

    async def agent_on_shutdown(self) -> None:
        """Clean up Parsl on agent shutdown."""
        logger.info('Cleaning up Parsl')
        if self.dfk:
            self.dfk.cleanup()
            self.dfk = None
        parsl.clear()

    @action
    async def fold_initial(
        self,
        target_sequence: str,
        binder_sequence: str,
        trial: int,
    ) -> str:
        """Perform initial forward folding on target-binder complex."""
        logger.info(f"Forward folding: Initial fold for trial {trial}")

        sequences = [target_sequence, binder_sequence]
        label = f"trial_{trial}"
        seq_label = "seq_0"

        structure = self.fold_alg(sequences, label, seq_label)
        logger.info(f"Initial structure folded: {structure}")

        return structure

    @action
    async def refold_sequences(
        self,
        target_sequence: str,
        sequences: list[str],
        trial: int,
    ) -> dict[int, dict[str, Any]]:
        """Refold new sequences with target."""
        logger.info(f"Forward folding: Refolding {len(sequences)} sequences for trial {trial}")

        folded_structures = {}

        structures = []
        for i, seq in enumerate(sequences):
            label = f"trial_{trial}"
            seq_label = f"seq_{i}"

            seqs = [target_sequence, seq]
            structures.append(asyncio.wrap_future(fold_sequence_task(self.fold_alg, seqs, label, seq_label)))

        structures = await asyncio.gather(*structures)
        
        folded_structures = {i: {
            'sequence': sequences[i],
            'structure': str(structures[i]),
            'energy': None,
            'rmsd': None
        } for i in range(len(structures))}

        logger.info(f"Folded {len(folded_structures)} structures")

        return folded_structures

    @action
    async def generate_sequences(
        self,
        fasta_in: Path,
        pdb_path: Path,
        fasta_out: Path,
        remodel_indices: list[int],
    ) -> list[str]:
        """Generate new sequences via inverse folding."""
        logger.info(f"Inverse folding: Generating sequences")
        
        sequences = await asyncio.wrap_future(inverse_fold_task(
            inv_fold_alg=self.inv_fold_alg,
            input_path=fasta_in,
            pdb_path=pdb_path,
            output_path=fasta_out,
            remodel_positions=remodel_indices
        ))

        logger.info(f"Generated {len(sequences)} sequences")
        return sequences


class QualityControlAgent(Agent):
    """Agent responsible for sequence quality control filtering."""

    def __init__(self, qc_filter: SequenceQualityControl) -> None:
        super().__init__()
        self.qc_filter = qc_filter

    @action
    async def filter_sequences(self, sequences: list[str]) -> list[str]:
        """Filter sequences based on quality control criteria."""
        logger.info(f"Quality control: Filtering {len(sequences)} sequences")

        filtered_sequences = []

        for seq in sequences:
            if self.qc_filter(seq):
                filtered_sequences.append(seq)

        logger.info(
            f"Quality control: {len(filtered_sequences)} / {len(sequences)} "
            "sequences passed QC"
        )

        return filtered_sequences


class AnalysisAgent(Agent):
    """Agent responsible for structure analysis and filtering."""

    def __init__(self, energy_alg: EnergyCalculation) -> None:
        super().__init__()
        self.energy_alg = energy_alg

    @action
    async def evaluate_structures(
        self,
        folded_structures: dict[int, dict[str, Any]],
        energy_threshold: float = -10.0,
    ) -> tuple[dict[int, dict[str, Any]], list[str]]:
        """Analyze folded structures and filter based on energy."""
        logger.info(f"Analysis: Evaluating {len(folded_structures)} structures")

        evaluated_structures = {}
        passing_structures = []

        for idx, struct_data in folded_structures.items():
            logger.info(f'Analyzing: {idx}, {struct_data["structure"]}')
            try:
                energy = self.energy_alg(Path(struct_data["structure"]))
                struct_data["energy"] = energy
                evaluated_structures[idx] = struct_data

                if energy < energy_threshold:
                    passing_structures.append(struct_data["structure"])
            except Exception as e:
                logger.warning(f"Energy calculation failed for structure {idx}: {e}")

        logger.info(
            f"Analysis: {len(passing_structures)} / {len(evaluated_structures)} "
            "structures passed filtering"
        )

        return evaluated_structures, passing_structures


class PeptideDesignCoordinator(Agent):
    """Coordinator agent that orchestrates the peptide design workflow."""

    def __init__(
        self,
        fold_agent: Handle[FoldingAgent],
        qc_agent: Handle[QualityControlAgent],
        analyzer_agent: Handle[AnalysisAgent],
        nseqs: int,
        retries: int,
    ) -> None:
        super().__init__()
        self.fold_agent = fold_agent
        self.qc_agent = qc_agent
        self.analyzer_agent = analyzer_agent
        self.nseqs = nseqs
        self.retries = retries

    @action
    async def prepare_run(self,
                          target_sequence: str,
                          binder_sequence: str,):
        structure = await self.fold_agent.fold_initial(
            target_sequence, binder_sequence, 0
        )

    @action
    async def run_design_cycle(
        self,
        target_sequence: str,
        binder_sequence: str,
        fasta_in: Path,
        pdb_path: Path,
        fasta_out: Path,
        remodel_indices: list[int],
        trial: int,
    ) -> dict[str, Any]:
        """Run one complete design cycle."""
        logger.info(f"Coordinator: Starting design cycle for trial {trial}")
        print("about to fold")
        try:
            filtered_sequences = []
            i = 0

            while len(filtered_sequences) < self.nseqs and i < self.retries:
                # Step 1: Inverse folding
                generated_sequences = await self.fold_agent.generate_sequences(
                    fasta_in, pdb_path, fasta_out, remodel_indices
                )

                # Step 2: Quality control
                filtered_sequences += await self.qc_agent.filter_sequences(
                    generated_sequences
                )

                i += 1

            if not filtered_sequences:
                logger.warning("No sequences passed quality control, max retries attempted.")
                return {
                    "success": False,
                    "error": "No sequences passed QC",
                    "trial": trial,
                }

            # Step 3: Refolding
            folded_structures = await self.fold_agent.refold_sequences(
                target_sequence, filtered_sequences, trial
            )

            # Step 4: Analysis and filtering
            evaluated_structures, passing_structures = (
                await self.analyzer_agent.evaluate_structures(folded_structures)
            )

            logger.info(
                f"Coordinator: Cycle {trial} complete. "
                f"{len(passing_structures)} structures passed filtering"
            )

            return {
                "success": True,
                "trial": trial,
                "generated_sequences": len(generated_sequences),
                "filtered_sequences": len(filtered_sequences),
                "folded_structures": len(folded_structures),
                "passing_structures": passing_structures,
                "evaluated_structures": evaluated_structures,
            }

        except Exception as e:
            logger.error(f"Coordinator: Error in design cycle {trial}: {e}")
            return {
                "success": False,
                "error": str(e),
                "trial": trial,
            }

    @action
    async def run_full_workflow(
        self,
        target_sequence: str,
        binder_sequence: str,
        fasta_base_path: Path,
        pdb_base_path: Path,
        remodel_indices: list[int],
        num_rounds: int = 3,
    ) -> dict[str, Any]:
        """Run the complete peptide design workflow."""
        logger.info(f"Coordinator: Starting full workflow for {num_rounds} rounds")

        results = {
            "success": True,
            "rounds_completed": 0,
            "total_sequences_generated": 0,
            "total_sequences_filtered": 0,
            "best_energy": float("inf"),
            "all_cycles": [],
            "error_message": "",
        }
        print(results)

        (fasta_base_path / 'trial_0').mkdir(exist_ok=True)
        (pdb_base_path / 'trial_0').mkdir(exist_ok=True)
        await self.prepare_run(target_sequence, binder_sequence)

        for trial in range(1, num_rounds + 1):
            # Construct paths for this trial
            last_trial = trial - 1
            fasta_in = fasta_base_path / f"trial_{last_trial}"
            fasta_out = fasta_base_path / f"trial_{trial}"
            pdb_path = pdb_base_path / f"trial_{last_trial}"

            cycle_result = await self.run_design_cycle(
                target_sequence,
                binder_sequence,
                fasta_in,
                pdb_path,
                fasta_out,
                remodel_indices,
                trial,
            )

            results["all_cycles"].append(cycle_result)

            if not cycle_result["success"]:
                logger.warning(f"Design cycle {trial} failed: {cycle_result.get('error')}")
                results["success"] = False
                results["error_message"] = cycle_result.get("error", "Unknown error")
                break

            # Update metrics
            results["rounds_completed"] += 1
            results["total_sequences_generated"] += cycle_result.get(
                "generated_sequences", 0
            )
            results["total_sequences_filtered"] += cycle_result.get(
                "filtered_sequences", 0
            )

        logger.info(f"Coordinator: Workflow complete. {results['rounds_completed']} rounds completed")
        return results

