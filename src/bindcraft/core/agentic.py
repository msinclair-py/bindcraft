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

import asyncio
import logging
from pathlib import Path
from typing import Any

from academy.agent import Agent, action
from academy.handle import Handle

from .folding import Folding
from .inverse_folding import InverseFolding
from ..analysis.energy import EnergyCalculation, SimpleEnergy
from ..util.quality_control import SequenceQualityControl


logger = logging.getLogger(__name__)


class ForwardFoldingAgent(Agent):
    """Agent responsible for forward folding (structure prediction)."""

    def __init__(self, fold_alg: Folding) -> None:
        super().__init__()
        self.fold_alg = fold_alg

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

        return str(structure)

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
        max_fold = min(4, len(sequences))  # Limit to 4 per round

        for i, seq in enumerate(sequences[:max_fold]):
            label = f"trial_{trial}"
            seq_label = f"seq_{i}"

            seqs = [target_sequence, seq]
            structure = self.fold_alg(seqs, label, seq_label)

            folded_structures[i] = {
                "sequence": seq,
                "structure": str(structure),
                "energy": None,
                "rmsd": None,
            }

        logger.info(f"Refolded {len(folded_structures)} structures")
        return folded_structures


class InverseFoldingAgent(Agent):
    """Agent responsible for inverse folding (sequence generation)."""

    def __init__(self, inv_fold_alg: InverseFolding) -> None:
        super().__init__()
        self.inv_fold_alg = inv_fold_alg

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

        try:
            sequences = self.inv_fold_alg(
                input_path=fasta_in,
                pdb_path=pdb_path,
                output_path=fasta_out,
                remodel_positions=remodel_indices,
            )
            logger.info(f"Generated {len(sequences)} sequences")
            return sequences
        except Exception as e:
            logger.error(f"Inverse folding failed: {e}")
            return []


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
        energy_threshold: float = -50.0,
    ) -> tuple[dict[int, dict[str, Any]], list[str]]:
        """Analyze folded structures and filter based on energy."""
        logger.info(f"Analysis: Evaluating {len(folded_structures)} structures")

        evaluated_structures = {}
        passing_structures = []

        for idx, struct_data in folded_structures.items():
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
        forward_folder: Handle[ForwardFoldingAgent],
        inverse_folder: Handle[InverseFoldingAgent],
        qc_agent: Handle[QualityControlAgent],
        analyzer: Handle[AnalysisAgent],
    ) -> None:
        super().__init__()
        self.forward_folder = forward_folder
        self.inverse_folder = inverse_folder
        self.qc_agent = qc_agent
        self.analyzer = analyzer

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
        is_initial: bool = False,
    ) -> dict[str, Any]:
        """Run one complete design cycle."""
        logger.info(f"Coordinator: Starting design cycle for trial {trial}")

        try:
            # Step 1: Forward folding (initial only)
            if is_initial:
                initial_pdb = await self.forward_folder.fold_initial(
                    target_sequence, binder_sequence, trial
                )
                logger.info(f"Coordinator: Initial fold complete: {initial_pdb}")

            # Step 2: Inverse folding
            generated_sequences = await self.inverse_folder.generate_sequences(
                fasta_in, pdb_path, fasta_out, remodel_indices
            )

            if not generated_sequences:
                logger.warning("No sequences generated in inverse folding")
                return {
                    "success": False,
                    "error": "No sequences generated",
                    "trial": trial,
                }

            # Step 3: Quality control
            filtered_sequences = await self.qc_agent.filter_sequences(
                generated_sequences
            )

            if not filtered_sequences:
                logger.warning("No sequences passed quality control")
                return {
                    "success": False,
                    "error": "No sequences passed QC",
                    "trial": trial,
                }

            # Step 4: Refolding
            folded_structures = await self.forward_folder.refold_sequences(
                target_sequence, filtered_sequences, trial
            )

            # Step 5: Analysis and filtering
            evaluated_structures, passing_structures = (
                await self.analyzer.evaluate_structures(folded_structures)
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
        n_rounds: int = 3,
    ) -> dict[str, Any]:
        """Run the complete peptide design workflow."""
        logger.info(f"Coordinator: Starting full workflow for {n_rounds} rounds")

        results = {
            "success": True,
            "rounds_completed": 0,
            "total_sequences_generated": 0,
            "total_sequences_filtered": 0,
            "best_energy": float("inf"),
            "all_cycles": [],
            "error_message": "",
        }

        for trial in range(n_rounds):
            is_initial = trial == 0

            # Construct paths for this trial
            last_trial = trial - 1 if trial > 0 else 0
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
                is_initial=is_initial,
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

