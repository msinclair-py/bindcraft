"""
Agentic workflow for peptide design using Academy agents with LLM decision-making.

This module implements a multi-agent system for peptide design that uses an LLM
(via Argo) to make intelligent decisions about workflow continuation, restarting,
or branching based on design metrics.
"""

import asyncio
import json
import logging
import requests
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from academy.agent import Agent, action
from academy.handle import Handle

from bindcraft.core.folding import Folding
from bindcraft.core.inverse_folding import InverseFolding
from bindcraft.analysis.energy import EnergyCalculation
from bindcraft.util.quality_control import SequenceQualityControl

logger = logging.getLogger(__name__)


class ForwardFoldingAgent(Agent):
    """Agent responsible for forward folding (structure prediction)."""

    def __init__(self, fold_alg: Folding):
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
        try:
            pdb = self.fold_alg(
                seqs=[target_sequence, binder_sequence],
                exp_label=f"trial_{trial}",
                out_label=f"initial_fold",
            )
            logger.info(f"Initial fold complete: {pdb}")
            return str(pdb)
        except Exception as e:
            logger.error(f"Forward folding failed: {e}")
            raise

    @action
    async def refold_sequences(
        self,
        target_sequence: str,
        binder_sequences: list[str],
        trial: int,
    ) -> dict[int, dict[str, Any]]:
        """Refold new sequences with target."""
        logger.info(f"Forward folding: Refolding {len(binder_sequences)} sequences for trial {trial}")
        folded_structures = {}

        for idx, binder_seq in enumerate(binder_sequences):
            try:
                pdb = self.fold_alg(
                    seqs=[target_sequence, binder_seq],
                    exp_label=f"trial_{trial}",
                    out_label=f"seq_{idx}",
                )
                folded_structures[idx] = {
                    "pdb_path": str(pdb),
                    "sequence": binder_seq,
                }
            except Exception as e:
                logger.warning(f"Refolding failed for sequence {idx}: {e}")

        logger.info(f"Refolded {len(folded_structures)} structures")
        return folded_structures


class InverseFoldingAgent(Agent):
    """Agent responsible for inverse folding (sequence generation)."""

    def __init__(self, inv_fold_alg: InverseFolding):
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

    def __init__(self, qc_alg: SequenceQualityControl):
        self.qc_alg = qc_alg

    @action
    async def filter_sequences(self, sequences: list[str]) -> list[str]:
        """Filter sequences based on quality control criteria."""
        logger.info(f"QC: Filtering {len(sequences)} sequences")
        try:
            filtered = self.qc_alg(sequences)
            logger.info(f"QC: {len(filtered)} sequences passed filtering")
            return filtered
        except Exception as e:
            logger.error(f"QC filtering failed: {e}")
            return []


class AnalysisAgent(Agent):
    """Agent responsible for structure analysis and energy evaluation."""

    def __init__(self, energy_calc: EnergyCalculation):
        self.energy_calc = energy_calc

    @action
    async def evaluate_structures(
        self, folded_structures: dict[int, dict[str, Any]]
    ) -> tuple[list[dict], list[dict]]:
        """Evaluate folded structures and filter by energy."""
        logger.info(f"Analysis: Evaluating {len(folded_structures)} structures")

        evaluated = []
        for idx, struct in folded_structures.items():
            try:
                energy = self.energy_calc(struct["pdb_path"])
                struct["energy"] = energy
                evaluated.append(struct)
            except Exception as e:
                logger.warning(f"Energy calculation failed for structure {idx}: {e}")

        # Filter by energy threshold
        passing = [s for s in evaluated if s.get("energy", float("inf")) < 100.0]
        logger.info(f"Analysis: {len(passing)} structures passed energy threshold")

        return evaluated, passing


class LLMDecisionAgent(Agent):
    """Agent that uses LLM (via Argo) to make workflow decisions."""

    def __init__(self, argo_url: str, argo_user: str, model: str = "gpt4o"):
        self.argo_url = argo_url
        self.argo_user = argo_user
        self.model = model

    def _call_argo(self, prompt: str, system_prompt: str) -> str:
        """Synchronous call to Argo API."""
        data = {
            "user": self.argo_user,
            "model": self.model,
            "system": system_prompt,
            "prompt": [prompt],
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 500,
        }

        payload = json.dumps(data)
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.argo_url, data=payload, headers=headers, timeout=60)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Argo API call failed: {e}")
            raise

    def _parse_recommendation(self, response: str) -> dict[str, Any]:
        """Parse LLM response into structured recommendation."""
        response_lower = response.lower()

        if "restart" in response_lower:
            action = "restart"
        elif "branch" in response_lower:
            action = "branch"
        else:
            action = "continue"

        return {
            "action": action,
            "reasoning": response,
        }

    @action
    async def analyze_workflow(
        self,
        cycle_history: list[dict],
        current_trial: int,
        max_trials: int,
        initial_energy: float,
        improved_count: int,
    ) -> dict[str, Any]:
        """Analyze workflow progress and recommend next action."""
        logger.info(f"LLM: Analyzing workflow at trial {current_trial}")

        best_energy = cycle_history[-1]["energy"] if cycle_history else initial_energy
        energy_improvement = ((initial_energy - best_energy) / initial_energy * 100) if initial_energy > 0 else 0

        prompt = f"""
Peptide design workflow analysis:

Current Status:
- Trial: {current_trial}/{max_trials}
- Initial energy: {initial_energy:.2f}
- Best energy achieved: {best_energy:.2f}
- Energy improvement: {energy_improvement:.1f}%
- Structures with better energy: {improved_count}

Cycle History:
{json.dumps(cycle_history[-3:], indent=2)}

Based on this progress, should we:
1. CONTINUE - keep iterating with current sequence
2. RESTART - start over with a different initial sequence
3. BRANCH - save current best and try a variation

Provide your recommendation and brief reasoning.
"""

        system_prompt = """You are an expert in peptide design optimization. 
Analyze the workflow metrics and recommend whether to continue, restart, or branch the design process.
Consider convergence, improvement rate, and exploration efficiency."""

        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                response = await loop.run_in_executor(
                    executor,
                    lambda: self._call_argo(prompt, system_prompt),
                )

            recommendation = self._parse_recommendation(response)
            logger.info(f"LLM recommendation: {recommendation['action']}")
            return recommendation

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {"action": "continue", "reasoning": f"Error in LLM analysis: {e}"}


class PeptideDesignCoordinator(Agent):
    """Coordinator that orchestrates the multi-agent peptide design workflow."""

    def __init__(
        self,
        forward_folder: Handle[ForwardFoldingAgent],
        inverse_folder: Handle[InverseFoldingAgent],
        qc_agent: Handle[QualityControlAgent],
        analyzer: Handle[AnalysisAgent],
        llm_decision: Handle[LLMDecisionAgent],
    ):
        self.forward_folder = forward_folder
        self.inverse_folder = inverse_folder
        self.qc_agent = qc_agent
        self.analyzer = analyzer
        self.llm_decision = llm_decision
        self.cycle_history = []
        self.initial_energy = None
        self.best_energy = float("inf")
        self.best_trial = 0
        self.improved_count = 0

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
            if is_initial:
                initial_pdb = await self.forward_folder.fold_initial(
                    target_sequence, binder_sequence, trial
                )
                logger.info(f"Coordinator: Initial fold complete: {initial_pdb}")

            generated_sequences = await self.inverse_folder.generate_sequences(
                fasta_in, pdb_path, fasta_out, remodel_indices
            )

            if not generated_sequences:
                logger.warning("No sequences generated in inverse folding")
                return {"success": False, "error": "No sequences generated", "trial": trial}

            filtered_sequences = await self.qc_agent.filter_sequences(generated_sequences)

            if not filtered_sequences:
                logger.warning("No sequences passed quality control")
                return {"success": False, "error": "No sequences passed QC", "trial": trial}

            folded_structures = await self.forward_folder.refold_sequences(
                target_sequence, filtered_sequences, trial
            )

            evaluated_structures, passing_structures = await self.analyzer.evaluate_structures(
                folded_structures
            )

            # Track metrics
            cycle_improved = sum(1 for s in evaluated_structures if s.get("energy", float("inf")) < self.initial_energy)
            self.improved_count += cycle_improved

            if evaluated_structures:
                cycle_best = min(s.get("energy", float("inf")) for s in evaluated_structures)
                if cycle_best < self.best_energy:
                    self.best_energy = cycle_best
                    self.best_trial = trial

            self.cycle_history.append({
                "trial": trial,
                "energy": self.best_energy,
                "improved_count": self.improved_count,
                "sequences_generated": len(generated_sequences),
                "structures_passing_qc": len(filtered_sequences),
            })

            logger.info(f"Coordinator: Cycle {trial} complete. {len(passing_structures)} structures passed")

            return {
                "success": True,
                "trial": trial,
                "generated_sequences": len(generated_sequences),
                "filtered_sequences": len(filtered_sequences),
                "folded_structures": len(folded_structures),
                "passing_structures": len(passing_structures),
            }

        except Exception as e:
            logger.error(f"Coordinator: Error in design cycle {trial}: {e}")
            return {"success": False, "error": str(e), "trial": trial}

    @action
    async def run_full_workflow(
        self,
        target_sequence: str,
        binder_sequence: str,
        fasta_base_path: Path,
        pdb_base_path: Path,
        remodel_indices: list[int],
        n_rounds: int = 3,
        llm_check_interval: int = 3,
    ) -> dict[str, Any]:
        """Run the complete peptide design workflow with LLM decision-making."""
        logger.info(f"Coordinator: Starting full workflow for {n_rounds} rounds")

        results = {
            "success": True,
            "rounds_completed": 0,
            "total_sequences_generated": 0,
            "total_sequences_filtered": 0,
            "best_energy": float("inf"),
            "all_cycles": [],
            "llm_decisions": [],
            "error_message": "",
        }

        for trial in range(n_rounds):
            is_initial = trial == 0

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

            results["rounds_completed"] += 1
            results["total_sequences_generated"] += cycle_result.get("generated_sequences", 0)
            results["total_sequences_filtered"] += cycle_result.get("filtered_sequences", 0)

            # Check with LLM at intervals
            if trial > 0 and trial % llm_check_interval == 0:
                if self.initial_energy is None:
                    self.initial_energy = self.best_energy

                llm_rec = await self.llm_decision.analyze_workflow(
                    cycle_history=self.cycle_history,
                    current_trial=trial,
                    max_trials=n_rounds,
                    initial_energy=self.initial_energy,
                    improved_count=self.improved_count,
                )

                results["llm_decisions"].append({
                    "trial": trial,
                    "action": llm_rec["action"],
                    "reasoning": llm_rec["reasoning"],
                })

                if llm_rec["action"] == "restart":
                    logger.info("LLM recommends restart - stopping workflow")
                    results["recommendation"] = "restart"
                    break
                elif llm_rec["action"] == "branch":
                    logger.info("LLM recommends branch - continuing from best trial")
                    results["recommendation"] = "branch"

        results["best_energy"] = self.best_energy
        logger.info(f"Coordinator: Workflow complete. {results['rounds_completed']} rounds completed")
        return results

