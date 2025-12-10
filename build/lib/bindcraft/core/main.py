import dill as pickle
import MDAnalysis as mda
from MDAnalysis.analysis.rms import rmsd
import numpy as np
from pathlib import Path
from rust_simulation_tools import kabsch_align
from string import Template
from typing import Any, Union
from .folding import Folding
from .inverse_folding import InverseFolding
from ..analysis.energy import EnergyCalculation, SimpleEnergy
from ..util.quality_control import SequenceQualityControl

class BindCraft:
    def __init__(self,
                 cwd: Path,
                 target: str,
                 binder: str,
                 fold_alg: Folding,
                 inv_fold_alg: InverseFolding,
                 energy_alg: EnergyCalculation = None,
                 qc_filter: SequenceQualityControl = None,
                 chk_file: str = 'checkpoint.pkl',
                 n_rounds: int = 1,
                 dist_cutoff: float = 4.0,
                 rmsd_cutoff: float = 5.0,
                 energy_cutoff: float = -50.0,
                 **kwargs):
        self.cwd = cwd
        self.target = target
        self.binder = binder
        self.fold = fold_alg
        self.inv_fold = inv_fold_alg
        self.energy = energy_alg if energy_alg is not None else SimpleEnergy()
        self.qc = qc_filter if qc_filter is not None else SequenceQualityControl()
        self.chk_file = Path(chk_file)
        self.n_rounds = n_rounds
        self.dist_cutoff = dist_cutoff
        self.rmsd_cutoff = rmsd_cutoff
        self.energy_cutoff = energy_cutoff

        for k, v in kwargs.items():
            setattr(self, k, v)
        

        self.label = Template('trial_$trial')
        self.seq_label = Template('seq_$seq')
        self.trial = 1

        self.ff_path = self.cwd / 'folds'
        self.if_path = self.cwd / 'inverse_folds'

    def run_inference(self):
        """Runs the main inference loop. First checks for a restart, then prepares
        all internal data objects and finally performs sampling, checkpointing after each
        round.
        """
        if self.chk_file.exists():
            current_pdbs = self.restart_run()
            current_round = self.trial
        else:
            current_pdbs = str(self.ff_path / 'trial_0')
            current_round = 0

        self.prepare()

        for _ in range(self.n_rounds - current_round):
            print(current_pdbs)
            new_structures = self.cycle(current_pdbs)
            current_pdbs = self.appraise(new_structures)
            self.checkpoint()
            self.trial += 1

    def prepare(self) -> dict[str, dict]:
        """Performs first forward fold and initializes data structure for runs.
        If this is a restart we will have already loaded the previous data structure
        and we only need to obtain the reference coordinates and interface of the parent.
        """
        label = self.label.substitute(trial=0)
        seq_label = self.seq_label.substitute(seq=0)

        if not hasattr(self, 'structures'):
            self.fold.out = self.fold.out / label
            self.fold.out.mkdir(exist_ok=True)
            (self.if_path / label).mkdir(exist_ok=True)

            structure = self.fold([self.target, self.binder], label, seq_label)
            
            energy = self.measure_energy(structure)

            self.structures = {
                0: {
                    0: {
                        'sequence': self.binder, 
                        'structure': str(structure), 
                        'tref_rmsd': 0.0, 
                        'bref_rmsd': 0.0,
                        'energy': energy
                    }
                }
            }

        else:
            structure = str(self.ff_path / label / f'{seq_label}.pdb')
        
        self.get_reference_coords(structure)
        self.remodel = self.get_interface(structure)

    def cycle(self,
              pdbs: Path) -> dict[str, dict]:
        """Perform 1 cycle of sampling. Core loop is:
            (i)    Scrape seqs from last round by reading in PDB
            (ii)   Identify interface, inverse fold new seqs
            (iii)  QC new seqs, fold passing seqs

        Args:
            pdbs (list[Path]): List of passing PDBs from last round to 
                iterate through design cycle.

        Returns:
            (dict[str, dict]): 
        """
        label = self.label.substitute(trial=self.trial)
        last_label = self.label.substitute(trial=self.trial - 1)

        fasta_in = self.if_path / last_label
        fasta_out = self.if_path / label
        structure_out = self.ff_path / label
        
        fasta_out.mkdir(exist_ok=True, parents=True)
        structure_out.mkdir(exist_ok=True, parents=True)

        self.fold.out = structure_out
        
        # scale number of seqs by number of pdbs
        n_pdbs = len(list((self.ff_path / last_label).glob('*.pdb')))
        n_seqs = self.inv_fold.num_seq * n_pdbs
        filtered_seqs = []
        current_fasta = ''
        i = 0
        while len(filtered_seqs) < n_seqs and i < self.inv_fold.max_retries:
            print(f'{len(filtered_seqs)} quality sequences!')

            inverse_fold_seqs = self.inv_fold(
                fasta_in,       # input_path (Path)
                pdbs,           # pdb_path (Path)
                fasta_out,      # output_path (Path)
                self.remodel    # indices (list[int])
            )
        
            filtered_seqs += [seq for seq in inverse_fold_seqs if self.qc(seq)]

            i += 1
                
            if i == 1:
                print(fasta_in)
                print(fasta_out)
                print(list((fasta_out / 'seqs').glob('seq_*.fa')))
                fa = list((fasta_out / 'seqs').glob('seq_*.fa'))[0]
                current_fasta = fa.name

            fa.rename(fasta_out / 'seqs' / f'{i-1}.{current_fasta}')

        assert len(filtered_seqs) > 0, 'Inverse folding failed!'
        
        max_fold = 4 if len(filtered_seqs) >= 4 else len(filtered_seqs)
        #structures = {bnum: {} for bnum in range(len(filtered_seqs))}
        structures = {bnum: {} for bnum in range(max_fold)}
        for i, seq in enumerate(filtered_seqs[:max_fold]):
            seq_label = self.seq_label.substitute(seq=i)
            structure = self.fold([self.target, seq], label, seq_label)
            
            structures[i] = {
                'sequence': seq, 
                'structure': str(structure), 
                'tref_rmsd': np.nan, 
                'bref_rmsd': np.nan, 
                'energy': np.nan
            }

        return structures

    def checkpoint(self) -> None:
        """Writes out current state to a pickle file.
        """
        with open(self.chk_file, 'wb') as fout:
            pickle.dump(self.structures, fout)

    def appraise(self, structures: dict[str, Any]) -> Path:
        """Performs structural appraisal in the form of RMSD relative to the
        parent binder conformation and binder to target interaction energy.
        Stores all measurements but returns list of only the PDBs which pass 
        these metrics for further development.
        """
        current_trial = self.label.substitute(trial=self.trial)
        fail_path = self.ff_path / f'failed_{current_trial}'
        fail_path.mkdir(exist_ok=True)
        
        for key, val in structures.items():
            structure = Path(val['structure'])
            coords = self.get_coords(structure)
            
            #align = np.squeeze(kabsch_align(coords[np.newaxis, :, :], self.ref_pos, self.target_idx))
            #displacement_rmsd = rmsd(align[self.binder_idx], self.ref_pos[self.binder_idx])
            
            align = np.squeeze(kabsch_align(coords[np.newaxis, :, :], self.ref_pos, self.binder_idx))
            binder_rmsd = rmsd(align[self.binder_idx], self.ref_pos[self.binder_idx])
            energy = self.measure_energy(structure)
            #val['tref_rmsd'] = displacement_rmsd
            val['tref_rmsd'] = np.nan
            val['bref_rmsd'] = binder_rmsd
            val['energy'] = energy

            print(val)
            if energy > self.energy_cutoff:
            #if displacement_rmsd > self.rmsd_cutoff or energy > self.energy_cutoff:
                moved = fail_path / structure.name
                structure.rename(moved)
                structure = moved
        
        print(structures)
        self.structures[self.trial] = structures

        return self.ff_path / current_trial

    def measure_energy(self, structure: Path) -> float:
        return self.energy(structure)

    def get_reference_coords(self, pdb: Path) -> None:
        u = mda.Universe(str(pdb))
        self.ref_pos = u.select_atoms('name CA').positions.astype(np.float32)
        self.target_idx = np.array(
            [i for i in range(len(u.select_atoms('name CA and chainID A')))], dtype=np.int32
        )
        self.binder_idx = np.array(
            [i for i in range(self.ref_pos.shape[0]) if i not in self.target_idx], dtype=np.int32
        )

    def get_coords(self, pdb: Path) -> np.ndarray:
        u = mda.Universe(str(pdb))
        return u.select_atoms('name CA').positions.astype(np.float32)

    def get_interface(self, pdb: Path) -> None:
        u = mda.Universe(str(pdb))
        sel = u.select_atoms(f'chainID B and around {self.dist_cutoff} chainID A')
        resIDs = np.unique(sel.residues.resids)
        return [i for i in range(1, len(self.binder) + 1) if i not in resIDs]

    def restart_run(self) -> list[str]:
        with open(self.chk_file, 'rb') as fin:
            self.structures = pickle.load(fin)
        
        self.trial = max(self.structures.keys())
        
        return self.ff_path / self.label.substitute(trial=self.trial)
