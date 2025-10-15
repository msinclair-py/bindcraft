import dill as pickle
import MDAnalysis as mda
from MDAnalysis.analysis.rms import rmsd
import numpy as np
from pathlib import Path
from string import Template
from typing import Any, Union
from .core.folding import Folding
from .core.inverse_folding import InverseFolding
from .analysis.energy import EnergyCalculation, SimpleEnergy
from .util.quality_control import SequenceQualityControl

class BindCraft:
    def __init__(self,
                 target: str,
                 binder: str,
                 fold_alg: Folding,
                 inv_fold_alg: InverseFolding,
                 energy_alg: EnergyCalculation = None,
                 qc_filter: SequenceQualityControl = None,
                 chk_file: str = 'checkpoint.pkl',
                 n_rounds: int = 1,
                 cutoff: float = 5.0,
                 **kwargs):
        self.target = target
        self.binder = binder
        self.cutoff = cutoff
        self.chk_file = Path(chk_file)

        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self.fold = fold_alg
        self.inv_fold = inv_fold_alg
        self.energy = energy_alg if energy_alg else SimpleEnergy()
        self.qc = qc_filter if qc_filter else SequenceQualityControl()

        self.n_rounds = n_rounds
        self.label = Template('trial_$trial')
        self.seq_label = Template('seq_$seq')
        self.trial = 0
        self.bnum = 1
        self.remodel = None

        if self.chk_file.exists():
            self.restart_run()

    def run_inference(self):
        structures = self.prepare()
        rounds = 0

        while rounds < self.n_rounds:
            new_structures = self.cycle()
            structures.update(self.appraise(new_structures))
            self.checkpoint(structures)
            self.trial += 1
            rounds += 1
        
        return structures

    def prepare(self) -> dict[str, dict]:
        label = self.label.substitute(trial='initial', seq='guess')
        structure = self.fold([self.target, self.binder], label)
        
        self.reference = self.get_binder_coords(structure)
        self.get_interface(structure)
        energy = self.measure_energy(structure)

        return {
            'initial_guess': {
                'sequence': self.binder, 
                'structure': str(structure), 
                'rmsd': 0.0, 
                'energy': energy
            }
        }

    def cycle(self) -> dict[str, dict]:
        if self.trial == 0:
            fasta_in = Path('.')
            pdb_path = Path('.')
            fasta_out = Path('inverse_folds') / f'trial_{self.trial}'
            structure_out = Path('folds') / f'trial_{self.trial}'
        else:
            fasta_in = Path('inverse_folds') / f'trial_{self.trial-1}'
            pdb_path = input_path
            fasta_out = Path('inverse_folds') / f'trial_{self.trial}'
            structure_out = Path('folds') / f'trial_{self.trial}'
        
        fasta_out.mkdir(exist_ok=True, parents=True)
        structure_out.mkdir(exist_ok=True, parents=True)

        self.fold.out = structure_out
        
        inverse_fold_seqs = self.inv_fold(
            fasta_in,
            pdb_path,
            fasta_out,
            self.remodel
        )
        
        filtered_seqs = [seq for seq in inverse_fold_seqs if self.qc(seq)]
        
        structures = {}
        for i, seq in enumerate(filtered_seqs):
            label = self.label.substitute(trial=self.trial)
            seq_label = self.seq_label.substitute(seq=i)
            structure = self.fold([self.target, seq], label, seq_label)
            
            structures[self.bnum] = {
                'sequence': seq, 
                'structure': str(structure), 
                'rmsd': np.nan, 
                'energy': np.nan
            }
            self.bnum += 1

        return structures

    def checkpoint(self, structures: dict[str, Any]) -> None:
        with open(self.chk_file, 'wb') as fout:
            pickle.dump(structures, fout)

    def appraise(self, structures: dict[str, Any]) -> dict[str, Any]:
        for pep_id, values in structures.items():
            if np.isnan(values['rmsd']):
                structure = Path(values['structure'])
                coords = self.get_binder_coords(structure)
                rmsd_value = rmsd(coords, self.reference)
                energy = self.measure_energy(structure)
                values['rmsd'] = rmsd_value
                values['energy'] = energy
        
        return structures

    def measure_energy(self, structure: Path) -> float:
        return self.energy(structure)

    def restrict(self):
        pass

    def get_binder_coords(self, pdbs: Union[list[Path], Path]) -> np.ndarray:
        if not isinstance(pdbs, list):
            pdbs = [pdbs]

        coords = np.zeros((len(pdbs), len(self.binder), 3))
        for i, pdb in enumerate(pdbs):
            u = mda.Universe(str(pdb))
            coords[i, :, :] = u.select_atoms('name CA and chainID B').positions
            del u
        
        return coords.squeeze()

    def get_interface(self, pdb: Path) -> None:
        u = mda.Universe(str(pdb))
        sel = u.select_atoms(f'chainID B and around {self.cutoff} chainID A')
        resIDs = np.unique(sel.residues.resids)
        self.remodel = [i for i in range(1, len(self.binder) + 1) 
                       if i not in resIDs]
        del u

    def restart_run(self) -> dict:
        with open(self.chk_file, 'rb') as fin:
            structures = pickle.load(fin)
        
        max_bnum = max([int(k) for k in structures.keys() 
                       if isinstance(k, (int, str)) and str(k).isdigit()], 
                      default=0)
        self.bnum = max_bnum + 1
        
        trials = []
        for data in structures.values():
            if 'structure' in data:
                structure_path = data['structure']
                if 'trial_' in structure_path:
                    trial_str = structure_path.split('trial_')[1].split('_')[0]
                    if trial_str.isdigit():
                        trials.append(int(trial_str))
        
        self.trial = max(trials, default=0) + 1
        return structures
