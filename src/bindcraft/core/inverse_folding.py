from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import sys

class InverseFolding(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def postprocessing(self):
        pass

class ProteinMPNN(InverseFolding):
    def __init__(self,
                 proteinmpnn_path: Path,
                 num_seq: int = 1,
                 sampling_temp: str = '0.1',
                 batch_size: int = 2,
                 model_name: str = '',
                 model_weights: str = ''):
        self.proteinmpnn_path = Path(proteinmpnn_path)
        self.run_py = self.proteinmpnn_path / 'protein_mpnn_run.py'
        self.helpers = [
            self.proteinmpnn_path / 'helper_scripts' / 'parse_multiple_chains.py',
            self.proteinmpnn_path / 'helper_scripts' / 'assign_fixed_chains.py'
        ]
        self.num_seq = num_seq
        self.sampling_temp = sampling_temp
        self.batch_size = batch_size
        self.model_name = model_name
        self.model_weights = model_weights
        self.file_intermediates = ['parsed_pdbs.jsonl', 'chain_B_design.jsonl']

    def prepare(self,
                pdb_path: Path,
                file1: Path,
                file2: Path) -> None:
        args = [
            [
                '--input_path', str(pdb_path), 
                '--output_path', str(file1),
            ],
            [
                '--input_path', str(file1),
                '--output_path', str(file2),
                '--chain_list', 'B'
            ]
        ]

        for helper, arg in zip(self.helpers, args):
            subprocess.run([sys.executable, str(helper), *arg], check=True)

    def __call__(self,
                 input_path: Path,
                 pdb_path: Path,
                 output_path: Path,
                 remodel_positions: list[int] = None):
        input_path = Path(input_path)
        pdb_path = Path(pdb_path)
        output_path = Path(output_path)
        
        input_path.mkdir(exist_ok=True, parents=True)
        output_path.mkdir(exist_ok=True, parents=True)
        
        jsonls = [input_path / fi for fi in self.file_intermediates]
        self.prepare(pdb_path, *jsonls)
        self.run(jsonls, output_path)
        return self.postprocessing(output_path)

    def run(self,
            jsonls: list[Path],
            output: Path) -> None:
        cmd = [
            sys.executable, str(self.run_py),
            '--jsonl_path', str(jsonls[0]),
            '--chain_id_jsonl', str(jsonls[1]),
            '--out_folder', str(output),
            '--num_seq_per_target', str(self.num_seq),
            '--sampling_temp', self.sampling_temp,
            '--batch_size', str(self.batch_size),
        ]
        
        if self.model_name:
            cmd.extend(['--model_name', self.model_name])
        if self.model_weights:
            cmd.extend(['--path_to_model_weights', self.model_weights])

        subprocess.run(cmd, check=True)

    def postprocessing(self,
                       output_dir: Path) -> list[str]:
        seqs_dir = output_dir / 'seqs'
        if not seqs_dir.exists():
            return []
            
        fas = seqs_dir.glob('*.fa')
        seqs = []
        for fa in fas:
            seqs.extend(self.process_single_file(fa))
        
        return seqs

    def process_single_file(self,
                            path: Path) -> list[str]:
        with open(path) as f:
            raw_data = f.readlines()
        
        seqs = [line.strip() for line in raw_data[3::2]]
        return seqs


class ESMIF1(InverseFolding):
    def __init__(self):
        pass
    
    def prepare(self):
        pass
    
    def __call__(self):
        pass
    
    def postprocessing(self):
        pass
