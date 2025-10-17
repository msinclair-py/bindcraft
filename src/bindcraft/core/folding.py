from abc import ABC, abstractmethod
from chai_lab.chai1 import run_inference
import gemmi
from pathlib import Path
from string import Template
import tempfile

class Folding(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def prepare(self):
        pass
    
    @abstractmethod
    def __call__(self):
        pass
    
    @abstractmethod
    def postprocess(self):
        pass

class Chai(Folding):
    def __init__(self,
                 fasta_dir: Path,
                 out: Path,
                 device: str='xpu:0'):
        self.fasta_dir = Path(fasta_dir)
        self.out = Path(out)
        self.device = device

        self.devshm = Path('/dev/shm')
        self.template_fasta = Template('>protein|target\n$target\n>protein|binder\n$binder')
        
        self.fasta_dir.mkdir(exist_ok=True, parents=True)
        self.out.mkdir(exist_ok=True, parents=True)

    def prepare(self,
                seqs: list[str],
                label: str) -> Path:
        fasta_str = self.template_fasta.substitute(target=seqs[0], binder=seqs[1])
        fasta_path = self.fasta_dir / f'{label}.fa'
        fasta_path.write_text(fasta_str)
        return fasta_path

    def __call__(self, 
                 seqs: list[str],
                 exp_label: str,
                 out_label: str) -> Path:
        fasta = self.prepare(seqs, f'{exp_label}_{out_label}')
        out = self.devshm / exp_label
        out.mkdir(exist_ok=True, parents=True)

        print(fasta)
        
        with tempfile.TemporaryDirectory(dir=str(out)) as tmpdir:
            tmp = Path(tmpdir)
            run_inference(
                fasta_file=fasta,
                output_dir=tmp,
                device=self.device,
                use_esm_embeddings=True,
            )

            pdb = self.postprocess(tmp, out_label)

        return pdb

    def postprocess(self,
                    in_path: Path,
                    out_name: str) -> Path:
        best_model = in_path / 'pred.model_idx_0.cif'
        final_path = self.out / f'{out_name}.pdb'

        structure = gemmi.read_structure(str(best_model))
        structure.write_pdb(str(final_path))

        return final_path

class Boltz(Folding):
    def __init__(self):
        pass

    def prepare(self):
        pass

    def __call__(self):
        pass
    
    def postprocess(self):
        pass
