from abc import ABC, abstractmethod
from chai_lab.chai1 import run_inference
from pathlib import Path
import shutil
from string import Template

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
                 out: Path):
        self.fasta_dir = Path(fasta_dir)
        self.out = Path(out)
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
                 label: str) -> Path:
        fasta = self.prepare(seqs, label)
        out = self.devshm / label
        out.mkdir(exist_ok=True, parents=True)

        run_inference(
            fasta_file=str(fasta),
            output_dir=str(out),
            device='xpu:0',
            use_esm_embeddings=True,
        )

        return self.postprocess(out)

    def postprocess(self,
                    in_path: Path) -> Path:
        best_model = in_path / 'pred.model_idx_0.cif'
        final_path = self.out / best_model.name
        shutil.move(str(best_model), str(final_path))
        shutil.rmtree(str(in_path))
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
