import subprocess, tempfile, os, shutil
from node import FASTAStringNode, SentenceNode
from reward import Reward

HMM_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/fluorescence/PF01353.hmm"))

class ProteinFluorescenceReward(Reward):
    def __init__(self, e0=1e-5, sigma=2.0):
        """
        Args:
            e0: Used for e-value scaling. 0.5 if e_value=e0
            sigma: Slope for e-value scaling.
        """
        self.e0= e0
        self.sigma = sigma

    # implement
    def objective_functions(self):
        def e_value(node: FASTAStringNode | SentenceNode):
            if issubclass(node.__class__, FASTAStringNode):
                fasta = node.string
            else:
                fasta = node.key()
            return self.hmmer_similarity_hmmsearch(fasta)

        return [e_value]

    # implement
    def reward_from_objective_values(self, objective_values):
        e_value = objective_values[0]
        return self.e_value_to_score(e_value)
    
    @classmethod
    def hmmer_similarity_hmmsearch(cls, seq: str) -> float:
        """
        Returns e value.
        """
        hmmsearch_bin = "hmmsearch"
        if not shutil.which(hmmsearch_bin):
            raise FileNotFoundError(f"{hmmsearch_bin} not found.")
        if not os.path.exists(HMM_PATH):
            raise FileNotFoundError(f"HMM not found: {HMM_PATH}")

        with tempfile.TemporaryDirectory() as tmp:
            fasta = os.path.join(tmp, "query.faa")
            tblout = os.path.join(tmp, "out.tbl")
            with open(fasta, "w") as f:
                f.write(">query\n")
                f.write(seq.strip() + "\n")

            cmd = [
                hmmsearch_bin, "--noali", "--tblout", tblout, HMM_PATH, fasta
            ]
            run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if run.returncode not in (0, 1):
                raise RuntimeError(f"hmmsearch failed (code {run.returncode}).\nSTDERR:\n{run.stderr}")

            best = cls._parse_hmmer_tblout_best(tblout)
            if best is None:
                e = float("inf")
            else:
                e = float(best["full_evalue"])

        return e

    @staticmethod
    def _parse_hmmer_tblout_best(tbl_path: str):
        """
        Parse HMMER3 tblout, then return best hit.
        """
        best = None
        with open(tbl_path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split()
                rec = {
                    "query": parts[0],
                    "query_acc": parts[1],
                    "target": parts[2],
                    "target_acc": parts[3],
                    "full_evalue": float(parts[4]), # e value (smaller - better)
                    "full_score": float(parts[5]),  # bit score (bigger - better)
                    "full_bias": float(parts[6]),
                    "c_evalue": float(parts[7]),
                    "c_score": float(parts[8]),
                    "c_bias": float(parts[9]),
                }
                if best is None or rec["full_evalue"] < best["full_evalue"]:
                    best = rec
        return best

    def e_value_to_score(self, e_value, floor=1e-180):
        import math
        e = max(e_value, floor)
        z = (math.log10(e) - math.log10(self.e0)) / self.sigma
        return 1.0 / (1.0 + 10**z)