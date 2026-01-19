import medchem as mc
from rdkit.Chem import Mol
from rdkit import Chem

from filter import MolFilter
from utils.linker_utils import link_linker

class StructuralAlertFilter(MolFilter):
    def __init__(self, cores=None):
        self.cores = cores

    def mol_check(self, mol):
        alerts = mc.structural.CommonAlertsFilters()
        if self.cores is not None:
            try:
                mol = link_linker(self.cores, mol)
                Chem.SanitizeMol(mol)
            except:
                return False
        try:
            results = alerts(mols=[mol],
                            n_jobs=-1,
                            progress=True,
                            progress_leave=True,
                            scheduler="auto")
        except:
            return False
        return results["reasons"][0] == None

    def mol_check_linker(self, mol: Mol, link_mol: Mol) -> bool:
        alerts = mc.structural.CommonAlertsFilters()
        try:
            results = alerts(mols=[link_mol],
                            n_jobs=-1,
                            progress=True,
                            progress_leave=True,
                            scheduler="auto")
        except:
            return False
        return results["reasons"][0] == None