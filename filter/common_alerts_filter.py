import medchem as mc
from chemtsv3.filter import MolFilter

class CommonAlertsFilter(MolFilter):
    """
    Requires: medchem==2.0.5, rdkit==2025.9.3 (rdkit needs to be upgraded)
    The filter excludes molecules that contain substructures listed under “Common Alerts” in the medchem package.
    """
    def mol_check(self, mol):
        alerts = mc.structural.CommonAlertsFilters()
        try:
            results = alerts(mols=[mol],
                            n_jobs=-1,
                            progress=False,
                            progress_leave=False,
                            scheduler="auto")
        except:
            return False
        return results["reasons"][0] == None