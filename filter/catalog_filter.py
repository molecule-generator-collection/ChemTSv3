from rdkit.Chem import Mol, FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams
from filter import MolFilter

class CatalogFilter(MolFilter):
    VALID_CATALOGS = {
        "PAINS_A": FilterCatalogParams.FilterCatalogs.PAINS_A, 
        "PAINS_B": FilterCatalogParams.FilterCatalogs.PAINS_B, 
        "PAINS_C": FilterCatalogParams.FilterCatalogs.PAINS_C, 
        "PAINS":   FilterCatalogParams.FilterCatalogs.PAINS, 
        "BRENK":   FilterCatalogParams.FilterCatalogs.BRENK, 
        "CHEMBL":  FilterCatalogParams.FilterCatalogs.CHEMBL, 
        "CHEMBL_BMS":  FilterCatalogParams.FilterCatalogs.CHEMBL_BMS, 
        "CHEMBL_DUNDEE":  FilterCatalogParams.FilterCatalogs.CHEMBL_Dundee, 
        "CHEMBL_GLAXO":  FilterCatalogParams.FilterCatalogs.CHEMBL_Glaxo, 
        "CHEMBL_INPHARMATICA":  FilterCatalogParams.FilterCatalogs.CHEMBL_Inpharmatica, 
        "CHEMBL_LINT":  FilterCatalogParams.FilterCatalogs.CHEMBL_LINT, 
        "CHEMBL_MLSMR":  FilterCatalogParams.FilterCatalogs.CHEMBL_MLSMR, 
        "CHEMBL_SURECHEMBL":  FilterCatalogParams.FilterCatalogs.CHEMBL_SureChEMBL, 
        "NIH":     FilterCatalogParams.FilterCatalogs.NIH, 
        "ZINC":    FilterCatalogParams.FilterCatalogs.ZINC, 
        "ALL":     FilterCatalogParams.FilterCatalogs.ALL
    }
    
    def __init__(self, catalogs: list[str]=["A", "B", "C"]):
        self.catalogs = [f.upper() for f in catalogs]
        
        params = FilterCatalogParams()
        for f in self.catalogs:
            if f in self.VALID_CATALOGS:
                params.AddCatalog(self.VALID_CATALOGS[f])
            else:
                raise ValueError(f"Unknown catalog name: {f}.")
        self.filter_catalogs = FilterCatalog.FilterCatalog(params)

    # implement
    def mol_check(self, mol: Mol) -> bool:
        return not self.filter_catalogs.HasMatch(mol)