import json
import re
import xml.etree.ElementTree as ET
from rdkit import Chem
from rdkit.Chem import Mol

class MonomersLib():
    def __init__(self, monomers_lib: dict={}, cap_group_mols: dict={}):
        self.lib = monomers_lib
        self.cap_group_mols = cap_group_mols # smiles - mol
        #self.__class__.strip_namespace(monomers_lib.getroot())
    
    #load xml in ChEMBL format
    #one instance can load multiple libraries
    def load_xml(self, monomers_lib_path: str):
        root = ET.parse(monomers_lib_path).getroot()
        MonomersLib.strip_namespace(root)
        polymers = root.find("PolymerList")
        lib = self.lib or {}
        cap_group_mols = self.cap_group_mols or {}
        
        for pt in polymers:
            polymer_type = pt.get("polymerType")
            lib[polymer_type] = lib.get(polymer_type) or {}
            for m in pt:
                for m_tag in m:
                    if m_tag.tag == "MonomerID": #should be earlier than below
                        monomer_token = m_tag.text
                        lib[polymer_type][monomer_token] = lib[polymer_type].get(monomer_token) or {}
                    elif m_tag.tag == "MonomerSmiles":
                        monomer_smiles = m_tag.text
                        lib[polymer_type][monomer_token]["MonomerSmiles"] = monomer_smiles
                    elif m_tag.tag == "Attachments":
                        lib[polymer_type][monomer_token]["Attachments"] = lib[polymer_type][monomer_token].get("Attachments") or {}
                        for a in m_tag:
                            for a_tag in a:
                                if a_tag.tag == "AttachmentLabel": #should be earlier than below
                                    attachment_label = a_tag.text
                                elif a_tag.tag == "CapGroupSmiles":
                                    cap_group_smiles = a_tag.text
                                    lib[polymer_type][monomer_token]["Attachments"][attachment_label] = cap_group_smiles
                                    if not cap_group_smiles in cap_group_mols:
                                        cap_group_mols[cap_group_smiles] = MonomersLib.prepare_attachment_cap(cap_group_smiles)
                                    
        self.lib = lib
        self.cap_group_mols = cap_group_mols
    
    #load json in Pistoia Alliance format
    def load_json(self, monomers_lib_path:str):
        lib = self.lib or {}
        cap_group_mols = self.cap_group_mols or {}
        with open(monomers_lib_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for m in data:
            polymer_type = m.get("polymerType")
            lib[polymer_type] = lib.get(polymer_type) or {}
            monomer_token = m.get("symbol")
            lib[polymer_type][monomer_token] = lib[polymer_type].get(monomer_token) or {}
            lib[polymer_type][monomer_token]["MonomerSmiles"] = MonomersLib.atom_mapped_to_cx(m.get("smiles"))
            lib[polymer_type][monomer_token]["Attachments"] = lib[polymer_type][monomer_token].get("Attachments") or {}
            for r in m.get("rgroups"):
                attachment_label = r.get("label")
                cap_group_smiles = r.get("capGroupSmiles") or r.get("capGroupSMILES") #can be both
                lib[polymer_type][monomer_token]["Attachments"][attachment_label] = cap_group_smiles
                if not cap_group_smiles in cap_group_mols:
                    cap_group_mols[cap_group_smiles] = MonomersLib.prepare_attachment_cap_from_atom_mapped_smiles(cap_group_smiles, attachment_label)
            if polymer_type == "RNA" and monomer_token in ["r", "p"]: #for convenience: both capitalizations are commonly used
                lib[polymer_type][monomer_token.upper()] = lib[polymer_type][monomer_token]
    
        self.lib = lib
        self.cap_group_mols = cap_group_mols        
    
    # remove namespace from xml
    @classmethod
    def strip_namespace(cls, element: ET.Element):
        element.tag = element.tag.split("}", 1)[-1] if "}" in element.tag else element.tag
        for child in element:
            cls.strip_namespace(child)

    @staticmethod
    def standardize_monomer_token(token: str):
        if token.startswith("["):
            token = token[1:-1]
        return token
    
    #convert atom mapped SMILES to CXSMILES
    @staticmethod
    def atom_mapped_to_cx(smiles: str) -> str:
        smiles = smiles.replace("null", "H")
        p = Chem.SmilesParserParams()
        p.removeHs = False
        mol = Chem.MolFromSmiles(smiles, p)
        rw  = Chem.RWMol(mol)

        for idx in range(rw.GetNumAtoms()):
            a = rw.GetAtomWithIdx(idx)
            atom_map = a.GetAtomMapNum()
            if atom_map:
                a.SetAtomicNum(0)
                a.SetAtomMapNum(0)
                a.SetNoImplicit(True)
                a.SetNumExplicitHs(0)
                a.SetProp("atomLabel", f"_R{atom_map}")
                for nb in list(a.GetNeighbors()):
                    if nb.GetAtomicNum() == 1:
                        rw.RemoveBond(a.GetIdx(), nb.GetIdx())
                        rw.RemoveAtom(nb.GetIdx())

        mol = rw.GetMol()
        Chem.SanitizeMol(mol)

        cx = Chem.MolToCXSmiles(mol)
        return cx

    @staticmethod
    def prepare_attachment_cap(smiles: str) -> Mol:
        cap = Chem.MolFromSmiles(smiles)
        for a in cap.GetAtoms():
            if a.HasProp("atomLabel"):
                a.SetAtomMapNum(1)
        return cap
    
    #receives "R1" as attachment_label, not "_R1"
    @staticmethod
    def prepare_attachment_cap_from_atom_mapped_smiles(smiles: str, attachment_label: str) -> Mol:
        cap = Chem.MolFromSmiles(smiles)
        for a in cap.GetAtoms():
            if a.GetAtomicNum() == 0:
                a.SetAtomMapNum(1)
                a.SetProp("atomLabel", "_" + attachment_label)
        return cap

    def get_monomer_smiles(self, polymer_type: str, monomer_token: str):
        monomer_token = self.standardize_monomer_token(monomer_token)
        if monomer_token in self.lib[polymer_type]:
            return self.lib[polymer_type][monomer_token]["MonomerSmiles"]
        else: #inline SMILES
            if "$" in monomer_token:
                return monomer_token
            else:
                return MonomersLib.atom_mapped_to_cx(monomer_token)

    def get_cap_group_smiles(self, polymer_type: str, monomer_token: str, attachment_label: str):
        monomer_token = self.standardize_monomer_token(monomer_token)
        if monomer_token in self.lib[polymer_type]:
            return self.lib[polymer_type][monomer_token]["Attachments"][attachment_label]
        else: #Inline SMILES or undefined monomer token
            return "[*][H] |$_R1;$|"

class HELMConverter():
    polymer_types = ["PEPTIDE", "RNA", "CHEM", "BLOB"]
    skip_tokens = [".", "{", "(", ")"]

    def __init__(self, monomers_lib: MonomersLib):
        self.lib = monomers_lib
    
    def convert(self, helm: str):
        try:
            return self._convert(helm)
        except:
            return None
    
    def _convert(self, helm: str, close=True, verbose=False):
        helm_parts = helm.rsplit("$", 4)
        parsed_polymers = self.parse_polymers(helm_parts[0])
        parsed_bonds = self.parse_bonds(helm_parts[1])
        mol = None        

        for p in parsed_polymers:
            if mol is None:
                mol = self.mol_from_single_polymer(p)
            else:
                mol2 = self.mol_from_single_polymer(p)
                mol = Chem.molzip(mol, mol2)
        
        for b in parsed_bonds:
            mol = self.add_bond(mol, *b)

        if close: 
            mol = self.close_residual_attachment_points(mol)
            mol = Chem.RemoveHs(mol)
        return mol

    @staticmethod
    def split_helm(helm: str):
        pattern = (
            r"("
            r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]"
            r"|PEPTIDE\d+"
            r"|RNA\d+"
            r"|CHEM\d+"
            r"|BLOB\d+"
            r"|R\d"
            r"|pair"
            r"|V2.0"
            r"|[A-Z]"
            r"|[a-z]"
            r"|\||\(|\)|\{|\}|-|\$|:|,|\."
            r"|\d{2}|\d"
            r")"
        )
        tokens = re.findall(pattern, helm)
        assert helm == "".join(tokens)
        return tokens

    # should be called only if each r_num is unique within that mol
    @staticmethod
    def combine_monomers_with_unique_r_nums(m1: Mol, m1_r_num: int, m2: Mol, m2_r_num: int) -> Mol:
        m1_r_str = "_R" + str(m1_r_num)
        m2_r_str = "_R" + str(m2_r_num)
        for a in m1.GetAtoms():
            if a.HasProp("atomLabel") and a.GetProp("atomLabel").endswith(m1_r_str):
                a.SetAtomMapNum(1)
        for a in m2.GetAtoms():
            if a.HasProp("atomLabel") and a.GetProp("atomLabel").endswith(m2_r_str):
                a.SetAtomMapNum(1)
        return Chem.molzip(m1, m2)

    @classmethod
    def combine_backbone_monomers(cls, m_left: Mol, m_right: Mol) -> Mol:
        return cls.combine_monomers_with_unique_r_nums(m_left, 2, m_right, 1)

    def generate_mol(self, polymer_type: str, polymer_name: str, monomer_token: str, monomer_idx: int) -> Mol:
        smiles = self.lib.get_monomer_smiles(polymer_type, monomer_token)
        mol = Chem.MolFromSmiles(smiles)

        for a in mol.GetAtoms():
            if a.HasProp("atomLabel"):
                attachment_label = a.GetProp("atomLabel")[1:]
                cap_group_smiles = self.lib.get_cap_group_smiles(polymer_type, monomer_token, attachment_label)
                a.SetProp("capGroupSmiles", cap_group_smiles)
                a.SetProp("polymerName", polymer_name)
                a.SetProp("monomerIndex", str(monomer_idx)) # int can't be passed
        
        return mol
    
    #mol form splitted POLYMERNAME{......}
    def mol_from_single_polymer(self, helm_tokens_list: list[str]) -> Mol:
        polymer_type = None
        polymer_name = None
        monomer_idx = 1 # 1-based index

        for i, t in enumerate(helm_tokens_list):
            if t in self.skip_tokens:
                continue
            if t == "}":
                break
            if polymer_type is None:
                for pt in self.polymer_types:
                    if t.startswith(pt):
                        polymer_type = pt
                        polymer_name = t
                        continue
            elif monomer_idx == 1:
                mol = self.generate_mol(polymer_type, polymer_name, t, monomer_idx)
                if helm_tokens_list[i+1] == "(":
                    branch_mol = self.generate_mol(polymer_type, polymer_name, helm_tokens_list[i+2], -1)
                    mol = self.combine_monomers_with_unique_r_nums(mol, 3, branch_mol, 1)
                monomer_idx += 1
            else:
                if helm_tokens_list[i-1] == "(":
                    continue
                last_mol = self.generate_mol(polymer_type, polymer_name, t, monomer_idx)
                if helm_tokens_list[i+1] == "(":
                    branch_mol = self.generate_mol(polymer_type, polymer_name, helm_tokens_list[i+2], -1)
                    last_mol = self.combine_monomers_with_unique_r_nums(last_mol, 3, branch_mol, 1)
                mol = self.combine_backbone_monomers(mol, last_mol)
                monomer_idx += 1

        return mol
    
    @staticmethod
    def add_bond(polymer: Mol, initial_polymer_name_1: str, initial_monomer_idx_1: str, attachment_label_1: str, initial_polymer_name_2: str, initial_monomer_idx_2: str, attachment_label_2: str) -> Mol:
        idx_1 = idx_2 = idx_r_1 = idx_r_2 = None
        for a in polymer.GetAtoms():
            if a.HasProp("polymerName") and a.GetProp("polymerName") == initial_polymer_name_1:
                if a.GetProp("monomerIndex") == initial_monomer_idx_1:
                    if a.GetProp("atomLabel").endswith(attachment_label_1):
                        a.ClearProp("attachmentID") # used to detect remaining attatchment points
                        idx_r_1 = a.GetIdx()
                        idx_1 = a.GetNeighbors()[0].GetIdx()
            if a.HasProp("polymerName") and a.GetProp("polymerName") == initial_polymer_name_2:
                if a.GetProp("monomerIndex") == initial_monomer_idx_2:
                    if a.GetProp("atomLabel").endswith(attachment_label_2):
                        a.ClearProp("attachmentID")
                        idx_r_2 = a.GetIdx()
                        idx_2 = a.GetNeighbors()[0].GetIdx()
        if idx_1 is None or idx_2 is None:
            return None
        else:
            emol = Chem.EditableMol(polymer)
            emol.AddBond(idx_1, idx_2, Chem.rdchem.BondType.SINGLE)
            emol.RemoveAtom(idx_r_1) # changes indices
            emol.RemoveAtom(idx_r_2 - 1 if idx_r_2 > idx_r_1 else idx_r_2)
            return emol.GetMol()
    
    def parse_polymers(self, polymer_part: list):
        polymer_tokens = HELMConverter.split_helm(polymer_part)
        parsed_polymers = []
        for i in range(len(polymer_tokens)):
            if polymer_tokens[i].startswith(tuple(self.polymer_types)):
                j = i+2
                while polymer_tokens[j] != "}":
                    j += 1
                parsed_polymers.append(polymer_tokens[i:j+1])
                
        return parsed_polymers
        
    def parse_bonds(self, bond_part: str) -> list[tuple[str, str, str, str, str, str]]:
        bond_tokens = HELMConverter.split_helm(bond_part)
        parsed_bonds = [] # list[(initial_polymer_name_1: str, initial_monomer_idx_1: str, attachment_label_1: str, initial_polymer_name_2: str, initial_monomer_idx_2: str, attachment_label_2: str)]
        for i in range(len(bond_tokens)):
            if len(bond_tokens) - 1 < i + 10:
                break
            if bond_tokens[i].startswith(tuple(self.polymer_types)) and bond_tokens[i+2].startswith(tuple(self.polymer_types)):
                parsed_bonds.append((bond_tokens[i], bond_tokens[i+4], bond_tokens[i+6], bond_tokens[i+2], bond_tokens[i+8], bond_tokens[i+10]))

        return parsed_bonds
    
    def close_residual_attachment_points(self, mol: Mol) -> Mol:
        remaining = True
        while remaining:
            remaining = False
            for a in mol.GetAtoms():
                if a.HasProp("capGroupSmiles"):
                    remaining = True
                    a.SetAtomMapNum(1)
                    cap_group_smiles = a.GetProp("capGroupSmiles")
                    cap_mol = self.lib.cap_group_mols[cap_group_smiles]
                    mol = Chem.molzip(mol, cap_mol)
                    break
        return mol