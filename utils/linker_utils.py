import re
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from vina import Vina
from rdkit.Geometry import Point3D
from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule

def calc_MorganCount(mol, r=2, dimension=500):
    info = {}
    _fp = AllChem.GetMorganFingerprint(mol, r, bitInfo=info)
    count_list = [0] * dimension
    for key in info:
        pos = key % dimension
        count_list[pos] += len(info[key])
    return count_list

def add_atom_index_in_wildcard(smiles: str):
    c = iter(range(1, smiles.count('*')+1))
    labeled_smiles = re.sub(r'\*', lambda _: f'[*:{next(c)}]', smiles)
    return labeled_smiles

def link_linker(cores, linker, linker_type="mol", output_type="mol"):
    if linker_type == "mol":
        smi = Chem.MolToSmiles(linker)
    elif linker_type == "smiles":
        smi = linker
    mol_ = Chem.MolFromSmiles(add_atom_index_in_wildcard(smi))
    rwmol = Chem.RWMol(mol_)
    if type(cores) is list:
        cores_mol = [Chem.MolFromSmiles(s) for s in cores]
    else:
        cores_mol = [Chem.MolFromSmiles(s) for s in [cores['ligand_1'], cores['ligand_2']]]
    for m in cores_mol:
        rwmol.InsertMol(m)
    prod = Chem.MolToSmiles(rwmol)
    prod = Chem.molzip(rwmol)
    if output_type == "smiles":
        return Chem.MolToSmiles(prod)
    return prod

def calc_vina_score(mol, vina_sf_name, vina_cpu, vina_verbosity, vina_pdbqt_file_name, vina_center, vina_box_size, vina_spacing, vina_exhaustiveness, vina_n_poses, vina_min_rmsd, vina_max_evals, output_dir, count):
    try:
        v = Vina(sf_name=vina_sf_name, cpu=vina_cpu, verbosity=vina_verbosity)
        v.set_receptor(rigid_pdbqt_filename=vina_pdbqt_file_name)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        mol_conf = mol.GetConformer(-1)
        centroid = list(rdMolTransforms.ComputeCentroid(mol_conf))
        tr = [vina_center[i] - centroid[i] for i in range(3)]
        for i, p in enumerate(mol_conf.GetPositions()):
            mol_conf.SetAtomPosition(i, Point3D(p[0] + tr[0], p[1] + tr[1], p[2] + tr[2]))
        mol_prep = MoleculePreparation()
        setups = mol_prep.prepare(mol)                  
        ms = setups[0]
        writer = PDBQTWriterLegacy()                            
        mol_pdbqt, ok, msg = writer.write_string(ms)
        if not ok:
            print(f"PDBQTWriterLegacy failed: {msg}")
            return 0
        if mol_pdbqt is None or not mol_pdbqt.strip():
            print("mol_pdbqt is empty")
            return 0
        v.set_ligand_from_string(mol_pdbqt)
        v.compute_vina_maps(center=vina_center,
                            box_size=vina_box_size,
                            spacing=vina_spacing)
        _ = v.optimize()
        v.dock(exhaustiveness=vina_exhaustiveness,
            n_poses=vina_n_poses,
            min_rmsd=vina_min_rmsd,
            max_evals=vina_max_evals)
        scores = v.energies()
        min_inter_score = 1000
        best_model = 1
        for m, ene in enumerate(scores):
            if ene[0] < min_inter_score:
                min_inter_score = ene[0]
                best_model = m
        pose_dir = output_dir+"/vina/3Dpose_"+count
        if not os.path.exists(pose_dir):
            os.makedirs(pose_dir)

        v.write_poses(f"{pose_dir}/vina_temp_out.pdbqt",
                    n_poses=vina_n_poses,
                    overwrite=True)
        pdbqt_mol = PDBQTMolecule.from_file(f"{pose_dir}/vina_temp_out.pdbqt", skip_typing=True)
        for pose in pdbqt_mol:
            if pose.pose_id == best_model:
                pose.write_pdbqt_file(f"{pose_dir}/vina_best_model.pdbqt")
    except Exception as e:
        print(f"Error SMILES: {Chem.MolToSmiles(mol)}")
        print(e)
        return 0
    return min_inter_score
