import pickle
import re
from typing import Self
from rdkit import Chem
from rdkit.Chem import Mol
from language import DynamicMolLanguage

class SMILES(DynamicMolLanguage):
  #implement
  def sentence2tokens(self, sentence):
    pass

  #implement
  @staticmethod
  def sentence2mol(sentence: str) -> Mol:
    return Chem.MolFromSmiles(sentence)