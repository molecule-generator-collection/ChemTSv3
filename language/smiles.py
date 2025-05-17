from language import DynamicMolLanguage
import re
from rdkit import Chem
from rdkit.Chem import Mol
import pickle
from typing import Self

class SMILES(DynamicMolLanguage):
  #implement
  def sentence2tokens(self, sentence):
    pass

  #implement
  @staticmethod
  def sentence2mol(sentence: str) -> Mol:
    return Chem.MolFromSmiles(sentence)