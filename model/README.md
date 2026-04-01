Detailed model parameters and the training settings for autoregressive models can be found in `config.json` and `setting.yaml` of the corresponding directory.

# SMILES autoregressive models
## smiles/drugs_zinc/~
- Trained with: `250k_rndm_zinc_drugs_clean.smi` (https://github.com/molecule-generator-collection/ChemTSv3/tree/main/data/smiles)
- `gpt2_0.2m`: GPT-2 model with 0.2m parameters
- `gpt2_0.5m`: GPT-2 model with 0.5m parameters
- `gpt2_1.2m`: GPT-2 model with 1.2m parameters
- `gru`: GRU model
- `lstm`: LSTM model
- `tf25_ported`: GRU model, ported to PyTorch from ChemTSv2's Tensorflow model: https://github.com/molecule-generator-collection/ChemTSv2/tree/master/model

## smiles/chembl/~
- Trained with: `ChEMBL_220K.smi` (https://github.com/molecule-generator-collection/ChemTSv3/tree/main/data/smiles)
- `gru`: GRU model

## smiles/pubchem_qc/~
- Trained with: `2019PubChemQC_can_nocharge.smi` (https://github.com/molecule-generator-collection/ChemTSv3/tree/main/data/smiles)
- `gru`: GRU model

# SELFIES autoregressive models
## selfies/drugs_zinc/~
- Trained with: `250k_rndm_zinc_drugs_clean.smi` (https://github.com/molecule-generator-collection/ChemTSv3/tree/main/data/smiles)
- `gru`: GRU model

# HELM autoregressive models
## helm/chembl_peptide/~
- Trained with the dataset curated from the ChEMBL34 HELM dataset (https://github.com/molecule-generator-collection/ChemTSv3/tree/main/data/helm)
- Original dataset source: https://chembl.blogspot.com/2024/04/chembl-34-is-out.html
- `gru`: GRU model

# FASTA autoregressive models
## fasta/gfp
- LSTM model trained with the GFP dataset (https://github.com/molecule-generator-collection/ChemTSv3/tree/main/data/fasta)
- Original dataset (paper): https://www.nature.com/articles/nature17995

# Reward models

## reward/dyramo_lgb_models.json, reward/dyramo_lgb_models_wo_approved_v1.json
- DyRAMO reward model (LightGBM) 
- Trained with / without approved drugs respectively
- Ported to json for the compatibility with newer versions of LightGBM
- Original (paper): https://www.nature.com/articles/s41467-025-57582-3
- Original (source): https://github.com/ycu-iil/DyRAMO/tree/main/data

## reward/d_score_lgb_models.json
- DScore reward model (LightGBM)
- Original (paper): https://pubs.acs.org/doi/full/10.1021/acs.jcim.2c00787
- Original (source): https://github.com/molecule-generator-collection/ChemTSv2/tree/master/data/model
