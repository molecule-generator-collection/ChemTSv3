# SMILES
## smiles_drugs_zinc_1.2m
class: GPT2Transition  
lang: smiles_drugs_zinc.lang  
dataset: 250k_rndm_zinc_drugs_clean.smi  
num_params: 1207552  
n_embd=128, n_layer=6, n_head=4  
test_size=0.1  

# HELM
## pep_noperiod_1.2m
class: GPT2Transition  
lang: helm_pep_noperiod.lang  
dataset: chembl34_protein_helm_only.helm  
has_period = False  
num_params: 1207552  
n_embd=128, n_layer=6, n_head=4  
test_size=0.1  

## pep_period_1.2m
class: GPT2Transition  
lang: helm_pep_period.lang  
dataset: chembl34_protein_helm_only.helm  
has_period = True  
num_params: 1207552  
n_embd=128, n_layer=6, n_head=4  
test_size=0.1  