import random
from rdkit import Chem
from rdkit.Chem import AllChem
from chemtsv3.filter import Filter
from chemtsv3.node import CanonicalSMILESStringNode
from chemtsv3.transition import TemplateTransition


class ForwardReactionTransition(TemplateTransition):
    """Generate products using one- or two-reactant reaction rules."""

    def __init__(self, reaction_templates_path: str, building_blocks_path: str, max_children: int=25, max_expansion_tries: int=250, check_reversibility: bool=False, record_actions: bool=True, filters: list[Filter]=None, top_p=None, logger=None):
        """
        Args:
            reaction_templates_path: Path to a file containing one unary or binary reaction SMARTS/SMIRKS per line. Empty lines and text after ``##`` are ignored.
            building_blocks_path: Path to a SMILES file. The first whitespace-separated field of each line is used.
            max_children: Maximum number of unique child nodes generated during expansion.
            max_expansion_tries: Maximum number of sampled reaction choices tried during expansion.
            check_reversibility: If True, keep a product only when the reverse template recovers the input reactant(s).
            record_actions: If True, the reaction and current-molecule role are recorded in child nodes, together with the selected building block for binary reactions.
        """
        if max_children <= 0:
            raise ValueError("max_children must be greater than 0.")
        if max_expansion_tries <= 0:
            raise ValueError("max_expansion_tries must be greater than 0.")
        self.max_children = max_children
        self.max_expansion_tries = max_expansion_tries
        self.check_reversibility = check_reversibility
        self.record_actions = record_actions
        self.load_reaction_templates(reaction_templates_path)
        self.load_building_blocks(building_blocks_path)
        self.prepare_compatible_building_blocks()
        super().__init__(filters=filters, top_p=top_p, logger=logger)

    def load_reaction_templates(self, path: str):
        self.reaction_templates = []
        with open(path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                smirks = line.split("##", 1)[0].strip()
                if not smirks:
                    continue
                try:
                    reaction = AllChem.ReactionFromSmarts(smirks)
                    reverse_reaction = AllChem.ChemicalReaction()
                    for product_template in reaction.GetProducts():
                        reverse_reaction.AddReactantTemplate(product_template)
                    for reactant_template in reaction.GetReactants():
                        reverse_reaction.AddProductTemplate(reactant_template)
                except Exception as e:
                    raise ValueError(
                        f"Invalid reaction template at line {line_number}: {smirks}"
                    ) from e
                if reaction is None or reverse_reaction is None:
                    raise ValueError(
                        f"Invalid reaction template at line {line_number}: {smirks}"
                    )
                n_reactants = reaction.GetNumReactantTemplates()
                if n_reactants not in (1, 2):
                    raise ValueError(
                        "ForwardReactionTransition supports one- or two-reactant templates; "
                        f"line {line_number} has {n_reactants}: {smirks}"
                    )
                if reaction.GetNumProductTemplates() != 1:
                    raise ValueError(
                        "ForwardReactionTransition supports one-product templates; "
                        f"line {line_number} has {reaction.GetNumProductTemplates()}: {smirks}"
                    )
                self.reaction_templates.append((smirks, reaction, reverse_reaction))

        if not self.reaction_templates:
            raise ValueError(f"No reaction templates found in: {path}")

    def load_building_blocks(self, path: str):
        building_blocks = {}
        with open(path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                smiles = line.split()[0]
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError(f"Invalid building block at line {line_number}: {smiles}")
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                building_blocks[canonical_smiles] = mol

        if not building_blocks:
            raise ValueError(f"No building blocks found in: {path}")
        self.building_blocks = list(building_blocks.items())

    def prepare_compatible_building_blocks(self):
        self.compatible_building_blocks = []
        for _, reaction, _ in self.reaction_templates:
            if reaction.GetNumReactantTemplates() == 1:
                self.compatible_building_blocks.append([])
                continue

            compatible_for_reaction = []
            for reactant_index in range(2):
                pattern = reaction.GetReactantTemplate(reactant_index)
                compatible = [i for i, (_, mol) in enumerate(self.building_blocks) if mol.HasSubstructMatch(pattern)]
                compatible_for_reaction.append(compatible)
            self.compatible_building_blocks.append(compatible_for_reaction)

    @staticmethod
    def try_sanitize(mol):
        try:
            Chem.SanitizeMol(mol)
            return True
        except Exception:
            return False

    @staticmethod
    def reactant_smiles(reactants):
        return sorted(Chem.MolToSmiles(mol, canonical=True) for mol in reactants)

    def is_reversible(self, product, reactants, reverse_reaction):
        expected_reactants = self.reactant_smiles(reactants)
        for previous_reactants in reverse_reaction.RunReactants((product,)):
            if not all(self.try_sanitize(mol) for mol in previous_reactants):
                continue
            if self.reactant_smiles(previous_reactants) == expected_reactants:
                return True
        return False

    def run_reaction(self, reaction, reverse_reaction, reactants):
        products = {}
        for product_tuple in reaction.RunReactants(reactants):
            product = product_tuple[0]
            if not self.try_sanitize(product):
                continue
            if self.check_reversibility and not self.is_reversible(
                    product, reactants, reverse_reaction):
                continue
            smiles = Chem.MolToSmiles(product, canonical=True)
            products[smiles] = product
        return list(products.keys())

    # implement
    def _next_nodes_impl(self, node: CanonicalSMILESStringNode, for_rollout: bool=False) -> list[CanonicalSMILESStringNode]:
        try:
            mol = node.mol(save_cache=False)
            reaction_choices = []
            for reaction_index, (_, reaction, _) in enumerate(self.reaction_templates):
                if reaction.GetNumReactantTemplates() == 1:
                    if mol.HasSubstructMatch(reaction.GetReactantTemplate(0)):
                        reaction_choices.append([reaction_index, 0, [None], 1])
                    continue

                for current_reactant_index in range(2):
                    partner_reactant_index = 1 - current_reactant_index
                    compatible = self.compatible_building_blocks[reaction_index][partner_reactant_index]
                    if compatible and mol.HasSubstructMatch(
                            reaction.GetReactantTemplate(current_reactant_index)):
                        partner_choices = list(compatible)
                        random.shuffle(partner_choices)
                        reaction_choices.append([
                            reaction_index, current_reactant_index, partner_choices,
                            len(compatible)
                        ])

            if not reaction_choices:
                return []

            raw_result = {}
            target_children = 1 if for_rollout else self.max_children
            n_reaction_choices = len(reaction_choices)
            n_tries = 0
            while (len(raw_result) < target_children and reaction_choices and
                   n_tries < self.max_expansion_tries):
                n_tries += 1
                reaction_choice = random.choice(reaction_choices)
                reaction_index, current_reactant_index, partner_choices, n_partners = reaction_choice
                partner_index = partner_choices.pop()
                if not partner_choices:
                    reaction_choices.remove(reaction_choice)

                smirks, reaction, reverse_reaction = self.reaction_templates[reaction_index]
                if partner_index is None:
                    partner_smiles = None
                    reactants = (mol,)
                else:
                    partner_smiles, partner_mol = self.building_blocks[partner_index]
                    if current_reactant_index == 0:
                        reactants = (mol, partner_mol)
                    else:
                        reactants = (partner_mol, mol)
                products = self.run_reaction(reaction, reverse_reaction, reactants)
                if not products:
                    continue

                smiles = random.choice(products)
                if smiles == node.string:
                    continue
                probability = 1 / n_reaction_choices / n_partners / len(products)
                action = None
                if self.record_actions:
                    action = f"{smirks} // current_reactant={current_reactant_index}"
                    if partner_smiles is not None:
                        action += f" // building_block={partner_smiles}"
                if smiles in raw_result:
                    raw_result[smiles][1] += probability
                else:
                    raw_result[smiles] = [action, probability]

            total = sum(probability for _, probability in raw_result.values())
            if total == 0:
                return []
            return [CanonicalSMILESStringNode(string=smiles, parent=node, last_action=action, last_prob=probability / total) for smiles, (action, probability) in raw_result.items()]
        except Exception:
            return []
