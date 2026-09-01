# RGFN reaction data

This directory contains a matched set of two-reactant reaction templates and building blocks for `ForwardReactionTransition`.

- `reactions.smirks`: 66 reaction SMARTS from the `Reactions_NoDocking` sheet of RGFN's `data/chemistry.xlsx` (17 reaction families).
- `building_blocks.smi`: the corresponding `Fragments_NoDocking` building blocks.

RGFN expands the 66 base reactions to 132 anchored templates by assigning each of the two reactant positions to the current molecule. `ForwardReactionTransition` handles both positions dynamically, so the swapped copies are not stored here.

Source: [koziarskilab/RGFN](https://github.com/koziarskilab/RGFN) (MIT License)

Reference: Koziarski et al., “RGFN: Synthesizable Molecular Generation Using
GFlowNets,” *Advances in Neural Information
Processing Systems 37* (2024),
[paper](https://papers.nips.cc/paper_files/paper/2024/hash/53704142f230054140418ecd8857f391-Abstract-Conference.html).
