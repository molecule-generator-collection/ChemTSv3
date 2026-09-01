# SMIRKS rule sets

The files documented below contain SMIRKS rules for use with `SMIRKSTransition`.

## `gbga.txt`

This rule set is based on the molecular mutation operations from Jan H. Jensen's graph-based genetic algorithm (GB-GA). For an implementation that more closely reproduces the original GB-GA mutation procedure, use `GBGATransition`.

- Original implementation (MIT License): [jensengroup/GB_GA](https://github.com/jensengroup/GB_GA)
- Reference: Jan H. Jensen, “A graph-based genetic algorithm and generative model/Monte Carlo tree search for the exploration of chemical space,” *Chemical Science* **10**, 3567–3572 (2019), [doi:10.1039/C8SC05372C](https://doi.org/10.1039/C8SC05372C).

## `chembl17_ki_mmp_cliff.txt`

This file contains 13,485 molecular transformation rules derived from matched
molecular pairs (MMPs) of activity cliffs in ChEMBL. It is the MMP-based rule
set used for lead optimization in the ChemTSv3 paper.