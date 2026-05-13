# MPP
MolWalk-SSM: Random-Walk State Space Modeling with Message Passing for Molecular Property Prediction

Dataset

Training

python training.py

# MolWalk-SSM

MolWalk-SSM: Random-Walk State Space Modeling with Message Passing for Molecular Property Prediction.

## Overview

MolWalk-SSM represents molecular graphs using sampled random walks and encodes these walk sequences using a state space model. The learned walk representations are aggregated back to molecular graph nodes and combined with local message passing and global virtual-node propagation for molecular property prediction.

## Datasets

The code supports MoleculeNet-style datasets:

- Binary classification: BBBP, BACE, HIV
- Multi-task classification: Tox21, SIDER, ClinTox
- Regression: ESOL, FreeSolv, Lipophilicity

Place datasets in the following format:

```text
data/
  BBBP/
    refined_BBBP.csv
  Tox21/
    refined_Tox21.csv
