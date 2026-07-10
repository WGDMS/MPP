# MolWalk-SSM

MolWalk-SSM: Random-Walk State Space Modelling with Message Passing for Molecular Property Prediction.

MolWalk-SSM is a molecular property prediction framework that combines chemistry-aware random-walk sequence modelling with local GINE message passing and global virtual-node propagation. The model operates directly on 2D molecular graphs and uses a bidirectional Mamba encoder to capture path-aware structural dependencies along sampled atom–bond walks.

## Datasets 
MoleculeNet datasets:
- Binary classification: BBBP, BACE, HIV
- Multi-task classification: Tox21, SIDER, ClinTox
- Regression: ESOL, FreeSolv, Lipophilicity


## Configuration

Edit 'config.py' to specify: the dataset, task type, split type, and hyperparameters as mentioned in the supporting document.

## Training

Run the following command:

```bash
python training.py
