# MolWalk-SSM

MolWalk-SSM: Random-Walk State Space Modeling with Message Passing for Molecular Property Prediction.

## Datasets 
MoleculeNet datasets:
- Binary classification: BBBP, BACE, HIV
- Multi-task classification: Tox21, SIDER, ClinTox
- Regression: ESOL, FreeSolv, Lipophilicity


## Configuration

Edit config.py to select the dataset, task type, split type, and hyperparameters.

## Training

```bash
python training.py
