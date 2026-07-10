# MolWalk-SSM

MolWalk-SSM: Random-Walk State Space Modelling with Message Passing for Molecular Property Prediction.

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
