# MolWalk-SSM

MolWalk-SSM: Random-Walk State Space Modeling with Message Passing for Molecular Property Prediction.

## Datasets 
MoleculeNet datasets:
- Binary classification: BBBP, BACE, HIV
- Multi-task classification: Tox21, SIDER, ClinTox
- Regression: ESOL, FreeSolv, Lipophilicity
  
For access to the curated MoleculeNet datasets used in this work, please contact Dr. Binh P. Nguyen at: binh.nguyen@vuw.ac.nz

## Configuration

Edit config.py to select the dataset, task type, split type, and hyperparameters.

## Training

```bash
python training.py
