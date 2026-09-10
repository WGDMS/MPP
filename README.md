# MolWalk-SSM

**MolWalk-SSM: Chemistry-Aware Random-Walk State Space Modelling with Message Passing for Molecular Property Prediction**

MolWalk-SSM is a molecular property prediction framework that combines chemistry-aware random-walk sequence modelling with local GINE message passing and global virtual-node propagation. The model operates directly on 2D molecular graphs and uses a bidirectional Mamba encoder to capture path-aware structural dependencies along sampled atom-bond walks.

## Datasets 
Experiments are conducted on nine **MoleculeNet** benchmark datasets.

- Binary classification: BBBP, BACE, HIV
- Multi-task classification: Tox21, SIDER, ClinTox
- Regression: ESOL, FreeSolv, Lipophilicity


## Configuration

Edit 'config.py' to specify the dataset, task type, split strategy, random seed, and hyperparameters. The dataset-specific hyperparameter settings are reported in Table S1 of the Supporting Document. Random or scaffold splits are generated automatically according to the 'split' setting in 'config.py'.

## Reproducing the experiments

1. Install the environment using the provided environment file.
2. Use seeds 0, 1, 2, 3, and 4 for the five experimental runs.
3. Generate the random and scaffold splits using the provided splitting procedure.
4. Train MolWalk-SSM using the dataset-specific settings reported in Table S1.
5. During training, random walks are resampled for each batch.
6. For validation and testing, one fixed walk set is generated for each molecule using the corresponding experimental seed (0-4) and reused throughout evaluation.
7. Select checkpoints using validation ROC-AUC for classification and validation RMSE for regression.
8. Evaluate the selected checkpoint once on the corresponding held-out test set.
   
## Training

Run:

```bash
python training.py
