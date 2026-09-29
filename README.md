# MolWalk-SSM

**MolWalk-SSM: Chemistry-Aware Random-Walk State Space Modelling with Message Passing for Molecular Property Prediction**

MolWalk-SSM is a molecular property prediction framework that combines chemistry-aware random-walk sequence modelling with local GINE message passing and global virtual-node propagation. The model operates directly on 2D molecular graphs and uses a bidirectional Mamba encoder to capture path-aware structural dependencies along sampled atom-bond walks.

## Datasets

Experiments were conducted on nine **MoleculeNet** benchmark datasets.

- Binary classification: BACE, BBBP, HIV
- Multi-task classification: Tox21, SIDER, ClinTox
- Regression: ESOL, FreeSolv, Lipophilicity

## Installation

Create and activate a Python 3.10 environment:

```bash
conda create -n molwalk_mamba python=3.10
conda activate molwalk_mamba
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The experiments reported in the paper were conducted using:

- Python 3.10
- PyTorch 2.2.2 with CUDA 12.1
- PyTorch Geometric 2.8.0
- RDKit 2022.09.5
- Mamba-SSM 2.2.2

## Configuration

Edit `config.py` to specify the dataset, task type, split strategy, random seeds, and hyperparameters.

The main configuration options include:

- `dataset`
- `task_type`
- `split_type`
- `batch_sizes`
- `lrs`
- `epochs_list`
- `patiences`
- `seeds`
- `walk_lengths`
- `num_layers_list`
- `sampling_modes`

The hyperparameter search space and dataset-specific settings used in the reported experiments are described in Section 1.1 and Table S1 of the Supporting Information.

Random and scaffold splits are generated automatically according to the `split_type` setting in `config.py`.

## Reproducing the Experiments

The reported experiments use five predefined experimental seeds:

```text
0, 1, 2, 3, 4
```

For each experimental seed, the corresponding train, validation, and test partitions are generated using either random or scaffold splitting.

During training, random walks are resampled for each batch.

For validation and testing, deterministic fixed walk sets are generated and reused throughout evaluation. Separate walk seeds are used for validation and testing:

```text
Validation walk seed = 10000 + experimental seed
Test walk seed       = 20000 + experimental seed
```

For example, for experimental seed `0`, the validation walk seed is `10000` and the test walk seed is `20000`.

Checkpoint selection is based exclusively on validation performance:

- ROC-AUC is maximised for classification tasks.
- RMSE is minimised for regression tasks.

For each run, the checkpoint with the best validation performance is selected and evaluated once on the corresponding held-out test set. Test performance is not used for checkpoint selection.

The final reported results are calculated across the five predefined experimental runs.

## Training

After specifying the required dataset and experimental settings in `config.py`, run:

```bash
python training.py
```

The script runs the configurations and seeds defined in `config.py` and stores the resulting outputs in the configured results directory.

## Reproducibility

The repository provides the source code, dataset-processing procedures, configuration settings, random seeds, fixed evaluation-walk procedure, and dependency information required to reproduce the MolWalk-SSM experiments.

For the experiments reported in the paper, the five experimental seeds are `0`, `1`, `2`, `3`, and `4`. Training walks are sampled dynamically, whereas validation and test walks are fixed for each experimental run to ensure deterministic evaluation.

## Citation

If you use this code in your research, please cite the MolWalk-SSM paper:

```text
W. G. D. M. Samankula, J. Peng, and B. P. Nguyen,
"MolWalk-SSM: Chemistry-Aware Random-Walk State Space Modelling
with Message Passing for Molecular Property Prediction."
```

## Authors

- W. G. D. M. Samankula
- Jiajie Peng
- Binh P. Nguyen
