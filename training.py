import os
import random
import itertools
import copy
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from model import GraphModel
from dataprocessing import create_dataset
from config import CONFIG

from utils import (
    train_func_binary, test_binary, get_perform_binary,
    train_func_multitask, test_multitask, get_perform_multitask,
    train_func_regre, test_regre, get_perform_regression,
    random_split, random_scaffold_split, make_collate_fn
)
from torch_geometric.data import Batch
from transforms import RandomWalkSampler


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + self.eps)


dataset = CONFIG["dataset"]
task_type = CONFIG["task_type"]
split_type = CONFIG["split_type"]
n_output = CONFIG["n_output"]

batch_sizes = CONFIG["batch_sizes"]
lrs = CONFIG["lrs"]
epochs_list = CONFIG["epochs_list"]
patiences = CONFIG["patiences"]
seeds = CONFIG["seeds"]

walk_lengths = CONFIG.get("walk_lengths", [50])
num_layers_list = CONFIG.get("num_layers_list", [3])
window_size = CONFIG.get("window_size", 8)
sample_rate = CONFIG.get("sample_rate", 1.0)

walk_encoder = CONFIG["walk_encoder"]
models_dir = CONFIG["models_dir"]
results_dir = CONFIG["results_dir"]

tasks = CONFIG.get("tasks", None)
sampling_modes = CONFIG.get("sampling_modes", ["uniform"])

os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Using device:", device)
print(f"Training dataset: {dataset}")
print(f"Task type: {task_type}")
print(f"Split type: {split_type}")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_fixed_walk_dataset(
    dataset_subset,
    walk_seed,
    walk_length,
    sampling_mode,
    ):
    """
    Generate and cache one deterministic walk set for each molecule.
    """
    fixed_rng = np.random.RandomState(walk_seed)

    sampler = RandomWalkSampler(
        length=walk_length,
        sample_rate=sample_rate,
        backtracking=False,
        strict=False,
        pad_idx=-1,
        window_size=window_size,
        sampling_mode=sampling_mode,
        w_conj=CONFIG.get("w_conj", 0.5),
        w_ring=CONFIG.get("w_ring", 0.3),
        rng=fixed_rng,
    )

    fixed_dataset = []

    for index in range(len(dataset_subset)):
        graph = copy.deepcopy(dataset_subset[index])
        graph_with_walks = sampler(graph)
        fixed_dataset.append(graph_with_walks)

    return fixed_dataset

def get_split(data, smiles_list, seed):
    if split_type == "random":
        return random_split(
            data,
            ratio_test=0.1,
            ration_valid=0.1,
            random_seed=seed
        )

    if split_type == "scaffold":
        return random_scaffold_split(
            data,
            smiles_list,
            ratio_test=0.1,
            ration_valid=0.1,
            random_seed=seed
        )

    raise ValueError(f"Unknown split_type: {split_type}")


def get_task_setup():
    if task_type == "binary":
        return {
            "criterion": torch.nn.BCEWithLogitsLoss(),
            "train_func": train_func_binary,
            "test_func": test_binary,
            "get_perform": get_perform_binary,
            "metric_name": "roc",
            "mode": "max",
        }

    if task_type == "multitask":
        return {
            "criterion": torch.nn.BCEWithLogitsLoss(reduction="none"),
            "train_func": train_func_multitask,
            "test_func": test_multitask,
            "get_perform": get_perform_multitask,
            "metric_name": "roc",
            "mode": "max",
        }

    if task_type == "regression":
        return {
            "criterion": RMSELoss(),
            "train_func": train_func_regre,
            "test_func": test_regre,
            "get_perform": get_perform_regression,
            "metric_name": "rmse",
            "mode": "min",
        }

    raise ValueError(f"Unknown task_type: {task_type}")


def is_better(current, best, mode):
    if mode == "max":
        return current >= best
    if mode == "min":
        return current <= best
    raise ValueError(f"Unknown mode: {mode}")


#def run_one_config(batch_size, lr, max_epochs, patience, seed):
def run_one_config(
    batch_size,
    lr,
    max_epochs,
    patience,
    seed,
    walk_length,
    num_layers, sampling_mode="uniform"):

    set_seed(seed)

    # Training walks remain stochastic and are regenerated for each batch.
    train_collate_fn = make_collate_fn(
        walk_length=walk_length,
        sample_rate=sample_rate,
        window_size=window_size,
        sampling_mode=sampling_mode,
        w_conj=CONFIG.get("w_conj", 0.5),
        w_ring=CONFIG.get("w_ring", 0.3),
    )

    data, smiles_list = create_dataset(dataset)
    train_data, valid_data, test_data = get_split(
        data,
        smiles_list,
        seed,
    )

    # Fixed and separate walk seeds for validation and testing.
    valid_walk_seed = 10_000 + seed
    test_walk_seed = 20_000 + seed

    fixed_valid_data = create_fixed_walk_dataset(
        dataset_subset=valid_data,
        walk_seed=valid_walk_seed,
        walk_length=walk_length,
        sampling_mode=sampling_mode,
    )

    fixed_test_data = create_fixed_walk_dataset(
        dataset_subset=test_data,
        walk_seed=test_walk_seed,
        walk_length=walk_length,
        sampling_mode=sampling_mode,
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=0,
    )

    valid_loader = torch.utils.data.DataLoader(
        fixed_valid_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=Batch.from_data_list,
        num_workers=0,
    )

    test_loader = torch.utils.data.DataLoader(
        fixed_test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=Batch.from_data_list,
        num_workers=0,
    )
    model = GraphModel(
    n_output=n_output,
    walk_encoder=walk_encoder,
    num_layers=num_layers,
    walk_length=walk_length,
    window_size=window_size,
    ).to(device)

    trainable_params = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
    )

    trainable_params_m = trainable_params / 1_000_000

    print(
    f"Trainable parameters: {trainable_params:,} "
    f"({trainable_params_m:.3f} M)"
    )
    
    setup = get_task_setup()
    criterion = setup["criterion"]
    train_func = setup["train_func"]
    test_func = setup["test_func"]
    get_perform = setup["get_perform"]
    metric_name = setup["metric_name"]
    mode = setup["mode"]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=1e-6
    )

    best_metric = -float("inf") if mode == "max" else float("inf")
    best_state = None
    best_test_metrics = None
    early_stop = 0
    epoch_train_times = []
    
    for epoch in range(max_epochs):
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        epoch_start_time = time.perf_counter()
    
        if task_type == "multitask":
            train_func(
                epoch, model, optimizer, criterion, train_loader,
                scheduler=None, tasks=tasks, device=device
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            epoch_train_time = time.perf_counter() - epoch_start_time
            
            val_out = test_func(
                epoch, model, criterion, valid_loader,
                tasks=tasks, device=device
            )
            val_metrics = get_perform(val_out[2], val_out[1], tasks)
            current_metric = val_metrics[0]

        else:
            train_func(
                epoch, model, optimizer, criterion, train_loader,
                scheduler=None
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            epoch_train_time = time.perf_counter() - epoch_start_time
            
            val_out = test_func(epoch, model, criterion, valid_loader)

            if task_type == "regression":
                val_metrics = get_perform(val_out[2], val_out[1])
                current_metric = val_metrics[0]
            else:
                val_metrics = get_perform(val_out[2], val_out[1])
                current_metric = val_metrics[0]


        if epoch > 0:
            epoch_train_times.append(epoch_train_time)

        print(
            f"[seed={seed}] Epoch {epoch} "
            f"training time: {epoch_train_time:.3f} s"
        )
    
        if is_better(current_metric, best_metric, mode):
            best_metric = current_metric
            best_state = copy.deepcopy(model.state_dict())
            early_stop = 0

            if task_type == "regression":
                print(
                    f"[seed={seed}] Epoch {epoch} "
                    f"BEST_VAL_RMSE={best_metric:.4f}"
                )
            else:
                print(
                    f"[seed={seed}] Epoch {epoch} "
                    f"BEST_VAL_AUC={best_metric:.4f}"
                )

        else:
            early_stop += 1

            if early_stop >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        scheduler.step()
    
    if epoch_train_times:
        avg_train_time_per_epoch = float(np.mean(epoch_train_times))
        std_train_time_per_epoch = float(np.std(epoch_train_times, ddof=1))
    else:
        avg_train_time_per_epoch = float("nan")
        std_train_time_per_epoch = float("nan")

    print(
    f"[seed={seed}] Average training time per epoch: "
    f"{avg_train_time_per_epoch:.3f} ± "
    f"{std_train_time_per_epoch:.3f} s"
)

    if best_state is None:
        raise RuntimeError(
            f"No valid checkpoint was selected for seed {seed}."
        )

    # Load the checkpoint selected using validation performance.
    model.load_state_dict(best_state)

    # Evaluate the held-out test set exactly once.
    if task_type == "multitask":
        test_out = test_func(
            epoch,
            model,
            criterion,
            test_loader,
            tasks=tasks,
            device=device,
        )
        best_test_metrics = get_perform(
            test_out[2],
            test_out[1],
            tasks,
        )

    else:
        test_out = test_func(
            epoch,
            model,
            criterion,
            test_loader,
        )
        best_test_metrics = get_perform(
            test_out[2],
            test_out[1],
        )

    if task_type == "regression":
        print(
            f"[seed={seed}] "
            f"BEST_VAL_RMSE={best_metric:.4f} "
            f"FINAL_TEST_RMSE={best_test_metrics[0]:.4f} "
            f"MAE={best_test_metrics[1]:.4f} "
            f"R2={best_test_metrics[2]:.4f}"
        )
    else:
        print(
            f"[seed={seed}] "
            f"BEST_VAL_AUC={best_metric:.4f} "
            f"FINAL_TEST_AUC={best_test_metrics[0]:.4f} "
            f"PR_AUC={best_test_metrics[1]:.4f}"
        )

    base_result = {
    "dataset": dataset,
    "task_type": task_type,
    "split_type": split_type,
    "batch_size": batch_size,
    "lr": lr,
    "max_epochs": max_epochs,
    "patience": patience,
    "seed": seed,
    "walk_length": walk_length,
    "num_layers": num_layers,
    "window_size": window_size,
    "sample_rate": sample_rate,
    "sampling_mode": sampling_mode, 
    "valid_walk_seed": valid_walk_seed,
    "test_walk_seed": test_walk_seed,    
    "trainable_params": trainable_params,
    "trainable_params_m": trainable_params_m,
    "avg_train_time_per_epoch": avg_train_time_per_epoch,
    "std_train_time_per_epoch": std_train_time_per_epoch,
    }

    if task_type == "regression":
        base_result.update({
            "best_val_rmse": best_metric,
            "test_rmse": best_test_metrics[0] if best_test_metrics else None,
            "test_mae": best_test_metrics[1] if best_test_metrics else None,
            "test_r2": best_test_metrics[2] if best_test_metrics else None,
        })
    else:
        base_result.update({
            "best_val_roc": best_metric,
            "test_roc": best_test_metrics[0] if best_test_metrics else None,
            "test_prc": best_test_metrics[1] if best_test_metrics else None,
            "test_acc": best_test_metrics[2] if best_test_metrics else None,
            "test_ba": best_test_metrics[3] if best_test_metrics else None,
            "test_mcc": best_test_metrics[4] if best_test_metrics else None,
            "test_ck": best_test_metrics[5] if best_test_metrics else None,
            "test_sens": best_test_metrics[6] if best_test_metrics else None,
            "test_spec": best_test_metrics[7] if best_test_metrics else None,
            "test_prec": best_test_metrics[8] if best_test_metrics else None,
            "test_f1": best_test_metrics[9] if best_test_metrics else None,
        })

    return base_result


def summarize_results(df):
    group_cols = [
    "sampling_mode",
    "walk_length",
    "num_layers",
    "batch_size",
    "lr",
    "max_epochs",
    "patience",
    ]


    if task_type == "regression":
        summary = (
            df.groupby(group_cols)
              .agg(
                  best_val_rmse_mean=("best_val_rmse", "mean"),
                  best_val_rmse_std=("best_val_rmse", "std"),
                  test_rmse_mean=("test_rmse", "mean"),
                  test_rmse_std=("test_rmse", "std"),
                  test_mae_mean=("test_mae", "mean"),
                  test_mae_std=("test_mae", "std"),
                  test_r2_mean=("test_r2", "mean"),
                  test_r2_std=("test_r2", "std"),
                  trainable_params_m=("trainable_params_m", "first"),
                  train_time_mean=("avg_train_time_per_epoch", "mean"),
                  train_time_std=("avg_train_time_per_epoch", "std"),
              )
              .reset_index()
              .sort_values("test_rmse_mean", ascending=True)
        )
    else:
        summary = (
            df.groupby(group_cols)
              .agg(
                  best_val_roc_mean=("best_val_roc", "mean"),
                  best_val_roc_std=("best_val_roc", "std"),
                  test_roc_mean=("test_roc", "mean"),
                  test_roc_std=("test_roc", "std"),
                  test_prc_mean=("test_prc", "mean"),
                  test_prc_std=("test_prc", "std"),
                  test_acc_mean=("test_acc", "mean"),
                  test_acc_std=("test_acc", "std"),
                  test_ba_mean=("test_ba", "mean"),
                  test_ba_std=("test_ba", "std"),
                  test_mcc_mean=("test_mcc", "mean"),
                  test_mcc_std=("test_mcc", "std"),
                  test_f1_mean=("test_f1", "mean"),
                  test_f1_std=("test_f1", "std"),
              )
              .reset_index()
              .sort_values("test_roc_mean", ascending=False)
        )

        for col in ["best_val_roc", "test_roc"]:
            summary[f"{col}_mean_pct"] = summary[f"{col}_mean"] * 100.0
            summary[f"{col}_std_pct"] = summary[f"{col}_std"] * 100.0
            summary[f"{col}_pct_str"] = summary.apply(
                lambda r: f"{r[f'{col}_mean_pct']:.2f} ± {r[f'{col}_std_pct']:.2f}%",
                axis=1
            )

    return summary


def main():
    all_results = []

    for sampling_mode, walk_length, num_layers, bs, lr, max_ep, pat in itertools.product(
        sampling_modes,
        walk_lengths,
        num_layers_list,
        batch_sizes,
        lrs,
        epochs_list,
        patiences):
        print("\n" + "=" * 80)
        print(
            f"CONFIG: T={walk_length} L={num_layers} "
            f"batch={bs} lr={lr} epochs={max_ep} patience={pat}"
        )
        print("=" * 80)

        for seed in seeds:
            result = run_one_config(
                batch_size=bs,
                lr=lr,
                max_epochs=max_ep,
                patience=pat,
                seed=seed,
                walk_length=walk_length,
                num_layers=num_layers,
                sampling_mode=sampling_mode, 
            )
            all_results.append(result)

    df = pd.DataFrame(all_results)

   # file_prefix = f"{dataset}_{task_type}_{split_type}_{walk_encoder}"
    file_prefix = f"{dataset}_{task_type}_{split_type}_{walk_encoder}_walkL_layerAblation"

    raw_path = os.path.join(results_dir, f"grid_search_results_{file_prefix}.csv")
    summary_path = os.path.join(results_dir, f"grid_search_summary_{file_prefix}.csv")

    df.to_csv(raw_path, index=False)

    summary = summarize_results(df)
    summary.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(" -", raw_path)
    print(" -", summary_path)

    if task_type == "regression":
        print("\nTop 5 configs by mean TEST_RMSE:")
    else:
        print("\nTop 5 configs by mean TEST_ROC_AUC:")

    print(summary.head(5).to_string(index=False))


if __name__ == "__main__":
    main()