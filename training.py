import os
import random
import itertools
import copy

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
    random_split, random_scaffold_split, collate
)


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

walk_encoder = CONFIG["walk_encoder"]
models_dir = CONFIG["models_dir"]
results_dir = CONFIG["results_dir"]

tasks = CONFIG.get("tasks", None)

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


def run_one_config(batch_size, lr, max_epochs, patience, seed):
    set_seed(seed)

    data, smiles_list = create_dataset(dataset)
    train_data, valid_data, test_data = get_split(data, smiles_list, seed)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    model = GraphModel(
        n_output=n_output,
        walk_encoder=walk_encoder
    ).to(device)

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

    for epoch in range(max_epochs):

        if task_type == "multitask":
            train_func(
                epoch, model, optimizer, criterion, train_loader,
                scheduler=None, tasks=tasks, device=device
            )
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
            val_out = test_func(epoch, model, criterion, valid_loader)

            if task_type == "regression":
                val_metrics = get_perform(val_out[2], val_out[1])
                current_metric = val_metrics[0]
            else:
                val_metrics = get_perform(val_out[2], val_out[1])
                current_metric = val_metrics[0]

        if is_better(current_metric, best_metric, mode):
            best_metric = current_metric
            best_state = copy.deepcopy(model.state_dict())
            early_stop = 0

            if task_type == "multitask":
                test_out = test_func(
                    epoch, model, criterion, test_loader,
                    tasks=tasks, device=device
                )
                best_test_metrics = get_perform(test_out[2], test_out[1], tasks)

            else:
                test_out = test_func(epoch, model, criterion, test_loader)
                best_test_metrics = get_perform(test_out[2], test_out[1])

            if task_type == "regression":
                print(
                    f"[seed={seed}] Epoch {epoch} "
                    f"VAL_RMSE={best_metric:.4f} "
                    f"TEST_RMSE={best_test_metrics[0]:.4f} "
                    f"MAE={best_test_metrics[1]:.4f} "
                    f"R2={best_test_metrics[2]:.4f}"
                )
            else:
                print(
                    f"[seed={seed}] Epoch {epoch} "
                    f"VAL_AUC={best_metric:.4f} "
                    f"TEST_AUC={best_test_metrics[0]:.4f} "
                    f"PR_AUC={best_test_metrics[1]:.4f}"
                )

        else:
            early_stop += 1
            if early_stop >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    base_result = {
        "dataset": dataset,
        "task_type": task_type,
        "split_type": split_type,
        "batch_size": batch_size,
        "lr": lr,
        "max_epochs": max_epochs,
        "patience": patience,
        "seed": seed,
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
    group_cols = ["batch_size", "lr", "max_epochs", "patience"]

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

    for bs, lr, max_ep, pat in itertools.product(
        batch_sizes, lrs, epochs_list, patiences
    ):
        print("\n" + "=" * 80)
        print(f"CONFIG: batch={bs} lr={lr} epochs={max_ep} patience={pat}")
        print("=" * 80)

        for seed in seeds:
            result = run_one_config(bs, lr, max_ep, pat, seed)
            all_results.append(result)

    df = pd.DataFrame(all_results)

    file_prefix = f"{dataset}_{task_type}_{split_type}_{walk_encoder}"

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