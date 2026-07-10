import os
from torch_geometric.data import InMemoryDataset, DataLoader, Batch, Data
from torch_geometric import data as DATA
import torch
import numpy as np
import subprocess
from math import sqrt
from sklearn.metrics import average_precision_score
from scipy import stats
#from munch import Munch
import pandas as pd 
import time
from tqdm import tqdm
from torch.utils.data import Subset
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix, precision_recall_curve
from sklearn.metrics import roc_auc_score, cohen_kappa_score, balanced_accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from transforms import RandomWalkSampler
from torch_geometric.data import Batch
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold

WALK_PAD_IDX = -1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def make_collate_fn(walk_length=50, sample_rate=1.0, window_size=8,
                    backtracking=False, strict=False, pad_idx=WALK_PAD_IDX,
                    sampling_mode="uniform", w_conj=0.5, w_ring=0.0):  # NEW
    walk_sampler = RandomWalkSampler(
        length=walk_length, sample_rate=sample_rate,
        backtracking=backtracking, strict=strict,
        pad_idx=pad_idx, window_size=window_size,
        sampling_mode=sampling_mode, w_conj=w_conj, w_ring=w_ring,  # NEW
    )
    def collate(data_list):
        return Batch.from_data_list([walk_sampler(m) for m in data_list])
    return collate

class MolDataset(InMemoryDataset):
    def __init__(self, root='data', dataset='ESOL', y=None,
                 transform=None, pre_transform=None,
                 mol_graph=None, drug_key=None, pre_filter=None):

        self.dataset = dataset
        self.mol_graph = mol_graph
        self.drug_key = drug_key
        self.y = y

        super().__init__(root, transform, pre_transform, pre_filter)

        processed_path = self.processed_paths[0]
        if os.path.exists(processed_path):
           self.data_mol = torch.load(processed_path, weights_only=False)

        else:
            self.process()  # calls the no-arg version below
            torch.save((self.data_mol), processed_path)

    @property
    def raw_file_names(self):
       
        return []

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt']

    def process(self):  
        drug_key = self.drug_key
        y = self.y
        mol_graph = self.mol_graph

        assert drug_key is not None and y is not None and mol_graph is not None
        assert len(drug_key) == len(y)

        data_list_mol = []
 

        for i in range(len(drug_key)):
            d_key = drug_key[i]
            if d_key not in mol_graph and str(d_key) in mol_graph:
                d_key = str(d_key)
            if d_key not in mol_graph:
                raise KeyError(f"drug_key[{i}]={drug_key[i]} not found in mol_graph keys.")

            #label = float(y[i])
            label = torch.tensor(y[i], dtype=torch.float).view(1, -1)
            
            (atom_features, bond_list, bond_features) = mol_graph[d_key]

            # ---- molecule graph ----
            x = torch.tensor(atom_features, dtype=torch.float)
            if len(bond_list) == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 12), dtype=torch.float)
            else:
                edge_index = torch.tensor(bond_list, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(bond_features, dtype=torch.float).view(-1, 12)

           # mol_data = Data(
            #    x=x,
            #    edge_index=edge_index,
            #    edge_attr=edge_attr,
            #    y=torch.tensor([label], dtype=torch.float)
          #  )
     
            mol_data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=label.clone())
            
            data_list_mol.append(mol_data)


        self.data_mol = data_list_mol
      

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx]


#binary classification

def train_func_binary(epoch, model, optimizer, criterion, train_loader, scheduler=None,
                      device="cuda", min_batch_size=2, grad_clip=None):
    """
    Binary classification training loop.
    
    """
    model.train()
    start_time = time.time()

    total_loss = 0.0
    total_samples = 0

    for batch_idx, data in enumerate(train_loader):
        data_mol = data.to(device)


        # labels: [-1,1] -> [0,1]
        labels = data_mol.y.view(-1, 1).float()
        if labels.min().item() < 0:   # handles {-1,1}
            labels = (labels + 1.0) / 2.0

        # move labels to device
        labels = labels.to(device)

        # optional: skip tiny batches (e.g., last batch)
        if labels.size(0) < min_batch_size:
            continue

        optimizer.zero_grad(set_to_none=True)

        logits = model(data_mol)
        if logits.dim() == 1:
            logits = logits.view(-1, 1)

        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs   # weighted by batch size
        total_samples += bs

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / max(total_samples, 1)
    print(f"====> Epoch: {epoch}, training time {time.time()-start_time:.2f}s, "
          f"Average Train Loss: {avg_loss:.4f}")

    return avg_loss


def test_binary(epoch, model, criterion, test_loader, device="cuda"):
    """
    Binary classification eval loop.
  
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            data_mol = data.to(device)
       

            labels = data_mol.y.view(-1, 1).float()
            if labels.min().item() < 0:
                labels = (labels + 1.0) / 2.0
            labels = labels.to(device)

            logits = model(data_mol)
            if logits.dim() == 1:
                logits = logits.view(-1, 1)

            loss = criterion(logits, labels)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            probs = torch.sigmoid(logits)

            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)
    probs = np.concatenate(all_probs, axis=0).reshape(-1)
    labels_np = np.concatenate(all_labels, axis=0).reshape(-1)

    return avg_loss, probs, labels_np
    

   #regression
   
def train_func_regre(epoch, model, optimizer, criterion, train_loader, scheduler):
    model.train()
    train_loss = 0
    start_time = time.time()

    for batch_idx, data in enumerate(train_loader):
        data_mol = data.to(device)
      

        optimizer.zero_grad()
        output = model(data_mol)
        loss = criterion(output, data_mol.y.view(-1, 1).float())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print('====> Epoch: {}, training time {},  Average Train Loss: {:.4f}'.format(
        epoch, time.time() - start_time, train_loss / len(train_loader)
    ))
    train_loss = train_loss / len(train_loader.dataset)
    return train_loss


# predict
def test_regre(current_iter, model, criterion, test_loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(test_loader.dataset)))
    with torch.no_grad():
        for data in test_loader:
            data_mol = data.to(device)
          
            output = model(data_mol)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
            
    #test_rmse = rmse(np.array(pred), np.array(labels_))
    test_results = get_perform_regression(total_labels.numpy().flatten(), total_preds.numpy().flatten())
    test_rmse= test_results[0]
    # print("Performance of model at epoch {} on test dataset:  {}".format(current_iter, test_rmse))
    # print('====> Epoch: {} Average Test Loss: {:.4f}'.format(current_iter, test_rmse))
    return test_rmse, total_preds.numpy().flatten(), total_labels.numpy().flatten()
    
# defined the required metrics 

#multitask
def train_func_multitask(epoch, model, optimizer, criterion, train_loader,
                         scheduler=None, tasks=None, device="cuda",
                         min_batch_size=2, grad_clip=None):
    model.train()
    start_time = time.time()

    total_loss = 0.0
    total_samples = 0
    num_tasks = len(tasks)

    for batch_idx, data in enumerate(train_loader):
        data_mol = data.to(device)

        labels = data_mol.y.float().to(device).view(-1, num_tasks)

        if labels.size(0) < min_batch_size:
            continue

        optimizer.zero_grad(set_to_none=True)

        outputs = model(data_mol)
        loss = get_loss_multitask(criterion, outputs, labels, device=device)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    avg_loss = total_loss / max(total_samples, 1)

    print(f"====> Epoch: {epoch}, training time {time.time()-start_time:.2f}s, "
          f"Average Train Loss: {avg_loss:.4f}")

    return avg_loss

def test_multitask(epoch, model, criterion, test_loader, tasks=None, device="cuda", use_test=True):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    num_tasks = len(tasks)

    preds = [[] for _ in range(num_tasks)]
    labels_all = [[] for _ in range(num_tasks)]

    with torch.no_grad():
        for data in test_loader:
            data_mol = data.to(device)
          
            labels = data_mol.y.float().to(device).view(-1, num_tasks)

            outputs = model(data_mol)
            loss = get_loss_multitask(criterion, outputs, labels, device=device)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            pred_list, label_list = get_prob_multitask(outputs, labels, device=device)

            for i in range(num_tasks):
                if len(label_list[i]) > 0:
                    preds[i].extend(pred_list[i])
                    labels_all[i].extend(label_list[i])

    avg_loss = total_loss / max(total_samples, 1)

    if use_test:
        valid_tasks = sum(1 for i in range(num_tasks) if len(labels_all[i]) > 0)
        assert valid_tasks > 0, "No valid task labels found in evaluation."

    return avg_loss, preds, labels_all
    
def random_split(dataset, random_seed=8, ratio_test=0.1, ration_valid=0.1):
    print('Random split ...........')
    indices = list(range(len(dataset)))

    idx_train_val, idx_test = train_test_split(
        indices, test_size=ratio_test, random_state=random_seed
    )
    idx_train, idx_val = train_test_split(
        idx_train_val, test_size=ration_valid, random_state=random_seed
    )

    assert len(idx_train) + len(idx_val) + len(idx_test) == len(indices)
    print(f'Num train: {len(idx_train)}, Num val {len(idx_val)}, Num test {len(idx_test)}')

    train_dataset = Subset(dataset, idx_train)
    valid_dataset = Subset(dataset, idx_val)
    test_dataset  = Subset(dataset, idx_test)

    return train_dataset, valid_dataset, test_dataset

def generate_scaffold(smiles, include_chirality=False):
    """
    Bemis-Murcko scaffold from smiles

    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold
    
def random_scaffold_split(dataset, smiles_list, random_seed= 8, ratio_test= 0.1, ration_valid= 0.1):
    print('Random scaffold split ...........')
    rng = np.random.RandomState(random_seed)
    scaffolds = defaultdict(list)
    for ind, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)
    idxs= list(scaffolds.keys())
    idxs = rng.permutation(idxs)
    scaffold_sets = [scaffolds[idx] for idx in idxs]

    n_total_valid = int(ration_valid * len(dataset) * (1-ratio_test))
    n_total_test = int(ratio_test * len(dataset))
    print('Num train: {}, Num val {}, Num test {}'.format(len(smiles_list)-n_total_test-n_total_valid, n_total_valid, n_total_test))
    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        elif len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0
    assert len(set(train_idx)) + len(set(test_idx))+ len(set(valid_idx)) == len(smiles_list), 'total not match'

    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    test_dataset  = Subset(dataset, test_idx)
    
    return train_dataset, valid_dataset, test_dataset


def get_perform_binary(labels, probs, task=None):
    trn_roc = roc_auc_score(labels, probs)
    trn_prc = metrics.auc(precision_recall_curve(labels, probs)[1],
                        precision_recall_curve(labels, probs)[0])
    predicted_labels = []
    for prob in probs: 
        predicted_labels.append(np.round(prob))

    trn_acc = accuracy_score(labels, predicted_labels)
    trn_ba  = balanced_accuracy_score(labels, predicted_labels)
    trn_mcc = matthews_corrcoef(labels, predicted_labels)
    trn_ck  = cohen_kappa_score(labels, predicted_labels)
    
    tn, fp, fn, tp = confusion_matrix(labels, predicted_labels).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision   = tp / (tp + fp)
    f1 = 2*precision*sensitivity / (precision + sensitivity)
   
    perform = [trn_roc, trn_prc, trn_acc, trn_ba, trn_mcc, trn_ck, sensitivity, specificity, precision, f1]
    # print(f"AUC= {trn_roc} , PR_AUC={trn_prc}")
    return perform

    
def get_perform_regression(labels, probs, task=None):
    actual_arr = np.array(labels)
    predicted_arr = np.array(probs)
    # rmse = np.sqrt(np.mean((actual_arr - predicted_arr)**2))
    rmse = np.sqrt(mean_squared_error(actual_arr, predicted_arr))
    r2= r2_score(actual_arr, predicted_arr)
    mae = mean_absolute_error(actual_arr, predicted_arr)
    # print("RMSE:", rmse)
    return rmse, mae, r2

def get_perform_multitask(labels_list, probs_list, tasks=None, return_each_task=False):
    
    per_task_metrics = []
    valid_task_names = []

    for i in range(len(labels_list)):
        labels = labels_list[i]
        probs = probs_list[i]

        if len(labels) == 0 or len(probs) == 0:
            continue

        # skip degenerate cases with only one class present
        uniq = np.unique(labels)
        if len(uniq) < 2:
            print(f"Skipping task {tasks[i] if tasks else i} because only one class is present.")
            continue

        metrics_i = get_perform_binary(labels, probs)
        per_task_metrics.append(metrics_i)
        valid_task_names.append(tasks[i] if tasks else str(i))

    if len(per_task_metrics) == 0:
        mean_metrics = [np.nan] * 10
        return (mean_metrics, {}) if return_each_task else mean_metrics

    per_task_metrics = np.array(per_task_metrics, dtype=float)
    mean_metrics = per_task_metrics.mean(axis=0).tolist()

    if return_each_task:
        per_task_dict = {
            valid_task_names[i]: per_task_metrics[i].tolist()
            for i in range(len(valid_task_names))
        }
        return mean_metrics, per_task_dict

    return mean_metrics
    

def get_loss_multitask(criterion, outputs, labels, device="cuda"):
    if isinstance(outputs, (list, tuple)):
        num_tasks = len(outputs)
    else:
        num_tasks = outputs.size(1)

    total_loss = torch.tensor(0.0, device=device)
    valid_task_count = 0

    for t in range(num_tasks):
        if isinstance(outputs, (list, tuple)):
            y_pred = outputs[t].view(-1)
        else:
            y_pred = outputs[:, t].view(-1)

        y_true = labels[:, t].view(-1).float()

        valid_mask = (y_true == -1) | (y_true == 1) | (y_true == 0)

        if valid_mask.sum().item() == 0:
            continue

        y_pred = y_pred[valid_mask]
        y_true = y_true[valid_mask]

        if y_true.min().item() < 0:
            y_true = (y_true + 1.0) / 2.0

        loss = criterion(y_pred, y_true).mean()

        total_loss = total_loss + loss
        valid_task_count += 1

    if valid_task_count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / valid_task_count


def get_prob_multitask(outputs, labels, device="cuda"):
    
    if isinstance(outputs, (list, tuple)):
        num_tasks = len(outputs)
    else:
        num_tasks = outputs.size(1)

    prob_list = [[] for _ in range(num_tasks)]
    label_list = [[] for _ in range(num_tasks)]

    for t in range(num_tasks):
        if isinstance(outputs, (list, tuple)):
            logits_t = outputs[t].view(-1)
        else:
            logits_t = outputs[:, t].view(-1)

        y_true = labels[:, t].view(-1).float()

        valid_mask = (y_true == -1) | (y_true == 1) | (y_true == 0)

        if valid_mask.sum().item() == 0:
            continue

        logits_t = logits_t[valid_mask]
        y_true = y_true[valid_mask]

        if y_true.min().item() < 0:
            y_true = (y_true + 1.0) / 2.0

        probs_t = torch.sigmoid(logits_t)

        prob_list[t].extend(probs_t.detach().cpu().numpy().tolist())
        label_list[t].extend(y_true.detach().cpu().numpy().tolist())

    return prob_list, label_list   
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
   
