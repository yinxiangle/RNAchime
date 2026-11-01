import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import Subset
import dgl
import networkx as nx


def dgl_to_nx(g_dgl, edge_map):
    hot_to_label = {v:k for k,v in edge_map.items()}
    print(hot_to_label)
    hots = g_dgl.edata['one_hot'].detach().numpy()
    G = dgl.to_networkx(g_dgl) 
    labels = {e:hot_to_label[i] for i,e in zip(hots, G.edges())}
    print(labels)
    nx.set_edge_attributes(G,labels, 'label')
    print(G.nodes())
    G = nx.to_undirected(G)
    return G


def makedir(path, permissive=True):
    try:
        os.makedirs(path)
    except FileExistsError:
        if not permissive:
            raise ValueError("This name is already taken !")


def mkdirs(dir, name1, name2, permissive=True):
    """
    Try to make the logs folder
    :param name:
    :param permissive: If True will overwrite existing files (good for debugging)
    :return:
    """
    results_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    save_path = os.path.join(results_root, dir, name1, name2)
    try:
        os.makedirs(save_path)
    except FileExistsError:
        if not permissive:
            raise ValueError('This name is already taken !')
    save_name = os.path.join(save_path, name2 + '.pth')
    return save_path, save_name


def get_external_split_data(loader, split_dir, k_folds=5):
    dataset = loader.dataset
    graph_to_idx = {name: idx for idx, name in enumerate(dataset.all_graphs)}

    if os.path.isfile(split_dir):
        split_paths = [split_dir]
    else:
        split_paths = [
            os.path.join(split_dir, f"fold_{fold_id}.p")
            for fold_id in range(1, k_folds + 1)
        ]

    for fold_id, split_path in enumerate(split_paths, start=1):
        with open(split_path, "rb") as f:
            split = pickle.load(f)

        train_names = [str(x) for x in split["train"]]
        test_names = [str(x) for x in split["test"]]

        missing_train = [name for name in train_names if name not in graph_to_idx]
        missing_test = [name for name in test_names if name not in graph_to_idx]

        if missing_train or missing_test:
            raise ValueError(
                f"Split file {split_path} contains graph names not found in annotated_data. "
                f"Missing train examples: {missing_train[:5]}, "
                f"missing test examples: {missing_test[:5]}"
            )

        train_indices = [graph_to_idx[name] for name in train_names]
        test_indices = [graph_to_idx[name] for name in test_names]

        train_set = Subset(dataset, train_indices)
        test_set = Subset(dataset, test_indices)

        train_loader = loader.make_loader(train_set, shuffle=True)
        test_loader = loader.make_loader(test_set)

        print(f">>> external split {fold_id}: train={len(train_set)}, test={len(test_set)}")

        yield train_loader, test_loader


def get_hyperparameter_summary(args, loader, model, device, used_gpus_count, run_time):

    summary_lines = []
    summary_lines.append("=====================================")
    summary_lines.append("RNAchime Training Run Summary")
    summary_lines.append(f"Timestamp: {run_time}")
    summary_lines.append("=====================================\n")
    summary_lines.append("---------- Command-Line Arguments ----------")
    for key, value in vars(args).items():
        summary_lines.append(f"{key:<20}: {value}")
    summary_lines.append("")
    summary_lines.append("---------- Hardware & Data ----------")
    summary_lines.append(f"{'GPUs Used':<20}: {used_gpus_count}")
    summary_lines.append(f"{'Device':<20}: {device}")
    summary_lines.append(f"{'Num Edge Types':<20}: {loader.num_edge_types}")
    summary_lines.append("")
    summary_lines.append("---------- Model Architecture ----------")
    summary_lines.append(f"{'Model Class':<20}: {model.__class__.__name__}")
    summary_lines.append(f"{'Criterion':<20}: {model.criterion}")
    summary_lines.append("")

    return "\n".join(summary_lines)


def save_hyperparameter_summary(args, loader, model, device, used_gpus_count, result_folder, fold, run_time):
    try:
        summary_text = get_hyperparameter_summary(args, loader, model, device, used_gpus_count, run_time)
        summary_file_path = os.path.join(result_folder, f'hyperparameters_{fold}_{run_time}.txt')
        with open(summary_file_path, 'w') as f:
            f.write(summary_text)
        print(f"Hyperparameter summary saved to: {summary_file_path}")
    except Exception as e:
        print(f"Error saving hyperparameter summary: {e}")


def plot_loss_curve(log_file_path, save_path, run_time):
    try:
        log_df = pd.read_csv(log_file_path)
        plot_file_path = os.path.join(os.path.dirname(save_path), f"loss_curve_{run_time}.png")
        plt.figure(figsize=(12, 7))
        plt.plot(log_df['epoch'], log_df['train_loss'], label='Train Loss')
        plt.plot(log_df['epoch'], log_df['test_loss'], label='Test Loss')
        plt.title('Training and Test Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xlim(1, log_df['epoch'].max()+1)
        plt.ylim(0, max(log_df['train_loss'].max(), log_df['test_loss'].max()) * 1.1)
        plt.legend(loc = 'best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_file_path)
        plt.close()
    except Exception as e:
        print(f"Unable to generate loss curve graph: {e}")


def plot_auc_curve(log_file_path, save_path, run_time):
    try:
        log_df = pd.read_csv(log_file_path)
        auc_plot_path = os.path.join(os.path.dirname(save_path), f"auc_curve_{run_time}.png")
        plt.figure(figsize=(12, 7))
        epochs = [0] + log_df['epoch'].tolist()
        train_aucs = [0] + log_df['train_auc'].tolist()
        test_aucs = [0] + log_df['test_auc'].tolist()
        plt.plot(epochs, train_aucs, label='Train AUC', color='blue')
        plt.plot(epochs, test_aucs, label='Test AUC', color='orange')
        plt.title('Test AUC Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.xlim(0, max(epochs)+1)
        plt.ylim(0, 1.05)
        plt.legend(loc = 'best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(auc_plot_path)
        plt.close()
    except Exception as e:
        print(f"Unable to generate AUC curve graph: {e}")



if __name__ == '__main__':
    pass
