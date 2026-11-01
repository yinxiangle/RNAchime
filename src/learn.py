import os
import time
import dgl
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from .utils import plot_loss_curve, plot_auc_curve


def send_graph_to_device(g, device):
    
    g = g.to(device, non_blocking=True)
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)
    return g


def print_gradients(model):
    """Print parameter gradients for debugging."""
    for name, parameter in model.named_parameters():
        print(name, parameter.grad)


def _compute_auc(y_true, y_pred):
    y_true, y_scores = np.vstack(y_true), np.vstack(y_pred)
    if y_true.shape != y_scores.shape:
        raise ValueError('AUC targets and predictions must have matching shapes.')
    if not np.isfinite(y_true).all() or not np.isfinite(y_scores).all():
        raise ValueError('AUC targets and predictions must be finite.')
    if not np.all((y_true == 0) | (y_true == 1)):
        return float('nan')
    aucs = [roc_auc_score(y_true[:, i], y_scores[:, i]) for i in range(y_true.shape[1]) if np.unique(y_true[:, i]).size > 1]
    return float(np.mean(aucs)) if aucs else float('nan')


def _forward_loss(model, graph, fp):
    if isinstance(model.criterion, torch.nn.BCEWithLogitsLoss) and not torch.all((fp >= 0) & (fp <= 1)):
        raise ValueError('BCEWithLogitsLoss targets must be finite values in [0, 1].')
    prediction, _ = model(graph)
    if prediction.shape != fp.shape:
        raise ValueError(f'Prediction shape {tuple(prediction.shape)} does not match target shape {tuple(fp.shape)}.')
    loss = model.fp_loss(fp, prediction)
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite loss encountered; check input data, model outputs, and learning rate.')
    return prediction, loss


def _compute_fp_stats(loader, device):
   
    count, mean, squared_deviations = 0, None, None
    for _, fp, _ in loader:
        batch = fp.detach().to(device='cpu', dtype=torch.float64)
        if not torch.isfinite(batch).all():
            raise ValueError('Training fingerprints contain non-finite values.')
        batch_count = batch.shape[0]
        batch_mean = batch.mean(dim=0)
        batch_deviations = ((batch - batch_mean) ** 2).sum(dim=0)
        total = count + batch_count
        if count == 0:
            mean, squared_deviations = batch_mean, batch_deviations
        else:
            delta = batch_mean - mean
            squared_deviations += batch_deviations + delta.square() * (count * batch_count / total)
            mean += delta * (batch_count / total)
        count = total
    if count == 0:
        raise ValueError('Training loader produced no samples for fingerprint statistics.')
    std = (squared_deviations / count).clamp_min(0).sqrt().clamp_min(1e-6)
    return mean.to(device=device, dtype=torch.float32), std.to(device=device, dtype=torch.float32)


@torch.no_grad()
def test(model, test_loader, device):
    """Return sample-weighted validation loss and macro AUC, or NaN if AUC is undefined."""
    if len(test_loader) == 0:
        raise ValueError('Validation loader must contain at least one batch.')
    was_training = model.training
    model.eval()
    total_loss, sample_count = 0.0, 0
    y_true, y_pred = [], []
    try:
        for graph, fp, _ in test_loader:
            graph = send_graph_to_device(graph, device)
            fp = fp.to(device, non_blocking=True)
            prediction, loss = _forward_loss(model, graph, fp)
            total_loss += loss.item() * fp.shape[0]
            sample_count += fp.shape[0]
            y_true.append(fp.cpu().numpy())
            y_pred.append(prediction.cpu().numpy())
        if sample_count == 0:
            raise ValueError('Validation loader produced no samples.')
        return total_loss / sample_count, _compute_auc(y_true, y_pred)
    finally:
        model.train(was_training)


def train_model(model, optimizer, device, train_loader, test_loader, save_path,
                writer=None, num_epochs=25, wall_time=None, early_stop_threshold=10, run_time=''):
    """Train and save the lowest-validation-loss checkpoint; wall_time is a soft limit in hours."""
    start_time = time.monotonic()
    deadline = None if wall_time is None else start_time + wall_time * 3600

    if isinstance(model.criterion, torch.nn.HuberLoss):
        model.set_fp_stats(*_compute_fp_stats(train_loader, device))

    save_path = os.path.abspath(os.fspath(save_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    run_time = run_time.strip() or time.strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(os.path.dirname(save_path), f'training_log_{run_time}.csv')
    with open(log_file_path, 'w') as handle:
        handle.write('epoch,train_loss,test_loss,train_auc,test_auc\n')
    best_loss, epochs_from_best = float('inf'), 0

    for epoch in range(num_epochs):
        if deadline is not None and time.monotonic() >= deadline:
            if not np.isfinite(best_loss):
                raise TimeoutError('Time limit reached before a complete training epoch.')
            print('Training stopped at the time limit.')
            break
        print(f'Epoch {epoch + 1}/{num_epochs}\n----------')
        model.train()
        running_loss, sample_count = 0.0, 0
        y_true, y_pred = [], []
        num_batches = len(train_loader)
        for batch_idx, (graph, fp, _) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            graph = send_graph_to_device(graph, device)
            fp = fp.to(device, non_blocking=True)
            prediction, loss = _forward_loss(model, graph, fp)
            y_true.append(fp.detach().cpu().numpy())
            y_pred.append(prediction.detach().cpu().numpy().copy())
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            running_loss += batch_loss * fp.shape[0]
            sample_count += fp.shape[0]
            if batch_idx % 10 == 0:
                elapsed = time.monotonic() - start_time
                print(f'Train Epoch: {epoch + 1} [{batch_idx + 1}/{num_batches} batches, {sample_count} samples] Loss: {batch_loss:.6f} Time: {elapsed:.2f}')
            if writer is not None:
                writer.add_scalar('Training batch loss', batch_loss, epoch * num_batches + batch_idx)

        if sample_count == 0:
            raise ValueError('Training loader produced no samples.')
        train_loss, train_auc = running_loss / sample_count, _compute_auc(y_true, y_pred)
        test_loss, test_auc = test(model, test_loader, device)
        print(f'>> train loss {train_loss:.6f} | train auc {train_auc:.4f}')
        print(f'>> test loss {test_loss:.6f} | test auc {test_auc:.4f}')
        if writer is not None:
            writer.add_scalar('Training epoch loss', train_loss, epoch)
            writer.add_scalar('Test loss during training', test_loss, epoch)
            if np.isfinite(train_auc):
                writer.add_scalar('Training epoch AUC', train_auc, epoch)
            if np.isfinite(test_auc):
                writer.add_scalar('Test AUC during training', test_auc, epoch)
        with open(log_file_path, 'a') as handle:
            handle.write(f'{epoch + 1},{train_loss},{test_loss},{train_auc},{test_auc}\n')

        improved = test_loss < best_loss
        if improved:
            best_loss, epochs_from_best = test_loss, 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'loss': best_loss}, save_path)
        else:
            epochs_from_best += 1
        # plot_loss_curve(log_file_path, save_path, run_time)
        # if np.isfinite(train_auc) or np.isfinite(test_auc):
        #     plot_auc_curve(log_file_path, save_path, run_time)
        if not improved and epochs_from_best >= max(1, early_stop_threshold):
            print(f'Training stopped after {epochs_from_best} epochs without validation loss improvement.')
            break
    return best_loss
