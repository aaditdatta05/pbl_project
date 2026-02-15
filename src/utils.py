import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def round_predictions(preds):
    # preds: numpy array floats, clamp and round to 0..4
    preds = np.rint(preds).astype(int)
    preds = np.clip(preds, 0, 4)
    return preds

def quadratic_weighted_kappa(y_true, y_pred):
    # expects integer labels
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def compute_pos_weight_from_labels(labels, num_classes=5, beta=0.9999, eps=1e-12):
    """
    Compute per-threshold pos_weight for ordinal regression using effective number.
    For each threshold k, positives are labels > k and negatives are labels <= k.
    """
    labels = np.asarray(labels)
    pos_weights = []

    for k in range(num_classes - 1):
        pos_count = np.sum(labels > k)
        neg_count = np.sum(labels <= k)

        pos_count = max(int(pos_count), 0)
        neg_count = max(int(neg_count), 0)

        pos_eff = (1.0 - beta) / max(1.0 - (beta ** max(pos_count, 1)), eps)
        neg_eff = (1.0 - beta) / max(1.0 - (beta ** max(neg_count, 1)), eps)

        pos_weight = pos_eff / max(neg_eff, eps)
        pos_weights.append(pos_weight)

    return torch.tensor(pos_weights, dtype=torch.float32)

def load_state_dict_compatible(model, state_dict):
    """
    Load only matching keys/shapes from a checkpoint state_dict.
    Returns (missing_keys, skipped_keys).
    """
    model_state = model.state_dict()
    compatible = {}
    skipped = []

    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)

    model_state.update(compatible)
    model.load_state_dict(model_state)

    missing = [key for key in model_state.keys() if key not in compatible]
    return missing, skipped
