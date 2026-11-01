### modified from https://github.com/cgoliver/RNAmigos/blob/master/learning/decoy_utils.py

import os
import pickle
import numpy as np
from scipy.spatial.distance import jaccard, euclidean

def get_decoys(mode='pdb', annots_dir='../data/annotated/pockets_nx'):
    """
    Build decoys set for validation.
    """
    if mode == 'pdb-whole':
        fp_dict = pickle.load(open('data/all_ligs_pdb_maccs.p', 'rb'))
        return fp_dict

    if mode=='pdb':
        fp_dict = {}
        for g in sorted(os.listdir(annots_dir)):
            graph_path = os.path.join(annots_dir, g)
            if not os.path.isfile(graph_path):
                continue
            try:
                lig_id = g.split(":")[2]
            except Exception as e:
                print(f"failed on {g}, {e}")
                continue
            with open(graph_path, 'rb') as handle:
                _, _, _, fp = pickle.load(handle)
            fp_dict[lig_id] = fp
        decoy_dict = {k:(v, [f for lig,f in fp_dict.items() if lig != k]) for k,v in fp_dict.items()}
        return decoy_dict
    pass

def distance_rank(active, pred, decoys, dist_func=jaccard):
    pred_dist = dist_func(active, pred)
    rank = 0
    for lig in decoys:
        d = dist_func(active, lig)
        if d < pred_dist:
            rank += 1
    return 1-(rank / len(decoys))


def decoy_test_(fp_pred, true_fp, decoys):
    fp_pred = np.asarray(fp_pred)
    fp_pred = (fp_pred > 0.5).astype(int)
    rank = distance_rank(true_fp, fp_pred, decoys)
    return rank


def euclidean_decoy_test_(fp_pred, true_fp, decoys):
    fp_pred, true_fp, decoys = (np.asarray(values, dtype=np.float64) for values in (fp_pred, true_fp, decoys))
    if fp_pred.ndim != 1 or fp_pred.shape != true_fp.shape or decoys.ndim != 2 or decoys.shape[1:] != fp_pred.shape or not len(decoys):
        raise ValueError('True, predicted, and decoy fingerprints must have matching widths and a nonempty decoy pool.')
    if not all(np.isfinite(values).all() for values in (fp_pred, true_fp, decoys)):
        raise ValueError('Fingerprints contain non-finite values.')
    pred_dist = euclidean(true_fp, fp_pred)
    decoy_distances = [euclidean(fp_pred, lig) for lig in decoys]
    all_distances = decoy_distances + [pred_dist]
    min_dist = min(all_distances)
    max_dist = max(all_distances)
    norm_pred_dist = (pred_dist - min_dist) / (max_dist - min_dist + 1e-8)
    norm_decoy_distances = [(d - min_dist) / (max_dist - min_dist + 1e-8) for d in decoy_distances]
    rank = 0
    for dist in norm_decoy_distances:
        if dist < norm_pred_dist:
            rank += 1
    rank = 1-(rank / len(decoys))
    return rank
