### modified from https://github.com/camlab-ethz/GEMS/blob/main/PDBbind_dataset_filtering/remove_train_redundancy.py

import json
import os
import random
from collections import Counter
import numpy as np


def parse_complex_id(complex_id):
    pdb_id,ligand_id = complex_id.split("_",1)
    return pdb_id,ligand_id.split(":")[1]


def remove_redundancy(train_dataset,complexes,similarity_matrix_tm,similarity_matrix_tanimoto,similarity_matrix_rmsd,tm_threshold,s_threshold):
    complex_to_index = {complex_id:index for index,complex_id in enumerate(complexes)}
    
    valid_pairs = np.isfinite(similarity_matrix_tm) & np.isfinite(similarity_matrix_rmsd)
    if len(complexes) > 1:
        np.fill_diagonal(valid_pairs,False)
    valid_pockets = np.any(valid_pairs,axis=1)
    for complex_id in train_dataset:
        if not valid_pockets[complex_to_index[complex_id]]:
            print(f"Excluding {complex_id}: no valid TM-score/RMSD comparisons with other pockets")
    train_dataset = [complex_id for complex_id in train_dataset if valid_pockets[complex_to_index[complex_id]]]
    if not train_dataset:
        return []

    train_indices = [complex_to_index[complex_id] for complex_id in train_dataset]
    selection = np.ix_(train_indices,train_indices)
    tm_matrix = similarity_matrix_tm[selection]
    tanimoto_matrix = similarity_matrix_tanimoto[selection]
    rmsd_matrix = similarity_matrix_rmsd[selection]
    ligand_similarity_matrix = tanimoto_matrix + (1 - rmsd_matrix)
    # Pairs with 
    # --- TM-score > TM_threshold 
    # --- tanimoto + (1-RMSE) > S_threshold
    adjacency_matrix = (tm_matrix > tm_threshold) & (ligand_similarity_matrix > s_threshold)
    adjacency_matrix = np.maximum(adjacency_matrix,adjacency_matrix.T)
    np.fill_diagonal(adjacency_matrix,0)

    removed_indices = set()
    column_sums = np.sum(adjacency_matrix,axis=0)
    complex_info = [parse_complex_id(complex_id) for complex_id in train_dataset]

    while True:
        max_sum_value = np.max(column_sums)
        if max_sum_value == 0:
            break

        candidate_indices = np.flatnonzero(column_sums == max_sum_value).tolist()
        remaining_indices = [index for index in range(len(train_dataset)) if index not in removed_indices]
        ligand_counts = Counter(complex_info[index][1] for index in remaining_indices)
        pdb_counts = Counter(complex_info[index][0] for index in remaining_indices)
        max_ligand_count = max(ligand_counts[complex_info[index][1]] for index in candidate_indices)
        candidate_indices = [index for index in candidate_indices if ligand_counts[complex_info[index][1]] == max_ligand_count]
        max_pdb_count = max(pdb_counts[complex_info[index][0]] for index in candidate_indices)
        candidate_indices = [index for index in candidate_indices if pdb_counts[complex_info[index][0]] == max_pdb_count]
        to_remove_idx = random.choice(candidate_indices)
        
        similarities = adjacency_matrix[to_remove_idx,:].copy()
        similar_indices = np.flatnonzero(similarities)
        removed_indices.add(to_remove_idx)

        pdb_id,ligand_name = complex_info[to_remove_idx]
        print(f"Removing {train_dataset[to_remove_idx]} with {max_sum_value} redundant connections, ligand count {ligand_counts[ligand_name]}, PDB count {pdb_counts[pdb_id]}")
        for index in similar_indices:
            print(f"--- {train_dataset[index]}: Tanimoto={tanimoto_matrix[to_remove_idx,index]:.3f}, TM-score={tm_matrix[to_remove_idx,index]:.3f}, RMSD={rmsd_matrix[to_remove_idx,index]:.3f}")
        # update
        column_sums -= similarities
        column_sums[to_remove_idx] = 0
        adjacency_matrix[to_remove_idx,:] = 0
        adjacency_matrix[:,to_remove_idx] = 0

    return [complex_id for index,complex_id in enumerate(train_dataset) if index not in removed_indices]


def main():
    random.seed(42)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir,"dataset/rnasmol_data")
    similarity_dir = os.path.join(script_dir,"US_similarity_results/rnasmol_data/riboswitch")
    
    input_train_file = os.path.join(dataset_dir,"riboswitch_pocket_ids.json")
    output_train_file = os.path.join(dataset_dir,"riboswitch_pocket_ids_us_nonredundant_new.json")
    
    tm_threshold = 0.8
    s_threshold = 1.3

    with open(input_train_file,"r") as file:
        train_dataset = json.load(file)
    with open(os.path.join(similarity_dir,"pairwise_similarity_complexes.json"),"r") as file:
        complexes = json.load(file)

    similarity_matrix_tm = np.load(os.path.join(similarity_dir,"pairwise_similarity_matrix_tm.npy"))
    similarity_matrix_tanimoto = np.load(os.path.join(similarity_dir,"pairwise_similarity_matrix_tanimoto.npy"))
    similarity_matrix_rmsd = np.load(os.path.join(similarity_dir,"pairwise_similarity_matrix_rmsd.npy"))

    print(f"Initial training complexes: {len(train_dataset)}")
    print(f"Thresholds: TM-score > {tm_threshold}, Tanimoto + (1 - RMSD) > {s_threshold}")
    filtered_train_dataset = remove_redundancy(train_dataset,complexes,similarity_matrix_tm,similarity_matrix_tanimoto,similarity_matrix_rmsd,tm_threshold,s_threshold)
    print(f"Final training complexes: {len(filtered_train_dataset)}")
    print(f"Removed complexes: {len(train_dataset) - len(filtered_train_dataset)}")

    with open(output_train_file,"w") as file:
        json.dump(filtered_train_dataset,file,indent=4)
    print(f"Saved filtered training set to {output_train_file}")


if __name__ == "__main__":
    main()
