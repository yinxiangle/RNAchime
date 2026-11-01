### modified from https://github.com/camlab-ethz/GEMS/blob/main/PDBbind_data/similarity/pairwise_similarity_matrix/pairwise_similarity_tm_rmsd.py

import argparse
import csv
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import subprocess
import tempfile
from time import time

from joblib import Parallel, delayed, effective_n_jobs
import numpy as np
from numpy.lib.format import open_memmap
from rdkit import Chem, rdBase
from scipy.spatial import cKDTree


logger = logging.getLogger("pairwise_similarity_us_rmsd")
logger.setLevel(logging.ERROR)
logger.propagate = False

GRAPH_NAME_PATTERN = re.compile(
    r"^(?P<pdb_id>[A-Za-z0-9]{4})_#[^:]+:(?P<ligand_chain>[^:]+):"
    r"(?P<ligand_name>[A-Za-z0-9]{3}):(?P<position>-?\d+)_BIND\.nx$"
)


def configure_logging(output_dir):
    log_file = os.path.abspath(os.path.join(output_dir, "pairwise_similarity_us_rmsd_error.log"))
    for handler in logger.handlers:
        if getattr(handler, "baseFilename", None) == log_file:
            return

    handler = RotatingFileHandler(log_file, mode="a", maxBytes=5 * 1024 * 1024, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    handler.setLevel(logging.ERROR)
    logger.addHandler(handler)
    # MODIFIED: Write RDKit parser diagnostics to the result-directory error log.
    rdBase.AttachFileToLog("rdApp.error", log_file, 0)


def get_complex_ids(folder_path, pocket_ids_file):
    """Select only listed pocket IDs that have an SDF file."""
    with open(pocket_ids_file) as file:
        pocket_ids = json.load(file)
    if len(pocket_ids) != len(set(pocket_ids)):
        raise ValueError(f"Duplicate pocket IDs found in {pocket_ids_file}")

    missing = [pocket_id for pocket_id in pocket_ids if not os.path.isfile(os.path.join(folder_path, pocket_id + ".sdf"))]
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} listed SDF files in {folder_path}: {preview}")
    return sorted(pocket_ids)


# MODIFIED: Map each ligand pocket ID to the primary RNA auth chain selected from its graph.
def load_pocket_chains(path, complexes):
    mapping = {}
    with open(path, newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        if reader.fieldnames != ["graph_name", "auth_chain_id", "sequence"]:
            raise ValueError(f"Unexpected pocket-chain TXT header: {path}")
        for row in reader:
            match = GRAPH_NAME_PATTERN.fullmatch(row["graph_name"])
            if match is None:
                raise ValueError(f"Cannot parse graph name in {path}: {row['graph_name']}")
            complex_id = (
                f"{match.group('pdb_id').lower()}_{match.group('ligand_chain')}:"
                f"{match.group('ligand_name').upper()}:{int(match.group('position'))}"
            )
            if complex_id in mapping:
                raise ValueError(f"Duplicate pocket ID in {path}: {complex_id}")
            mapping[complex_id] = row["auth_chain_id"]

    missing = [complex_id for complex_id in complexes if complex_id not in mapping]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Missing {len(missing)} pocket-chain mappings in {path}: {preview}")
    return mapping


def get_pdb_chain(complex_id, pocket_chains):
    pdb_id = complex_id.split("_", 1)[0].lower()
    return pdb_id, pocket_chains[complex_id]


def group_complexes(complexes, pocket_chains):
    groups = {}
    for index, complex_id in enumerate(complexes):
        groups.setdefault(get_pdb_chain(complex_id, pocket_chains), []).append(index)
    return groups


def parse_ligands(folder_path, complexes):
    coords = []
    trees = []
    failed_sdf_files = []
    for complex_id in complexes:
        sdf_path = os.path.join(folder_path, complex_id + ".sdf")
        try:
            # Coordinates do not require sanitization; relaxed parsing recovers more SDF files.
            supplier = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=True, strictParsing=False)
            mol = supplier[0] if len(supplier) else None
            if mol is None:
                raise ValueError("RDKit returned no molecule")
            if not mol.GetNumConformers():
                raise ValueError("SDF molecule has no 3D conformer")
            positions = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
            if not len(positions):
                raise ValueError("SDF molecule has no atoms")
            coords.append(positions)
            trees.append(cKDTree(positions))
        except Exception as error:
            coords.append(None)
            trees.append(None)
            failed_sdf_files.append(sdf_path)
            logger.error("RDKit failed to parse SDF %s: %s", sdf_path, error)
    return coords, trees, failed_sdf_files


# Parse current USalign Structure_1/Structure_2 output.
def parse_usalign_output(output):
    scores = []
    for line in output.splitlines():
        if "TM-score=" in line and "normalized by length of Structure_" in line:
            scores.append(float(line.split("=", 1)[1].split()[0]))
    if not scores:
        raise ValueError("USalign output does not contain a TM-score")
    return max(scores)


def parse_transform(filename):
    rotation = []
    translation = []
    with open(filename) as file:
        parsing = False
        for line in file:
            if "------ The rotation matrix to rotate Structure_1 to Structure_2 ------" in line:
                parsing = True
                continue
            if parsing and line.strip() and "m" not in line:
                parts = line.split()
                translation.append(float(parts[1]))
                rotation.append([float(value) for value in parts[2:]])
                if len(rotation) == 3:
                    break

    if len(rotation) != 3:
        raise ValueError("USalign matrix output does not contain a complete transform")
    return np.asarray(rotation), np.asarray(translation)


# Run USalign on the requested auth chain from each mmCIF file.
def run_usalign(usalign_path, pdb1, chain1, pdb2, chain2, output_dir):
    descriptor, matrix_path = tempfile.mkstemp(prefix="usalign_", suffix="_matrix.txt", dir=output_dir)
    os.close(descriptor)
    try:
        command = [
            usalign_path, "-chain1", chain1, pdb1, "-chain2", chain2, pdb2,
            "-infmt1", "3", "-infmt2", "3", "-mol", "RNA", "-atom", " C3'", "-m", matrix_path,
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"USalign exited with code {result.returncode}: {message[-500:]}")
        score = parse_usalign_output(result.stdout)
        rotation, translation = parse_transform(matrix_path)
        return score, rotation, translation
    finally:
        if os.path.exists(matrix_path):
            os.remove(matrix_path)


def ligand_rmsd_block(indices1, indices2, coords, trees, rotation, translation, symmetric=False):
    block = np.full((len(indices1), len(indices2)), np.nan, dtype=np.float32)
    for row, index1 in enumerate(indices1):
        if coords[index1] is None:
            continue
        moved1 = coords[index1] @ rotation.T + translation
        moved_tree = cKDTree(moved1)
        start = row if symmetric else 0
        for column in range(start, len(indices2)):
            index2 = indices2[column]
            if coords[index2] is None:
                continue
            distances1 = trees[index2].query(moved1)[0]
            distances2 = moved_tree.query(coords[index2])[0]
            rmsd = max(np.sqrt(np.mean(distances1 ** 2)), np.sqrt(np.mean(distances2 ** 2)))
            block[row, column] = rmsd
            if symmetric:
                block[column, row] = rmsd
    return block


def process_pdb_chain_pair(structure1, structure2, indices1, indices2, coords, trees, pdb_cif_dir, usalign_path, output_dir):
    try:
        pdb1, chain1 = structure1
        pdb2, chain2 = structure2
        if structure1 == structure2:
            score = 1.0
            rotation = np.eye(3)
            translation = np.zeros(3)
        else:
            path1 = os.path.join(pdb_cif_dir, pdb1 + ".cif")
            path2 = os.path.join(pdb_cif_dir, pdb2 + ".cif")
            score, rotation, translation = run_usalign(usalign_path, path1, chain1, path2, chain2, output_dir)

        rmsd = ligand_rmsd_block(indices1, indices2, coords, trees, rotation, translation, symmetric=structure1 == structure2)
        return score, rmsd, None
    except Exception as error:
        label1 = "_".join(structure1)
        label2 = "_".join(structure2)
        message = f"Error processing PDB-chain pair {label1}, {label2}: {error}"
        logger.error(message)
        shape = (len(indices1), len(indices2))
        return np.nan, np.full(shape, np.nan, dtype=np.float32), message


def write_manifest(path, values):
    if os.path.exists(path):
        with open(path) as file:
            existing = json.load(file)
        if existing != values:
            raise ValueError(f"Existing manifest does not match current input: {path}")
        return
    with open(path, "w") as file:
        json.dump(values, file, indent=2)


def open_result_matrix(path, shape, fill_value):
    if os.path.exists(path):
        matrix = np.load(path, mmap_mode="r+")
        if matrix.shape != shape or matrix.dtype != np.float32:
            raise ValueError(f"Existing matrix has an incompatible shape or dtype: {path}")
        return matrix, True

    matrix = open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    matrix[:] = fill_value
    matrix.flush()
    return matrix, False


def open_status_matrix(path, structure_ids, groups, tm_matrix, rmsd_matrix, reuse_results):
    shape = (len(structure_ids), len(structure_ids))
    if os.path.exists(path):
        status = np.load(path, mmap_mode="r+")
        if status.shape != shape or status.dtype != np.uint8:
            raise ValueError(f"Existing status matrix has an incompatible shape or dtype: {path}")
        if not reuse_results:
            status[:] = 0
        status[status == 2] = 0
        status.flush()
        return status

    status = open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)
    status[:] = 0
    if reuse_results:
        for i, structure1 in enumerate(structure_ids):
            rows = groups[structure1]
            for j in range(i, len(structure_ids)):
                columns = groups[structure_ids[j]]
                tm_block = tm_matrix[np.ix_(rows, columns)]
                rmsd_block = rmsd_matrix[np.ix_(rows, columns)]
                if np.all(np.isfinite(tm_block)) and np.all(tm_block > 0) and np.all(np.isfinite(rmsd_block)):
                    status[i, j] = status[j, i] = 1
    status.flush()
    return status


def save_json_rows(output_dir, complexes, tm_matrix, rmsd_matrix):
    for index, complex_id in enumerate(complexes):
        tm_data = dict(zip(complexes, np.asarray(tm_matrix[index]).tolist()))
        rmsd_data = dict(zip(complexes, np.asarray(rmsd_matrix[index]).tolist()))
        with open(os.path.join(output_dir, f"{complex_id}_similarities_tm_scores.json"), "w") as file:
            json.dump(tm_data, file)
        with open(os.path.join(output_dir, f"{complex_id}_similarities_rmsd.json"), "w") as file:
            json.dump(rmsd_data, file)


def main(folder_path, pdb_cif_dir, usalign_path, output_dir, pocket_ids_file, pocket_chains_file, n_jobs=96, save_as_json=False):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    configure_logging(output_dir)

    complexes = get_complex_ids(folder_path, pocket_ids_file)
    if not complexes:
        raise ValueError(f"No pocket IDs found in {pocket_ids_file}")
    pocket_chains = load_pocket_chains(pocket_chains_file, complexes)
    groups = group_complexes(complexes, pocket_chains)
    structure_ids = sorted(groups)
    structure_labels = ["_".join(structure_id) for structure_id in structure_ids]
    print(f"Number of complexes: {len(complexes)}", flush=True)
    print(f"Number of unique PDB-chain IDs: {len(structure_ids)}", flush=True)

    pdb_ids = sorted({pdb_id for pdb_id, _ in structure_ids})
    missing = [pdb_id for pdb_id in pdb_ids if not os.path.isfile(os.path.join(pdb_cif_dir, pdb_id + ".cif"))]
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} PDB CIF files in {pdb_cif_dir}: {preview}")

    complexes_file = os.path.join(output_dir, "pairwise_similarity_complexes.json")
    pocket_chains_manifest = os.path.join(output_dir, "pairwise_similarity_pocket_chains.json")
    structures_file = os.path.join(output_dir, "pairwise_similarity_pdb_chains.json")
    config_file = os.path.join(output_dir, "pairwise_similarity_config.json")
    write_manifest(complexes_file, complexes)
    write_manifest(pocket_chains_manifest, [{"complex_id": complex_id, "auth_chain_id": pocket_chains[complex_id]} for complex_id in complexes])
    write_manifest(structures_file, structure_labels)

    result_names = ("pairwise_similarity_matrix_tm.npy", "pairwise_similarity_matrix_rmsd.npy", "pairwise_similarity_pdb_chain_pair_status.npy")
    if not os.path.exists(config_file) and any(os.path.exists(os.path.join(output_dir, name)) for name in result_names):
        raise ValueError("Existing results have no mode metadata; use a new output directory")
    write_manifest(config_file, {
        "aligner": "USalign", "identifier": "pdb_primary_rna_auth_chain",
        "chain_source": "pocket_primary_chains.txt", "molecule": "RNA", "atom": " C3'",
    })

    shape = (len(complexes), len(complexes))
    tm_file = os.path.join(output_dir, "pairwise_similarity_matrix_tm.npy")
    rmsd_file = os.path.join(output_dir, "pairwise_similarity_matrix_rmsd.npy")
    status_file = os.path.join(output_dir, "pairwise_similarity_pdb_chain_pair_status.npy")
    tm_matrix, tm_exists = open_result_matrix(tm_file, shape, np.nan)
    rmsd_matrix, rmsd_exists = open_result_matrix(rmsd_file, shape, np.nan)
    status = open_status_matrix(status_file, structure_ids, groups, tm_matrix, rmsd_matrix, tm_exists and rmsd_exists)

    print("Parsing ligand coordinates...", flush=True)
    coords, trees, failed_sdf_files = parse_ligands(folder_path, complexes)
    print(f"Ligand coordinates parsed: {len(complexes) - len(failed_sdf_files)} succeeded, {len(failed_sdf_files)} failed", flush=True)
    if failed_sdf_files:
        print(f"RDKit errors saved to {os.path.join(output_dir, 'pairwise_similarity_us_rmsd_error.log')}", flush=True)

    tic = time()
    # Queue all unfinished pairs together to keep workers busy across PDB-chain rows.
    pending_pairs = [(i, j) for i in range(len(structure_ids)) for j in range(i, len(structure_ids)) if status[i, j] == 0]
    worker_count = effective_n_jobs(n_jobs)
    batch_size = 4 * worker_count
    total_batches = (len(pending_pairs) + batch_size - 1) // batch_size
    print(f"Pending PDB-chain pairs: {len(pending_pairs)}, workers: {worker_count}, batch size: {batch_size}", flush=True)

    with Parallel(n_jobs=n_jobs, prefer="threads") as parallel:
        for batch_index, start in enumerate(range(0, len(pending_pairs), batch_size), start=1):
            batch = pending_pairs[start:start + batch_size]
            results = parallel(
                delayed(process_pdb_chain_pair)(
                    structure_ids[i], structure_ids[j], groups[structure_ids[i]], groups[structure_ids[j]], coords, trees,
                    pdb_cif_dir, usalign_path, output_dir,
                ) for i, j in batch
            )

            for (i, j), (score, rmsd_block, error) in zip(batch, results):
                rows = groups[structure_ids[i]]
                columns = groups[structure_ids[j]]
                tm_matrix[np.ix_(rows, columns)] = score
                rmsd_matrix[np.ix_(rows, columns)] = rmsd_block
                if i != j:
                    tm_matrix[np.ix_(columns, rows)] = score
                    rmsd_matrix[np.ix_(columns, rows)] = rmsd_block.T
                status[i, j] = status[j, i] = 2 if error else 1

            # Persist every batch so an interrupted run can resume safely.
            tm_matrix.flush()
            rmsd_matrix.flush()
            status.flush()
            print(f"Batch {batch_index}/{total_batches} complete: {start + len(batch)}/{len(pending_pairs)} pairs, {time() - tic:.1f} seconds total", flush=True)

    np.fill_diagonal(tm_matrix, 1.0)
    np.fill_diagonal(rmsd_matrix, 0.0)
    tm_matrix.flush()
    rmsd_matrix.flush()

    failures = int(np.count_nonzero(np.triu(status == 2)))
    if save_as_json:
        save_json_rows(output_dir, complexes, tm_matrix, rmsd_matrix)
    print(f"TM-score matrix saved to {tm_file}", flush=True)
    print(f"RMSD matrix saved to {rmsd_file}", flush=True)
    print(f"Failed SDF files: {len(failed_sdf_files)}", flush=True)
    print(f"Failed PDB-chain pairs: {failures}", flush=True)
    print(f"Elapsed time: {time() - tic:.2f} seconds", flush=True)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    parser = argparse.ArgumentParser(description="Compute chain-aware USalign TM-scores and ligand position RMSDs")
    parser.add_argument("--folder_path", default=os.path.join(script_dir, "cifandsdf/hariboss_data"))
    parser.add_argument("--pdb_cif_dir", default=os.path.join(project_dir, "dataset/hariboss_data/data/pdb.cif"))
    parser.add_argument("--pocket_ids_file", default=os.path.join(script_dir, "dataset/hariboss_data/pocket_ids.json"))
    parser.add_argument("--pocket_chains_file", default=os.path.join(project_dir, "dataset/hariboss_data/data/pocket_primary_chains.txt"))

    parser.add_argument("--usalign_path", default=os.path.join(script_dir, "USalign"))
    parser.add_argument("--output_dir", default=os.path.join(script_dir, "US_similarity_results/hariboss_data/all_new"))
    parser.add_argument("--n_jobs", type=int, default=48)
    parser.add_argument("--save_as_json", action="store_true")
    args = parser.parse_args()
    main(
        args.folder_path, args.pdb_cif_dir, args.usalign_path, args.output_dir, args.pocket_ids_file, args.pocket_chains_file,
        n_jobs=args.n_jobs, save_as_json=args.save_as_json,
    )
