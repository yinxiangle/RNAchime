
"""Generate ligand permutations for RNA pocket graphs.

Run from the RNAchime directory:

python scripts/permute_ligands.py \
    --nx_dir data/pocket_nx \
    --output outputs/ligand_permutations.csv \
    --num_permutations 10 \
    --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import pickle
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PocketRecord:
    filename: str
    path: Path
    pdb_id: str
    model: str
    chain: str
    ligand: str
    ligand_position: str
    graph_hash: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Permute ligand labels while excluding known binders of the recipient RNA.")
    parser.add_argument("--nx_dir", type=Path, required=True, help="Input directory containing .nx graph files.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV base path; permutation and seed suffixes are added.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42).")
    parser.add_argument("--num_permutations", type=int, default=10, help="Number of distinct constrained permutations (default: 10).")
    parser.add_argument(
        "--forbid_scope", choices=("pdb", "graph", "pdb-and-graph"), default="pdb-and-graph",
        help="Exclude binders of the same PDB, identical attributed graphs, or both (default: pdb-and-graph).",
    )
    parser.add_argument("--max_attempts", type=int, default=500, help="Maximum attempts per permutation (default: 500).")
    return parser.parse_args()


def parse_filename(filename):
    suffix = "_BIND.nx"
    prefix = filename[: -len(suffix)]
    pdb_id, remainder = prefix.split("_#", 1)
    fields = remainder.split(":", 3)
    if len(fields) != 4:
        raise ValueError(f"Unsupported nx filename: {filename}")
    model, chain, ligand, ligand_position = fields
    return pdb_id.lower(), model, chain, ligand, ligand_position


def strict_graph_hash(graph):
    nodes = tuple(sorted((node, attrs["nt"]) for node, attrs in graph.nodes(data=True)))
    edges = []
    for source, target, attrs in graph.edges(data=True):
        left, right = sorted((source, target))
        edges.append((left, right, attrs["label"]))
    canonical = nodes, tuple(sorted(edges))
    payload = repr(canonical).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_records(nx_dir):
    if not nx_dir.is_dir():
        raise FileNotFoundError(f"nx directory does not exist: {nx_dir}")
    paths = sorted(nx_dir.glob("*.nx"), key=lambda path: path.name)
    if not paths:
        raise ValueError(f"No .nx files found in: {nx_dir}")
    records = []
    for path in paths:
        pdb_id, model, chain, ligand, ligand_position = parse_filename(path.name)
        with path.open("rb") as handle:
            graph = pickle.load(handle)
        if not hasattr(graph, "nodes") or not hasattr(graph, "edges"):
            raise TypeError(f"File does not contain a NetworkX-like graph: {path}")
        records.append(
            PocketRecord(
                filename=path.name,
                path=path,
                pdb_id=pdb_id,
                model=model,
                chain=chain,
                ligand=ligand,
                ligand_position=ligand_position,
                graph_hash=strict_graph_hash(graph),
            )
        )
    return records


def build_forbidden_sets(records, scope):
    pdb_ligands: dict[str, set[str]] = defaultdict(set)
    graph_ligands: dict[str, set[str]] = defaultdict(set)
    for record in records:
        pdb_ligands[record.pdb_id].add(record.ligand)
        graph_ligands[record.graph_hash].add(record.ligand)

    forbidden_sets = []
    for record in records:
        forbidden = {record.ligand}
        if scope in {"pdb", "pdb-and-graph"}:
            forbidden.update(pdb_ligands[record.pdb_id])
        if scope in {"graph", "pdb-and-graph"}:
            forbidden.update(graph_ligands[record.graph_hash])
        forbidden_sets.append(frozenset(forbidden))
    return forbidden_sets


def weighted_choice(candidates, remaining, rng):
    total = sum(remaining[ligand] for ligand in candidates)
    ticket = rng.randrange(total)  
    for ligand in candidates:
        ticket -= remaining[ligand]
        if ticket < 0:
            return ligand 
    raise AssertionError("Weighted choice exhausted without selecting a ligand")


def assign_ligands_once(records, forbidden_sets, seed, max_attempts):
    """Return assigned ligand names for one frequency-preserving permutation."""
    original_counts = Counter(record.ligand for record in records)
    ligand_types = sorted(original_counts)

    allowed_capacity = [
        sum(count for ligand, count in original_counts.items() if ligand not in forbidden)
        for forbidden in forbidden_sets
    ]
    impossible = [
        records[index].filename
        for index, capacity in enumerate(allowed_capacity)
        if capacity == 0
    ]
    if impossible:
        preview = ", ".join(impossible[:5])
        raise RuntimeError(f"No valid replacement ligand for: {preview}")

    for attempt in range(max_attempts):
        rng = random.Random(seed + attempt * 10)
        remaining = original_counts.copy()
        assignments: list[str | None] = [None] * len(records)

        # Constrained recipients are assigned first. Random tie-breaking gives
        # reproducible but different solutions for different seeds.
        recipients = list(range(len(records)))
        rng.shuffle(recipients)
        recipients.sort(key=lambda index: allowed_capacity[index])

        failed = False
        for index in recipients:
            candidates = [
                ligand
                for ligand in ligand_types
                if remaining[ligand] > 0 and ligand not in forbidden_sets[index]
            ]
            if not candidates:
                failed = True
                break
            rng.shuffle(candidates)
            chosen = weighted_choice(candidates, remaining, rng)
            assignments[index] = chosen
            remaining[chosen] -= 1

        if failed:
            continue

        assigned = [ligand for ligand in assignments if ligand is not None]
        if len(assigned) != len(records) or Counter(assigned) != original_counts:
            raise AssertionError("Internal error: permutation changed ligand frequencies")

        return assigned

    raise RuntimeError(
        f"Could not construct a valid permutation after {max_attempts} attempts. "
        "Increase --max_attempts or relax --forbid_scope."
    )


def validate_permutation(records, forbidden_sets, assigned_ligands):
    if len(assigned_ligands) != len(records):
        raise AssertionError("Permutation length differs from input length")
    if Counter(assigned_ligands) != Counter(record.ligand for record in records):
        raise AssertionError("Ligand occurrence counts changed during permutation")

    violations = [
        records[index].filename
        for index, ligand in enumerate(assigned_ligands)
        if ligand in forbidden_sets[index]
    ]
    if violations:
        preview = ", ".join(violations[:5])
        raise AssertionError(f"Known-binder exclusion violated for: {preview}")


def output_rows(records, forbidden_sets, ligands):
    for index, record in enumerate(records):
        yield {
            "rna_filename": record.filename,
            "swapped_ligand": ligands[index],
            "forbidden_ligands": ";".join(sorted(forbidden_sets[index])),
        }


def permutation_output_path(requested_output, permutation_id, permutation_seed, number_of_permutations):
    suffix = requested_output.suffix or ".csv"
    if number_of_permutations == 1:
        return requested_output.with_name(
            f"{requested_output.stem}_seed_{permutation_seed}{suffix}"
        )
    return requested_output.with_name(
        f"{requested_output.stem}_perm_{permutation_id:03d}"
        f"_seed_{permutation_seed}{suffix}"
    )


def main() -> None:
    args = parse_args()
    if args.num_permutations < 1:
        raise ValueError("--num_permutations must be at least 1")
    if args.max_attempts < 1:
        raise ValueError("--max_attempts must be at least 1")

    records = load_records(args.nx_dir.expanduser().resolve())
    forbidden_sets = build_forbidden_sets(records, args.forbid_scope)
    original_counts = Counter(record.ligand for record in records)
    allowed_occurrences = [
        sum(
            count
            for ligand, count in original_counts.items()
            if ligand not in forbidden
        )
        for forbidden in forbidden_sets
    ]

    permutations = []
    seen_signatures: set[tuple[str, ...]] = set()
    for permutation_index in range(args.num_permutations):
        uniqueness_attempt = 0
        while True:
            permutation_seed = args.seed + permutation_index * 100 + uniqueness_attempt * 100
            ligands = assign_ligands_once(
                records,
                forbidden_sets,
                permutation_seed,
                args.max_attempts,
            )
            validate_permutation(records, forbidden_sets, ligands)
            signature = tuple(ligands)
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                break
            uniqueness_attempt += 1
            if uniqueness_attempt >= args.max_attempts:
                raise RuntimeError(
                    "Could not generate the requested number of distinct permutations"
                )
        permutations.append((permutation_index + 1, permutation_seed, ligands))

    requested_output = args.output.expanduser().resolve()
    requested_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["rna_filename", "swapped_ligand", "forbidden_ligands"]
    output_paths = []
    for permutation_id, permutation_seed, ligands in permutations:
        output = permutation_output_path(
            requested_output,
            permutation_id,
            permutation_seed,
            len(permutations),
        )
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows(records, forbidden_sets, ligands))
        output_paths.append((output, permutation_seed))

    print(f"Input files: {len(records)}")
    print(f"Distinct PDBs: {len({record.pdb_id for record in records})}")
    print(f"Distinct strict RNA graphs: {len({record.graph_hash for record in records})}")
    print(f"Distinct ligands: {len(original_counts)}")
    print(f"Forbid scope: {args.forbid_scope}")
    print(
        "Forbidden ligand types per RNA (min/median/max): "
        f"{min(map(len, forbidden_sets))}/"
        f"{statistics.median(map(len, forbidden_sets)):.1f}/"
        f"{max(map(len, forbidden_sets))}"
    )
    print(
        "Allowed ligand occurrences per RNA (min/median/max): "
        f"{min(allowed_occurrences)}/"
        f"{statistics.median(allowed_occurrences):.1f}/"
        f"{max(allowed_occurrences)}"
    )
    print(f"Permutations written: {len(permutations)}")
    print(f"Rows per output file: {len(records)}")
    for output, permutation_seed in output_paths:
        print(f"Output (seed {permutation_seed}): {output}")


if __name__ == "__main__":
    main()
