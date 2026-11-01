# Environment setup: https://unimol-tools.readthedocs.io/en/latest/installation.html

import pickle
import sys

import numpy as np
from unimol_tools import UniMolRepr

INPUT_PATH = "in-vitro_canonical.smi"
OUTPUT_PATH = "in-vitro_unimol.p"


def load_smiles_with_names(file_path):
	pairs = []
	with open(file_path, "r", encoding="utf-8") as handle:
		for line_no, line in enumerate(handle, 1):
			line = line.strip()
			if not line or line.startswith("#"):
				continue
			parts = line.split()
			if len(parts) < 2:
				print(f"Skip line {line_no}: expected 2 columns, got {len(parts)}", file=sys.stderr)
				continue
			pairs.append((parts[0], parts[1]))
	return pairs


def main():
	pairs = load_smiles_with_names(INPUT_PATH)
	if not pairs:
		raise ValueError("No valid SMILES entries found in input file")

	smiles_list = [smiles for _, smiles in pairs]
	names = [name for name, _ in pairs]

	model = UniMolRepr(data_type="molecule", remove_hs=False)
	reprs = model.get_repr(smiles_list, return_atomic_reprs=False)
	# Store plain lists to avoid NumPy-specific pickle formats.
	emb_list = [np.asarray(emb, dtype=np.float32).tolist() for emb in reprs]
	# print(emb_list[0:3])

	if len(names) != len(emb_list):
		raise ValueError("Embedding count does not match input count")

	embeddings = {name: emb for name, emb in zip(names, emb_list)}

	with open(OUTPUT_PATH, "wb") as handle:
		pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

	print(f"Saved {len(embeddings)} embeddings to {OUTPUT_PATH}")


if __name__ == "__main__":
	main()
