import os
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator

def morgan_fp(smiles_file):
    
    if not os.path.exists(smiles_file):
        print(f"Error: File not found at {smiles_file}")
        return


    all_ligand = {}

    print(f"Reading from {smiles_file}...")
    
    with open(smiles_file, 'r') as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} entries...")
 
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) < 2:
            print(f"Skipping invalid line: {line}")
            continue

  
        ligand_id = parts[0]
        smile = parts[1]   

        mol = Chem.MolFromSmiles(smile)
        if mol:
            # Morgan Fingerprint (Radius=2, nBits=2048)
            fp = morgan_gen.GetFingerprint(mol)
            all_ligand[ligand_id] = fp
        else:
            print(f"Warning: Could not parse SMILES for ID {ligand_id}: {smile}")
    return all_ligand

 
if __name__ == "__main__":
    all_ligs = morgan_fp(smiles_file='in-vitro_canonical.smi')
    pickle.dump(all_ligs, open("in-vitro_morgan.p", "wb"))