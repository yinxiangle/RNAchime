### copy from https://github.com/cgoliver/RNAmigos/blob/master/data_processor/pybel_fp.py


import pickle
import numpy as np
from openbabel import pybel


def fp_dict(smiles_file, bits=True, fptype='FP2'):                                                                                                     

    nbits = {'maccs': 166, 'FP2': 1024}
    fps = {}

    with open(smiles_file, "r") as sms:
        for line in sms:
            s = line.strip()
            if len(s.split()) != 2:
                print(f"Line '{s}' is not in the expected format, skipping...")
            name, smile = s.split()
            
            try:
                mol = pybel.readstring('smi', smile)
            except Exception as e:
                print(f"Could not read SMILES for {name}, skipping...")
                print("error:", e)
                continue
            fp = mol.calcfp(fptype=fptype)
            if bits:
                fp = index_to_vec(fp.bits, nbits=nbits[fptype])
            fps[name] = fp
    return fps

def index_to_vec(fp, nbits=1024):
    """
    Convert list of 1 indices to numpy binary vector.

    Returns:
        `array`: 1x1024 binary numpy vector
    """
    vec = np.zeros(nbits)
    vec[fp] = 1
    return vec

if __name__ == "__main__":

    input_smi = "in-vitro_canonical.smi"

    # Generate MACCS fingerprints (166 bits)
    print("Generating MACCS fingerprints...")
    maccs = fp_dict(input_smi, bits=True, fptype='maccs')
    maccs_out = "in-vitro_maccs.p"
    pickle.dump(maccs, open(maccs_out, "wb"))
    print(f"MACCS fingerprints saved to {maccs_out} ({len(maccs)} ligands)")

    # Generate FP2 fingerprints (1024 bits)
    print("Generating FP2 fingerprints...")
    fp2 = fp_dict(input_smi, bits=True, fptype='FP2')
    fp2_out = "in-vitro_fp2.p"
    pickle.dump(fp2, open(fp2_out, "wb"))
    print(f"FP2 fingerprints saved to {fp2_out} ({len(fp2)} ligands)")