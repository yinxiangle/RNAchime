import csv
import re
from collections import Counter
from pathlib import Path


kwd_dict = {
    'RIBOSOM': 0,'RIBOSOME': 0,'RIBOSOMAL': 0,'A SITE': 0,'P SITE': 0,'E SITE': 0,'DECODING': 0,'AMINOGLYCOSIDE': 0,'RRNA': 0,
    'VIRAL': 1,'VIRUS': 1,'HIV': 1,'MOSAIC': 1,'CORONA': 1,
    'RIBOZYME': 2,
    'RIBOSWITCH': 3,
    'APTAMER': 4,'MANGO': 4,'CORN': 4,
    'PRIMER': 5,'MEIMPG': 5,'METHYLIMIDAZOLE': 5,'POLYMERASE': 5,'TRANSCRIPTION': 5,'SIGNAL RECOGNITION PARTICLE': 5,
    'SPLICING': 6,'SPLICEOSOME': 6,
}

rev_kwd_dict = {
    0: 'ribosomal rna',
    1: 'viral rna',
    2: 'ribozyme',
    3: 'riboswitch',
    4: 'aptamer',
    5: 'primer complex',
    6: 'splicing-related',
    7: 'other structure',
}

CIF_HEADER_BYTES = 2_000_000


def normalize_text(text):
    clean_text = re.sub('[^A-Z0-9 ]', ' ', str(text).upper())
    return re.sub(r'\s+', ' ', clean_text).strip()


def match_rna_types(text):
    clean_text = f' {normalize_text(text)} '
    matched_types = set()

    for keyword, rna_type in kwd_dict.items():
        keyword_text = normalize_text(keyword)
        if f' {keyword_text} ' in clean_text:
            matched_types.add(rna_type)

    return matched_types


def classify_rna_structure(path_pdb):
    type_set = set()

    with open(path_pdb, 'r') as f_in:
        for line in f_in:
            line_list = normalize_text(line).split()
            if not line_list:
                continue

            line_id = line_list[0]
            if line_id == 'ATOM':
                break

            if line_id in ['TITLE', 'COMPND', 'KEYWDS', 'JRNL']:
                type_set.update(match_rna_types(line))

    return sorted(type_set)


def classify_rna_structure_cif(path_cif):
    type_set = set()
    try:
        with open(path_cif, 'r', errors='replace') as f_in:
            header_text = f_in.read(CIF_HEADER_BYTES)
    except Exception as e:
        print(f'Error reading {path_cif}: {e}')
        return []

    atom_site_index = header_text.find('\n_atom_site.')
    if atom_site_index != -1:
        header_text = header_text[:atom_site_index]

    type_set.update(match_rna_types(header_text))
    return sorted(type_set)


def adjust_rna_types(type_list):
    type_list = list(type_list)
    if len(type_list) == 0:
        type_list.append(7)

    if 3 in type_list and 4 in type_list:
        type_list.remove(4)

    if 1 in type_list and len(type_list) > 1:
        type_list.remove(1)

    return type_list


def extract_pdb_id(filename):
    left_part = filename.split('_', 1)[0]
    return left_part[:4].lower()


def find_cif_file(pdb_id, cif_dirs):
    for cif_dir in cif_dirs:
        path_cif = Path(cif_dir) / f'{pdb_id}.cif'
        if path_cif.exists():
            return path_cif
    return None


def classify_pocket_files(pocket_dir, cif_dirs):
    results = []
    type_cache = {}

    for pocket_file in sorted(Path(pocket_dir).iterdir()):
        if not pocket_file.is_file():
            continue

        filename = pocket_file.name
        pdb_id = extract_pdb_id(filename)

        if pdb_id not in type_cache:
            path_cif = find_cif_file(pdb_id, cif_dirs)
            if path_cif is None:
                type_cache[pdb_id] = ['cif not found']
            else:
                type_list = classify_rna_structure_cif(path_cif)
                adjusted_type_list = adjust_rna_types(type_list)
                type_cache[pdb_id] = [rev_kwd_dict[t] for t in adjusted_type_list]

        results.append((filename, '; '.join(type_cache[pdb_id])))

    return results


def write_results(results, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['filename', 'rna_type'])
        writer.writerows(results)


def count_rna_types(results):
    counts = Counter()

    for _, rna_types in results:
        for rna_type in rna_types.split('; '):
            if rna_type:
                counts[rna_type] += 1

    return counts


def main(pocket_dir, cif_dirs, output_file):
    results = classify_pocket_files(
        pocket_dir=pocket_dir,
        cif_dirs=[Path(path) for path in cif_dirs],
    )
    write_results(results, output_file)

    counts = count_rna_types(results)

    print(f'Wrote {len(results)} rows to {output_file}')
    print('RNA type pocket counts:')
    for rna_type in list(rev_kwd_dict.values()) + sorted(set(counts) - set(rev_kwd_dict.values())):
        print(f'{rna_type}: {counts.get(rna_type, 0)}')


if __name__ == '__main__':
    pocket_dir = Path('dataset/inputs/pockets')
    cif_dirs = (Path('dataset/inputs/cifs'))
    output_file = Path('dataset/outputs/rna_types.csv')
    main(pocket_dir, cif_dirs, output_file)
