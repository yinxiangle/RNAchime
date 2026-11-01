import re
import sys
import time
from pathlib import Path
from pypdb import get_ligand_instance_file

POCKET_NAME_PATTERN = re.compile(r"^(?P<pdb_id>[A-Za-z0-9]{4})_#[^:]+:(?P<chain_id>[^:]+):(?P<ligand_name>[A-Za-z0-9]{3}):(?P<position>-?\d+)_BIND\.nx$")


def parse_pocket_filename(filename):
    match = POCKET_NAME_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(f"Cannot parse binding-pocket filename: {filename}")
    return match.group("pdb_id").lower(),match.group("chain_id"),match.group("ligand_name").upper(),int(match.group("position"))


def download_sdf(pdb_id,chain_id,position,max_retries=3,retry_delay=2):
    for attempt in range(max_retries + 1):
        try:
            sdf = get_ligand_instance_file(pdb_id,chain_id,position,filetype="sdf",verbosity=False)
            if sdf:
                return sdf
            raise RuntimeError("RCSB returned no SDF data")
        except Exception:
            if attempt == max_retries:
                raise
            print(f"Retrying {pdb_id}_{chain_id}:{position} ({attempt + 1}/{max_retries})...",file=sys.stderr)
            time.sleep(retry_delay)


def download_sdf_files(pocket_dir,output_dir):

    pocket_files = sorted(pocket_dir.glob("*.nx"))
    if not pocket_files:
        print(f"No .nx files found: {pocket_dir}",file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    failed = []

    for index,pocket_file in enumerate(pocket_files,1):
        try:
            pdb_id,chain_id,ligand_name,position = parse_pocket_filename(pocket_file.name)
            output_file = output_dir / f"{pdb_id}_{chain_id}:{ligand_name}:{position}.sdf"
            if output_file.exists():
                skipped += 1
                print(f"[{index}/{len(pocket_files)}] Skipped: {output_file.name}")
                continue
            sdf = download_sdf(pdb_id,chain_id,position)
            output_file.write_text(sdf,encoding="utf-8")
            downloaded += 1
            print(f"[{index}/{len(pocket_files)}] Downloaded: {output_file.name}")
        except Exception as error:
            failed.append(f"{pocket_file.name}: {error}")
            print(f"[{index}/{len(pocket_files)}] Failed: {pocket_file.name}: {error}",file=sys.stderr)

    print(f"Downloaded: {downloaded}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {len(failed)}")
    print(f"Output directory: {output_dir}")
    return 1 if failed else 0


def main():
    project_root = Path(__file__).resolve().parents[1]
    pocket_dir = project_root / "dataset/hariboss_data/data/pocket_nx/binding_pocket_nx"
    output_dir = project_root / "GEMS/cifandsdf/hariboss_data"
    return download_sdf_files(pocket_dir,output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
