import os
import pandas as pd

ROOT = "/content"

datasets = ["BRCA", "BLCA", "OV"]
splits = ["10", "20", "30", "40", "50"]

for dataset in datasets:

    print("\n" + "="*80)
    print(f"DATASET: {dataset}")
    print("="*80)

    dataset_path = os.path.join(ROOT, dataset)

    for split in splits:

        split_path = os.path.join(dataset_path, split)

        if not os.path.exists(split_path):
            print(f"\nSplit {split} not found.")
            continue

        dna = pd.read_csv(os.path.join(split_path, "DNA.csv"))
        mrna = pd.read_csv(os.path.join(split_path, "mRNA.csv"))
        mirna = pd.read_csv(os.path.join(split_path, "miRNA.csv"))

        dna_ids = set(dna.iloc[:,0])
        mrna_ids = set(mrna.iloc[:,0])
        mirna_ids = set(mirna.iloc[:,0])

        all_ids = dna_ids | mrna_ids | mirna_ids
        total = len(all_ids)

        print(f"\nSplit {split}")
        print(f"Total patients: {total}")

        for name, ids in [
            ("DNA", dna_ids),
            ("mRNA", mrna_ids),
            ("miRNA", mirna_ids)
        ]:

            available = len(ids)
            missing = total - available

            print(
                f"{name:6s}: "
                f"Available = {available:4d} | "
                f"Missing = {missing:4d} | "
                f"Missing Rate = {100*missing/total:.2f}%"
            )
