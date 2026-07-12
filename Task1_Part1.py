import os
import random

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from scipy.spatial.distance import cosine

# ==========================================================
# GLOBAL CONFIGURATION
# ==========================================================

DATASET_BASE_PATH = "/content"

DATASETS = ["BRCA", "BLCA", "OV"]

# These folders are treated as independent folds
FOLDS = ["10", "20", "30", "40", "50"]

PAIRS = [
    ("DNA", "mRNA"),
    ("DNA", "miRNA"),
    ("mRNA", "DNA"),
    ("mRNA", "miRNA"),
    ("miRNA", "DNA"),
    ("miRNA", "mRNA")
]


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# ==========================================================
# RESULT TABLES
# ==========================================================

metrics_summary = {
    "cosine": pd.DataFrame(
        index=[f"{s}->{t}" for s, t in PAIRS],
        columns=DATASETS
    ),
    "rmse": pd.DataFrame(
        index=[f"{s}->{t}" for s, t in PAIRS],
        columns=DATASETS
    ),
    "r2": pd.DataFrame(
        index=[f"{s}->{t}" for s, t in PAIRS],
        columns=DATASETS
    ),
}

os.makedirs("/content/saved_predictors", exist_ok=True)

# ==========================================================
# MODEL
# ==========================================================

class CrossOmicsPredictor(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(CrossOmicsPredictor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# ==========================================================
# MAIN LOOP
# ==========================================================

for dataset in DATASETS:

    print(f"\n================ {dataset} ================")

    pair_accumulators = {
        f"{s}->{t}": {
            "cos": [],
            "rmse": [],
            "r2": []
        }
        for s, t in PAIRS
    }

    for fold in FOLDS:

        folder_path = os.path.join(DATASET_BASE_PATH, dataset, fold)

        if not os.path.exists(folder_path):
            continue

        print(f"Processing fold {fold}")

        current_seed = int(fold)
        set_seed(current_seed)

        raw_matrices = {}

        try:

            for modality in ["DNA", "mRNA", "miRNA"]:

                file_path = os.path.join(
                    folder_path,
                    f"{modality}.csv"
                )

                raw_matrices[modality] = (
                    pd.read_csv(file_path, index_col=0)
                    .dropna(how="all")
                )

        except FileNotFoundError:

            print(f"Missing files in {folder_path}")

            continue

        for source, target in PAIRS:

            pair_key = f"{source}->{target}"

            source_df = raw_matrices[source]
            target_df = raw_matrices[target]

            # Reviewer comment #3
            overlapping_patients = sorted(
                list(
                    set(source_df.index) &
                    set(target_df.index)
                )
            )

            if len(overlapping_patients) < 5:
                continue

            X_all = source_df.loc[
                overlapping_patients
            ].values

            Y_all = target_df.loc[
                overlapping_patients
            ].values

            # Reviewer comment #1
            X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = train_test_split(
                X_all,
                Y_all,
                test_size=0.20,
                random_state=42
            )

            # Determine the maximum valid number of PCA components
            n_components_x = min(
                128,
                X_train_raw.shape[0],
                X_train_raw.shape[1]
            )

            n_components_y = min(
                128,
                Y_train_raw.shape[0],
                Y_train_raw.shape[1]
            )

            pca_x = PCA(
                n_components=n_components_x,
                random_state=42
            )

            X_train = pca_x.fit_transform(X_train_raw)
            X_test = pca_x.transform(X_test_raw)

            pca_y = PCA(
                n_components=n_components_y,
                random_state=42
            )

            Y_train = pca_y.fit_transform(Y_train_raw)
            Y_test = pca_y.transform(Y_test_raw)

            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            X_train_t = torch.FloatTensor(X_train).to(device)
            Y_train_t = torch.FloatTensor(Y_train).to(device)
            X_test_t = torch.FloatTensor(X_test).to(device)

            model = CrossOmicsPredictor(
                input_dim=n_components_x,
                output_dim=n_components_y
            ).to(device)

            criterion = nn.MSELoss()

            optimizer = optim.Adam(
                model.parameters(),
                lr=0.001
            )

            # ==========================================================
            # TRAINING
            # ==========================================================

            model.train()

            # Explicit generator for reproducibility
            generator = torch.Generator(device=device)
            generator.manual_seed(current_seed)

            for epoch in range(80):

                permutation = torch.randperm(
                    len(X_train_t),
                    generator=generator,
                    device=device
                )

                for i in range(0, len(X_train_t), 16):

                    indices = permutation[i:i + 16]

                    batch_x = X_train_t[indices]
                    batch_y = Y_train_t[indices]

                    optimizer.zero_grad()

                    predictions = model(batch_x)

                    loss = criterion(predictions, batch_y)

                    loss.backward()

                    optimizer.step()

            # ==========================================================
            # EVALUATION
            # ==========================================================

            model.eval()

            with torch.no_grad():

                preds = model(X_test_t).cpu().numpy()

            # -----------------------------
            # Cosine Similarity
            # -----------------------------

            cosine_scores = []

            for true_vec, pred_vec in zip(Y_test, preds):

                if (
                    np.count_nonzero(true_vec) > 0
                    and np.count_nonzero(pred_vec) > 0
                ):

                    cosine_scores.append(
                        1 - cosine(true_vec, pred_vec)
                    )

                else:

                    cosine_scores.append(0.0)

            mean_cosine = np.mean(cosine_scores)

            # -----------------------------
            # RMSE
            # -----------------------------

            mean_rmse = root_mean_squared_error(
                Y_test,
                preds
            )

            # -----------------------------
            # R²
            # -----------------------------

            mean_r2 = r2_score(
                Y_test,
                preds,
                multioutput="uniform_average"
            )

            pair_accumulators[pair_key]["cos"].append(
                mean_cosine
            )

            pair_accumulators[pair_key]["rmse"].append(
                mean_rmse
            )

            pair_accumulators[pair_key]["r2"].append(
                mean_r2
            )

            torch.save(
                model.state_dict(),
                f"/content/saved_predictors/"
                f"{dataset}_fold{fold}_{source}_to_{target}.pth"
            )
                # ==========================================================
    # SUMMARIZE RESULTS ACROSS FOLDS
    # ==========================================================

    for pair in pair_accumulators.keys():

        cos_scores = pair_accumulators[pair]["cos"]
        rmse_scores = pair_accumulators[pair]["rmse"]
        r2_scores = pair_accumulators[pair]["r2"]

        if len(cos_scores) == 0:
            continue

        metrics_summary["cosine"].loc[pair, dataset] = (
            f"{np.mean(cos_scores):.4f} ± {np.std(cos_scores):.4f}"
        )

        metrics_summary["rmse"].loc[pair, dataset] = (
            f"{np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}"
        )

        metrics_summary["r2"].loc[pair, dataset] = (
            f"{np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}"
        )

# ==========================================================
# SAVE RESULTS
# ==========================================================

for metric_name, df in metrics_summary.items():

    print(
        f"\n================ "
        f"{metric_name.upper()} "
        f"================"
    )

    print(df)

    df.to_csv(
        f"/content/final_predictability_{metric_name}_scores.csv"
    )

print("\nPipeline execution completed successfully.")
print("Saved files:")

print("- /content/final_predictability_cosine_scores.csv")
print("- /content/final_predictability_rmse_scores.csv")
print("- /content/final_predictability_r2_scores.csv")
print("- /content/saved_predictors/")
