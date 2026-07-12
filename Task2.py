import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cosine

# =====================================================================
# GLOBAL CONFIGURATION & SEEDING
# =====================================================================
DATASET_BASE_PATH = "/content/drive/MyDrive/split_data"
DATASETS = ['BRCA', 'BLCA', 'OV']
SEEDS = ['10', '20', '30', '40', '50']

PAIRS = [
    ('DNA', 'mRNA'), ('DNA', 'miRNA'),
    ('mRNA', 'DNA'), ('mRNA', 'miRNA'),
    ('miRNA', 'DNA'), ('miRNA', 'mRNA')
]

RECONSTRUCTION_TRIADS = [
    ('mRNA', 'DNA', 'miRNA'),
    ('DNA', 'mRNA', 'miRNA'),
    ('miRNA', 'DNA', 'mRNA')
]

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
os.makedirs("/content/saved_predictors", exist_ok=True)

# =====================================================================
# ADAPTIVE NEURAL NETWORK ARCHITECTURE
# =====================================================================
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

def compute_cosine_similarity(y_true, y_pred):
    similarities = []
    for true_vec, pred_vec in zip(y_true, y_pred):
        if np.count_nonzero(true_vec) > 0 and np.count_nonzero(pred_vec) > 0:
            similarities.append(1.0 - cosine(true_vec, pred_vec))
        else:
            similarities.append(0.0)
    return np.mean(similarities)

# Initialize master metric tracking tables
results_data = {d: {p: {'mlp_cos': [], 'linear_cos': []} for p in [f"{pair[0]}->{pair[1]}" for pair in PAIRS]} for d in DATASETS}
recon_data = {d: {t[0]: {'s1_cos': [], 's2_cos': [], 'dual_cos': []} for t in RECONSTRUCTION_TRIADS} for d in DATASETS}

# =====================================================================
# PIPELINE RUNTIME EXECUTION
# =====================================================================
for dataset in DATASETS:
    print(f"\n================ Processing Dataset Matrix: {dataset} ================")

    for seed in SEEDS:
        folder_path = os.path.join(DATASET_BASE_PATH, dataset, seed)
        if not os.path.exists(folder_path):
            continue

        print(f" -> Processing Seed Folder: {seed}")

        raw_matrices = {}
        try:
            for modality in ['DNA', 'mRNA', 'miRNA']:
                file_path = os.path.join(folder_path, f"{modality}.csv")
                raw_matrices[modality] = pd.read_csv(file_path, index_col=0).dropna(how='all')
        except FileNotFoundError:
            print(f"    [Warning] Missing CSV files in {dataset}/{seed}. Skipping seed folder.")
            continue

        # Storage dictionary to preserve PCA fits from Task 1 for proper usage in Task 2
        fitted_pcas = {}

        # -------------------------------------------------------------
        # TASK 1: PAIRWISE MAPPING WITH LINEAR BASELINES & ERROR BARS
        # -------------------------------------------------------------
        for source, target in PAIRS:
            pair_key = f"{source}->{target}"
            source_df = raw_matrices[source]
            target_df = raw_matrices[target]

            overlapping_patients = sorted(list(set(source_df.index) & set(target_df.index)))
            if len(overlapping_patients) < 5:
                continue

            X_all = source_df.loc[overlapping_patients].values
            Y_all = target_df.loc[overlapping_patients].values

            X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = train_test_split(
                X_all, Y_all, test_size=0.20, random_state=42
            )

            max_components_x = min(X_train_raw.shape[0], X_train_raw.shape[1])
            n_comp_x = min(128, max_components_x)

            max_components_y = min(Y_train_raw.shape[0], Y_train_raw.shape[1])
            n_comp_y = min(128, max_components_y)

            # Anti-leakage isolated PCA fitting
            pca_x = PCA(n_components=n_comp_x, random_state=42)
            X_train = pca_x.fit_transform(X_train_raw)
            X_test = pca_x.transform(X_test_raw)

            pca_y = PCA(n_components=n_comp_y, random_state=42)
            Y_train = pca_y.fit_transform(Y_train_raw)
            Y_test = pca_y.transform(Y_test_raw)

            # Save the trained PCA fits mapped under the current unique key
            fitted_pcas[pair_key] = (pca_x, pca_y)

            # Baseline: Fit standard Linear Regression model
            linear_baseline = LinearRegression()
            linear_baseline.fit(X_train, Y_train)
            linear_baseline_preds = linear_baseline.predict(X_test)
            linear_cos = compute_cosine_similarity(Y_test, linear_baseline_preds)
            results_data[dataset][pair_key]['linear_cos'].append(linear_cos)

            # Deep Learning: Fit dynamic MLP model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_train_t = torch.FloatTensor(X_train).to(device)
            Y_train_t = torch.FloatTensor(Y_train).to(device)
            X_test_t = torch.FloatTensor(X_test).to(device)

            set_seed(int(seed))
            model = CrossOmicsPredictor(input_dim=n_comp_x, output_dim=n_comp_y).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            model.train()
            for epoch in range(80):
                permutation = torch.randperm(len(X_train_t))
                for i in range(0, len(X_train_t), 16):
                    indices = permutation[i:i + 16]
                    batch_x, batch_y = X_train_t[indices], Y_train_t[indices]
                    optimizer.zero_grad()
                    loss = criterion(model(batch_x), batch_y)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                mlp_preds = model(X_test_t).cpu().numpy()

            mlp_cos = compute_cosine_similarity(Y_test, mlp_preds)
            results_data[dataset][pair_key]['mlp_cos'].append(mlp_cos)

            torch.save(model.state_dict(), f"/content/saved_predictors/{dataset}_seed{seed}_{source}_to_{target}.pth")

        # -------------------------------------------------------------
        # TASK 2: MULTI-SOURCE MISSING MODALITY RECONSTRUCTION
        # -------------------------------------------------------------
        triplet_patients = sorted(list(set(raw_matrices['DNA'].index) &
                                       set(raw_matrices['mRNA'].index) &
                                       set(raw_matrices['miRNA'].index)))

        if len(triplet_patients) >= 5:
            _, test_patients = train_test_split(triplet_patients, test_size=0.20, random_state=42)

            for target, source1, source2 in RECONSTRUCTION_TRIADS:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Retrieve original PCA engines to manage transformations
                pca_s1_engine, pca_t_engine1 = fitted_pcas[f"{source1}->{target}"]
                pca_s2_engine, pca_t_engine2 = fitted_pcas[f"{source2}->{target}"]

                raw_target = raw_matrices[target].loc[test_patients].values
                raw_s1 = raw_matrices[source1].loc[test_patients].values
                raw_s2 = raw_matrices[source2].loc[test_patients].values

                # Transform sources using respective baseline configurations
                s1_test = torch.FloatTensor(pca_s1_engine.transform(raw_s1)).to(device)
                s2_test = torch.FloatTensor(pca_s2_engine.transform(raw_s2)).to(device)

                m1 = CrossOmicsPredictor(input_dim=pca_s1_engine.n_components_, output_dim=pca_t_engine1.n_components_).to(device)
                m1.load_state_dict(torch.load(f"/content/saved_predictors/{dataset}_seed{seed}_{source1}_to_{target}.pth"))
                m1.eval()

                m2 = CrossOmicsPredictor(input_dim=pca_s2_engine.n_components_, output_dim=pca_t_engine2.n_components_).to(device)
                m2.load_state_dict(torch.load(f"/content/saved_predictors/{dataset}_seed{seed}_{source2}_to_{target}.pth"))
                m2.eval()

                with torch.no_grad():
                    pred_s1_pca = m1(s1_test).cpu().numpy()
                    pred_s2_pca = m2(s2_test).cpu().numpy()

                # FIX: Invert PCA embeddings back to original space to safely resolve shape disparities
                pred_s1_raw = pca_t_engine1.inverse_transform(pred_s1_pca)
                pred_s2_raw = pca_t_engine2.inverse_transform(pred_s2_pca)

                # Compute mathematical average in the aligned original dimensional space
                pred_dual_raw = (pred_s1_raw + pred_s2_raw) / 2.0

                # Establish ground truth and project values into a shared workspace for comparison
                t_test_s1 = pca_t_engine1.transform(raw_target)
                t_test_s2 = pca_t_engine2.transform(raw_target)
                pred_dual_pca = pca_t_engine1.transform(pred_dual_raw)

                recon_data[dataset][target]['s1_cos'].append(compute_cosine_similarity(t_test_s1, pred_s1_pca))
                recon_data[dataset][target]['s2_cos'].append(compute_cosine_similarity(t_test_s2, pred_s2_pca))
                recon_data[dataset][target]['dual_cos'].append(compute_cosine_similarity(t_test_s1, pred_dual_pca))

# =====================================================================
# METRIC AGGREGATION & REPORT COMPILATION
# =====================================================================
final_rows = []
for pair_key in [f"{p[0]}->{p[1]}" for p in PAIRS]:
    row = {'Mapping Direction': pair_key}
    for dataset in DATASETS:
        mlp_vals = results_data[dataset][pair_key]['mlp_cos']
        lin_vals = results_data[dataset][pair_key]['linear_cos']

        row[f"{dataset} (MLP)"] = f"{np.mean(mlp_vals):.4f}±{np.std(mlp_vals):.4f}" if mlp_vals else "N/A"
        row[f"{dataset} (Linear Baseline)"] = f"{np.mean(lin_vals):.4f}±{np.std(lin_vals):.4f}" if lin_vals else "N/A"
    final_rows.append(row)

df_pairs = pd.DataFrame(final_rows)
df_pairs.to_csv("/content/final_pairwise_with_baselines.csv", index=False)

recon_rows = []
for target in ['mRNA', 'DNA', 'miRNA']:
    row = {'Missing Target Modality': target}
    for dataset in DATASETS:
        s1 = recon_data[dataset][target]['s1_cos']
        s2 = recon_data[dataset][target]['s2_cos']
        dl = recon_data[dataset][target]['dual_cos']

        row[f"{dataset} (Source 1 Alone)"] = f"{np.mean(s1):.4f}±{np.std(s1):.4f}" if s1 else "N/A"
        row[f"{dataset} (Source 2 Alone)"] = f"{np.mean(s2):.4f}±{np.std(s2):.4f}" if s2 else "N/A"
        row[f"{dataset} (Dual-Source Averaged)"] = f"{np.mean(dl):.4f}±{np.std(dl):.4f}" if dl else "N/A"
    recon_rows.append(row)

df_recon = pd.DataFrame(recon_rows)
df_recon.to_csv("/content/missing_modality_reconstruction.csv", index=False)

print("\n=== TASK 1: PAIRWISE PREDICTION WITH LINEAR BASELINE ===")
print(df_pairs.to_string(index=False))

print("\n=== TASK 2: ONE SOURCE VS TWO SOURCES RECONSTRUCTION VALUE ===")
print(df_recon.to_string(index=False))
