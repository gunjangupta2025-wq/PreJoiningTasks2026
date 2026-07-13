import numpy as np
import matplotlib.pyplot as plt
import torch

# ---------------------------------------------------------
# GLOBAL CONFIGURATION: Predictability scores from Task 1 & 2
# ---------------------------------------------------------
PREDICTABILITY_SCORES = {
    'BRCA': {
        # Pairwise (Source -> Target)
        ('DNA', 'mRNA'): 0.7185, ('DNA', 'miRNA'): 0.5259,
        ('mRNA', 'DNA'): 0.7560, ('mRNA', 'miRNA'): 0.5266,
        ('miRNA', 'DNA'): 0.6632, ('miRNA', 'mRNA'): 0.7588,
        # Dual-Source Averaged (Target)
        ('Dual', 'mRNA'): 0.9444, ('Dual', 'DNA'): 0.8680, ('Dual', 'miRNA'): 0.8073
    },
    'BLCA': {
        ('DNA', 'mRNA'): 0.6907, ('DNA', 'miRNA'): 0.4868,
        ('mRNA', 'DNA'): 0.8177, ('mRNA', 'miRNA'): 0.5717,
        ('miRNA', 'DNA'): 0.7808, ('miRNA', 'mRNA'): 0.7496,
        ('Dual', 'mRNA'): 0.8825, ('Dual', 'DNA'): 0.9135, ('Dual', 'miRNA'): 0.8413
    },
    'OV': {
        ('DNA', 'mRNA'): 0.4991, ('DNA', 'miRNA'): 0.3459,
        ('mRNA', 'DNA'): 0.5444, ('mRNA', 'miRNA'): 0.4479,
        ('miRNA', 'DNA'): 0.4133, ('miRNA', 'mRNA'): 0.4939,
        ('Dual', 'mRNA'): 0.8756, ('Dual', 'DNA'): 0.9180, ('Dual', 'miRNA'): 0.8744
    }
}

def fill_patient(available_modalities, dataset_name, trained_models, inverse_pca_transformers=None):
    """
    Fills a single missing modality based on remaining available ones.

    Parameters:
    - available_modalities: dict of {modality_name: embedding_tensor}
    - dataset_name: str ('BRCA', 'BLCA', 'OV')
    - trained_models: dict of dicts { (source, target): trained_model_object }
    - inverse_pca_transformers: dict of {modality_name: pca_object} (Optional, if mapping to raw space)

    Returns:
    - filled_dict: dict of {modality: (embedding, confidence)}
    """
    all_modalities = {'DNA', 'mRNA', 'miRNA'}
    present = set(available_modalities.keys())
    missing = all_modalities - present

    # Initialize output dictionary with confidence=1.0 for genuinely present modalities
    filled_dict = {mod: (emb, 1.0) for mod, emb in available_modalities.items()}

    # 1. No-op Condition: No modalities are missing
    if len(missing) == 0:
        return filled_dict

    missing_modality = list(missing)[0]
    scores = PREDICTABILITY_SCORES[dataset_name]

    # 2. Case A: Two source modalities are available (Dual-Source Prediction)
    if len(present) == 2:
        sources = list(present)
        src1, src2 = sources[0], sources[1]

        # Get individual predictions
        model1 = trained_models[(src1, missing_modality)]
        model2 = trained_models[(src2, missing_modality)]

        # Predict (Assuming inputs are torch tensors or numpy arrays; handling as torch here)
        with torch.no_grad():
            pred1 = model1(available_modalities[src1])
            pred2 = model2(available_modalities[src2])

            # Blend using the validated averaging routine in the uncompressed space if transformers exist
            if inverse_pca_transformers and missing_modality in inverse_pca_transformers:
                pca = inverse_pca_transformers[missing_modality]
                raw_pred1 = pca.inverse_transform(pred1.cpu().numpy())
                raw_pred2 = pca.inverse_transform(pred2.cpu().numpy())
                raw_avg = (raw_pred1 + raw_pred2) / 2.0
                filled_embedding = torch.tensor(pca.transform(raw_avg))
            else:
                filled_embedding = (pred1 + pred2) / 2.0

        # Assign blended confidence score from the Task 2 dual-source benchmarks
        confidence = scores[('Dual', missing_modality)]

    # 3. Case B: Only one source modality is available (Single-Source Prediction)
    elif len(present) == 1:
        src = list(present)[0]
        model = trained_models[(src, missing_modality)]

        with torch.no_grad():
            filled_embedding = model(available_modalities[src])

        # Assign individual pairwise predictability score
        confidence = scores[(src, missing_modality)]

    filled_dict[missing_modality] = (filled_embedding, confidence)
    return filled_dict

# ---------------------------------------------------------
# Calibration Check and Plotting Utility
# ---------------------------------------------------------
def run_calibration_check(dataset_name, triplet_patients_data, trained_models, inverse_pca_transformers=None):
    """
    Simulates missing modalities on perfect data to evaluate confidence vs reconstruction error.
    """
    confidences = []
    errors = []

    modalities = ['DNA', 'mRNA', 'miRNA']

    for patient_data in triplet_patients_data:
        # patient_data is a dict of {'DNA': emb, 'mRNA': emb, 'miRNA': emb}
        for target_to_drop in modalities:
            # Simulate missingness by removing one modality
            available = {mod: emb for mod, emb in patient_data.items() if mod != target_to_drop}
            ground_truth = patient_data[target_to_drop]

            # Fill via our utility
            filled = fill_patient(available, dataset_name, trained_models, inverse_pca_transformers)
            predicted_embedding, confidence = filled[target_to_drop]

            # Calculate actual reconstruction error (Mean Squared Error)
            if isinstance(predicted_embedding, torch.Tensor):
                pred_np = predicted_embedding.detach().cpu().numpy()
                gt_np = ground_truth.detach().cpu().numpy()
            else:
                pred_np, gt_np = predicted_embedding, ground_truth

            mse_error = np.mean((gt_np - pred_np) ** 2)

            confidences.append(confidence)
            errors.append(mse_error)

    # Plotting the Calibration curve
    plt.figure(figsize=(7, 5))
    plt.scatter(confidences, errors, alpha=0.6, color='crimson', edgecolors='k')

    # Fit a trendline to show negative correlation
    if len(set(confidences)) > 1:
        z = np.polyfit(confidences, errors, 1)
        p = np.poly1d(z)
        plt.plot(sorted(confidences), p(sorted(confidences)), "r--", label="Trendline")

    plt.title(f"Confidence vs. Reconstruction Error ({dataset_name})")
    plt.xlabel("Predicted Confidence (Informational Predictability)")
    plt.ylabel("Actual Reconstruction Error (MSE)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(f"{dataset_name}_confidence_calibration.png", dpi=300)
    plt.close()

    print(f" Saved calibration plot for {dataset_name}.")
