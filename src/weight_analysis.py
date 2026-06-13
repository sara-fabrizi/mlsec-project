import os
import gc
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt  
from scipy.stats import ks_2samp, entropy, gaussian_kde
 

def load_weights_from_path(path, device):
    
    model = get_resnet18(pretrained=False).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
 
def analyze_layer_distributions(model_wm, model_clean):
    
    results = {}
 
    layers_to_check = {
        "layer1_conv1": model_wm.layer1[0].conv1,
        "layer2_conv1": model_wm.layer2[0].conv1,
        "layer3_conv1": model_wm.layer3[0].conv1,
        "layer4_conv2": model_wm.layer4[1].conv2
    }
 
    layers_clean = {
        "layer1_conv1": model_clean.layer1[0].conv1,
        "layer2_conv1": model_clean.layer2[0].conv1,
        "layer3_conv1": model_clean.layer3[0].conv1,
        "layer4_conv2": model_clean.layer4[1].conv2
    }
 
    print(f"{'LAYER':<15} | {'KS STAT':<8} | {'P-VALUE':<10} | {'COSINE SIM':<10} | {'L2 DIFF':<8} | {'ENTROPY DIFF':<12}")
    print("-" * 85)
 
    np.random.seed(42)
 
    for name in layers_to_check.keys():
        w_wm_full = layers_to_check[name].weight.detach().cpu().numpy().flatten()
        w_clean_full = layers_clean[name].weight.detach().cpu().numpy().flatten()
 
        # L2 Norms
        norm_wm = np.linalg.norm(w_wm_full)
        norm_clean = np.linalg.norm(w_clean_full)
        l2_diff = np.abs(norm_wm - norm_clean)
 
        # Sampling
        sample_size = min(50000, len(w_wm_full))
        indices = np.random.choice(len(w_wm_full), size=sample_size, replace=False)
 
        # Explicit memory copy to decouple array slices from parent tensor scope
        w_wm_sample = np.copy(w_wm_full[indices])
        w_clean_sample = np.copy(w_clean_full[indices])
 
        # KS test
        ks_stat, p_value = ks_2samp(w_wm_sample, w_clean_sample)
 
        # Cosine Similarity
        dot_product = np.dot(w_wm_full, w_clean_full)
        cosine_sim = dot_product / (norm_wm * norm_clean) if (norm_wm * norm_clean) > 0 else 0.0
 
        # Shannon Entropy Discrepancy
        hist_wm, _ = np.histogram(w_wm_sample, bins=100, density=True)
        hist_clean, _ = np.histogram(w_clean_sample, bins=100, density=True)
        
        # Add epsilon boundary to prevent undefined log(0) mathematical states
        ent_wm = entropy(hist_wm + 1e-12)
        ent_clean = entropy(hist_clean + 1e-12)
        entropy_diff = np.abs(ent_wm - ent_clean)
 
        results[name] = {
            "ks_stat": ks_stat,
            "p_value": p_value,
            "cosine_sim": cosine_sim,
            "l2_norm_diff": l2_diff,
            "entropy_diff": entropy_diff,
            "w_wm_sample": w_wm_sample,  
            "w_clean_sample": w_clean_sample
        }
 
        print(f"{name:<15} | {ks_stat:<8.4f} | {p_value:<10.4e} | {cosine_sim:<10.4f} | {l2_diff:<8.4f} | {entropy_diff:<12.4f}")
 
    return results
 
def plot_forensic_results(results_scratch, results_pretrained):

    layers = list(results_scratch.keys())
    x = np.arange(len(layers))
    width = 0.35
 
    # -----------------------------------------------------------------
    # FIGURE 1: GEOMETRIC ALIGNMENT (COSINE SIMILARITY BAR CHART)
    # Question: Are the weight vectors spatially aligned?
    # -----------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    cos_scratch = [results_scratch[l]["cosine_sim"] for l in layers]
    cos_pre = [results_pretrained[l]["cosine_sim"] for l in layers]
 
    plt.bar(x - width/2, cos_scratch, width, label='FromScratch vs Clean Baseline', color='#e74c3c', edgecolor='black', alpha=0.9)
    plt.bar(x + width/2, cos_pre, width, label='Pretrained vs Clean Baseline', color='#2ecc71', edgecolor='black', alpha=0.9)
 
    plt.ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
    plt.title('Figure 1 - Forensic Geometric Inspection: Spatial Feature Alignment', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(x, layers, fontsize=11)
    plt.ylim(0, 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(fontsize=11, loc='lower left')
    plt.tight_layout()
    plt.savefig('figure1_cosine_similarity.png', dpi=300)
    plt.close()
    print("[Figure 1] Generated: 'figure1_cosine_similarity.png'")
 
    # -----------------------------------------------------------------
    # FIGURE 2: WEIGHT DISTRIBUTION DENSITY ANALYSIS (KDE + HIST)
    # Question: Are the continuous parameter distributions deviating?
    # -----------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    l4_data_wm = results_pretrained["layer4_conv2"]["w_wm_sample"]
    l4_data_clean = results_pretrained["layer4_conv2"]["w_clean_sample"]
 
    # Step 1: Render light underlying histograms for baseline transparency
    plt.hist(l4_data_clean, bins=100, density=True, alpha=0.25, label='Clean Baseline Hist', color='#34495e', histtype='stepfilled')
    plt.hist(l4_data_wm, bins=100, density=True, alpha=0.25, label='Pretrained Hist', color='#2ecc71', histtype='stepfilled')
 
    # Step 2: Compute and overlay smooth Kernel Density Estimations (KDE)
    kde_range = np.linspace(min(l4_data_clean.min(), l4_data_wm.min()), max(l4_data_clean.max(), l4_data_wm.max()), 1000)
    plt.plot(kde_range, gaussian_kde(l4_data_clean)(kde_range), color='#2c3e50', linewidth=3, label='Clean Baseline KDE')
    plt.plot(kde_range, gaussian_kde(l4_data_wm)(kde_range), color='#2ecc71', linewidth=3, label='Pretrained KDE')
 
    plt.xlabel("Weight Value", fontsize=12, fontweight='bold')
    plt.ylabel("Density", fontsize=12, fontweight='bold')
    plt.title("Figure 2 - Weight Distribution Analysis (layer4_conv2)", fontsize=13, fontweight='bold', pad=15)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure2_weight_distribution.png", dpi=300)
    plt.close()
    print("[Figure 2] Generated: 'figure2_weight_distribution.png'")
 
    # -----------------------------------------------------------------
    # FIGURE 3: GEOMETRIC ENERGY DELTA (L2 NORM DIFFERENCE)
    # Question: How far have the models drifted in terms of overall energy?
    # -----------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    l2_scratch = [results_scratch[l]["l2_norm_diff"] for l in layers]
    l2_pre = [results_pretrained[l]["l2_norm_diff"] for l in layers]
 
    plt.bar(x - width/2, l2_scratch, width, label="FromScratch", color='#e74c3c', edgecolor='black', alpha=0.9)
    plt.bar(x + width/2, l2_pre, width, label="Pretrained", color='#2ecc71', edgecolor='black', alpha=0.9)
 
    plt.xticks(x, layers, fontsize=11)
    plt.ylabel("$\Delta L_2$ Norm", fontsize=12, fontweight='bold')
    plt.title("Figure 3 - Energy Difference Between Clean and Watermarked Models", fontsize=13, fontweight='bold', pad=15)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure3_l2_difference.png", dpi=300)
    plt.close()
    print("[Figure 3] Generated: 'figure3_l2_difference.png'")
 
    # -----------------------------------------------------------------
    # FIGURE 4: DUAL-AXIS LAYER-WISE FORENSIC SIGNATURE (INSIGHTS)
    # Question: Where is the watermark footprint concentrated along the network?
    # -----------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(11, 6))
 
    ks_scratch = [results_scratch[l]["ks_stat"] for l in layers]
    ks_pre = [results_pretrained[l]["ks_stat"] for l in layers]
    ent_scratch = [results_scratch[l]["entropy_diff"] for l in layers]
    ent_pre = [results_pretrained[l]["entropy_diff"] for l in layers]
 
    # Left Axis: Kolmogorov-Smirnov Shape Statistic
    ax1.plot(x, ks_scratch, 'o--', color='#e74c3c', linewidth=2.5, markersize=8, label='KS: FromScratch')
    ax1.plot(x, ks_pre, 'o-', color='#2c3e50', linewidth=2.5, markersize=8, label='KS: Pretrained')
    ax1.set_ylabel("KS Statistic", fontsize=12, fontweight='bold', color='#2c3e50')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers, fontsize=11)
    ax1.grid(linestyle=':', alpha=0.5)
 
    # Right Axis: Twin axis for Shannon Entropy Discrepancy
    ax2 = ax1.twinx()
    ax2.plot(x, ent_scratch, 's--', color='#f39c12', linewidth=2, markersize=7, label='Entropy: FromScratch')
    ax2.plot(x, ent_pre, 's-', color='#2ecc71', linewidth=2, markersize=7, label='Entropy: Pretrained')
    ax2.set_ylabel("Entropy Difference", fontsize=12, fontweight='bold', color='#2ecc71')
 
    # Unified layout legends across different axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
 
    plt.title("Figure 4 - Layer-wise Forensic Signature: Shape Shift vs Information Disorder", fontsize=13, fontweight='bold', pad=15)
    fig.tight_layout()
    plt.savefig("figure4_forensic_signature.png", dpi=300)
    plt.close()
    print("[Figure 4] Generated: 'figure4_forensic_signature.png'")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Executing Optimized Forensic Weight Inspection on Destination Device: [{device}]")
 
    path_scratch = "../checkpoints/model_fromscratch.pth"
    path_pretrained = "../checkpoints/model_pretrained.pth"
    path_clean = "../checkpoints/model_nowm.pth"  
 
    if not (os.path.exists(path_scratch) and os.path.exists(path_pretrained) and os.path.exists(path_clean)):
        print("Critical Error: Core ResNet-18 checkpoint files missing. Execution halted.")
        return
 
    print("\n[1/3] Anchoring Reference Control Model (CLEAN)...")
    model_clean = load_weights_from_path(path_clean, device)
 
    print("[2/3] Loading Target Checkpoint: FROMSCRATCH (Watermarked)...")
    model_scratch = load_weights_from_path(path_scratch, device)
 
    print("\n" + "="*95)
    print("FORENSIC ANALYSIS: FROM-SCRATCH PARADIGM (FROMSCRATCH) vs CLEAN BASELINE")
    print("="*95)
    results_scratch = analyze_layer_distributions(model_scratch, model_clean)
 
    del model_scratch
    gc.collect()
    torch.cuda.empty_cache()
 
    print("\n[3/3] Loading Target Checkpoint: PRETRAINED (Transfer Learning + Watermarked)...")
    model_pretrained = load_weights_from_path(path_pretrained, device)
 
    print("\n" + "="*95)
    print("FORENSIC ANALYSIS: PRE-TRAINED PARADIGM (PRETRAINED) vs CLEAN BASELINE")
    print("="*95)
    results_pretrained = analyze_layer_distributions(model_pretrained, model_clean)
 
    print("\n" + "="*75)
    print("GENERATING PRESENTATION-OPTIMIZED FORENSIC GRAPH SUITE")
    print("="*75)
    plot_forensic_results(results_scratch, results_pretrained)
 
    del model_pretrained, model_clean
    torch.cuda.empty_cache()
    gc.collect()
    print("\nForensic inspection pipeline successfully finalized. Workstation memory flushed.")
 
if __name__ == '__main__':
    main()