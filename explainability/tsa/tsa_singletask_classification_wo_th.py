"""
TSA Explainability Script (Gas Sensors Only)
===========================================

This script mirrors `tsa_singletask_classification.py` but targets the
sensor-only SNN variant trained without temperature or humidity features.
It performs:
    ✅ Temporal Spike Attribution (TSA) for a selected sample
    ✅ TSA for random samples
    ✅ Aggregated TSA over multiple samples
    ✅ Class-specific TSA comparison plots
    ✅ Perturbation curves (top-k deletion vs random deletion)

Usage:
    python tsa_singletask_classification_wo_th.py \
        --checkpoint ../../trained_models/wine_timeseries_snn_wo_th_seq500_beta0.9_bs16_lr0.001_ep100.pth

Ensure spike data is generated with:
    python ../../multitask_net/utils/generate_spike_data.py --output_dir ../../data/spike_data_wo_th
"""

import argparse
import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1. MODEL DEFINITION (must match training script)
# -------------------------------------------------------
class TimeSeriesWineSNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, beta=0.9):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

        self.lif1 = snn.Leaky(beta=beta)
        self.lif2 = snn.Leaky(beta=beta)
        self.lif3 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk1_list, spk2_list, spk3_list, mem3_list = [], [], [], []

        for t in range(x.size(0)):
            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk1_list.append(spk1)
            spk2_list.append(spk2)
            spk3_list.append(spk3)
            mem3_list.append(mem3)

        return (
            torch.stack(mem3_list),
            torch.stack(spk3_list),
            torch.stack(spk1_list),
            torch.stack(spk2_list)
        )


# -------------------------------------------------------
# 2. TSA HELPERS
# -------------------------------------------------------

def compute_N_from_spike_times(spike_times_list, t_prime, beta, fan_in):
    N = 0.0
    # TSAS part
    for t_k in spike_times_list:
        if t_k <= t_prime:
            N += beta ** (t_prime - t_k)

    # TSANS part
    spike_set = set(spike_times_list)
    for t_k in range(t_prime + 1):
        if t_k not in spike_set:
            N -= (1.0 / fan_in) * (beta ** (t_prime - t_k))

    return N


@torch.no_grad()
def compute_tsa(model, x, beta, device):
    """
    Computes full TSA attributions: [T, features, classes]
    """
    x = x.to(device)
    mem_rec, _, h1_spk, h2_spk = model(x.unsqueeze(1))

    T = mem_rec.shape[0]
    F = x.shape[1]
    H1 = h1_spk.shape[2]
    H2 = h2_spk.shape[2]
    C = mem_rec.shape[2]

    W1_T = model.fc1.weight.t()
    W2_T = model.fc2.weight.t()
    W3_T = model.fc3.weight.t()

    # Extract spike times (work on device where data is)
    inp_spk = [torch.where(x[:, i] > 0)[0].tolist() for i in range(F)]
    h1_spk_t = [torch.where(h1_spk[:, 0, i] > 0)[0].tolist() for i in range(H1)]
    h2_spk_t = [torch.where(h2_spk[:, 0, i] > 0)[0].tolist() for i in range(H2)]
    out_mem = mem_rec.squeeze(1)

    fan1, fan2, fan3 = F, H1, H2
    atts = []

    for t in range(T):
        P_t = torch.softmax(out_mem[t], dim=0)
        P_diag = torch.diag(P_t)

        N0 = torch.tensor([compute_N_from_spike_times(inp_spk[i], t, beta, fan1) for i in range(F)], device=device, dtype=torch.float32)
        N1 = torch.tensor([compute_N_from_spike_times(h1_spk_t[i], t, beta, fan2) for i in range(H1)], device=device, dtype=torch.float32)
        N2 = torch.tensor([compute_N_from_spike_times(h2_spk_t[i], t, beta, fan3) for i in range(H2)], device=device, dtype=torch.float32)

        CI_t = (((((torch.diag(N0) @ W1_T) @ torch.diag(N1)) @ W2_T) @ torch.diag(N2)) @ W3_T) @ P_diag
        atts.append(CI_t)

    return torch.stack(atts)  # shape (T, F, C)


@torch.no_grad()
def predict_mem(model, x, device):
    mem_rec, _, _, _ = model(x.unsqueeze(1).to(device))
    membrane_sums = mem_rec.sum(dim=0)[0].cpu()
    pred = int(torch.argmax(membrane_sums))
    return pred, membrane_sums


# -------------------------------------------------------
# 3. VISUALIZATION UTILITIES
# -------------------------------------------------------
def plot_heatmap(att2d, feature_names, title="TSA", save_path=None):
    arr = att2d.cpu().numpy()
    plt.figure(figsize=(14, 6))
    sns.heatmap(arr.T, cmap="coolwarm", center=0,
                xticklabels=50, yticklabels=feature_names)
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Feature")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.show()
    plt.close()


def plot_class_comparison(att_full, feature_names, class_names, save_dir=None):
    C = att_full.shape[-1]
    for c in range(C):
        avg_att = att_full[:, :, :, c].mean(dim=0)
        save_path = None
        if save_dir:
            save_path = save_dir / f"tsa_class{class_names[c]}.png"
        plot_heatmap(avg_att, feature_names, title=f"Avg TSA for class {class_names[c]}", save_path=save_path)


def plot_perturbation_curve(model, x, attribution, device, feature_names, save_path=None):
    T, F = attribution.shape
    flat_imp = attribution.abs().flatten().cpu()

    x_cpu = x.cpu()
    spike_mask = (x_cpu > 0.5).flatten()
    spike_indices = torch.nonzero(spike_mask).flatten()

    if spike_indices.numel() == 0:
        print("No spikes to test perturbation.")
        return None

    k_values = list(range(1, min(50, spike_indices.numel()), 5))
    top_k_effects = []
    rand_k_effects = []

    base_pred, base_mem = predict_mem(model, x, device)
    base_val = base_mem[base_pred].item()

    perturbation_data = {
        'description': 'Analysis of how removing spikes affects model predictions',
        'baseline': {
            'predicted_class_idx': int(base_pred),
            'membrane_potentials': {
                f'class_{i}': float(base_mem[i]) for i in range(len(base_mem))
            },
            'predicted_class_membrane_potential': float(base_val)
        },
        'k_values_tested': k_values,
        'top_k_deletions': {
            'description': 'Removing the k most important spikes (highest TSA attribution)',
            'results': []
        },
        'random_k_deletions': {
            'description': 'Removing k random spikes for comparison',
            'results': []
        }
    }

    for k in k_values:
        imp_values = flat_imp[spike_indices]
        top_order = torch.topk(imp_values, k).indices
        top_idx = spike_indices[top_order]

        x_edit = x.clone()
        for idx in top_idx:
            t = (idx // F).item()
            f = (idx % F).item()
            x_edit[t, f] = 0.0

        pred_edit, mem_edit = predict_mem(model, x_edit, device)
        top_k_effect = base_val - mem_edit[base_pred].item()
        top_k_effects.append(top_k_effect)

        perturbation_data['top_k_deletions']['results'].append({
            'k': k,
            'predicted_class_idx': int(pred_edit),
            'predicted_class_changed': bool(pred_edit != base_pred),
            'membrane_potentials': {
                f'class_{i}': float(mem_edit[i]) for i in range(len(mem_edit))
            },
            'original_class_membrane_potential': float(mem_edit[base_pred]),
            'membrane_potential_drop': float(top_k_effect),
            'membrane_potential_drop_percentage': float(100 * top_k_effect / base_val) if base_val != 0 else 0.0
        })

        rand_perm = spike_indices[torch.randperm(spike_indices.numel())[:k]]
        x_rand = x.clone()
        for idx in rand_perm:
            t = (idx // F).item()
            f = (idx % F).item()
            x_rand[t, f] = 0.0
        pred_rand, mem_rand = predict_mem(model, x_rand, device)
        rand_k_effect = base_val - mem_rand[base_pred].item()
        rand_k_effects.append(rand_k_effect)

        perturbation_data['random_k_deletions']['results'].append({
            'k': k,
            'predicted_class_idx': int(pred_rand),
            'predicted_class_changed': bool(pred_rand != base_pred),
            'membrane_potentials': {
                f'class_{i}': float(mem_rand[i]) for i in range(len(mem_rand))
            },
            'original_class_membrane_potential': float(mem_rand[base_pred]),
            'membrane_potential_drop': float(rand_k_effect),
            'membrane_potential_drop_percentage': float(100 * rand_k_effect / base_val) if base_val != 0 else 0.0
        })

    perturbation_data['summary'] = {
        'top_k_average_drop': float(np.mean(top_k_effects)),
        'top_k_max_drop': float(np.max(top_k_effects)),
        'random_k_average_drop': float(np.mean(rand_k_effects)),
        'random_k_max_drop': float(np.max(rand_k_effects)),
        'top_k_more_effective': bool(np.mean(top_k_effects) > np.mean(rand_k_effects))
    }

    plt.figure(figsize=(10, 5))
    plt.plot(k_values, top_k_effects, label="Top-K deletion", linewidth=2)
    plt.plot(k_values, rand_k_effects, label="Random-K deletion", linewidth=2)
    plt.xlabel("K (number of deleted spikes)")
    plt.ylabel("Reduction in membrane potential")
    plt.title("Perturbation Curve")
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.show()
    plt.close()

    return perturbation_data


# -------------------------------------------------------
# 4. MAIN EXPLAINABILITY PIPELINE
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="TSA Explainability for sensor-only SNN wine classification"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../../trained_models/wine_timeseries_snn_wo_th_seq500_beta0.9_bs16_lr0.001_ep100.pth",
        help="Path to model checkpoint (relative to repo root or absolute)"
    )
    parser.add_argument(
        "--spike_data_dir",
        type=str,
        default="../../data/spike_data_wo_th",
        help="Directory containing spike data (default: ../../data/spike_data_wo_th)"
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Index of test sample to analyze (default: 0)"
    )
    parser.add_argument(
        "--num_random_samples",
        type=int,
        default=5,
        help="Number of random samples for aggregated TSA (default: 5)"
    )
    parser.add_argument(
        "--full_dataset_tsa",
        action="store_true",
        help="If set, compute TSA over the entire dataset for stable class-specific maps. Warning: Can be slow."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    script_dir = Path(__file__).parent
    results_base_dir = script_dir / "results" / "classification_time_series_wo_th"
    results_base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%H%M_%d%m%Y")
    results_dir = results_base_dir / timestamp
    results_dir.mkdir(exist_ok=True)

    print(f"Results will be saved to: {results_dir}\n")

    run_summary = {
        'timestamp': timestamp,
        'datetime': datetime.now().isoformat(),
        'device': str(device),
        'args': vars(args)
    }

    repo_root = script_dir.parent.parent

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (script_dir / checkpoint_path).resolve()

    spike_data_dir = Path(args.spike_data_dir)
    if not spike_data_dir.is_absolute():
        spike_data_dir = (script_dir / spike_data_dir).resolve()

    print("=" * 80)
    print("TSA EXPLAINABILITY PIPELINE (SENSORS ONLY)")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Spike data directory: {spike_data_dir}")
    print()

    print("Loading model checkpoint...")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    input_size = ckpt["input_size"]
    hidden1 = ckpt["hidden1"]
    hidden2 = ckpt["hidden2"]
    output_size = ckpt["output_size"]
    beta = ckpt.get("beta", 0.9)

    model = TimeSeriesWineSNN(input_size, hidden1, hidden2, output_size, beta).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✓ Model loaded: {input_size} -> {hidden1} -> {hidden2} -> {output_size}")
    print(f"  Beta: {beta}")
    print()

    run_summary['model'] = {
        'checkpoint': str(checkpoint_path),
        'input_size': input_size,
        'hidden1': hidden1,
        'hidden2': hidden2,
        'output_size': output_size,
        'beta': beta
    }

    print("Loading spike data...")
    if not spike_data_dir.exists():
        raise FileNotFoundError(
            f"Spike data directory not found: {spike_data_dir}\n"
            f"Please run: python {repo_root}/multitask_net/utils/generate_spike_data.py --output_dir ../../data/spike_data_wo_th"
        )

    spike_train_path = spike_data_dir / 'spike_train.npy'
    spike_test_path = spike_data_dir / 'spike_test.npy'
    y_train_path = spike_data_dir / 'y_train.npy'
    y_test_path = spike_data_dir / 'y_test.npy'
    config_path = spike_data_dir / 'config.npy'

    if not spike_test_path.exists():
        raise FileNotFoundError(f"Spike test data not found: {spike_test_path}")
    if not spike_train_path.exists():
        raise FileNotFoundError(f"Spike train data not found: {spike_train_path}")

    spike_train = np.load(spike_train_path)
    spike_test = np.load(spike_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)
    config = np.load(config_path, allow_pickle=True).item()

    print(f"✓ Spike train data loaded: {spike_train.shape}")
    print(f"✓ Spike test data loaded: {spike_test.shape}")
    print(f"  Train labels: {y_train.shape}")
    print(f"  Test labels: {y_test.shape}")
    print()

    feature_names = config.get('feature_names', [f"f{i}" for i in range(input_size)])
    class_names = config.get('label_encoder_classes', [f"c{i}" for i in range(output_size)])

    print(f"Feature names: {feature_names}")
    print(f"Class names: {class_names}")
    print()

    if args.sample_index >= len(spike_test):
        print(f"Warning: sample_index {args.sample_index} >= {len(spike_test)}, using 0")
        sample_index = 0
    else:
        sample_index = args.sample_index

    spikes = spike_test[sample_index]
    x = torch.tensor(spikes, dtype=torch.float32)
    true_label = y_test[sample_index]

    print(f"Analyzing sample {sample_index}:")
    print(f"  Shape: {x.shape}")
    print(f"  True label: {class_names[true_label]}")
    print()

    run_summary['selected_sample'] = {
        'index': sample_index,
        'true_label': class_names[true_label],
        'true_label_idx': int(true_label),
        'shape': list(x.shape)
    }

    print("=" * 80)
    print("A. TSA FOR SELECTED SAMPLE")
    print("=" * 80)

    att_full = compute_tsa(model, x, beta, device)
    pred, mem = predict_mem(model, x, device)
    att_pred = att_full[:, :, pred]

    print(f"Predicted class: {class_names[pred]}")
    print(f"Membrane potentials: {mem.numpy()}")
    print()

    save_path = results_dir / f"tsa_sample{sample_index}.png"
    plot_heatmap(att_pred, feature_names,
                title=f"TSA for sample {sample_index} (pred: {class_names[pred]})",
                save_path=save_path)

    run_summary['sample_analysis'] = {
        'predicted_class': class_names[pred],
        'predicted_class_idx': int(pred),
        'membrane_potentials': mem.numpy().tolist(),
        'correct_prediction': bool(pred == true_label)
    }

    print("\n" + "=" * 80)
    print(f"B. TSA FOR {args.num_random_samples} RANDOM SAMPLES")
    print("=" * 80)

    rnd_indices = np.random.choice(len(spike_test), size=min(args.num_random_samples, len(spike_test)), replace=False)
    rnd_atts = []
    random_samples_info = []

    for i, idx in enumerate(rnd_indices):
        rnd_spikes = spike_test[idx]
        rnd_x = torch.tensor(rnd_spikes, dtype=torch.float32)
        rnd_att = compute_tsa(model, rnd_x, beta, device)
        rnd_atts.append(rnd_att)

        rnd_pred, rnd_mem = predict_mem(model, rnd_x, device)
        print(f"  Sample {idx}: true={class_names[y_test[idx]]}, pred={class_names[rnd_pred]}")

        random_samples_info.append({
            'index': int(idx),
            'true_class': class_names[y_test[idx]],
            'true_class_idx': int(y_test[idx]),
            'predicted_class': class_names[rnd_pred],
            'predicted_class_idx': int(rnd_pred),
            'membrane_potentials': rnd_mem.numpy().tolist(),
            'correct_prediction': bool(rnd_pred == y_test[idx])
        })

    run_summary['random_samples'] = random_samples_info

    print("\n" + "=" * 80)
    print("C. AGGREGATED DATASET-LEVEL TSA")
    print("=" * 80)

    all_atts = torch.stack([att_full] + rnd_atts)
    avg_att = all_atts.mean(dim=0)[:, :, pred]

    save_path = results_dir / f"tsa_aggregate_{args.num_random_samples}samples.png"
    plot_heatmap(avg_att, feature_names,
                title="Aggregated TSA (Average over samples)",
                save_path=save_path)

    print("\n" + "=" * 80)
    print("D. CLASS-SPECIFIC TSA COMPARISON")
    print("=" * 80)

    if args.full_dataset_tsa:
        print("Computing TSA for the entire dataset (train + test) - this may take a while...")

        all_spike_data = np.concatenate([spike_train, spike_test], axis=0)
        all_labels = np.concatenate([y_train, y_test], axis=0)

        print(f"Total samples to process: {len(all_spike_data)} (train: {len(spike_train)}, test: {len(spike_test)})")

        class_atts = {i: [] for i in range(output_size)}

        for i in range(len(all_spike_data)):
            sample_spikes = all_spike_data[i]
            sample_x = torch.tensor(sample_spikes, dtype=torch.float32)

            sample_att_full = compute_tsa(model, sample_x, beta, device)

            true_class_idx = all_labels[i]
            class_atts[true_class_idx].append(sample_att_full)

            if (i + 1) % 10 == 0 or i == len(all_spike_data) - 1:
                print(f"  Processed {i+1}/{len(all_spike_data)} samples", end='\r')

        print("\n✓ Full dataset TSA computation complete.\n")

        class_specific_info = {}

        for c_idx, att_list in class_atts.items():
            if att_list:
                avg_class_att_full = torch.stack(att_list).mean(dim=0)
                avg_att_for_class_c = avg_class_att_full[:, :, c_idx]

                save_path = results_dir / f"tsa_class{class_names[c_idx]}.png"
                plot_heatmap(avg_att_for_class_c, feature_names,
                             title=f"Average TSA for True Class: {class_names[c_idx]} ({len(att_list)} samples)",
                             save_path=save_path)

                class_specific_info[class_names[c_idx]] = {
                    'num_samples': len(att_list),
                    'attribution_stats': {
                        'mean': float(avg_att_for_class_c.mean().cpu()),
                        'std': float(avg_att_for_class_c.std().cpu()),
                        'min': float(avg_att_for_class_c.min().cpu()),
                        'max': float(avg_att_for_class_c.max().cpu())
                    }
                }
            else:
                print(f"Warning: No samples found for class {class_names[c_idx]}")
                class_specific_info[class_names[c_idx]] = {
                    'num_samples': 0,
                    'attribution_stats': None
                }

        run_summary['class_specific_tsa'] = {
            'full_dataset': True,
            'total_samples_analyzed': len(all_spike_data),
            'train_samples': len(spike_train),
            'test_samples': len(spike_test),
            'classes': class_specific_info
        }
    else:
        print("Using random sample subset for class comparison.")
        print("WARNING: The following plots will vary on each run.")
        print("         Use --full_dataset_tsa flag for stable, dataset-wide results.\n")
        plot_class_comparison(all_atts, feature_names, class_names, save_dir=results_dir)

        run_summary['class_specific_tsa'] = {
            'full_dataset': False,
            'num_samples_used': len(all_atts),
            'note': 'Random subset - results will vary on each run'
        }

    print("\n" + "=" * 80)
    print("E. PERTURBATION ANALYSIS")
    print("=" * 80)
    print("Testing importance of top-k vs. random-k spike deletions...\n")

    save_path = results_dir / "perturbation_analysis.png"
    perturbation_data = plot_perturbation_curve(model, x, att_pred, device, feature_names, save_path=save_path)

    run_summary['perturbation_analysis'] = perturbation_data

    print("\n" + "=" * 80)
    print("F. SAVING RUN SUMMARY")
    print("=" * 80)

    summary_path = results_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(run_summary, f, indent=2)

    print(f"✓ Summary saved to: {summary_path}")

    print("\n" + "=" * 80)
    print("✓ ALL TSA ANALYSIS COMPLETE (SENSORS ONLY)!")
    print(f"✓ Results saved to: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
