# Wavenet Torch Implementation for Gravitational Wave Classification

## 0 Introduction
This project is a PyTorch implementation of the Wavenet architecture, specifically designed for the classification of gravitational waves. Based on arXiv:2306.15728, this implementation focuses on providing a robust solution for the classification of gravitational wave signals.

### Installation and Usage
To get started with this project, clone the repository and install the required dependencies:
```
git clone https://github.com/victoria-tiki/Wavenet_torch.git
cd Wavenet_torch
```
This project includes slurm scripts for use in high-performance computing environments. To train the model, execute:
```
sbatch slurm_train.sh
```

## 1  Model Variants
| Name | Components | When to use |
|------|------------|-------------|
| **CNN + PinSage_Attn** (default) | 2-channel CNN front-end (processing Livingston and Hanford signals separately) → PinSage **attention-based** aggregator | Best overall; lets the network weight L1 vs H1 features dynamically, empirically found to reduce FPs|
| **CNN + PinSage** | 2-channel CNN front-end → PinSage **graph** aggregator | For ablation or if you prefer deterministic pooling. |

---

## 2  Training Pipeline Overview
We start with a library of **synthetic binary-merger waveforms** and a stash of **real LIGO noise**.  
During training the *data generator*:

1. Picks a waveform (or none, for noise-only cases).  
2. Picks a noise chunk.  
3. Injects / mixes them at a chosen SNR.  
4. Whitens, band-passes, and windows the result.  

The resulting `[4096 × 2]` tensor (H1 & L1) and a binary target mask are fed to the model through a PyTorch `DataLoader`.

---

## 3  Data-Generation Modes

|                     | **`data_generation` (NEW)** | **`data_generation_old` (LEGACY)** |
|---------------------|-----------------------------|------------------------------------|
| **Noise on disk**   | Raw, **un-whitened**. | Already **whitened**. |
| **SNR definition**  | Matched-filter SNR. | Custom “training SNR” defined via relative time-domain std. |
| **Curriculum**      | Probabilistic boost of chosen SNR range (eg. ∈ [15, 40]) that **decays** over time into realistic distribution. | User provides an explicit per-epoch schedule. |
| **Key args**        | `p_higher_init`, `p_higher_fin`, `boosted_range`, `decay_epochs`, `bandpass`, `noise_prob` | `snr_schedule`, `snr_bins`, `noise_prob` in snr_scheduler|
| **Why keep it?**    | Physics-aligned SNRs, smoother training, fewer knobs. | Exact reproducibility of older experiments, fine-grained control. |
| **Band-pass filtering**   | Built-in Butterworth (default **10–1000 Hz**) applied **after** whitening to remove artifacts. 10-1000 Hz keeps it closer to LIGO processing| Assumes data are already whitened & filtered; no extra band-pass inside the loader |
---

## 4  Hyper-parameter Guide – “What to tweak & why”

### 4.0  How the two data-loaders handle SNR

* **`data_generation` (NEW)**  
  *Uses raw (un-whitened) noise.*  
  During training it **boosts** injections with SNR in a configurable range (default **15 – 40**).  
  The boost probability starts at `p_higher_init`, decays exponentially, and bottoms out at `p_higher_fin` after `decay_epochs`.

* **`data_generation_old` (LEGACY)**  
  *Uses pre-whitened noise.*  
  No built-in boost. Instead you pass an **external `snr_schedule`** that specifies the target SNR for each epoch.  
  Because SNR varies abruptly between bins, you often have to co-tune the LR schedule. This does allow for fine grained control since
  we're not relying on the realistic matched filter-distribution but introduces several hyperparameters (3 for each bin) 

---

### 4.1  Quick knobs (first things to try)

| Parameter | Default | Loader | Where to change | Why / when to change |
|-----------|---------|--------|-----------------|----------------------|
| `noise_prob` | **0.65** | both | `GWDataset` arg | ↑ to cut false-positives; ↓ to cut false-negatives. Sweet-spot 0.60 – 0.70. |
| `p_higher_init` | **0.50** | NEW | `GWDataset` arg | Raise if early epochs miss *loud* events. |
| `p_higher_fin`  | **0.05** | NEW | `GWDataset` arg | Raise if the *final* model still misses loud events. |
| `patience` (ReduceLR) | **3** | both | `LightningModel.configure_optimizers` | Tweak if LR drops too early / too late. |
| `snr_bins` / `bin_size` | project-specific | LEGACY | build your `snr_schedule` | Finer bins = more control but tighter LR coupling. |

> **Tip:** set `plot_samples=True` in `GWDataset` to save 5 example windows and visually verify any change.

---

### 4.2  Other knobs (tune after the quick ones)

| Parameter | Default | Loader | Where to change | Why / when to change |
|-----------|---------|--------|-----------------|----------------------|
| `decay_epochs` | **10** | NEW | `GWDataset` arg | Lengthen if training is unstable; shorten if convergence is slow. |
| `boosted_range` | **[15, 40]** | NEW | inside `__data_generation` | Shift / widen around the SNRs that matter for your search; keep it broad enough to avoid data leakage. |
| `reset_epoch` | **9999** (≫ epochs) | both | callback `CustomLRSchedulerCallback` | High value = disable reset (**recommended start**). Set ≈`decay_epochs` to kick LR once the boost ends. |
| `new_lr` | `5e-4` | both | same callback | < initial LR → conservative late training; ≥ initial LR → aggressive fine-tuning on hard samples. |
| `bandpass` | 10 – 1000 Hz | both | butter filter in `GWDataset` | Advanced: tighten to 20 – 800 Hz to test robustness vs. PSD mismatch. |

---

### 4.3  SNR boost formula (NEW loader, for reference)

```python
# GWDataset.__init__
tau = -decay_epochs / np.log(p_higher_fin / p_higher_init)
p_higher(epoch) = p_higher_init * exp(-epoch / tau)
```

τ is **derived**; you only set `p_higher_*` and `decay_epochs`.

### Training workflow suggestion
1. **Start** with loader defaults and a *very high* `reset_epoch` (essentially disabling LR-reset).  
2. Tweak the **Quick knobs** until the **validation statistics**  looks reasonable (don't rely too much on the loss alone)
3. Adjust the **Other knobs** one at a time.  

> **Validation vs Test**  
> • Tune every hyper-parameter **only on the validation set** (eg synthetic signals + real noise).  
> • Once satisfied, evaluate once on the **held-out test set** (eg true events) to avoid data leakage.  
> • Make sure the PSD and noise statistics used in validation **match** those expected for the test data; mismatched noise can hide generalisation issues.

Happy tweaking!

