# FLIoMT Research Journal

Running notes on findings, decisions, and understanding. Not polished — meant to be useful.

---

## Project Goal

Federated learning for physiological anomaly detection on edge IoMT hardware. The clinical motivation is post-discharge cardiac monitoring: patients take devices home, devices train on their own baseline, anomalies are flagged without raw data ever leaving the device. The FL server only sees model weights, never raw signal.

Detection is **reconstruction-based and unsupervised**: models are trained on normal signal only. At inference time, anomaly score = MSE(x, x̂). Windows the model can't reconstruct well are flagged as anomalies. This means no labeled anomaly data is required for training — only normal baseline is needed.

---

## Datasets

### MIT-BIH Arrhythmia Database (primary)
- 48 ECG recordings, each 30 min at 360 Hz, resampled to 100 Hz
- Labels: normal sinus rhythm vs. annotated arrhythmia beats
- Client 0 (Orin #2): record 213 — normal train, arrhythmia test
- Client 1 (Pi 5): record 106 — same split
- Each client holds a different patient — realistic FL scenario
- **This is the dataset producing all meaningful results.** MIT-BIH gives clean, well-labeled ground truth.

### WESAD
- Wrist-worn physiological data, stress/affect protocol
- Tested early in the project (PatchTST, flat LR, 50 rounds on WESAD)
- Results were weaker — loss still 0.018 at r50 and declining
- Harder task: stress response is more subtle than cardiac arrhythmia
- Kept in the pipeline but MIT-BIH is the main benchmark

---

## Models

### CNNAutoencoder
Dilated residual 1D CNN encoder-decoder. Each encoder layer doubles the dilation rate, giving an exponentially growing receptive field (RF) without increasing parameter count proportionally.

RF formula: `1 + 2*(2^e_layers - 1)` at kernel_size=3
- e_layers=4: RF = 31 samples = 310 ms — captures QRS complex
- e_layers=5: RF = 63 samples = 630 ms — captures full P-QRS-T cycle at 70 bpm
- e_layers=6: RF = 127 samples = 1.27 s ≈ full seq_len=128 window (under investigation)

**Why this matters**: A normal heartbeat has a recognizable P-QRS-T morphology. If the receptive field only covers the QRS complex (the sharp spike), the model learns the spike but not the full context. Extending to 630 ms gives the model enough window to learn the full waveform, making reconstruction failure on arrhythmia more reliable.

**Key finding**: e_layers=6 (RF ~1.27s ≈ full seq_len=128 window) outperforms e_layers=5 (RF ~630ms) on both clients: AUROC 0.962/0.990 vs 0.959/0.970, score separation nearly doubled (10.55σ/9.13σ vs 5.81σ/4.63σ), and the model converges faster (best loss at r59 vs r67). This is now the production config. e_layers=5 was the previous production; e_layers=4 was the baseline.

**Quirk discovered**: The baseline run's score separation of 46.1σ on client 0 looked impressive but AUROC was only 0.863. Score separation = (mean_anomaly - mean_normal) / std_normal. A high σ can come from either a large numerator (good separation) or a tiny denominator (very tight normal scores). The old model had very tight normal scores but poor discrimination overall. The tuned models have 5–10σ separation with 0.96–0.99 AUROC — more meaningful.

### PatchTST
Patch-based transformer. Divides the time series into fixed-length patches and treats each patch as a token. With patch_len=16, stride=8, seq_len=128: 15 patches with 50% overlap.

**Why patches work for ECG**: A 160 ms patch at 100 Hz contains most of a QRS complex. The transformer attends across patches rather than individual timesteps, which aligns with how ECG morphology is naturally structured (the meaningful units are beats, not samples).

**Parameter that matters most**: `e_layers` controls the depth of the transformer stack. Going 3→4 added ~240K params (~44% increase) with the intention of improving the lower-scoring client.

**Confounding result resolved**: In the tuned run (e_layers=4, seq_len=128), client 0 dropped from 0.979 to 0.929 while client 1 improved from 0.897 to 0.981. Running seq_len=100 with e_layers=4 confirmed seq_len was the culprit: c0 recovered to 0.979, c1 dropped to 0.914. The effect is patient-specific — seq_len=128 benefits patient 106 (c1), seq_len=100 benefits patient 213 (c0). Average AUROC: seq_len=128 → 0.955, seq_len=100 → 0.946. Production stays at seq_len=128 (better average) but this sensitivity is worth reporting in the paper.

### iTransformer
Inverted transformer: where a standard transformer treats each timestep as a token (seq_len tokens of dim n_channels), iTransformer inverts this — each **channel** is a token of dim seq_len.

**Architectural mismatch for single-channel ECG**: With enc_in=1, the attention matrix is 1×1. `softmax([score]) = [1]` — the attention output is just the value unchanged. Self-attention is a literal no-op. All learning is in the FFN layers.

Despite this, iTransformer became the best-performing model with proper training. With d_model=128, d_ff=256, e_layers=3, flat LR, 200 rounds, local_epochs=2: AUROC 0.9989/0.9946.

**What this tells us**: The FFN stack (3 layers of 128→256→128 transformations on a 128-dim sequence representation) is sufficient to learn excellent reconstruction even without functional attention. The model is essentially a deep MLP applied to the full sequence.

**What would actually use the attention**: enc_in≥3. With ECG+PPG (3 channels), the attention matrix becomes 3×3 and the model can learn cross-channel relationships. This is worth a future experiment.

### TimesNet (excluded)
FFT → reshape to 2D → 2D convolution. Strong detection (AUROC 0.779/0.965) but impractical:
- 9.4M params, 37.5 MB model
- 350 ms per-sample CPU inference on Pi 5 (vs. <17 ms for all other models)
- 15.1 GB communication over 100 rounds (vs. 27 MB for CNN)
- Not viable for IoMT edge deployment

---

## Experiment Log

All runs on MIT-BIH, 2-client FL (Orin #2 + Pi 5), cosine LR unless noted.

| Date | Run ID | Model | Config | AUROC (c0/c1) | Notes |
|------|--------|-------|--------|---------------|-------|
| 2026-06-22 | 20260622_213419_fl_PatchTST | PatchTST | 100r, seq_len=100, e_layers=3 | 0.979 / 0.897 | Baseline; still declining at r100 |
| 2026-06-23 | 20260623_001440_fl_CNNAutoencoder | CNN | 100r, seq_len=100, e_layers=4 | 0.863 / 0.991 | Baseline; imbalanced clients |
| 2026-06-23 | 20260623_002919_fl_iTransformer | iTransformer | 100r, seq_len=100, e_layers=2 | 0.611 / 0.451 | **Failure** — cosine LR killed training |
| 2026-06-23 | 20260623_014236_fl_TimesNet | TimesNet | 100r, seq_len=100 | 0.779 / 0.965 | Excluded — edge-impractical |
| 2026-06-23 | 20260623_133450_fl_PatchTST | PatchTST | 150r, seq_len=128, e_layers=4 | 0.929 / 0.981 | Tuned; c0 regressed |
| 2026-06-23 | 20260623_133937_fl_CNNAutoencoder | CNN | 100r, seq_len=128, e_layers=5 | 0.959 / 0.970 | Tuned; more balanced |
| 2026-06-23 | 20260623_135731_fl_iTransformer | iTransformer | 200r, flat LR, e_layers=3 | **0.9989 / 0.9946** | Tuned; complete reversal |
| 2026-06-23 | 20260623_161544_fl_CNNAutoencoder | CNN | 100r, seq_len=128, e_layers=6 | 0.962 / 0.990 | **Production** — beats e_layers=5 on both clients |
| 2026-06-23 | 20260623_165016_fl_PatchTST | PatchTST | 150r, seq_len=100, e_layers=4 | 0.979 / 0.914 | Ablation — confirms seq_len=128 caused c0 regression |

Full per-round data in `results/<run_id>/fl_summary.json`.

---

## Key Technical Insights

### 1. The iTransformer failure was 100% training, not architecture
The baseline iTransformer had loss 0.041 at r100 — still actively converging. PatchTST had converged to 0.018 by then. The cosine LR schedule decayed to ~2.3e-5 by round 75/100 while iTransformer still needed full-rate gradient steps. Flat LR + 200 rounds + local_epochs=2 fixed it completely. Lesson: **don't use cosine decay for models whose convergence time you don't know yet.**

### 2. Cosine LR decay formula
`lr(r) = lr_min + 0.5*(lr_max - lr_min) * (1 + cos(π*(r-1)/(R-1)))`

At r=75/100, lr ≈ 2.3e-5. At r=50/100, lr ≈ 5.6e-5. By halfway through training the LR is already ~56% decayed. For fast-converging models this is fine. For models that need more steps, it's premature.

### 3. local_epochs is free gradient steps
In FedAvg, each round: server broadcasts weights → clients train locally → clients send weights back → server aggregates. Communication cost is fixed per round (one send + one receive). local_epochs=2 vs. 1 doubles gradient steps per round at no extra communication cost. The risk (client drift, where models overfit to local data before aggregation) is negligible at 2 epochs. For a slow-converging model, this is a high-value knob.

### 4. Communication cost scales with model size, not rounds
Per-round comm ≈ 2 × param_mb × n_clients (bidirectional, all clients).
- CNN (0.063 MB): 0.25 MB/round — runs 100 rounds for 27 MB total
- iTransformer (1.72 MB): 6.9 MB/round — runs 200 rounds for 1.38 GB total
- PatchTST (3.18 MB): 12.7 MB/round — runs 150 rounds for 3.44 GB total
- TimesNet (37.5 MB): 150 MB/round — at 100 rounds = 15.1 GB

Model size matters far more than round count for communication budget.

### 5. F1_PA vs F1_raw
Point-Adjust (PA) protocol: if any point in a contiguous anomaly segment is detected, all points in that segment are counted as correct. F1_PA is the accepted literature metric for time-series AD. F1_raw is point-wise and more honest.

All models achieve F1_PA ≈ 1.0 but F1_raw ≈ 0.14. The gap is large because arrhythmia segments are long — the model often detects a few windows in each segment, which PA counts as full detection. F1_raw = 0.14 means most individual anomaly windows are missed. This is expected for reconstruction-based AD on ECG: normal beats and arrhythmia beats can look similar at the window level; it's the sustained deviation that gets caught. **Use F1_PA as the primary metric, report F1_raw for transparency.**

### 6. What "fair comparison" means between architectures
Fixed hyperparameters is not a fair comparison — it advantages models with faster convergence dynamics. Fair comparison means: same task, same data, same evaluation protocol, same seq_len, same initial LR. Rounds, LR schedule, local_epochs, and batch_size can and should differ per model. The baseline run proved this: identical hyperparameters made iTransformer look like the worst model when it turned out to be the best.

### 7. Config hierarchy in run_client.py
Priority order: `_MODEL_PRESETS` → YAML config (if `--config` passed) → CLI args.
The YAML model config files (`configs/models/*.yaml`) are **not automatically used** by the FL scripts — they're documentation. Architecture defaults live in `_MODEL_PRESETS` in `fl/run_client.py`. CLI args via env vars (E_LAYERS, SEQ_LEN, etc.) override everything.

---

## Methodological Decisions (paper-relevant)

**Why reconstruction-based detection**: Avoids needing labeled anomaly data for training. Patients can contribute their own normal baseline; the model learns what's normal for them. Anomalies are implicitly detected by failed reconstruction.

**Why FedAvg**: Simplest, most established FL algorithm. Weights are averaged proportional to local dataset size. No extra communication overhead. Appropriate baseline before exploring more complex FL strategies.

**Why MIT-BIH for evaluation**: Has clean temporal separation between normal and arrhythmia segments in each recording. Allows training entirely on normal beats and testing on labeled arrhythmia. Well-established benchmark in the cardiac AD literature.

**Why temporal partitioning (not patient partitioning) for single-patient POC**: With one recording per client, each client trains on the first 70% of the time series and tests on the held-out 10%. This simulates a scenario where each device has its own patient's data.

**On single-seed results**: All current results are from single runs (seed=42). Statistical validity for a paper requires 3+ seeds with mean ± std. This is the most important pending experiment before submission.

---

## Open Questions

1. **Live data integration**: Sensors are deployed and working (AD8232 ECG, MAX30102 PPG on Pi 5). Problem: strapping sensors to a healthy person only produces normal data — can't validate anomaly detection without pathological data. Options: anomaly injection into normal ECG, keep using public datasets for evaluation, or use post-exercise ECG as a proxy anomaly. Waiting on professor's guidance.

2. **iTransformer + multi-channel**: With enc_in=3 (ECG + PPG red + IR), the inverted attention becomes a 3×3 matrix and the architecture actually does what it was designed for. Would the attention mechanism contribute meaningful signal beyond what the FFN alone achieves? One experiment to run eventually.

3. **PatchTST seq_len sensitivity**: ~~Is the c0 regression from seq_len or e_layers?~~ Resolved: seq_len=128 hurt patient 213, seq_len=100 hurts patient 106. Effect is patient-specific. Production stays at seq_len=128 (better average AUROC 0.955 vs 0.946). Worth mentioning in the paper.

4. **CNNAutoencoder e_layers=6**: ~~Does extending the RF to ~1.27s improve AUROC?~~ Resolved: yes, clearly. e_layers=6 is now production (AUROC 0.962/0.990 vs 0.959/0.970).

5. **FL vs. local ablation**: Need to show FL is better than (or at least comparable to) local training alone. Run each client independently for the same number of rounds and compare AUROC. Required for the paper.

6. **anomaly_ratio tuning**: All models use ratio=1.0 (threshold at 99th percentile). Per-model tuning could improve F1_raw without affecting F1_PA.

7. **More patients**: Only two MIT-BIH records used. More records as additional clients would make the FL claim more convincing.

---

## Things to Include in the Paper

- Two-phase experimental design: baseline (uniform) → tuned (per-model optimal). The baseline is valuable as evidence for why per-model tuning is necessary.
- iTransformer's architectural mismatch for single-channel ECG (1×1 attention) and the fact that it still outperformed others with proper training — this is a noteworthy finding.
- Communication efficiency table: CNN at 27 MB vs. TimesNet at 15.1 GB for the same task on the same data makes a strong case for architecture selection in FL.
- The distinction between F1_PA and F1_raw and why both should be reported.
- TimesNet exclusion with quantitative justification (not just "too big" but the specific numbers: 350 ms inference, 15.1 GB comm).
- Fairness of comparison section: explicitly state what was controlled and what was tuned per-model, and why this is the right approach.
