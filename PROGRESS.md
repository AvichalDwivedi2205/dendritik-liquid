# DendriticLiquid-ViT - Implementation Progress

**Target**: NeurIPS 2026 Submission
**Timeline**: 20 weeks (January - June 2026)
**Current Status**: Week 3 - Baseline ViT Complete ✅

---

## ✅ Completed Milestones

### Week 1: Environment Setup (COMPLETE)
- ✅ Python 3.12.3 environment with uv package manager
- ✅ PyTorch 2.10.0+cu128 with CUDA 12.8
- ✅ GPU: NVIDIA GeForce RTX 4090 verified
- ✅ All dependencies installed: PyTorch Lightning, timm, wandb, einops, etc.
- ✅ Project structure created
- ✅ CIFAR-10 and CIFAR-100 datasets downloaded (50k train, 10k test each)
- ✅ Git repository initialized

**Deliverables**:
- [x] Environment test passing
- [x] GPU working (RTX 4090)
- [x] Datasets ready
- [x] Project structure complete

---

### Week 3-4: Baseline Vision Transformer (COMPLETE)

**Implemented Files**:
1. `src/models/vit_base.py` - Complete ViT implementation
   - Patch embedding (Conv2d-based)
   - Multi-head self-attention
   - Transformer blocks with residual connections
   - CLS token classification
   - Attention map extraction for visualization
   - Three model sizes: tiny, small, base

2. `src/data/cifar_dataset.py` - CIFAR dataloaders
   - Standard augmentation (RandomCrop + HorizontalFlip)
   - Normalization with dataset statistics
   - Efficient data loading (pin_memory, persistent_workers)

3. `src/evaluation/metrics.py` - Evaluation utilities
   - FLOPs counting (fvcore)
   - Parameter counting
   - Inference latency measurement
   - Top-k accuracy computation
   - Model summary utility

4. `experiments/train_baseline.py` - PyTorch Lightning training
   - AdamW optimizer
   - Cosine annealing with linear warmup
   - Gradient clipping
   - Mixed precision (AMP) support
   - WandB logging integration
   - ModelCheckpoint callbacks

5. `experiments/configs/baseline_vit.yaml` - Configuration

**Model Stats (ViT-Small)**:
```
Parameters:     4,766,474 (4.8M)
FLOPs:         321,558,272 (0.32 GFLOPs)
Latency:       0.73ms per image
Throughput:    1,362 images/sec (RTX 4090)
Model Size:    18.18 MB
```

**Tests**:
- ✅ Forward pass and output shapes
- ✅ Gradient flow through all parameters
- ✅ Attention map extraction (6 layers × 8 heads)
- ✅ Dataloader functionality
- ✅ **Overfitting sanity check: 96.88% on single batch** (proves model learns)

**Deliverables**:
- [x] ViT implementation complete
- [x] Training script ready
- [x] All tests passing
- [x] Metrics utilities working

---

## 🚧 In Progress

### Week 3-4: Baseline Training
**Next Immediate Step**: Train baseline ViT to establish benchmark

**Command**:
```bash
python experiments/train_baseline.py \
    --model_size small \
    --epochs 50 \
    --batch_size 128 \
    --use_amp \
    --use_wandb
```

**Expected Results** (from plan):
- Train accuracy: ~98%
- **Val accuracy: >95%** ⭐ (target)
- Training time: ~2-3 hours on RTX 4090

**Metrics to Record**:
- Best validation accuracy
- Final FLOPs count
- Parameters count
- Training time
- Convergence curves

---

## 📋 Upcoming Milestones

### Week 5-7: Dendritic Module
**Objective**: Implement multi-branch dendritic gating

**Files to Create**:
- `src/models/dendritic_module.py`
  - `DendriticBranch`: Single branch (MLP with sigmoid)
  - `DendriticGating`: Multi-branch with max-pooling
  - Positive bias initialization (+2.0)

- `tests/test_dendritic.py`
  - Output shapes
  - Gradient flow
  - Positive bias verification

- `src/models/dendritic_vit.py`
  - ViT with dendritic gating (no sparsity yet)
  - Just compute importance scores

**Expected Results**:
- Accuracy: ~95.5% (slight improvement)
- FLOPs: ~98% of baseline (minimal overhead)
- **Key**: Importance scores should be meaningful (visualize!)

---

### Week 8-10: CfC Liquid Gate
**Objective**: Implement closed-form continuous-time gates

**Files to Create**:
- `src/models/cfc_gate.py`
  - CfC gate network
  - Importance-modulated blending
  - Learnable time constants

- `tests/test_cfc.py`
  - Shape tests
  - Gradient flow
  - Extreme case tests (gate=0, gate=1)

- `src/models/dendritic_cfc_vit.py`
  - ViT with dendritic + CfC (no sparsity)

**Critical**:
- ⚠️ Gradient clipping: `gradient_clip_val=1.0`
- ⚠️ Lower LR for CfC params: `5e-4` instead of `1e-3`
- ⚠️ Neutral bias initialization: `0.0`

---

### Week 11-13: Sparse Attention
**Objective**: Add sparse masked attention with STE

**Files to Create**:
- `src/models/sparse_attention.py`
  - `StraightThroughEstimator` (autograd.Function)
  - `SparseMaskedMultiHeadAttention`

- `src/training/scheduler.py`
  - Temperature annealing (5.0 → 0.1)
  - Threshold mode scheduler (soft → hard)

- `src/models/dendriticliquid_vit.py`
  - **Full model**: Dendritic + CfC + Sparse

**Critical**:
- ⚠️ Warmup: Soft masking epochs 0-5
- ⚠️ Transition: Gradual sharpening epochs 5-15
- ⚠️ Hard masking: STE from epoch 15+

---

### Week 14-16: Main Experiments
**Objective**: Train all ablations and record results

**Models to Train**:
1. Baseline ViT (already done)
2. Dendritic-only
3. CfC-only
4. Sparse-only
5. **DendriticLiquid (Full)**

**For Each Model** (3 seeds):
- Train on CIFAR-10
- Train on CIFAR-100 (optional)
- Record: accuracy, FLOPs, latency, params

**Results Table** (target):
| Model | CIFAR-10 Acc | FLOPs (%) | Params |
|-------|--------------|-----------|--------|
| ViT | 95.2 ± 0.2 | 100 | 4.8M |
| Dendritic | 95.4 ± 0.2 | 98 | 5.0M |
| CfC | 95.1 ± 0.3 | 99 | 5.1M |
| Sparse | 94.8 ± 0.2 | 65 | 4.8M |
| **DendriticLiquid** | **95.6 ± 0.2** | **62** | **5.3M** |

---

### Week 17-18: Robustness Evaluation
**Objective**: Evaluate on ImageNet-C / CIFAR-10-C

**Steps**:
1. Download CIFAR-10-C (Zenodo)
2. Create `src/data/corruptions.py`
3. Create `experiments/evaluate_robustness.py`
4. Evaluate all models on 15 corruption types

**Target Results**:
- Baseline ViT: ~82-85% on CIFAR-10-C
- **DendriticLiquid: ~88-92%** (+6-8% improvement) ⭐

---

### Week 19: Visualization & Interpretability
**Files to Create**:
- `src/visualization/attention_maps.py`
  - Visualize which patches are important
  - Heatmaps overlaid on images

- `src/visualization/cfc_evolution.py`
  - Plot CfC gate values across layers
  - Should show: low early, high late

- `src/visualization/dendritic_patterns.py`
  - Sparsity per layer
  - Token pruning analysis

**Expected Insights**:
- Early layers: ~10-20% pruning
- Middle layers: ~40-50% pruning
- Late layers: ~60% pruning
- CfC gates: gradual increase across layers

---

### Week 20: Paper Writing
**Sections**:
1. Abstract (250 words)
2. Introduction (1200 words)
3. Related Work (1500 words)
4. Method (2500 words)
5. Experiments (2500 words)
6. Discussion (800 words)
7. Conclusion (300 words)

**Figures & Tables**:
- Architecture diagram
- Main results table
- Ablation table
- Robustness comparison
- Importance visualizations
- CfC evolution plot
- Sparsity analysis

---

## 📊 Key Performance Indicators (KPIs)

**Must-Have for NeurIPS Acceptance**:
- [ ] DendriticLiquid-ViT ≥95% on CIFAR-10
- [ ] ≥30% FLOPs reduction vs baseline
- [ ] +5% accuracy on CIFAR-10-C (OOD robustness)
- [ ] All ablations complete
- [ ] Code runs without errors
- [ ] All visualizations generated
- [ ] Paper draft complete (8-10 pages)

**Nice-to-Have**:
- [ ] Results on CIFAR-100
- [ ] Results on ImageNet-1K
- [ ] Hardware profiling (actual GPU speedup)
- [ ] Transfer learning experiments

---

## 🔧 Technical Notes

### Current Setup
- **GPU**: NVIDIA GeForce RTX 4090 (24GB VRAM)
- **CUDA**: 12.8
- **PyTorch**: 2.10.0
- **Python**: 3.12.3

### Training Performance
- **Baseline ViT throughput**: 1,362 images/sec
- **Expected training time** (50 epochs, CIFAR-10):
  - ViT-Tiny: ~30 minutes
  - ViT-Small: ~1 hour
  - ViT-Base: ~2-3 hours

### Critical Implementation Details
1. **Gradient clipping**: ALWAYS use `gradient_clip_val=1.0`
2. **Separate LRs**: Dendritic & CfC at 50% of base LR
3. **Temperature annealing**: Essential for stable sparse training
4. **Positive bias**: Dendritic branches must start with bias=+2.0

---

## 🐛 Known Issues & Warnings

1. **Torchvision deprecation warning**:
   ```
   VisibleDeprecationWarning: dtype(): align should be passed as Python or NumPy boolean
   ```
   - **Impact**: None, can be ignored
   - **Cause**: NumPy 2.4 compatibility in torchvision

2. **FLOPs counting warnings**:
   ```
   Unsupported operator aten::mul encountered
   ```
   - **Impact**: None, FLOPs still counted correctly
   - **Cause**: fvcore doesn't recognize all PyTorch ops

---

## 📈 Progress Tracking

**Timeline**:
- Week 1: ✅ Environment setup (COMPLETE)
- Week 2: ⏭️ Literature reading (user task)
- Week 3-4: ✅ Baseline ViT implementation (COMPLETE)
- Week 3-4: 🚧 Baseline training (IN PROGRESS)
- Week 5-7: ⏱️ Dendritic module (UPCOMING)
- Week 8-10: ⏱️ CfC gate (UPCOMING)
- Week 11-13: ⏱️ Sparse attention (UPCOMING)
- Week 14-16: ⏱️ Experiments (UPCOMING)
- Week 17-18: ⏱️ Robustness (UPCOMING)
- Week 19: ⏱️ Visualization (UPCOMING)
- Week 20: ⏱️ Paper writing (UPCOMING)

**Overall Progress**: 15% complete (3/20 weeks)

---

## 🎯 Next Actions

1. **Immediate (Today)**:
   ```bash
   # Train baseline ViT to completion
   python experiments/train_baseline.py --model_size small --epochs 50 --use_amp --use_wandb
   ```

2. **Record baseline metrics**:
   - Best validation accuracy
   - Training curves
   - Final FLOPs count
   - Checkpoint location

3. **After baseline completes**:
   - Start Week 5: Implement dendritic module
   - Read dendritic ANNs paper (Chavlis et al., 2025)

---

**Last Updated**: January 29, 2026
**Status**: On track for NeurIPS 2026 submission ✅
