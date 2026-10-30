# SG2DiT

## ✨ Highlights

- **Signal–geometry representations.** Single-level wavelet subbands capture coarse intensity and directional details; structure tensors and gradients describe local spatial organization.
- **Source-aware reconstruction.** Hybrid convolution–attention encoders feed cross-modal shallow-skip gates, complemented by visible-band weighting and latent diffusion refinement.
- **Small-target evaluation.** A separately trained, frozen detector measures how fused images preserve target evidence on the public UAV test split.
- **Controlled analysis.** Retrained prior combinations, spatial interventions, and reconstruction-route studies examine where the improvements arise.

## 📊 Results

### Frozen-detector evaluation on LSS-UAV

The following **manuscript-reported** results use the **614-frame public LSS-UAV test split**, native-resolution fusion within the stated inference limit, and a shared frozen YOLOv8n detector. In this protocol, fusion methods use **RGB outputs**, while the single-modality source references use replicated grayscale images. Scores are percentages.

| Input | AP ↑ | AP50 ↑ | AP75 ↑ | AP<sub>s</sub> ↑ | AR<sub>s</sub> ↑ |
|:--|--:|--:|--:|--:|--:|
| Infrared | 49.83 | 78.62 | 55.22 | 39.41 | 58.18 |
| Visible luminance | 47.96 | 77.23 | 49.21 | 34.32 | 50.90 |
| CDDFuse | 56.76 | 81.97 | 58.50 | 33.44 | 47.89 |
| SeAFusion | 50.22 | 76.31 | 54.98 | 34.70 | 51.20 |
| LFDT-Fusion | 49.20 | 78.29 | 51.12 | 37.56 | 52.03 |
| **SG²DiT** | **60.12** | **88.94** | **65.38** | **42.16** | **61.72** |

**Reported improvements:** **+3.36 AP** over CDDFuse and **+4.60 AP<sub>s</sub>** over LFDT-Fusion in the protocol above.

> **Evaluation protocol matters.** The manuscript also includes separate recent-method comparisons, 256×256-input controls, and grayscale route experiments. Their results should be compared only within the corresponding protocols.

### Qualitative examples

The manuscript includes matched infrared, visible, and fused-image comparisons on airborne targets, general scenes, and aerial vehicles, together with target-region crops and prior-removal diagnostics.

<!-- Optional: add a verified montage at assets/qualitative.png and uncomment:
![Qualitative fusion comparisons](assets/qualitative.png)
-->

## 🗂️ Datasets

| Dataset | Pairs used | Role |
|:--|--:|:--|
| **MSRS (train)** | 1,083 | Fusion-model training |
| **RoadScene** | 50 | General IR–VIS fusion evaluation |
| **M3FD** | 300 | General IR–VIS fusion evaluation |
| **LSS-Plane** | 432 | Aircraft-scene fusion evaluation |
| **LSS-UAV (test)** | 614 | UAV fusion and small-target detection |
| **DroneVehicle** | 128 | Fixed selection from the test archive for qualitative analysis |

For LSS-UAV, the public splits described in the manuscript contain **4,604 train / 616 validation / 614 test** pairs. The detection test split includes **552 annotated UAV objects**; **75** satisfy the paper's *Tiny16* criterion (both native box dimensions below 16 pixels).

Please obtain each dataset from its official distributor and comply with the applicable license and usage conditions.

## ⚙️ Reproducibility

### Paper-reported configuration

| Item | Setting |
|:--|:--|
| Fusion training data | MSRS, 1,083 paired images |
| Training crops | 256 × 256 |
| Batch size | 8 |
| Optimizer | AdamW |
| Learning rate | 3 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| Training duration | 61 epochs |
| Main fusion seed | 2026 |
| Latent refinement | 5 DPM-Solver++ calls |
| Detector | YOLOv8n, trained on UAV training annotations and frozen for fusion evaluation |
| Main detection test | LSS-UAV, 614 frames |




🚀 Usage
1. Training

To train the model, run the train script:
python train.py


2. Testing

To test the model (IR mode), run the test-ir script:
python test-ir.py

LLS datasets：
https://pan.baidu.com/s/11V7J0yNHRJdtFjnQr5vr-Q?pwd=acmm 提取码: acmm
