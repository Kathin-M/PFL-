# PFL – Running on Google Colab

## Prerequisites

- A Google account with **Google Drive** access.
- The file `cicids2017_cleaned.csv` uploaded to **My Drive → PFL** folder.
- This repo pushed to GitHub (`https://github.com/Kathin-M/PFL-.git`).

---

## Step-by-Step Instructions

### 1. Create a new Colab notebook

Go to [colab.research.google.com](https://colab.research.google.com) → **New Notebook**.

### 2. Switch to High-RAM

**Runtime → Change runtime type → High-RAM → Save.**

The dataset is ~684 MB; High-RAM prevents the VM from being killed.

### 3. Cell 1 — Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Authorise when prompted. Your CSV will be at:
```
/content/drive/MyDrive/PFL/cicids2017_cleaned.csv
```

### 4. Cell 2 — Clone repo & run setup

```bash
%%bash
git clone https://github.com/Kathin-M/PFL-.git /content/PFL-
cd /content/PFL-
bash setup_colab.sh
```

> **⏱ This takes ~8-10 minutes** (GHCup install + Libtorch download).

### 5. Cell 3 — Build the project

```bash
%%bash
source /content/pfl_env.sh
cp /content/cabal.project.local /content/PFL-/
cd /content/PFL-
cabal build 2>&1
```

> First build pulls all Haskell dependencies (~2 min).

### 6. Cell 4 — Train

```bash
%%bash
source /content/pfl_env.sh
cd /content/PFL-
cabal run pfl-train 2>&1
```

**Expected output:**
```
[1/3] Loading and filtering benign samples...
Loaded 2830540 benign samples (52 features each)

[2/3] Initialising SimCLR model...

[3/3] Training...
  Epoch   1 / 20  |  Loss: 3.456789
  Epoch   2 / 20  |  Loss: 2.891234
  ...
  Epoch  20 / 20  |  Loss: 0.567890

✓ Encoder weights saved to: pfl_encoder_weights.pt
```

### 7. Cell 5 — Copy weights to Drive (optional)

```bash
%%bash
cp /content/PFL-/pfl_encoder_weights.pt /content/drive/MyDrive/PFL/
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `cabal: command not found` | Re-run `source /content/pfl_env.sh` at the top of the cell |
| `libtorch*.so not found` at runtime | Add `export LD_LIBRARY_PATH=/usr/local/libtorch/lib:$LD_LIBRARY_PATH` before `cabal run` |
| OOM / kernel crash | Switch to **High-RAM** runtime; reduce `cfgBatchSize` in `Training.hs` |
| CSV parse error | Verify the CSV path and make sure Drive is still mounted |

---

## Project Structure

```
PFL-/
├── setup_colab.sh          # Colab environment bootstrap
├── pfl.cabal               # Cabal package config
├── cabal.project           # Cabal project settings
├── COLAB_INSTRUCTIONS.md   # This file
├── src/
│   ├── DataPipeline.hs     # CSV loading, filtering, augmentation
│   ├── SimCLR.hs           # MLP encoder + projection head
│   └── Training.hs         # NT-Xent loss + training loop
└── app/
    └── Main.hs             # Entry-point: load → train → save
```
