# DGSA: Density-Guided Generative Augmentation for Active Learning in Cybersecurity
---

## About the Project

**DGSA** (Density-Guided Synthetic Augmentation) is a post-query augmentation framework that operates within a standard pool-based active learning loop. After each acquisition round, DGSA:

- Projects the current labeled and unlabeled pools into a representation space induced by the classifier.
- Selects class-dependent **anchor points** from the labeled pool using a diversity-aware selection procedure that spreads anchors across distinct local regions within each class.
- Retrieves **local neighborhoods** from the unlabeled pool around each anchor to estimate local support and identify regions where additional augmentation is most needed.
- Allocates augmentation **jointly by class imbalance and local density**: minority classes receive larger augmentation budgets, while anchors in sparse or isolated regions receive fewer generated samples to avoid unrealistic interpolation.
- Trains **local class-conditional generators** (TVAE, CTGAN, or RTF) on each anchor neighborhood and samples synthetic examples proportional to the anchor's support score.


This closed-loop, density-aware process adaptively shifts support toward underrepresented regions as the labeled pool evolves—leading to consistent gains in macro-F1 for minority classes without degrading majority-class performance. DGSA is agnostic to the underlying acquisition strategy and classifier architecture, and has been validated across Margin, PowerMargin, Entropy, Density, CLUE, and GALAXY query strategies with Random Forest, XGBoost, MLP, and TabM classifiers.

On cybersecurity datasets (CIC-IDS-2017, CIC-IDS-2018, BODMAS, APIGRAPH, ANDMAL) and non-cybersecurity tabular benchmarks (Shuttle, Cover, Satellite), DGSA consistently outperforms standard active learning and uniform generative augmentation baselines, achieving **3–7 average F1-point gains** over BaseAL and **4–10 points** over uniform generators at a 1% labeling budget.

---

## Motivation

When annotation budgets are tight, active learning often misses rare classes, leading to poor generalization. Existing class-conditional augmentation methods assume enough labeled examples per class to reliably estimate the class distribution—an assumption that routinely fails for minority classes under limited budgets. DGSA addresses this gap by estimating local structure around individual anchor points and selectively expanding underrepresented regions in the feature space using density-guided generative sampling.

---

## Built With

- **Python 3.x**
- Generative modeling: TVAE, CTGAN, RTF (RealTabFormer), TabDDPM
- Classifiers: MLP, Random Forest (RF), XGBoost (XGBC), TabM

---

## 🛠 Getting Started

### Prerequisites

- Python 3.x
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Setup

1. Clone this repo:
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   ```
2. Place your input CSV in `source_code/raw_data/`.
3. Edit `source_code/config.py` to define:
   - `LABEL_NAME`, `FEATURE_NAMES`, `DISCRETE_FEATURES`
   - `NUM_FEATURE`, `STANDARDIZE`, `NUM_CLASS`, etc.
4. Example config snippet:

   [config screenshot placeholder]

---

## Usage

### 1. Data Preprocessing

From `source_code/`, run:
```bash
python -m data_pre_process_pipeline \
  --input_csv adult.csv \
  --output_dir adult \
  --label_col income \
  --discrete_to_label
```
- `--discrete_to_label`: label-encodes discrete/categorical features. Use ONLY if your discrete features are not numerical.
- `output_dir` must match the reference name set in `config.py`.

Produces: `data/adult/train.npz`, `val.npz`, `test.npz`, and `label2id.json`.

### 2. Main Experiment Pipeline

Run:
```bash
python -m main \
  --al_method <base|DA|DGSA> \
  --al_function <random|entropy|lc|margin|coreset|galaxy|bald|powermargin|clue|diana|eada|upper|lower|density> \
  --classifier <MLP|RF|XGBC> \
  --dataset adult \
  --budget 5 \
  --random_state 42 \
  --generator CTGAN \
  --num_synthetic 3.0 \
  --filter_synthetic \
  --alpha 1.0 \
  --steepness 50.0
```

#### Brief argument descriptions

| Argument             | Description                                     |
|----------------------|-------------------------------------------------|
| `--al_method`         | Experiment type                                  |
| `--al_function`       | Active learning selection strategy               |
| `--classifier`        | Classification model                             |
| `--dataset`           | Reference dataset name                           |
| `--budget`            | Number of samples per AL round                  |
| `--random_state`      | Seed for reproducibility                         |
| `--generator`         | Generative model for augmentation                |
| `--num_synthetic`     | Synthetic sample multiplier                      |
| `--filter_synthetic`  | Only include filtered synthetic data             |
| `--alpha`, `--steepness` | Generation hyperparameters                   |
| `--decay_power`       | Controls generation for all classes, (higher decay focuses generation on minority classes) |        

❗ Note: `galaxy` and `clue` AL methods work **only** with the **MLP classifier**.

#### Example Runs

**Base Active Learning (no augmentation):**
```bash
python -m main --al_method base --al_function random --classifier MLP --dataset adult --budget 5
```

**With data augmentation only:**
```bash
python -m main --al_method DA --al_function random --generator CTGAN --classifier XGBC --dataset adult --budget 5
```

**Full DGSA pipeline:**
```bash
python -m main --al_method DGSA --al_function random --generator CTGAN --classifier XGBC --dataset adult --budget 5 --filter_synthetic --alpha 1 --steepness 50
```

---

## Project Structure

```
source_code/
├── active_learning_functions/
├── classifiers/
├── config.py
├── data_pre_process_pipeline.py
├── main.py
├── raw_data/
├── data/
└── results/
```

---

## Results & Benchmarks

Place outputs under `results/`. DGSA consistently outperforms standard methods, especially in class-imbalanced and distribution-shifted scenarios.

---
  --input_csv adult.csv \
  --output_dir adult \
  --label_col income \
  --discrete_to_label
```
- `--discrete_to_label`: label‑encodes discrete/categorical features. Use ONLY if your discrete features are not numerical
- `output_dir` must match the reference name set in `config.py`.

Produces: `data/adult/train.npz`, `val.npz`, `test.npz`, and `label2id.json`.

### 2. Main Experiment Pipeline

Run:
```bash
python -m main \
  --al_method <base|DA|DA+ALFA> \
  --al_function <random|entropy|lc|margin|coreset|galaxy|bald|powermargin|clue|diana|eada|upper|lower|density> \
  --classifier <MLP|RF|XGBC> \
  --dataset adult \
  --budget 5 \
  --random_state 42 \
  --generator CTGAN \
  --num_synthetic 3.0 \
  --filter_synthetic \
  --alpha 1.0 \
  --steepness 50.0
```

#### Brief argument descriptions

| Argument             | Description                                     |
|----------------------|-------------------------------------------------|
| `--al_method`         | Experiment type                                  |
| `--al_function`       | Active learning selection strategy               |
| `--classifier`        | Classification model                             |
| `--dataset`           | Reference dataset name                           |
| `--budget`            | Number of samples per AL round                  |
| `--random_state`      | Seed for reproducibility                         |
| `--generator`         | Generative model for augmentation                |
| `--num_synthetic`     | Synthetic sample multiplier                      |
| `--filter_synthetic`  | Only include filtered synthetic data             |
| `--alpha`, `--steepness` | Generation hyperparameters                   |

❗ Note: `galaxy` and `clue` AL methods work **only** with the **MLP classifier**.

#### Example Runs

**Base Active Learning (no augmentation):**
```bash
python -m main --al_method base --al_function random --classifier MLP --dataset adult --budget 5
```

**With data augmentation only:**
```bash
python -m main --al_method DA --al_function random --generator CTGAN --classifier XGBC --dataset adult --budget 5
```

**Full ALFA pipeline:**
```bash
python -m main --al_method DA+ALFA --al_function random --generator CTGAN --classifier XGBC --dataset adult --budget 5 --filter_synthetic --alpha 1 --steepness 50
```

---

## Project Structure

```
source_code/
├── active_learning_functions/
├── classifiers/
├── config.py
├── data_pre_process_pipeline.py
├── main.py
├── raw_data/
├── data/
└── results/
```

---

## Results & Benchmarks

Place outputs under `results/`. ALFA consistently outperforms standard methods, especially in class‑imbalanced and distribution‑shifted scenarios.

---
