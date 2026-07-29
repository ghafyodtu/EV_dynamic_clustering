# ⚡ Dynamic Clustering of Electric Vehicle Charging Sessions

This repository contains research code for a **dynamic, drift-aware clustering framework** for analyzing electric-vehicle (EV) charging behavior over time.

The framework processes charging data in consecutive monthly batches, monitors distributional changes, and updates or reuses clustering models depending on whether new data differ sufficiently from previously observed patterns. It is designed to support the identification and tracking of recurring, emerging, and changing charging-session patterns.

---

## 🔄 Method Overview

The workflow consists of four main steps.

### 1. 📅 Monthly Data Preparation

Charging sessions are grouped into monthly batches and prepared for analysis using features related to:

- Energy demand
- Plug-in time
- Plug-in duration
- Charging delay

### 2. 📈 Distributional Drift Detection

Kernel density estimation (KDE) is used to represent each monthly batch. Jensen–Shannon (JS) distance is then used to compare the current batch with reference distributions.

### 3. 🧠 Drift-Aware Clustering

When a meaningful distributional change is detected, a new clustering model may be trained. Otherwise, an existing clustering model is reused.

### 4. 🔗 Cluster Similarity Analysis

Cluster-level KDEs and JS distances are used to compare newly obtained clusters with previously stored clusters, allowing recurring patterns to retain consistent labels over time.

---

## ⚙️ Clustering Configuration

The current implementation uses:

- **Clustering algorithm:** K-means
- **Initialization:** k-means++
- **Cluster selection:** Silhouette score

---

## 📁 Repository Structure

```text
EV_dynamic_clustering/
├── main_01.ipynb
├── clustering_class.py
├── clustering_functions.py
├── functions_kde_js.py
├── plot_function_results.py
├── seed_fix.py
├── LICENSE
└── README.md
```

### 📄 File Description

| File | Description |
|------|-------------|
| `main_01.ipynb` | Main notebook containing the complete dynamic clustering workflow. |
| `clustering_class.py` | Custom K-Means implementation and clustering evaluation functions. |
| `clustering_functions.py` | Data preparation, monthly processing, model creation, model reuse, and workflow management. |
| `functions_kde_js.py` | KDE estimation, Jensen–Shannon distance calculation, drift detection, and cluster similarity analysis. |
| `plot_function_results.py` | Visualization functions for clustering and drift analysis. |
| `seed_fix.py` | Utility functions for setting random seeds to ensure reproducible experiments and consistent clustering results. |

---

## 📦 Requirements

**Python:** 3.11.9

| Package | Version |
|---|---|
| numpy | 2.1.3 |
| pandas | 2.3.1 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| scikit-learn | 1.7.1 |
| scipy | 1.16.1 |

---

## 📊 Dataset

This project uses the **ACN-Data** public EV charging dataset.

---

## 📖 Citation

If you use the ACN-Data dataset, please cite:

```bibtex
@inproceedings{lee_acndata_2019,
  author = {Lee, Zachary J. and Li, Tongxin and Low, Steven H.},
  title = {{ACN-Data}: {Analysis} and {Applications} of an {Open} {EV} {Charging} {Dataset}},
  booktitle = {Proceedings of the Tenth International Conference on Future Energy Systems},
  series = {e-Energy '19},
  month = jun,
  year = {2019},
  location = {Phoenix, Arizona}
}
```

---

## 📄 License

This project is distributed under the MIT License. See the `LICENSE` file for details.