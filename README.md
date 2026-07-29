EV Dynamic Clustering

This repository contains research code for a dynamic, drift-aware clustering framework for analyzing electric-vehicle (EV) charging behavior over time.

The framework processes charging data in consecutive monthly batches, monitors distributional changes, and updates or reuses clustering models depending on whether new data differ sufficiently from previously observed patterns. It is designed to support the identification and tracking of recurring, emerging, and changing charging-session patterns.

Method overview

The workflow consists of four main steps:

Monthly data preparation
Charging sessions are grouped into monthly batches and prepared for analysis using features related to energy demand, plug-in time, plug-in duration, and charging delay.
Distributional drift detection
Kernel density estimation (KDE) is used to represent each monthly batch. Jensen--Shannon (JS) distance is then used to compare the current batch with reference distributions.
Drift-aware clustering
When a meaningful distributional change is detected, a new clustering model may be trained. Otherwise, an existing clustering model is reused.
Cluster similarity analysis
Cluster-level KDEs and JS distances are used to compare newly obtained clusters with previously stored clusters, allowing recurring patterns to retain consistent labels over time.

The current implementation uses k-means clustering with k-means++ initialization. The number of clusters is selected using the Silhouette score.

Repository structure
EV_dynamic_clustering/
├── main_01.ipynb
├── clustering_class.py
├── clustering_functions.py
├── functions_kde_js.py
├── plot_function_results.py
├── LICENSE
└── README.md
main_01.ipynb
Main notebook containing the workflow for the dynamic clustering analysis.
clustering_class.py
Custom k-means class and clustering-quality evaluation functions.
clustering_functions.py
Functions supporting data preparation, monthly processing, model creation, model reuse, and clustering workflow management.
functions_kde_js.py
KDE estimation, Jensen--Shannon distance calculation, drift detection, and cluster similarity functions.
plot_function_results.py
Functions for visualization. 


Python: 3.11.9

| Package | Version |
|---|---|
| numpy | 2.1.3 |
| pandas | 2.3.1 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| scikit-learn | 1.7.1 |
| scipy | 1.16.1 |
