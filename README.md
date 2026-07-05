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
