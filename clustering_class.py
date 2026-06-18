from my_imports import KMeans as SKLearnKMeans


def bic_score(x, labels):
    """
    BIC score for the goodness of fit of clusters.
    This Python function is translated from the Golang implementation by the author of the paper.
    The original code is available here:

    https://github.com/bobhancock/goxmeans/blob/a78e909e374c6f97ddd04a239658c7c5b7365e5c/km.go#L778
    """
    from my_imports import np
    import math
    if not isinstance(x, np.ndarray):
        x = np.array(x).astype("float32")
    n_points = len(labels)
    n_clusters = len(set(labels))
    n_dimensions = x.shape[1]

    n_parameters = (n_clusters - 1) + (n_dimensions * n_clusters) + 1

    loglikelihood = 0
    for label_name in set(labels):
        x_cluster = x[labels == label_name]
        n_points_cluster = len(x_cluster)
        centroid = np.mean(x_cluster, axis=0)
        variance = np.sum((x_cluster - centroid) ** 2) / (len(x_cluster) - 1)
        loglikelihood += \
            n_points_cluster * np.log(n_points_cluster) \
            - n_points_cluster * np.log(n_points) \
            - n_points_cluster * n_dimensions / 2 * np.log(2 * math.pi * variance) \
            - (n_points_cluster - 1) / 2

    bic = loglikelihood - (n_parameters / 2) * np.log(n_points)
    return bic


def plot_evaluation_metrics(metrics_history, month=None):
    """Plots the evaluation metrics in a 1-row, n-column subplot layout and saves the figure."""
    import os
    from my_imports import plt

    metrics = list(metrics_history.evaluation_metrics.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    # Handle case where there's only one metric
    if len(metrics) == 1:
        axes = [axes]
    # Plot each metric
    for i, metric in enumerate(metrics):
        x_values = [x[0] for x in metrics_history.evaluation_metrics[metric]]
        y_values = [y[1] for y in metrics_history.evaluation_metrics[metric]]
        axes[i].plot(x_values, y_values, marker='o')
        axes[i].set_xlabel('Number of Clusters')
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].grid(True)
    # Add a title
    if month:
        fig.suptitle(f"Quality Index for {month}", fontsize=16)
    # Save the figure
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"evaluation_metrics_{month}.png")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    plt.savefig(output_path)
    # Show the plot
    plt.close()


class KmeansC(SKLearnKMeans):
    def __init__(self, n_clusters=8, init='k-means++', n_init=50, max_iter=300, tol=1e-4, random_state=42):
        super().__init__(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                         random_state=random_state)
        self.evaluation_metrics = {}

    @property
    def labels(self):
        """Returns the labels of the clusters."""
        return self.labels_

    @property
    def centers(self):
        """Returns the cluster centers."""
        return self.cluster_centers_

    def clustering_evaluation(self, x_, n1, n2, metrics=None):
        """
        Evaluates clustering using specified metrics over a range of clusters.

        Parameters:
        - x_: Data to cluster
        - n1: Minimum number of clusters to evaluate
        - n2: Maximum number of clusters to evaluate
        - metrics: List of metrics to evaluate. Options are 'silhouette', 'davies_bouldin', 'inertia', 'bic_score',
         'calinski_harabasz_score'.
                   Default is all metrics.

        Returns:
        - A dictionary with the evaluation results for the specified metrics.
        """
        from my_imports import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        if metrics is None:
            metrics = ['silhouette', 'davies_bouldin', 'inertia', "bic_score", "calinski_harabasz"]
        silhouette_scores, davies_bouldin_scores, inertia_scores, bic_scores, calinski_harabasz_scores = ([], [],
                                                                                                          [], [], [])
        for n_clusters in range(n1, n2 + 1):
            model = SKLearnKMeans(n_clusters=n_clusters, init=self.init, n_init=self.n_init,
                                  max_iter=self.max_iter, tol=self.tol, random_state=self.random_state)
            labels = model.fit_predict(x_)

            if 'silhouette' in metrics:
                silhouette_scores.append((n_clusters, silhouette_score(x_, labels)))
                self.evaluation_metrics["silhouette"] = silhouette_scores
            if 'davies_bouldin' in metrics:
                davies_bouldin_scores.append((n_clusters, davies_bouldin_score(x_, labels)))
                self.evaluation_metrics["davies_bouldin"] = davies_bouldin_scores
            if 'inertia' in metrics:
                inertia_scores.append((n_clusters, model.inertia_))
                self.evaluation_metrics["inertia"] = inertia_scores
            if 'bic_score' in metrics:
                bic_scores.append((n_clusters, bic_score(x_, labels)))
                self.evaluation_metrics["bic"] = bic_scores
            if 'calinski_harabasz' in metrics:
                calinski_harabasz_scores.append((n_clusters, calinski_harabasz_score(x_, labels)))
                self.evaluation_metrics["calinski_harabasz"] = calinski_harabasz_scores
