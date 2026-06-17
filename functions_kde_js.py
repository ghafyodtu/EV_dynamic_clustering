from my_imports import pd
import copy


def compute_joint_kdes(data, bandwidth, labels=None):
    """
    Compute joint KDEs for each cluster or a single KDE if no labels are provided.

    Parameters:
        data (pd.DataFrame or np.ndarray): Dataset (n_samples, n_features).
        bandwidth (float): Bandwidth for the KDE.
        labels (np.ndarray, optional): Cluster labels for each sample. Default is None.

    Returns:
        dict or KernelDensity: A dictionary mapping cluster labels to a tuple of their KDE estimators
                               and the number of occurrences, or a single KDE object if labels are not provided.
    """
    from sklearn.neighbors import KernelDensity
    from my_imports import np

    if labels is None:
        # If no labels are provided, compute a single KDE for the entire dataset
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(data)
        return kde

    unique_labels, counts = np.unique(labels, return_counts=True)
    kdes = {}

    for label, count in zip(unique_labels, counts):
        cluster_data = data[labels == label]
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(cluster_data)  # Fit KDE on the multi-dimensional data
        kdes[label] = [kde, count]  # Store KDE and count as a tuple

    return kdes

def compute_js_divergence(kde1, kde2, num_samples=3000):
    """
    Compute JS Divergence between two multi-dimensional KDEs using a fixed grid.

    Parameters:
        kde1, kde2 (KernelDensity): KDE estimators for the two distributions.
        grid (np.ndarray): A shared grid of points for evaluating both KDEs.

    Returns:
        float: JS Divergence score.
    """
    from my_imports import np
    from scipy.spatial.distance import jensenshannon
    samples1 = kde1.sample(num_samples, random_state=42)  # (500, 5)
    samples2 = kde2.sample(num_samples,  random_state=42)
    # grid = np.vstack([samples1, samples2])
    min_val = min(np.min(samples1), np.min(samples2))
    max_val = max(np.max(samples1), np.max(samples2))
    x_eval = np.linspace(min_val, max_val, num_samples).reshape(-1, 1)  # Reshape for sklearn
    # Evaluate KDEs at the same grid points
    log_prob1 = kde1.score_samples(x_eval)
    log_prob2 = kde2.score_samples(x_eval)

    prob1 = np.exp(log_prob1)
    prob2 = np.exp(log_prob2)

    # Normalize to ensure probabilities sum to 1
    prob1 /= prob1.sum()
    prob2 /= prob2.sum()

    return jensenshannon(prob1, prob2)


def fit_kde_per_feature(df, bandwidth=0.2, kernel="gaussian"):
    """
    Fits a KernelDensity estimator for each feature in the given DataFrame.

    Parameters:
        df (pd.DataFrame): Input data where each column is a feature.
        bandwidth (float): Bandwidth for the KDE (smoothing parameter).
        kernel (str): Kernel type (e.g., "gaussian", "tophat", "epanechnikov", etc.).

    Returns:
        list: A list of fitted KDE models (one for each feature).
    """
    from sklearn.neighbors import KernelDensity
    kde_list = []
    for col in df.columns:
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        kde.fit(df[[col]].values)  # Fit using single-column 2D array
        kde_list.append(kde)
    return kde_list


def compute_js_divergence_mD(kde1, kde2, num_samples=3000):
    """
    Compute JS Divergence between two multi-dimensional KDEs using a fixed grid.

    Parameters:
        kde1, kde2 (KernelDensity): KDE estimators for the two distributions.
        grid (np.ndarray): A shared grid of points for evaluating both KDEs.

    Returns:
        float: JS Divergence score.
    """
    from my_imports import np
    from scipy.spatial.distance import jensenshannon
    samples1 = kde1.sample(num_samples, random_state=42)  # (500, 5)
    samples2 = kde2.sample(num_samples,  random_state=42)
    grid = np.vstack([samples1, samples2])
    # Evaluate KDEs at the same grid points
    log_prob1 = kde1.score_samples(grid)
    log_prob2 = kde2.score_samples(grid)
    prob1 = np.exp(log_prob1)
    prob2 = np.exp(log_prob2)
    # Normalize to ensure probabilities sum to 1
    prob1 /= prob1.sum()
    prob2 /= prob2.sum()
    return jensenshannon(prob1, prob2)




def create_joint_similarity_matrix(kdes1, kdes2, num_points):
    """
    Create a similarity matrix based on JS Divergence for joint KDEs.

    Parameters:
        kdes1, kdes2 (dict): Dictionaries of KDEs for datasets 1 and 2.
        support_points (np.ndarray): Points for evaluating KDEs.

    Returns:
        np.ndarray: Similarity matrix.
    """
    import numpy as np
    labels1 = list(kdes1.keys())
    labels2 = list(kdes2.keys())
    similarity_matrix = np.zeros((len(labels1), len(labels2)))

    for i, label1 in enumerate(labels1):
        for j, label2 in enumerate(labels2):
            similarity_matrix[i, j] = compute_js_divergence_mD(
                kdes1[label1], kdes2[label2], num_points
            )
    return similarity_matrix


def create_months(start, end):
    from datetime import datetime, timedelta
    # Convert strings to datetime objects
    start = datetime.strptime(start, "%m-%Y")
    end = datetime.strptime(end, "%m-%Y")
    # Generate a list of months between start and end dates
    months = []
    current = start
    while current <= end:
        months.append(current.strftime("%m-%Y"))
        # Increment by one month
        current += timedelta(days=31)  # Ensure incrementing by a full month
        current = current.replace(day=1)  # Reset to the first day of the next month
    return months


def read_scale_monthly_kde(scaler, m1, m2, band_width_):
    from clustering_functions import load_filter_data
    x_train = load_filter_data(start_m=m1, end_m=m2)
    x_train1 = scaler.transform(x_train)
    x_train = pd.DataFrame(x_train1, columns=x_train.columns)
    # create kde of month i
    monthly_kde_ = fit_kde_per_feature(x_train, bandwidth=band_width_)
    return x_train, monthly_kde_


def read_scale_monthly_kde_mD(scaler, m1, m2, band_width_):
    from clustering_functions import load_filter_data
    x_train = load_filter_data(start_m=m1, end_m=m2)
    x_train1 = scaler.transform(x_train)
    x_train = pd.DataFrame(x_train1, columns=x_train.columns)
    # create kde of month i
    monthly_kde_ = compute_joint_kdes(x_train, bandwidth=band_width_)
    return x_train, monthly_kde_


def categorize_count(x_train, km_objects, cluster_kdes_, r_month, c_month, scaler):
    from my_imports import StandardScaler, np
    km_scaler = StandardScaler()
    _ = km_scaler.fit(scaler.inverse_transform(x_train))
    x_train_km = km_scaler.transform(scaler.inverse_transform(x_train))
    x_train_km = pd.DataFrame(x_train_km, columns=x_train.columns)
    km_labels = km_objects[r_month].predict(x_train_km)
    x_train["km_labels"] = km_labels
    x_train.sort_values(by="km_labels", ascending=True, inplace=True)
    km_labels = x_train["km_labels"]
    x_train = x_train.drop(columns=["km_labels"])
    unique_labels, counts = np.unique(km_labels, return_counts=True)
    for (key, value), new_value in zip(cluster_kdes_[c_month].items(), counts.tolist()):
        cluster_kdes_[c_month][key][1] = new_value
    return x_train, cluster_kdes_


def cluster_similarity_analysis(similarity_matrix, cluster_objects_pool_, cluster_kdes_, js_lim_cluster, current_month):
    from my_imports import np
    import copy
    min_indices = np.argmin(similarity_matrix, axis=0)
    # Create a new matrix filled with 1s
    modified_matrix = np.ones_like(similarity_matrix)
    # Set the minimum values back to their original values
    for col in range(similarity_matrix.shape[1]):
        row = min_indices[col]
        modified_matrix[row, col] = similarity_matrix[row, col]
    similarity_matrix = modified_matrix
    indexes = np.where(similarity_matrix < js_lim_cluster)
    indexes = list(zip(indexes[0], indexes[1]))
    keys_list_cluster_pool = list(cluster_objects_pool_.keys())
    keys_list_current_month = list(cluster_kdes_[current_month].keys())
    placement_list = keys_list_current_month.copy()
    if indexes:
        # replace the similar clusters
        for item in indexes:
            #  Go  to new cluster and mark the new clusters
            placement_list[item[1]] = keys_list_cluster_pool[item[0]]
            a_ = cluster_kdes_[current_month].pop(keys_list_current_month[item[1]], False)
            if a_:
                cluster_kdes_[current_month][keys_list_cluster_pool[item[0]]] = a_
    sorted_dict = {key: cluster_kdes_[current_month][key] for key in placement_list if key in cluster_kdes_[current_month]}
    cluster_kdes_[current_month] = copy.deepcopy(sorted_dict)
    for key, value in cluster_kdes_[current_month].items():
        if key not in cluster_objects_pool_:
            cluster_objects_pool_[key] = value[0]
    return cluster_objects_pool_, cluster_kdes_


def _as_2d(X):
    import numpy as np
    """Ensure X is (n_samples, n_features)."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def scott_bandwidth(X):
    """
    Scott's rule for Gaussian KDE.

    Returns
    -------
    bw_per_dim : (d,) ndarray
        Per-dimension bandwidths h_j = factor * std_j.
    bw_isotropic : float
        Single scalar bandwidth using RMS(std) * factor (good for isotropic KDE).
    """
    import numpy as np
    X = _as_2d(X)
    n, d = X.shape
    if n < 2:
        raise ValueError("Need at least 2 samples.")
    std = X.std(axis=0, ddof=1)
    factor = n ** (-1.0 / (d + 4.0))
    bw_per_dim = factor * std
    # Isotropic scalar: match overall scale via RMS of per-dim stds
    bw_isotropic = float(factor * np.sqrt(np.mean(std**2)))
    return bw_per_dim, bw_isotropic


def silverman_bandwidth(X):
    """
    Silverman's rule of thumb for Gaussian KDE.

    Returns
    -------
    bw_per_dim : (d,) ndarray
        Per-dimension bandwidths h_j = factor * std_j.
    bw_isotropic : float
        Single scalar bandwidth using RMS(std) * factor (good for isotropic KDE).
    """
    import numpy as np
    X = _as_2d(X)
    n, d = X.shape
    if n < 2:
        raise ValueError("Need at least 2 samples.")
    std = X.std(axis=0, ddof=1)
    factor = (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0)) * n ** (-1.0 / (d + 4.0))
    bw_per_dim = factor * std
    bw_isotropic = float(factor * np.sqrt(np.mean(std**2)))
    return bw_per_dim, bw_isotropic


def find_mlcv_bandwidth(
        x_train,
        label_col=None,
        sample_size=15000,
        bandwidths=None,
        cv=5,
        kernel="gaussian",
):
    """
    Find the best KDE bandwidth using maximum likelihood cross-validation.

    Parameters
    ----------
    x_train : pandas.DataFrame
        Training data containing feature columns and optionally a label column.

    label_col : str
        Name of the label column to remove.

    sample_size : int or None
        Number of rows to use after scaling. If None, use all rows.

    bandwidths : array-like or None
        Candidate bandwidth values. If None, uses np.arange(0.02, 0.201, 0.02).

    cv : int
        Number of cross-validation folds.

    kernel : str
        Kernel type for KernelDensity.

    Returns
    -------
    h_mlcv : float
        Best bandwidth found by GridSearchCV.
    grid : GridSearchCV
        Fitted GridSearchCV object.
    scaler : StandardScaler
        Fitted scaler object.
    """
    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from my_imports import np
    if bandwidths is None:
        bandwidths = np.arange(0.04, 0.261, 0.02)
    # Remove labels column
    if label_col:
        X = x_train.drop(columns=[label_col])
    else:
        X = x_train
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Optionally limit sample size
    if sample_size is not None:
        X_scaled = X_scaled[:sample_size]
    # Maximum likelihood cross-validation
    grid = GridSearchCV(
        KernelDensity(kernel=kernel),
        {"bandwidth": bandwidths},
        cv=cv
    )
    grid.fit(X_scaled)
    h_mlcv = grid.best_params_["bandwidth"]
    return h_mlcv, grid, scaler


def run_md_sensitivity_analysis(
        global_start_date="01-2023",
        global_end_date="01-2025",
        js_thr_val_list=None,
        n_js_samples=500,
        plot=True,
):
    """
    Run sensitivity analysis for MD drift detection over different JS thresholds.

    Parameters
    ----------
    global_start_date : str
        Start month, e.g. "01-2023".

    global_end_date : str
        End month, e.g. "01-2025".

    js_thr_val_list : list or None
        List of JS thresholds to test.
        If None, uses [0.06, 0.08, ..., 0.24].

    n_js_samples : int
        Number of samples used when computing JS divergence.

    plot : bool
        Whether to plot threshold vs number of detected drifts.

    Returns
    -------
    results : dict
        Dictionary containing:
        - "js_thr_val_list"
        - "number_of_drifts_list"
        - "details_per_threshold"
    """
    from my_imports import StandardScaler, np
    from clustering_functions import load_filter_data
    if js_thr_val_list is None:
        js_thr_val_list = [round(x, 2) for x in np.arange(0.06, 0.26, 0.02)]

    number_of_drifts_list = []
    details_per_threshold = {}

    months = create_months(global_start_date, global_end_date)

    for js_lim_month in js_thr_val_list:
        monthly_kde = {}
        monthly_kde_mD = {}
        js_md_per_month = {}

        ref_month = global_start_date
        stream_order = []
        js_mat_time_order = []
        number_of_drifts = 0
        # Fit scaler on first month interval
        scaler = StandardScaler()
        x_train_first = load_filter_data(
            start_m=months[0],
            end_m=months[1]
        )
        _ = scaler.fit(x_train_first)
        # Bandwidth selection
        band_width, _, _ = find_mlcv_bandwidth(x_train_first)
        for idx in range(len(months) - 1):
            current_month_pair = (months[idx], months[idx + 1])
            current_month = current_month_pair[0]

            print(
                f"threshold={js_lim_month}, "
                f"current_month_pair={current_month_pair[0]} to {current_month_pair[1]}"
            )

            x_month, monthly_kde[current_month] = read_scale_monthly_kde(
                scaler,
                current_month_pair[0],
                current_month_pair[1],
                band_width
            )

            _, monthly_kde_mD[current_month] = read_scale_monthly_kde_mD(
                scaler,
                current_month_pair[0],
                current_month_pair[1],
                band_width
            )

            if current_month == global_start_date:
                stream_order.append(global_start_date)
                ref_month = current_month
                continue

            js_md_per_month[current_month] = compute_js_divergence_mD(
                monthly_kde_mD[current_month],
                monthly_kde_mD[ref_month],
                n_js_samples
            )

            if js_md_per_month[current_month] < js_lim_month:
                continue

            number_of_drifts += 1
            js_mat_time_order.append(current_month)

            state = False
            js_dict = {}

            for item in monthly_kde_mD:
                if item == current_month or item == ref_month:
                    continue

                js = compute_js_divergence_mD(
                    monthly_kde_mD[current_month],
                    monthly_kde_mD[item],
                    n_js_samples
                )

                js_dict[item] = [np.round(js, 5)]

                if js < js_lim_month:
                    ref_month = item
                    state = True
                    stream_order.append(ref_month)
                    js_mat_time_order.append(current_month)
                    break

            if not state:
                ref_month = current_month
                stream_order.append(current_month)

        print("threshold:", js_lim_month)
        print("number of drifts:", number_of_drifts)

        number_of_drifts_list.append(number_of_drifts)

        details_per_threshold[js_lim_month] = {
            "number_of_drifts": number_of_drifts,
            "stream_order": stream_order,
            "js_mat_time_order": js_mat_time_order,
            "js_md_per_month": js_md_per_month,
            "band_width": band_width,
        }

    if plot:
        from my_imports import plt
        plt.figure(figsize=(10 / 1.5, 6 / 1.5))
        plt.plot(js_thr_val_list, number_of_drifts_list, marker="o")
        plt.xlabel("Threshold")
        plt.ylabel("Number of Detected Drifts")
        plt.grid(True)
        plt.tight_layout()
        plt.xticks(np.arange(0.04, 0.28, 0.02))
        plt.yticks(np.arange(0, max(number_of_drifts_list) + 1, 2))
        plt.show()

    results = {
        "js_thr_val_list": js_thr_val_list,
        "number_of_drifts_list": number_of_drifts_list,
        "details_per_threshold": details_per_threshold,
    }

    return results


def run_cluster_threshold_sensitivity(
        all_clusters_kde,
        thresholds=None,
        n_samples=1000,
        plot=True,
):
    """
    Run sensitivity analysis for cluster matching over different JS thresholds.

    Parameters
    ----------
    all_clusters_kde : dict
        Dictionary of KDE objects where keys include month information,
        for example: "01-2023_cluster_0".

    thresholds : list or None
        JS thresholds to test.
        If None, uses [0.10, 0.11, ..., 0.24].

    n_samples : int
        Number of samples used in create_joint_similarity_matrix.

    plot : bool
        Whether to plot threshold vs number of distinctive clusters.

    Returns
    -------
    results : dict
        Dictionary containing:
        - "thresholds_clusters"
        - "clusters"
        - "details_per_threshold"
    """
    from my_imports import np, plt
    if thresholds is None:
        thresholds = [round(x, 2) for x in np.arange(0.10, 0.25, 0.01)]

    clusters = []
    details_per_threshold = {}

    for js_threshold in thresholds:
        # Group KDEs by month-year
        grouped_dicts = {}

        for key, kde in all_clusters_kde.items():
            month_year = key.split("_")[0]

            if month_year not in grouped_dicts:
                grouped_dicts[month_year] = {}

            grouped_dicts[month_year][key] = kde

        grouped_values = list(grouped_dicts.values())

        if len(grouped_values) < 2:
            raise ValueError(
                "Expected at least two month groups in all_clusters_kde."
            )

        # First group is the reference cluster set
        kde_dict_1 = grouped_values[0]

        # Remaining groups are compared against the reference/prototype set
        list_of_groups = grouped_values[1:]

        cluster_objects_pl = copy.deepcopy(kde_dict_1)

        for cl in list_of_groups:
            sim_mat = create_joint_similarity_matrix(
                cluster_objects_pl,
                cl,
                n_samples
            )

            min_indices = np.argmin(sim_mat, axis=0)

            new_cl_dict_keys = list(cl.keys())
            cl_obj_dict_keys = list(cluster_objects_pl.keys())

            for col_idx, row_idx in enumerate(min_indices):
                js_value = sim_mat[row_idx, col_idx]

                key_to_add = new_cl_dict_keys[col_idx]
                kde_2 = cl[key_to_add]

                if js_value >= js_threshold:
                    cluster_objects_pl[key_to_add] = kde_2

        number_of_clusters = len(cluster_objects_pl.keys())
        clusters.append(number_of_clusters)

        details_per_threshold[js_threshold] = {
            "number_of_clusters": number_of_clusters,
            "cluster_keys": list(cluster_objects_pl.keys()),
        }

    if plot:
        plt.figure(figsize=(10 / 1.5, 6 / 1.5))
        plt.plot(thresholds, clusters, marker="o")
        plt.xlabel("Threshold")
        plt.ylabel("Number of Clusters")
        plt.title("Number of Distinctive Clusters by Threshold")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    results = {
        "thresholds_clusters": thresholds,
        "clusters": clusters,
        "details_per_threshold": details_per_threshold,
    }

    return results

def fit_cluster_kdes_with_mlcv_bandwidth(
        x_train,
        km_labels,
        label_col="labels",
        n_eff_max=10000,
        bandwidths=None,
        cv=5,
        kernel="gaussian",
        min_samples=10,
        random_state=42,
):
    """
    Append cluster labels to x_train internally, select MLCV bandwidth
    per cluster/label, and return only the maximum bandwidth.
    """
    from my_imports import StandardScaler, np
    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV
    if bandwidths is None:
        bandwidths = np.arange(0.02, 0.201, 0.02)

    # Copy so the original x_train is not modified
    x_train_labeled = x_train.copy()
    x_train_labeled[label_col] = km_labels

    rng = np.random.default_rng(random_state)
    max_bandwidth = None

    for label_value in sorted(x_train_labeled[label_col].unique()):
        df_label = x_train_labeled[x_train_labeled[label_col] == label_value]

        X = df_label.drop(columns=[label_col])

        if len(X) < min_samples:
            continue

        scaler = StandardScaler()
        X_scaled_all = scaler.fit_transform(X)

        n_eff = min(n_eff_max, len(X_scaled_all))

        idx = rng.choice(
            len(X_scaled_all),
            size=n_eff,
            replace=False
        )

        X_scaled_bw = X_scaled_all[idx]

        grid = GridSearchCV(
            KernelDensity(kernel=kernel),
            {"bandwidth": bandwidths},
            cv=cv
        )

        grid.fit(X_scaled_bw)

        h_mlcv = grid.best_params_["bandwidth"]

        if max_bandwidth is None or h_mlcv > max_bandwidth:
            max_bandwidth = h_mlcv

    return max_bandwidth