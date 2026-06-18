
def drift_plot_v2(js_per_feature_mat, js_mat_time_order, end_date, drift_months):
    """
    :param js_per_feature_mat: per-feature JS values.
    :param js_mat_time_order: list of month labels in JS order.
    :param end_date: last x-axis label for step plot.
    :param drift_months: list of month labels OR indices where drift happens (same for all features).
    """
    from my_imports import np, plt

    js_per_feature_mat = np.array(js_per_feature_mat)

    # Remove duplicate months
    unique_indices = []
    seen = set()
    for idx, m in enumerate(js_mat_time_order):
        if m not in seen:
            seen.add(m)
            unique_indices.append(idx)

    filtered_months = [js_mat_time_order[i] for i in unique_indices]
    filtered_js_matrix = js_per_feature_mat[unique_indices, :]

    js_values = filtered_js_matrix
    x_months = filtered_months
    month_indices = np.arange(len(x_months))

    # Normalize drift_months input into indices
    drift_indices = []
    for dm in drift_months:
        if isinstance(dm, str):      # if user passes month labels
            if dm in x_months:
                drift_indices.append(x_months.index(dm))
        else:                         # if user passes indices
            drift_indices.append(dm)
    drift_indices = np.array(drift_indices)

    features = ['plugin_duration', 'plugin_hour_sine', 'plugin_hour_cosine', 'energy', 'delay']
    feature_name = ["Plugin duration", 'Plugin hour sine', "Plugin hour cosine", 'Energy', 'Delay']

    global_max = np.max(js_values)
    global_ylim = [0, global_max * 1.1]

    extended_month_indices = np.append(month_indices, month_indices[-1] + 1)
    extended_x_months = x_months + [end_date]

    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]

        extended_js_values = np.append(js_values[:, i], js_values[-1, i])

        # Step plot
        ax.step(extended_month_indices, extended_js_values, where='post', linestyle='-', marker='')

        # Add red drift dots (same positions for all features)
        ax.plot(drift_indices + 0.5,
                js_values[drift_indices, i],
                'ro', markersize=5, label="Drift Event")

        ax.set_title(feature_name[i])
        ax.set_ylabel("")
        ax.set_xticks(extended_month_indices)
        ax.set_xticklabels(extended_x_months, rotation=90, ha='center')
        ax.set_ylim(global_ylim)
        ax.grid(True)

    # Legend subplot
    legend_ax = axes[-1]
    fig.delaxes(legend_ax)
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax = fig.add_subplot(n_rows, n_cols, len(axes))
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center")

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    plt.show()


def styled_step_plot_with_dots(data_dict, threshold=0.18):
    """
    Plot a step plot styled like `drift_plot` and add red dots where values exceed the threshold.

    :param data_dict: Dictionary with keys as 'MM-YYYY' strings and values as floats.
    :param threshold: Float value above which red dots are shown.
    """
    from my_imports import plt, pd
    # Convert to Series and sort by datetime index
    series = pd.Series(data_dict)
    series.index = pd.to_datetime(series.index, format='%m-%Y')
    series = series.sort_index()

    # Extend x-axis indices and labels
    month_indices = pd.RangeIndex(len(series))
    extended_month_indices = month_indices.append(pd.RangeIndex(start=month_indices[-1] + 1, stop=month_indices[-1] + 2))
    extended_labels = list(series.index.strftime('%m-%Y'))
    next_month_label = (series.index[-1] + pd.DateOffset(months=1)).strftime('%m-%Y')
    extended_labels.append(next_month_label)

    # Extend values for step plot
    extended_values = series.values.tolist() + [series.values[-1]]

    # Compute midpoints and identify where value exceeds threshold
    midpoints = month_indices[:-1] + 0.5
    above_threshold = series.values > threshold
    # Create the plot
    plt.figure(figsize=(15/1.75, 8/2))
    plt.step(extended_month_indices, extended_values, where='post', linestyle='-', marker='')
    plt.plot(midpoints[above_threshold[:-1]], series.values[:-1][above_threshold[:-1]], 'ro', markersize=4)
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylabel("JS distance value")
    plt.xticks(extended_month_indices, extended_labels, rotation=90, ha='center')
    plt.ylim([0, series.values.max() * 1.1])  # Add 10% padding
    plt.legend()
    plt.title("")
    plt.tight_layout()
    plt.show()


def rename_cluster_keys(cluster_dicts):
    """
    Renames second-level keys in a nested dictionary with globally increasing numbers,
    keeping consistent mapping across all months. Pads all numbers to 2 digits (e.g., _01, _10).
    Args:
        cluster_dicts (dict): Original nested dictionary with month-wise cluster keys.

    Returns:
        tuple:
            - dict: New dictionary with renamed cluster keys.
            - dict: Mapping from old keys to new keys.
    """
    key_mapping = {}
    counter = 1
    renamed_data = {}
    name_translation = {}
    for month, clusters in cluster_dicts.items():
        new_clusters = {}
        for old_key, value in clusters.items():
            key_root = old_key.rsplit('_', 1)[0]
            if old_key not in key_mapping:
                key_mapping[old_key] = counter
                counter += 1
            new_number = key_mapping[old_key]
            suffix_str = f"{new_number:02d}"  # pad to 2 digits
            new_key = f"{key_root}_{suffix_str}"
            name_translation[old_key] = new_key
            new_clusters[new_key] = value
        renamed_data[month] = new_clusters
    return renamed_data, name_translation


def compute_relative_popularity(data_):
    relative_popularity = {}

    for month, clusters in data_.items():
        total_count = sum(value[1] for value in clusters.values())

        if total_count == 0:
            continue  # Avoid division by zero

        relative_popularity[month] = {
            cluster: count[1] / total_count for cluster, count in clusters.items()
        }

    return relative_popularity


def plot_cluster_relative_popularity(
    relative_popularity_dict,
    figsize=(10, 5),
    cmap_name="Blues",
    tick_values=(5, 10, 15, 20, 25, 30),
    show=True,
):
    """
    Plot cluster relative popularity over time.

    Parameters
    ----------
    relative_popularity_dict : dict
        Dictionary in the form:

        {
            "01-2023": {
                "01-2023_0": 0.12,
                "01-2023_1": 0.25,
            },
            "02-2023": {
                "01-2023_0": 0.10,
                "02-2023_1": 0.30,
            }
        }

    figsize : tuple, optional
        Figure size.

    cmap_name : str, optional
        Matplotlib colormap name.

    tick_values : tuple, optional
        Colorbar tick values in percent.

    show : bool, optional
        If True, calls plt.show().

    Returns
    -------
    fig, ax, corrected_data, key_old_new
        Matplotlib figure, axis, corrected dictionary, and key mapping.
    """
    from my_imports import plt, np
    from datetime import datetime
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    corrected_data = {}
    key_old_new = {}

    # -------------------------------------------------------------
    # Rename cluster keys:
    # "01-2023_0" -> "C0(01-2023)"
    # -------------------------------------------------------------
    for month, subdict in relative_popularity_dict.items():
        new_subdict = {}

        for old_key, value in subdict.items():
            key_month_year, suffix = old_key.split("_")
            new_key = f"C{suffix}({key_month_year})"

            new_subdict[new_key] = value
            key_old_new[old_key] = new_key

        corrected_data[month] = new_subdict

    data = corrected_data

    # -------------------------------------------------------------
    # Sort months chronologically
    # -------------------------------------------------------------
    sorted_months = sorted(
        data.keys(),
        key=lambda x: datetime.strptime(x, "%m-%Y")
    )

    # -------------------------------------------------------------
    # Extract all unique clusters
    # -------------------------------------------------------------
    clusters = set()

    for month_data in data.values():
        clusters.update(month_data.keys())

    clusters = sorted(clusters)

    # -------------------------------------------------------------
    # Map months and clusters to plotting positions
    # -------------------------------------------------------------
    cluster_pos = {
        cluster: i
        for i, cluster in enumerate(clusters)
    }

    month_pos = {
        month: i
        for i, month in enumerate(sorted_months)
    }

    # -------------------------------------------------------------
    # Normalize popularity values for color intensity
    # -------------------------------------------------------------
    all_popularities = [
        popularity
        for month_data in data.values()
        for popularity in month_data.values()
    ]

    min_pop = min(all_popularities)
    max_pop = max(all_popularities)

    # -------------------------------------------------------------
    # Create plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)

    cmap = plt.colormaps[cmap_name]

    # -------------------------------------------------------------
    # Plot each cluster-month block
    # -------------------------------------------------------------
    for month, clusters_dict in data.items():
        for cluster, popularity in clusters_dict.items():

            x_start = month_pos[month]
            y_pos = cluster_pos[cluster]

            normalized_popularity = (
                (popularity - min_pop) / (max_pop - min_pop)
                if max_pop > min_pop else 1
            )

            color = cmap(normalized_popularity)

            ax.broken_barh(
                [(x_start, 1)],
                (y_pos - 0.4, 0.8),
                facecolors=color,
                edgecolors="white",
                linewidth=1,
                antialiased=False,
            )

    # -------------------------------------------------------------
    # Configure axes
    # -------------------------------------------------------------
    ax.set_xticks(range(len(sorted_months)))
    ax.set_xticklabels(sorted_months, rotation=90)

    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels(clusters)

    ax.set_xlabel("Month")
    ax.set_ylabel("Cluster")

    # -------------------------------------------------------------
    # Add colorbar
    # -------------------------------------------------------------
    norm = mcolors.Normalize(vmin=min_pop, vmax=max_pop)
    scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])

    cbar = plt.colorbar(scalar_mappable, ax=ax)

    tick_values = np.array(tick_values)
    cbar.set_ticks(tick_values / 100)
    cbar.set_ticklabels([f"{v}%" for v in tick_values])

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax, corrected_data, key_old_new


def plot_feature_kde_changes(
    feature_name,
    feature_names,
    js_per_feature_mat,
    js_mat_time_order,
    all_monthly_kde,
    scaler,
    base_month="01-2023",
    threshold=0.1,
    months_to_compare=None,
    num_samples=10000,
    figsize_per_plot=(3.2, 4),
    fill=True,
    alpha=0.2,
    linewidth=3,
):
    """
    Plot pairwise KDE comparisons for one feature across selected months.
    The function only displays the plot and does not return anything.
    """
    from my_imports import plt, sns
    # -------------------------------------------------------------
    # Find months where each feature changed above the JS threshold
    # -------------------------------------------------------------
    feature_dict_js = {
        feature: []
        for feature in feature_names
    }

    for month_idx, month_values in enumerate(js_per_feature_mat):
        for feature_idx, value in enumerate(month_values):
            if value > threshold:
                feature_dict_js[feature_names[feature_idx]].append(
                    js_mat_time_order[month_idx]
                )

    # -------------------------------------------------------------
    # Validate selected feature
    # -------------------------------------------------------------
    if feature_name not in feature_names:
        raise ValueError(
            f"Unknown feature_name: {feature_name}. "
            f"Available features are: {feature_names}"
        )

    feature_index = feature_names.index(feature_name)

    # -------------------------------------------------------------
    # Select months automatically or use manually provided months
    # -------------------------------------------------------------
    if months_to_compare is None:
        months_to_compare = [base_month] + feature_dict_js[feature_name]

    # Remove duplicate months while preserving order
    months_to_compare = list(dict.fromkeys(months_to_compare))

    if len(months_to_compare) < 2:
        raise ValueError(
            "At least two months are needed for comparison. "
            "Pass months_to_compare manually or lower the threshold."
        )

    # -------------------------------------------------------------
    # Check that selected months exist in all_monthly_kde
    # -------------------------------------------------------------
    missing_months = [
        month
        for month in months_to_compare
        if month not in all_monthly_kde
    ]

    if missing_months:
        raise KeyError(
            f"These months are missing from all_monthly_kde: {missing_months}"
        )

    # -------------------------------------------------------------
    # Choose x-axis label based on feature
    # -------------------------------------------------------------
    xlabel_map = {
        "Plugin duration": "Hours",
        "Delay": "Hours",
        "Energy": "Energy (kWh)",
        "Plugin hour sine": "Plugin hour sine",
        "Plugin hour cosine": "Plugin hour cosine",
    }

    xlabel = xlabel_map.get(feature_name, feature_name)

    # -------------------------------------------------------------
    # Sample from KDEs and inverse-transform to original scale
    # -------------------------------------------------------------
    samples = {}

    for month in months_to_compare:
        sampled_scaled = all_monthly_kde[month][feature_index].sample(
            num_samples
        ).flatten()

        sampled_original = (
            sampled_scaled * scaler.scale_[feature_index]
        ) + scaler.mean_[feature_index]

        samples[month] = sampled_original

    # -------------------------------------------------------------
    # Create pairwise KDE plots
    # -------------------------------------------------------------
    num_plots = len(months_to_compare) - 1

    fig_width = figsize_per_plot[0] * num_plots
    fig_height = figsize_per_plot[1]

    fig, axes = plt.subplots(
        1,
        num_plots,
        figsize=(fig_width, fig_height),
        sharey=True,
    )

    if num_plots == 1:
        axes = [axes]

    for i in range(num_plots):
        month1 = months_to_compare[i]
        month2 = months_to_compare[i + 1]

        data1 = samples[month1]
        data2 = samples[month2]

        ax = axes[i]

        sns.kdeplot(
            data1,
            ax=ax,
            label=month1,
            fill=fill,
            alpha=alpha,
            linewidth=linewidth,
        )

        sns.kdeplot(
            data2,
            ax=ax,
            label=month2,
            fill=fill,
            alpha=alpha,
            linewidth=linewidth,
        )

        ax.grid(alpha=0.4)
        ax.legend()
        ax.set_title(f"{month1} vs {month2}")

    fig.supxlabel(xlabel, y=-0.02)

    plt.tight_layout()
    plt.show()


def convert_sine_cosine_to_hour(df, sine_col, cosine_col, hour_col):
    from my_imports import np
    if sine_col in df.columns and cosine_col in df.columns:
        df[hour_col] = np.arctan2(df[sine_col], df[cosine_col]) / (2 * np.pi) * 24
        df[hour_col] = df[hour_col].apply(lambda x: x if x >= 0 else x + 24)
        df = df.drop(columns=[sine_col, cosine_col])
    return df


def mean_feature_each_cluster(cluster_objs, scaler, features):
    from my_imports import pd
    # Dictionary to store the computed mean values for each cluster
    mean_values = {}

    for cluster_name, kde in cluster_objs.items():
        # Sample 2000 points from the KDE (resulting in a (2000, 5) array)
        samples = kde.sample(2000, random_state=42)
        # Compute the mean along axis 0, resulting in a vector of length 5
        means = samples.mean(axis=0)
        mean_values[cluster_name] = means

    # Create a DataFrame where each row corresponds to a cluster
    df_means = pd.DataFrame.from_dict(mean_values, orient='index', columns=features)

    # Reset the index so that the cluster names become a column rather than the index.
    df_means = df_means.reset_index().rename(columns={'index': 'cluster'})

    df_means[features] = scaler.inverse_transform(df_means[features])
    df_means = convert_sine_cosine_to_hour(df_means, 'plugin_hour_sine',
                                           'plugin_hour_cosine', 'plugin_hour')
    return df_means


def plot_clusters_delay(cluster_objs, scaler, features):
    """
    For each cluster (row in centroids_df), plot two horizontal bars:
      - A red bar representing the delay portion (starting at plugin_hour, width = delay).
      - A green bar representing the non-delayed portion (starting at plugin_hour + delay,
        width = plugin_duration - delay).

    The height of the bars equals the energy value.

    The x-axis spans 0 to 48 hours (i.e. two days) with ticks every 4 hours.
    Tick labels above 24 are adjusted by subtracting 24 to mimic a 24-hour clock.

    A global y-axis maximum (based on the maximum energy across clusters) is used in every subplot,
    and the y-axis ticks are set every 5 units.

    The DataFrame `centroids_df` must contain the following columns:
      - 'cluster'         : Cluster name (or identifier)
      - 'plugin_hour'     : Starting time of plugin (in hours)
      - 'plugin_duration' : Total duration (in hours)
      - 'energy'          : Energy value (used for the bar height)
      - 'delay'           : Delay time duration (in hours)
    """
    from my_imports import plt, np
    import math
    # Compute the global maximum energy to use as the y-axis limit for all subplots.
    centroids_df = mean_feature_each_cluster(cluster_objs, scaler, features)
    global_max_energy = centroids_df['energy'].max()

    # Number of clusters
    n_clusters = centroids_df.shape[0]
    n_cols = 2
    n_rows = math.ceil(n_clusters / n_cols)

    # Create subplots arranged in two columns.
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(12, 8),
                             sharex=False)
    axes = axes.flatten()  # flatten the grid for easy iteration

    # Set x-ticks on every subplot (every 4 hours from 0 to 48)
    for ax in axes:
        ax.set_xticks(range(0, 49, 4))

    # Define colors for the two segments.
    cmap = plt.colormaps['tab20']
    colors = [cmap(i % 20) for i in range(3)]
    delay_color = 'red'
    non_delay_color = 'green'

    # Loop over each cluster row and plot its two segments.
    for i, (_, row) in enumerate(centroids_df.iterrows()):
        ax = axes[i]
        cluster_name = row['cluster']
        plugin_hour = row['plugin_hour']
        plugin_duration = row['plugin_duration']
        energy = row['energy']
        delay = row['delay']
        non_delay_width = plugin_duration - delay
        # Draw the delayed portion.
        rect_delay = plt.Rectangle((plugin_hour, 0), delay, energy, color=colors[0], alpha=0.8)
        ax.add_patch(rect_delay)

        # Draw the non-delayed portion.
        rect_non_delay = plt.Rectangle((plugin_hour + delay, 0), non_delay_width, energy, color=colors[2], alpha=0.8)
        ax.add_patch(rect_non_delay)

        # Set x-axis limits to span two days.
        ax.set_xlim(0, 48)
        # Set y-axis limit using the global max (with a little extra padding).
        y_max = global_max_energy * 1.2
        ax.set_ylim(0, 45)
        # Set y-axis ticks every 5 units.
        ax.set_yticks(np.arange(0, 45, 5))

        ax.set_title(f"Cluster {cluster_name}")
        if i % n_cols == 0:
            ax.set_ylabel("Energy (kWh)")

        if i >= n_clusters - n_cols:
            ax.set_xlabel("Hour of Day")

        # Add the small color legend (patch) inside the subplot

        ax.grid(alpha=0.2)
    # Hide any unused subplots.
    for ax in axes[n_clusters:]:
        ax.axis('off')

    # Adjust x tick labels: for ticks > 24, subtract 24 to mimic a 24-hour clock.
    for ax in axes[:n_clusters]:
        xticks = ax.get_xticks()
        ax.set_xticklabels([int(xtick - 24) if xtick > 24 else int(xtick) for xtick in xticks])

        from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=colors[0], label='Delaying'),
        Patch(color=colors[2], label='Charging')
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    return centroids_df


def apply_new_names(original_dict, name_translation):
    """
    Renames the keys in a flat dictionary using a name translation mapping.

    Args:
        original_dict (dict): Dictionary with original keys.
        name_translation (dict): Mapping from old keys to new keys.

    Returns:
        dict: New dictionary with renamed keys.
    """
    renamed_dict = {}
    for old_key, value in original_dict.items():
        new_key = name_translation.get(old_key, old_key)  # fallback to old_key if not found
        renamed_dict[new_key] = value
    return renamed_dict


def sample_cluster_df(
        kde,
        scaler,
        features=None,
        n_samples=6000,
        random_state=42,
        clip_negative=True
):
    from my_imports import np, pd
    if features is None:
        features = [
            "plugin_duration",
            "plugin_hour_sine",
            "plugin_hour_cosine",
            "energy",
            "delay"
        ]

    samples = kde.sample(n_samples, random_state=random_state)
    samples_real = scaler.inverse_transform(samples)

    df = pd.DataFrame(samples_real, columns=features)

    df["plugin_hour"] = (
                                np.arctan2(
                                    df["plugin_hour_sine"],
                                    df["plugin_hour_cosine"]
                                )
                                / (2 * np.pi)
                                * 24
                        ) % 24

    if clip_negative:
        for col in ["plugin_duration", "energy", "delay"]:
            df[col] = df[col].clip(lower=0)

    return df[
        [
            "plugin_duration",
            "plugin_hour",
            "energy",
            "delay",
            "plugin_hour_sine",
            "plugin_hour_cosine"
        ]
    ]


def average_load_profile_from_energy(
        df,
        plugin_hour_col="plugin_hour",
        energy_col="energy",
        delay_col="delay",
        kwh_per_hour=11
):
    from my_imports import np, pd
    average_load = np.zeros(24)

    for _, row in df.iterrows():
        plugin_hour = row[plugin_hour_col]
        energy = row[energy_col]
        delay = row[delay_col]

        if energy <= 0:
            continue

        n_hours = int(np.ceil(energy / kwh_per_hour))
        start_hour = int(np.floor(plugin_hour + delay)) % 24

        for h in range(n_hours):
            average_load[(start_hour + h) % 24] += kwh_per_hour

    average_load = average_load / len(df)

    return pd.DataFrame({
        "hour": np.arange(24),
        "average_load": average_load
    })


def delay_per_hour_per_session_profile(
        df,
        plugin_hour_col="plugin_hour",
        delay_col="delay",
        normalize=True
):
    from my_imports import pd, np
    delay_profile = np.zeros(24)

    for _, row in df.iterrows():
        plugin_hour = row[plugin_hour_col]
        delay = row[delay_col]

        if delay <= 0:
            continue

        start_hour = int(np.floor(plugin_hour)) % 24
        n_delay_hours = int(np.ceil(delay))

        for h in range(n_delay_hours):
            delay_profile[(start_hour + h) % 24] += 1

    if normalize:
        delay_profile = delay_profile / len(df)

    return pd.DataFrame({
        "hour": np.arange(24),
        "delay_per_hour_per_session": delay_profile
    })


def build_joint_sampled_df(
        kdes,
        cluster_names,
        scaler,
        features=None,
        cluster_weights=None,
        n_samples_per_cluster=6000,
        random_state=42,
        clip_negative=True
):
    from my_imports import pd
    if features is None:
        features = [
            "plugin_duration",
            "plugin_hour_sine",
            "plugin_hour_cosine",
            "energy",
            "delay"
        ]

    if cluster_weights is None:
        cluster_weights = {c: 1 / len(cluster_names) for c in cluster_names}
    else:
        total_weight = sum(cluster_weights[c] for c in cluster_names)
        cluster_weights = {
            c: cluster_weights[c] / total_weight
            for c in cluster_names
        }

    sampled_parts = []
    total_target_samples = n_samples_per_cluster * len(cluster_names)

    for i, cluster_name in enumerate(cluster_names):
        if cluster_name not in kdes:
            raise KeyError(f"{cluster_name} not found in kdes.")

        weight = cluster_weights[cluster_name]
        n_samples = int(round(total_target_samples * weight))
        n_samples = max(n_samples, 1)

        seed = random_state + i if random_state is not None else None

        df_cluster = sample_cluster_df(
            kde=kdes[cluster_name],
            scaler=scaler,
            features=features,
            n_samples=n_samples,
            random_state=seed,
            clip_negative=clip_negative
        )

        df_cluster["cluster"] = cluster_name
        df_cluster["cluster_weight"] = weight
        sampled_parts.append(df_cluster)

    return pd.concat(sampled_parts, ignore_index=True)


def plot_average_load_and_delay_separate(
        kdes,
        cluster_names,
        scaler,
        features=None,
        cluster_weights=None,
        n_samples_per_cluster=6000,
        random_state=42,
        kwh_per_hour=11,
        clip_negative=True,
        figsize=(9, 3.2),
        ymax_load=11,
        ymax_delay=1
):
    """
    Creates one joint plot with twin y-axes.

    Left y-axis:
        Average load profile [kW]

    Right y-axis:
        Delay-window coverage [per session]

    Both profiles are plotted as hourly step plots.
    """
    from my_imports import plt
    df_joint = build_joint_sampled_df(
        kdes=kdes,
        cluster_names=cluster_names,
        scaler=scaler,
        features=features,
        cluster_weights=cluster_weights,
        n_samples_per_cluster=n_samples_per_cluster,
        random_state=random_state,
        clip_negative=clip_negative
    )

    load_profile = average_load_profile_from_energy(
        df=df_joint,
        plugin_hour_col="plugin_hour",
        energy_col="energy",
        delay_col="delay",
        kwh_per_hour=kwh_per_hour
    )

    delay_profile = delay_per_hour_per_session_profile(
        df=df_joint,
        plugin_hour_col="plugin_hour",
        delay_col="delay",
        normalize=True
    )

    avg_delay = df_joint["delay"].mean()
    median_delay = df_joint["delay"].median()

    print(f"Joint average delay: {avg_delay:.2f} h")
    print(f"Joint median delay:  {median_delay:.2f} h")

    for c in cluster_names:
        c_delay = df_joint.loc[df_joint["cluster"] == c, "delay"]
        print(f"{c} average delay: {c_delay.mean():.2f} h")

    load_plot = load_profile
    delay_plot = delay_profile

    fig, ax1 = plt.subplots(figsize=figsize)

    ax2 = ax1.twinx()

    # Put delay axis visually behind load axis
    ax2.set_zorder(1)
    ax1.set_zorder(2)
    ax1.patch.set_visible(False)

    # Right y-axis: delay-window coverage, drawn first and underneath
    line2, = ax2.plot(
        delay_plot["hour"],
        delay_plot["delay_per_hour_per_session"],
        linewidth=2.0,
        color="tab:blue",
        alpha=0.55,
        label="Share of delayed sessions",
        zorder=1,
        drawstyle="steps-pre"
    )

    ax2.fill_between(
        delay_plot["hour"],
        0,
        delay_plot["delay_per_hour_per_session"],
        color="tab:blue",
        alpha=0.05,
        zorder=0,
        step="pre"
    )

    ax2.set_ylabel(
        "Hourly delay-window coverage",
        fontsize=11,
        color="black"
    )
    ax2.set_ylim(0, ymax_delay)
    ax2.tick_params(axis="y", labelcolor="black", colors="black")

    # Left y-axis: average load, drawn on top
    line1, = ax1.plot(
        load_plot["hour"],
        load_plot["average_load"],
        linewidth=2.7,
        color="tab:orange",
        label="Average load profile",
        zorder=5,
        drawstyle="steps-pre"
    )

    ax1.set_xlabel("Hour of day", fontsize=12, color="black")
    ax1.set_ylabel("Average loading [kW]", fontsize=12, color="black")
    ax1.set_ylim(0, ymax_load)
    ax1.set_xlim(0, 23)
    ax1.set_xticks(range(0, 24, 2))
    ax1.tick_params(axis="x", labelsize=11.5, colors="black")
    ax1.tick_params(axis="y", labelsize=11.5, colors="black")
    ax1.grid(True, alpha=0.25, zorder=0)

    # Black spines
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color("black")

    # Joint legend, load first
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]

    ax1.legend(
        lines,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        fontsize=12
    )

    plt.tight_layout()
    plt.show()

    return load_profile, delay_profile, df_joint

