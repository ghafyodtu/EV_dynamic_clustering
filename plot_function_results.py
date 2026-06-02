
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

    plt.tight_layout(rect=[0, 0, 1, 0.96])
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



