from my_imports import pd, np, plt, sns, StandardScaler
import copy
import pytz
from clustering_class import KmeansC, plot_evaluation_metrics
from functions_kde_js import (compute_joint_kdes, compute_js_divergence, categorize_count,
                              cluster_similarity_analysis,
                              compute_js_divergence_mD, create_months,
                              read_scale_monthly_kde_mD, read_scale_monthly_kde,
                              create_joint_similarity_matrix, find_mlcv_bandwidth,
                              run_md_sensitivity_analysis, run_cluster_threshold_sensitivity,
                              fit_cluster_kdes_with_mlcv_bandwidth)

def primary_interval_filter(df1, start_date="01-2023", end_date="02-2023"):
    """
    This function keeps the time interval needed for the clustering.
    It adds "year_month" and "weekend" columns to the dataframe.
    :param df1: data_frame
    :param start_date: start month (first timestamp of the month)
    :param end_date: end month (first timestamp of the month)
    :return: Dataframe within certain start and end time.
    """
    from datetime import datetime, timedelta
    import pytz
    import holidays
    df1 = df1.copy()
    df1 = df1.drop(columns=["mean_power", "max_power", "end_charge"])
    df1 = df1.rename(columns={"percentile_90": "power"})
    # Define the time zones
    denmark = pytz.timezone('Europe/Copenhagen')
    # print(df1['plugin'].dtype, "The initial time zone of the dataframe")
    # Convert the 'plugin' and 'plugout' and "start_charge" columns to UTC
    df1[['plugin', 'plugout', 'start_charge']] = df1[['plugin', 'plugout', 'start_charge']].apply(
        lambda col: pd.to_datetime(col, utc=True))
    # Convert the 'plugin' and 'plugout' and "start_charge" columns to Denmark time
    df1[['plugin', 'plugout', 'start_charge']] = df1[['plugin', 'plugout', 'start_charge']].apply(
        lambda col: col.dt.tz_convert(denmark))
    # Sort the dataframe by 'CBID' and 'plugin' columns
    df1 = df1.sort_values(by=['CBID', 'plugin'])
    # Parse the start and end dates in Denmark time
    start_date = datetime.strptime(start_date, "%m-%Y")
    end_date = datetime.strptime(end_date, "%m-%Y")
    # Set the timezone for start and end dates to Denmark time
    start_date = denmark.localize(start_date) + timedelta(seconds=1)
    end_date = denmark.localize(end_date) - timedelta(seconds=1)  # Adjust end date to the end of the month
    # Print the start and end dates in Denmark time
    # print(
    #     f"{start_date} is the start date of the dataset (Denmark time)\n{end_date}"
    #     f" is the end date of the dataset (Denmark time)")
    # Filter the dataframe based on the 'plugin' and 'plugout' columns
    df1 = df1[(df1['plugin'] >= start_date) & (df1['plugout'] <= end_date)]
    df1['year_month'] = df1['plugin'].dt.to_period('M').astype(str)
    df1['weekend'] = (df1['start_charge'].dt.weekday >= 5).astype("int32")
    # Use the holidays library to fetch Denmark's holidays for the given period
    dk_holidays = holidays.country_holidays('DK')
    df1['holiday'] = df1['start_charge'].dt.date.apply(lambda x: 1 if x in dk_holidays else 0)
    df1['holiday'] = (df1['holiday'] | df1['weekend']).astype("int32")
    # Reset the index of the filtered dataframe
    # print(df1['plugin'].dtype, "The time zone of the dataframe")
    df1 = df1.reset_index(drop=True)
    return df1


def delay_threshold(df1, delay_thr=0.0):
    """
    This functions defines a minimum threshold for delay.
    :param df1: charging dataframe.
    :param delay_thr: in hours.
    :return: dataframe with modified delay column.
    """
    df1.loc[df1['delay'] < delay_thr, 'delay'] = 0.0
    return df1


def add_hour(df1):
    """
    This function turns the time values to sine and cosine values.
    :param df1: dataframe.
    :return: dataframe with time values transformed to sine and cosine.
    """
    from my_imports import np
    df1 = df1.sort_values(by=['CBID', 'plugin'])
    # Extract hour and minute from the 'plugin' column
    for item in ["plugin", "start_charge"]:
        # Combine hour and minute
        name_c = item + "_hour"
        df1[name_c] = df1[item].dt.hour + df1[item].dt.minute / 60
        df1[name_c + "_sine"] = np.sin(2 * np.pi * df1[name_c] / 24)
        df1[name_c + "_cosine"] = np.cos(2 * np.pi * df1[name_c] / 24)
    df1 = df1.reset_index(drop=True)
    return df1


def filter_outlier_per_feature(df1, quantiles):
    """
    Removes outliers from the dataset based on specified quantiles for each column.
    :param df1: Input dataframe.
    :param quantiles: Dictionary specifying (low, high) quantiles for each column.
                      Example: {'energy': (0.01, 0.99), 'plugin_duration': (0.01, 0.95)}
    :return: Filtered dataframe with outliers removed.
    """
    a1 = len(df1)
    # print(a1, "size of dataset before outlier removal")
    for column, (low, high) in quantiles.items():
        low_thresh = df1[column].quantile(low)
        high_thresh = df1[column].quantile(high)
        df1 = df1[(df1[column] >= low_thresh) & (df1[column] <= high_thresh)]
    a2 = len(df1)
    # print(a1 - a2, "Number of outlier sessions removed.")
    # df1 = df1[df1.plugin_duration < 30]
    # df1 = df1[df1.plugin_duration > 0.5]
    df1.reset_index(drop=True)
    return df1


def keep_columns(df1, columns_to_keep):
    """
    Keep only the specified columns in the DataFrame.
    Parameters:
        df1 (DataFrame): The input DataFrame.
        columns_to_keep (list): A list of column names to keep.
    Returns:
        DataFrame: A new DataFrame containing only the specified columns.
    """
    # Filter the DataFrame to keep only the specified columns
    new_df = df1[columns_to_keep].copy()
    return new_df


def df_scaler(df1, cols_1,  scaler):
    df2 = keep_columns(df1, cols_1)
    cols_2 = [col + "_sc" for col in cols_1]
    # scale
    scaled_data = scaler.fit_transform(df2)
    df2 = pd.DataFrame(scaled_data, columns=cols_2)
    df1 = pd.concat([df1, df2], axis=1)
    return df1, cols_2


def feature_scaling(df1, cols1_):
    from my_imports import StandardScaler, MinMaxScaler
    df1 = df1.reset_index(drop=True)
    not_scale_ = df1.select_dtypes(include=['int32', 'int64', 'object']).columns.tolist()
    df2 = df1[not_scale_].copy()  # keeping out the columns we don't want to scale
    df1 = df1.drop(columns=not_scale_)
    scaler_ = StandardScaler()
    print("-> standard scaler")
    a_, b_ = df_scaler(df1, cols1_, scaler_)
    df1 = pd.concat([a_, df2], axis=1)
    df1 = df1.sort_values(by=['CBID', 'plugin'])
    return df1


def load_filter_data(start_m, end_m):
    cols = ['plugin_duration', "plugin_hour_sine", "plugin_hour_cosine", "energy", "delay"]
    cols_ = [col + "_sc" for col in cols]
    dfo = pd.read_parquet("Data/EV_data_full_v2.parquet")
    df = primary_interval_filter(dfo, start_date=start_m, end_date=end_m)
    df = delay_threshold(df, delay_thr=0.0)
    df = add_hour(df)
    quantiles = {
        'energy': (0.05, 0.95),
        'plugin_duration': (0.01, 0.95),
        'free_time': (0.00, 1.00),
        'delay': (0.00, 0.95)}
    df = filter_outlier_per_feature(df, quantiles)
    df = keep_columns(df, cols)
    return df


def clustering_process(x_train, scaler, current_month, metric):
    kmeans = KmeansC()
    from my_imports import StandardScaler
    km_scaler = StandardScaler()
    _ = km_scaler.fit(scaler.inverse_transform(x_train))
    x_train_km = km_scaler.transform(scaler.inverse_transform(x_train))
    x_train_km = pd.DataFrame(x_train_km, columns=x_train.columns)
    kmeans.clustering_evaluation(x_train_km.sample(n=int(x_train_km.shape[0]*0.5), random_state=42), 5, 10, metrics=["silhouette"])
    plot_evaluation_metrics(kmeans, month=current_month)
    kmeans.n_clusters, metric_value = max(kmeans.evaluation_metrics[metric], key=lambda x: x[1])
    print(f"{kmeans.n_clusters}, Number of Optimal Clusters for Month {current_month}")
    # fit the clustering model
    _ = kmeans.fit(x_train_km)
    km_labels = kmeans.predict(x_train_km)
    x_train["km_labels"] = km_labels
    x_train.sort_values(by="km_labels", ascending=True, inplace=True)
    km_labels = x_train["km_labels"]
    x_train = x_train.drop(columns=["km_labels"])
    return x_train, km_labels, kmeans


def initialize_state(global_start_date):
    """
    Create all dictionaries/lists used by the streaming clustering process.
    """

    return {
        "cluster_objects_pool": {},
        "monthly_kde": {},
        "monthly_kde_mD": {},
        "cluster_kdes": {},
        "reference_clusters": {},
        "ref_month": global_start_date,
        "stream_order": [],
        "kmeans_objects_dict": {},
        "similarity_matrix": [],
        "js_per_feature_mat": [],
        "js_mat_time_order": [],
        "js_md_per_month": {},
        "all_monthly_kde": {},
        "all_clusters_kde": {},
    }


def rename_cluster_keys(cluster_kdes_for_month, month):
    """
    Rename cluster labels so that they contain the month.

    Example:
        0 -> "01-2023_0"
        1 -> "01-2023_1"
    """

    return {
        f"{month}_{cluster_id}": kde_value
        for cluster_id, kde_value in cluster_kdes_for_month.items()
    }


def load_current_month_kdes(state, scaler, current_month, next_month, bandwidth):
    """
    Load scaled monthly data and compute:
    1. feature-wise KDEs
    2. multidimensional KDE
    """

    x_train, kde_1d = read_scale_monthly_kde(
        scaler,
        current_month,
        next_month,
        bandwidth,
    )

    _, kde_mD = read_scale_monthly_kde_mD(
        scaler,
        current_month,
        next_month,
        bandwidth,
    )

    state["monthly_kde"][current_month] = kde_1d
    state["monthly_kde_mD"][current_month] = kde_mD
    state["all_monthly_kde"][current_month] = kde_1d.copy()

    return x_train


def compute_featurewise_js(state, current_month, ref_month, sample_size):
    """
    Compute JS divergence feature-by-feature between current month
    and the current reference month.
    """

    js_values = []

    n_features = len(state["monthly_kde"][current_month])

    for feature_idx in range(n_features):
        js = compute_js_divergence(
            state["monthly_kde"][current_month][feature_idx],
            state["monthly_kde"][ref_month][feature_idx],
            sample_size,
        )

        js_values.append(np.round(js, 5))

    return js_values


def store_featurewise_js(state, current_month, js_values):
    """
    Store feature-wise JS divergence results in time order.
    """

    state["js_per_feature_mat"].append(js_values)
    state["js_mat_time_order"].append(current_month)


def update_all_clusters_kde(state, month):
    """
    Save cluster KDE objects from a month into the global cluster history.
    """

    state["all_clusters_kde"].update({
        cluster_name: kde_info[0]
        for cluster_name, kde_info in state["cluster_kdes"][month].items()
    })


def update_cluster_objects_pool(state, month):
    """
    Add cluster KDE objects from a month to the global comparison pool.
    """

    state["cluster_objects_pool"].update({
        cluster_name: kde_info[0]
        for cluster_name, kde_info in state["cluster_kdes"][month].items()
    })


def create_new_month_clustering(state, x_train, scaler, current_month, metric, bw=None):
    """
    Create a new clustering model for the current month.

    This is used when the current month is not similar enough to:
    1. the reference month
    2. any previous stored month
    """

    x_train, km_labels, kmeans = clustering_process(
        x_train,
        scaler,
        current_month,
        metric,
    )

    if not bw:
        cluster_bandwidth = fit_cluster_kdes_with_mlcv_bandwidth(
        x_train=x_train,
        km_labels=km_labels,
        )
    else:
        cluster_bandwidth = bw

    cluster_kdes_for_month = compute_joint_kdes(
        x_train,
        bandwidth=cluster_bandwidth,
        labels=km_labels,
    )

    cluster_kdes_for_month = rename_cluster_keys(
        cluster_kdes_for_month,
        current_month,
    )

    state["cluster_kdes"][current_month] = cluster_kdes_for_month
    state["kmeans_objects_dict"][current_month] = kmeans
    state["ref_month"] = current_month
    state["stream_order"].append(current_month)
    state["cluster_bw"] = cluster_bandwidth
    update_all_clusters_kde(state, current_month)

    return x_train


def reuse_existing_clusters(state, x_train, scaler, current_month, ref_month):
    """
    Reuse cluster structure from a reference month.

    This happens when the current month is similar enough to an already
    known month.
    """

    state["cluster_kdes"][current_month] = copy.deepcopy(
        state["cluster_kdes"][ref_month]
    )

    x_train, state["cluster_kdes"] = categorize_count(
        x_train,
        state["kmeans_objects_dict"],
        state["cluster_kdes"],
        ref_month,
        current_month,
        scaler,
    )

    # Remove current month from active monthly KDE dictionaries because
    # this month is represented by the selected reference month.
    state["monthly_kde"].pop(current_month, None)
    state["monthly_kde_mD"].pop(current_month, None)

    state["stream_order"].append(ref_month)

    return x_train


def find_best_similar_previous_month(
    state,
    current_month,
    ref_month,
    js_lim_month,
    sample_size,
):
    """
    Search previous months and find the most similar one to the current month.

    Returns:
        best_month, best_js

    If no previous month is similar enough:
        None, None
    """

    candidates = {}

    for previous_month in state["monthly_kde_mD"]:
        if previous_month in {current_month, ref_month}:
            continue

        js = compute_js_divergence_mD(
            state["monthly_kde_mD"][current_month],
            state["monthly_kde_mD"][previous_month],
            sample_size,
        )

        if js < js_lim_month:
            candidates[previous_month] = js

    if not candidates:
        return None, None

    best_month = min(candidates, key=candidates.get)
    best_js = candidates[best_month]

    return best_month, best_js


def compare_new_clusters_with_pool(
    state,
    current_month,
    js_lim_cluster,
    sample_size,
):
    """
    Compare newly created clusters with the existing cluster pool.
    Then update the pool and cluster KDE dictionary.
    """

    current_cluster_kdes = {
        cluster_name: kde_info[0]
        for cluster_name, kde_info in state["cluster_kdes"][current_month].items()
    }

    current_cluster_kdes = copy.deepcopy(current_cluster_kdes)

    state["similarity_matrix"] = create_joint_similarity_matrix(
        state["cluster_objects_pool"],
        current_cluster_kdes,
        sample_size,
    )

    state["cluster_objects_pool"], state["cluster_kdes"] = cluster_similarity_analysis(
        state["similarity_matrix"],
        state["cluster_objects_pool"],
        state["cluster_kdes"],
        js_lim_cluster,
        current_month,
    )


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def run_dynamic_clustering(
    global_start_date,
    global_end_date,
    metric="silhouette",
    js_lim_month=0.18,
    js_lim_cluster=0.18,
    js_sample_size=1000,
):
    """
    Main dynamic clustering pipeline.

    Logic:
    1. Start with the first month.
    2. Fit scaler and monthly KDE bandwidth.
    3. Cluster the first month.
    4. For every later month:
        - compare it with the current reference month
        - if similar, reuse reference clusters
        - otherwise, search previous months
        - if no similar previous month exists, create new clusters
    """

    months = create_months(global_start_date, global_end_date)

    state = initialize_state(global_start_date)

    scaler = StandardScaler()
    state["scaler"] = scaler
    # Fit scaler and bandwidth using the first month pair.
    first_x_train = load_filter_data(
        start_m=months[0],
        end_m=months[1],
    )

    monthly_bandwidth, _, _ = find_mlcv_bandwidth(first_x_train)

    scaler.fit(first_x_train)

    for idx in range(len(months) - 1):
        current_month = months[idx]
        next_month = months[idx + 1]

        print("current_month_pair:", current_month, next_month)

        x_train = load_current_month_kdes(
            state=state,
            scaler=scaler,
            current_month=current_month,
            next_month=next_month,
            bandwidth=monthly_bandwidth,
        )

        # -------------------------------------------------------------
        # First month: create the initial reference clustering
        # -------------------------------------------------------------
        if current_month == global_start_date:
            print("Analyzing first month in the data.")

            x_train = create_new_month_clustering(
                state=state,
                x_train=x_train,
                scaler=scaler,
                current_month=current_month,
                metric=metric,
            )

            # First month initializes the global cluster pool.
            update_cluster_objects_pool(state, current_month)

            continue

        # -------------------------------------------------------------
        # Compare current month with current reference month
        # -------------------------------------------------------------
        ref_month = state["ref_month"]

        js_mD = compute_js_divergence_mD(
            state["monthly_kde_mD"][current_month],
            state["monthly_kde_mD"][ref_month],
            js_sample_size,
        )

        state["js_md_per_month"][current_month] = js_mD

        js_featurewise = compute_featurewise_js(
            state=state,
            current_month=current_month,
            ref_month=ref_month,
            sample_size=js_sample_size,
        )

        store_featurewise_js(
            state=state,
            current_month=current_month,
            js_values=js_featurewise,
        )

        # -------------------------------------------------------------
        # Case 1:
        # Current month is similar to the reference month
        # -------------------------------------------------------------
        if js_mD < js_lim_month:
            print(
                f"{current_month} is similar to reference month {ref_month}. "
                f"JS mD = {js_mD:.5f}"
            )

            x_train = reuse_existing_clusters(
                state=state,
                x_train=x_train,
                scaler=scaler,
                current_month=current_month,
                ref_month=ref_month,
            )

            continue

        # -------------------------------------------------------------
        # Case 2:
        # Current month is not similar to the reference month.
        # Search for another similar previous month.
        # -------------------------------------------------------------
        best_month, best_js = find_best_similar_previous_month(
            state=state,
            current_month=current_month,
            ref_month=ref_month,
            js_lim_month=js_lim_month,
            sample_size=js_sample_size,
        )

        if best_month is not None:
            print(
                f"The most similar previous data batch is {best_month}. "
                f"JS mD = {best_js:.5f}"
            )

            state["ref_month"] = best_month

            x_train = reuse_existing_clusters(
                state=state,
                x_train=x_train,
                scaler=scaler,
                current_month=current_month,
                ref_month=best_month,
            )

            continue

        # -------------------------------------------------------------
        # Case 3:
        # Current month is different from all known previous months.
        # Create a new clustering model.
        # -------------------------------------------------------------
        print(
            f"{current_month} is a new pattern. "
            f"JS mD = {js_mD:.5f}, threshold = {js_lim_month}"
        )

        x_train = create_new_month_clustering(
            state=state,
            x_train=x_train,
            scaler=scaler,
            current_month=current_month,
            metric=metric,
            bw=state["cluster_bw"],
        )

        compare_new_clusters_with_pool(
            state=state,
            current_month=current_month,
            js_lim_cluster=js_lim_cluster,
            sample_size=js_sample_size,
        )

    return state

