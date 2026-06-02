from my_imports import pd
from clustering_class import KmeansC, plot_evaluation_metrics


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


def load_scale_filter_data(start_m, end_m):
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
