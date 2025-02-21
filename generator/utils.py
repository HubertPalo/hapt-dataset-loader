from typing import List, Union, Dict
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
from pathlib import Path
import numpy as np
import os, shutil
from natsort import natsorted
from zipfile import ZipFile
from scipy import interpolate


# from librep.utils.visualization.multimodal_har import plot_windows_sample


def compare_metadata(dataset_normal, dataset_resampled, columns: Union[str, List[str]]):
    """This function compares the metadata of two datasets. The metadata is the information that is present in the specific columns of the dataset.
    You can compare the metadata of the index, user, serial, activity code and csv columns.

    Parameters
    ----------
    dataset_normal : pd.DataFrame
        The original dataset
    dataset_resampled : pd.DataFrame
        The resampled dataset
    columns : Union[str, List[str]]
        The columns that you want to compare. You can compare the metadata of the index, user, serial, activity code and csv columns.

    Returns
    -------
    bool
        True if the metadata is equal, False otherwise
    """
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if column == "index":
            # They all should result in the same fraction if they are linearly dependent
            if len((dataset_normal[column] / dataset_resampled[column]).unique()) != 1:
                print(f"Columns '{column}' are not equal")
                return False

        elif not np.all(dataset_normal[column] == dataset_resampled[column]):
            print(f"Columns '{column}' are not equal")
            return False
    return True


# def generate_plots(
#     *datasets,
#     sample_no: int = 0,
#     the_slice: slice = slice(None, None, None),
#     windows: List[str] = None,
#     height: int = 600,
#     width: int = 800,
#     names: List[str] = None,
#     vertical_spacing: float = 0.1,
#     title: str = "",
#     x_title: str = "x",
#     y_title: str = "y",
# ):
#     """Generate a plotly figure with the data from the datasets. The figure will have a subplot for each dataset.
#     The figure will have a subplot for each dataset. Each subplot will have a trace for each window.
#     The traces will be colored according to the window name.

#     Parameters
#     ----------
#     *datasets : pd.DataFrame
#         The datasets that you want to plot
#     sample_no : int, optional
#         The sample that you want to plot, by default 0
#     the_slice : slice, optional
#         The slice that you want to plot, by default slice(None, None, None)
#     windows : List[str], optional
#         The windows that you want to plot, by default None
#     height : int, optional
#         The height of the figure, by default 600
#     width : int, optional
#         The width of the figure, by default 800
#     names : List[str], optional
#         The names of the datasets, by default None
#     vertical_spacing : float, optional
#         The vertical spacing between the subplots, by default 0.1
#     title : str, optional
#         The title of the figure, by default ""
#     x_title : str, optional
#         The title of the x axis, by default "x"
#     y_title : str, optional
#         The title of the y axis, by default "y"

#     Returns
#     -------
#     fig : plotly.graph_objects.Figure
#         The plotly figure of the datasets
#     """
#     color10_16 = ["blue", "cyan", "magenta", "#636efa", "#00cc96", "#EF553B", "brown"]
#     if windows is None:
#         windows = datasets[0].window_names

#     if names is None:
#         names = [f"Dataset {i}" for i in range(len(datasets))]

#     fig = make_subplots(
#         rows=len(datasets),
#         cols=1,
#         start_cell="top-left",
#         subplot_titles=names,
#         vertical_spacing=vertical_spacing,
#         x_title=x_title,
#         y_title=y_title,
#     )

#     for i, dataset in enumerate(datasets):
#         traces, _ = plot_windows_sample(
#             dataset,
#             windows=windows,
#             sample_idx=sample_no,
#             the_slice=the_slice,
#             title=f"Dataset {i}",
#             return_traces_layout=True,
#         )
#         for j, trace in enumerate(traces):
#             trace.line.color = color10_16[j]
#             trace.legendgroup = str(j)
#             trace.showlegend = False if i > 0 else True

#         fig.add_traces(traces, rows=[i + 1] * len(traces), cols=[1] * len(traces))

#     fig.update_layout(title=title)
#     fig.update_layout(height=height, width=width)

#     return fig


# Create functions to read datasets
def read_kuhar(kuhar_dir_path: str) -> pd.DataFrame:
    """Read the Kuhar dataset and return a DataFrame with the data (coming from all CSV files)
    The returned dataframe has the following columns:
    - accel-x: Acceleration on the x axis
    - accel-y: Acceleration on the y axis
    - accel-z: Acceleration on the z axis
    - gyro-x: Angular velocity on the x axis
    - gyro-y: Angular velocity on the y axis
    - gyro-z: Angular velocity on the z axis
    - accel-start-time: Start time of the acceleration window
    - gyro-start-time: Start time of the gyroscope window
    - activity code: Activity code
    - index: Index of the sample coming from the csv
    - user: User code
    - serial: Serial number of the activity
    - csv: Name of the CSV file

    Parameters
    ----------
    kuhar_dir_path : str
        Path to the Kuhar dataset

    Returns
    -------
    pd.DataFrame
        DataFrame with the data from the Kuhar dataset
    """
    kuhar_dir_path = Path(kuhar_dir_path)

    # Create a dictionary with the data types of each column
    feature_dtypes = {
        "accel-start-time": np.float32,
        "accel-x": np.float32,
        "accel-y": np.float32,
        "accel-z": np.float32,
        "gyro-start-time": np.float32,
        "gyro-x": np.float32,
        "gyro-y": np.float32,
        "gyro-z": np.float32,
    }

    dfs = []
    cont = 0

    for i, f in enumerate(sorted(kuhar_dir_path.rglob("*.csv"))):
        # Get the name of the activity (folder name, e.g. 5.Lay)
        # Get the name of the CSV file (ex.: 1052_F_1.csv)
        # Split the activity number and the name (ex.: [5, 'Lay'])
        activity_no, activity_name = f.parents[0].name.split(".")
        activity_no = int(activity_no)

        # Split the user code, the activity type and the serial number (ex.: [1055, 'G', 1])
        csv_splitted = f.stem.split("_")
        user = int(csv_splitted[0])
        serial = "_".join(csv_splitted[2:])

        # Read the CSV file
        df = pd.read_csv(f, names=list(feature_dtypes.keys()), dtype=feature_dtypes)

        # Remove dataframes that contain NaN
        if df.isnull().values.any():
            cont += 1
            print(
                f"The dataframe contains {df.shape[0]} samples and {df.isnull().values.sum()} NaN values"
            )
            continue

        # Only reordering the columns (no column is removed)
        df = df[
            [
                "accel-x",
                "accel-y",
                "accel-z",
                "gyro-x",
                "gyro-y",
                "gyro-z",
                "accel-start-time",
                "gyro-start-time",
            ]
        ]

        # ----- Add auxiliary columns and metadata ------
        # Since it is a simple instant of time (without duration), the start and end time are the same
        df["accel-end-time"] = df["accel-start-time"]
        df["gyro-end-time"] = df["gyro-start-time"]
        # Add the activity code column
        df["activity code"] = activity_no
        # Add the index column (index of the sample in the CSV file)
        df["index"] = range(len(df))
        # Add the user column
        df["user"] = user
        # Add the serial column (the serial number of the activity)
        df["serial"] = serial
        # Add the csv column (the name of the CSV file)
        df["csv"] = "/".join(f.parts[-2:])
        # ----------------------------------------------------
        dfs.append(df)
    print("We removed {} dataframes".format(cont))
    return pd.concat(dfs)


def read_motionsense(motionsense_path: str) -> pd.DataFrame:
    """Read the MotionSense dataset and return a DataFrame with the data (coming from all CSV files)
    The returned dataframe has the following columns:
    - attitude.roll: Rotation around the x axis
    - attitude.pitch: Rotation around the y axis
    - attitude.yaw: Rotation around the z axis
    - gravity.x: Gravity around the x axis
    - gravity.y: Gravity around the y axis
    - gravity.z: Gravity around the z axis
    - rotationRate.x: Angular velocity around the x axis
    - rotationRate.y: Angular velocity around the y axis
    - rotationRate.z: Angular velocity around the z axis
    - userAcceleration.x: Acceleration on the x axis
    - userAcceleration.y: Acceleration on the y axis
    - userAcceleration.z: Acceleration on the z axis
    - activity code: Activity code
    - index: Index of the sample coming from the csv
    - user: User code
    - serial: Serial number of the activity
    - csv: Name of the CSV file

    Parameters
    ----------
    motionsense_path : str
        Path to the MotionSense dataset

    Returns
    -------
    pd.DataFrame
        DataFrame with the data from the MotionSense dataset
    """
    motionsense_path = Path(motionsense_path)
    activity_names = {0: "dws", 1: "ups", 2: "sit", 3: "std", 4: "wlk", 5: "jog"}
    activity_codes = {v: k for k, v in activity_names.items()}

    feature_dtypes = {
        "attitude.roll": np.float32,
        "attitude.pitch": np.float32,
        "attitude.yaw": np.float32,
        "gravity.x": np.float32,
        "gravity.y": np.float32,
        "gravity.z": np.float32,
        "rotationRate.x": np.float32,
        "rotationRate.y": np.float32,
        "rotationRate.z": np.float32,
        "userAcceleration.x": np.float32,
        "userAcceleration.y": np.float32,
        "userAcceleration.z": np.float32,
    }

    dfs = []
    cont = 0
    for i, f in enumerate(sorted(motionsense_path.rglob("*.csv"))):
        # Get the name of the activity (folder name, e.g. 5.Lay)
        activity_name = f.parents[0].name
        # Partition the name of the activity into the activity code and the serial number
        activity_name, serial = activity_name.split("_")
        activity_code = activity_codes[activity_name]

        user = int(f.stem.split("_")[1])
        df = pd.read_csv(
            f, names=list(feature_dtypes.keys()), dtype=feature_dtypes, skiprows=1
        )
        # Remove dataframes that contain NaN
        if df.isnull().values.any():
            cont += 1
            print(
                f"The dataframe contains {df.shape[0]} samples and {df.isnull().values.sum()} NaN values"
            )
            continue

        # ----- Add auxiliary columns and metadata ------
        df["activity code"] = activity_code
        df["index"] = range(len(df))
        df["user"] = user
        df["serial"] = serial
        df["csv"] = "/".join(f.parts[-2:])
        # ----------------------------------------------------
        dfs.append(df)
    print("We removed {} dataframes".format(cont))
    return pd.concat(dfs)


def read_uci(uci_path: str) -> pd.DataFrame:
    """Read the UCI-HAR dataset and return a DataFrame with the data (coming from all txt files)
    The returned dataframe has the following columns:
    - accel-x: Acceleration on the x axis
    - accel-y: Acceleration on the y axis
    - accel-z: Acceleration on the z axis
    - gyro-x: Angular velocity on the x axis
    - gyro-y: Angular velocity on the y axis
    - gyro-z: Angular velocity on the z axis
    - txt: Name of the TXT file
    - user: User code
    - serial: Serial number of the activity
    - activity code: Activity code
    - index: Index of the sample coming from the csv

    Parameters
    ----------
    uci_path : str
        Path to the UCI-HAR dataset

    Returns
    -------
    pd.DataFrame
        DataFrame with the data from the UCI-HAR dataset
    """
    activity_names = {
        1: "WALKING",
        2: "WALKING_UPSTAIRS",
        3: "WALKING_DOWNSTAIRS",
        4: "SITTING",
        5: "STANDING",
        6: "LAYING",
        7: "STAND_TO_SIT",
        8: "SIT_TO_STAND",
        9: "SIT_TO_LIE",
        10: "LIE_TO_SIT",
        11: "STAND_TO_LIE",
        12: "LIE_TO_STAND",
    }

    feature_columns = [
        "accel-x",
        "accel-y",
        "accel-z",
        "gyro-x",
        "gyro-y",
        "gyro-z",
    ]

    uci_path = Path(uci_path)

    df_labels = pd.read_csv(uci_path / "labels.txt", header=None, sep=" ")
    df_labels.columns = ["serial", "user", "activity code", "start", "end"]

    dfs = []
    cont = 0
    data_path = list(uci_path.glob("*.txt"))
    new_data_path = [elem.name.split("_") + [elem] for elem in sorted(data_path)]
    df = pd.DataFrame(new_data_path, columns=["sensor", "serial", "user", "file"])
    for key, df2 in df.groupby(["serial", "user"]):
        acc, gyr = None, None
        for row_index, row in df2.iterrows():
            data = pd.read_csv(row["file"], header=None, sep=" ")
            if row["sensor"] == "acc":
                acc = data
            else:
                gyr = data
        new_df = pd.concat([acc, gyr], axis=1)
        new_df.columns = feature_columns

        user = int(key[1].split(".")[0][4:])
        serial = int(key[0][3:])

        new_df["txt"] = row["file"]

        new_df["user"] = user
        new_df["serial"] = serial

        for row_index, row in df_labels.loc[
            (df_labels["serial"] == serial) & (df_labels["user"] == user)
        ].iterrows():
            start = row["start"]
            end = row["end"] + 1
            activity = row["activity code"]
            resumed_df = new_df.loc[start:end].copy()
            resumed_df["index"] = [i for i in range(start, end + 1)]
            resumed_df["activity code"] = activity

            # Remove dataframes that contain NaN
            if resumed_df.isnull().values.any():
                cont += 1
                print(
                    f"The dataframe contains {resumed_df.shape[0]} samples and {resumed_df.isnull().values.sum()} NaN values"
                )
                continue

            dfs.append(resumed_df)

    df = pd.concat(dfs)
    df['user'] = df['user'].astype(int)
    df.reset_index(inplace=True, drop=True)
    print("We removed {} dataframes".format(cont))
    return df


def read_wisdm(wisdm_path: str, interpol=True) -> pd.DataFrame:
    """Read the WISDM dataset and return a DataFrame with the data (coming from all txt files)
    The returned dataframe has the following columns:
    - accel-x: Acceleration on the x axis
    - accel-y: Acceleration on the y axis
    - accel-z: Acceleration on the z axis
    - gyro-x: Angular velocity on the x axis
    - gyro-y: Angular velocity on the y axis
    - gyro-z: Angular velocity on the z axis
    - user: User code
    - activity code: Activity code
    - window: Window number
    - timestamp-accel: Timestamp of the acceleration window
    - timestamp-gyro: Timestamp of the gyroscope window

    Parameters
    ----------
    wisdm_path : str
        Path to the WISDM dataset
        interpol : bool, optional
            If True, the data will be interpolated, by default True

    Returns
    -------
    pd.DataFrame
        DataFrame with the data from the WISDM dataset
    """
    feature_columns_acc = [
        "user",
        "activity code",
        "timestamp-accel",
        "accel-x",
        "accel-y",
        "accel-z",
    ]
    feature_columns_gyr = [
        "user",
        "activity code",
        "timestamp-gyro",
        "gyro-x",
        "gyro-y",
        "gyro-z",
    ]

    # List of capital letters from A to S without N
    labels: List[str] = [chr(i) for i in range(65, 84) if chr(i) != "N"]

    wisdm_path = Path(wisdm_path)

    dfs = []
    window = 1
    for user in range(1600, 1651):
        window = 1
        # Read the accelerometer data
        df_acc = pd.read_csv(
            wisdm_path / f"accel/data_{user}_accel_phone.txt",
            sep=",|;",
            header=None,
            engine="python",
        )
        df_acc = df_acc[df_acc.columns[0:-1]]
        df_acc.columns = feature_columns_acc
        df_acc["timestamp-accel"] = df_acc["timestamp-accel"].astype(np.int64)

        # Read the gyroscope data
        df_gyr = pd.read_csv(
            wisdm_path / f"gyro/data_{user}_gyro_phone.txt",
            sep=",|;",
            header=None,
            engine="python",
        )
        df_gyr = df_gyr[df_gyr.columns[0:-1]]
        df_gyr.columns = feature_columns_gyr
        df_gyr["timestamp-gyro"] = df_gyr["timestamp-gyro"].astype(np.int64)

        for activity in labels:
            # Get the data from the current activity
            acc = df_acc[df_acc["activity code"] == activity].copy()
            gyr = df_gyr[df_gyr["activity code"] == activity].copy()

            time_acc = np.array(acc["timestamp-accel"])
            time_gyr = np.array(gyr["timestamp-gyro"])

            # Flag to check if the data will be interpolated
            if interpol:
                # Set the initial time to 0
                if len(time_acc) > 0 and len(time_gyr) > 0:
                    time_acc = (time_acc - time_acc[0]) / 1000000000
                    time_gyr = (time_gyr - time_gyr[0]) / 1000000000

                    # Removing the intervals without samples (empty periods)
                    if np.any(np.diff(time_acc) < 0):
                        pos = np.nonzero(np.diff(time_acc) < 0)[0].astype(int)
                        for k in pos:
                            time_acc[k + 1 :] = time_acc[k + 1 :] + time_acc[k] + 1 / 20
                    if np.any(np.diff(time_gyr) < 0):
                        pos = np.nonzero(np.diff(time_gyr) < 0)[0].astype(int)
                        for k in pos:
                            time_gyr[k + 1 :] = time_gyr[k + 1 :] + time_gyr[k] + 1 / 20

                    # Interpolating the data to fix the sampling rate to 20 Hz
                    sigs_acc = []
                    sigs_gyr = []
                    for sig_acc, sig_gyr in zip(
                        acc[feature_columns_acc[2:]], gyr[feature_columns_gyr[2:]]
                    ):
                        fA = np.array(acc[sig_acc])
                        fG = np.array(gyr[sig_gyr])

                        intp1 = interpolate.interp1d(time_acc, fA, kind="cubic")
                        intp2 = interpolate.interp1d(time_gyr, fG, kind="cubic")
                        nt1 = np.arange(0, time_acc[-1], 1 / 20)
                        nt2 = np.arange(0, time_gyr[-1], 1 / 20)
                        sigs_acc.append(intp1(nt1))
                        sigs_gyr.append(intp2(nt2))

                    # Getting the minimum length of the signals (accelerometer and gyroscope)
                    tam = min(len(nt1), len(nt2))

                    new_acc = pd.DataFrame()
                    new_gyr = pd.DataFrame()

                    # Truncating the signals
                    for x, y in zip(sigs_acc, sigs_gyr):
                        x = x[:tam]
                        y = y[:tam]

                    # Truncating the timestamps
                    new_acc["timestamp-accel"] = nt1[:tam]
                    new_gyr["timestamp-gyro"] = nt2[:tam]

                    # Adding the other columns
                    for sig_acc, sig_gyr, column_acc, column_gyr in zip(
                        sigs_acc,
                        sigs_gyr,
                        feature_columns_acc[2:],
                        feature_columns_gyr[2:],
                    ):
                        new_acc[column_acc] = sig_acc[:tam]
                        new_gyr[column_gyr] = sig_gyr[:tam]
            else:
                tam = min(len(time_acc), len(time_gyr))
                new_acc = acc[feature_columns_acc[2:]].iloc[:tam]
                new_gyr = gyr[feature_columns_gyr[2:]].iloc[:tam]

            # Concatenating the accelerometer and gyroscope dataframes
            df = pd.concat([new_acc, new_gyr], axis=1)
            # Adding the other columns
            df["activity code"] = activity
            df["user"] = user
            df["window"] = window

            # Drop samples with NaN
            if df.isnull().values.any():
                nan = df.isnull().values.sum()
                print(
                    f"The dataframe contains {df.shape[0]} samples and {nan} NaN values"
                )
                print(
                    "We only removed the samples with NaN values, not the entire dataframe"
                )
            df = df.dropna()

            dfs.append(df)
    # Concatenating the dataframes
    df = pd.concat(dfs)
    df.reset_index(inplace=True, drop=True)

    # Converting the data types
    for column in feature_columns_acc[2:] + feature_columns_gyr[2:]:
        df[column] = df[column].astype(np.float32)
    df["user"] = df["user"].astype(np.int32)

    return df.dropna().reset_index(drop=True)


def read_realworld(workspace: str, users: List[str]) -> pd.DataFrame:
    """Read the RealWorld dataset and return a DataFrame with the data (coming from all files)
    The returned dataframe has the following columns:
    - accel-x: Acceleration on the x axis
    - accel-y: Acceleration on the y axis
    - accel-z: Acceleration on the z axis
    - gyro-x: Angular velocity on the x axis
    - gyro-y: Angular velocity on the y axis
    - gyro-z: Angular velocity on the z axis
    - user: User code
    - position: Position of the sensor
    - activity code: Activity code
    - index: Index of the sample coming from the csv

    Parameters
    ----------
    workspace : str
        Path to the RealWorld dataset
    users : List[str]
        List of users that you want to read

    Returns
    -------
    pd.DataFrame
        DataFrame with the data from the RealWorld dataset
    """

    # List of activities
    activities: List[str] = [
        "climbingdown",
        "climbingup",
        "jumping",
        "lying",
        "running",
        "sitting",
        "standing",
        "walking",
    ]

    # List to filter the positions
    positions: List[str] = [
        "chest",
        "forearm",
        "head",
        "shin",
        "thigh",
        "upperarm",
        "waist",
    ]

    # List of features
    feature_acc: List[str] = [
        "index",
        "accel-start-time",
        "accel-x",
        "accel-y",
        "accel-z",
    ]
    feature_gyr: List[str] = ["index", "gyro-start-time", "gyro-x", "gyro-y", "gyro-z"]

    # List to store the dataframes
    dfs: List[pd.DataFrame] = []

    workspace = Path(workspace)

    cont = 0
    for position in positions:
        for user in users:
            # List of files from the accelerometer and gyroscope sensors of the current user
            filesacc = sorted(
                os.listdir(workspace / "realworld2016_dataset_organized" / user / "acc")
            )
            filesgyr = sorted(
                os.listdir(workspace / "realworld2016_dataset_organized" / user / "gyr")
            )

            pos = []
            # Get the indexes of the files that contain the current position
            for i in range(len(filesacc)):
                if filesacc[i].find(position) > -1:
                    pos.append(i)

            for i in pos:
                # Read the accelerometer and gyroscope data
                acc = pd.read_csv(
                    workspace
                    / "realworld2016_dataset_organized"
                    / user
                    / "acc"
                    / filesacc[i]
                )
                acc.columns = feature_acc
                gyr = pd.read_csv(
                    workspace
                    / "realworld2016_dataset_organized"
                    / user
                    / "gyr"
                    / filesgyr[i]
                )
                gyr.columns = feature_gyr

                for activity in activities:
                    if filesacc[i].find(activity) > -1:
                        break

                # Work around to remove the samples that are less problematic (the samples that have a difference of tam_diff samples or more)

                tam_diff = 60
                if not abs(acc.shape[0] - gyr.shape[0]) < tam_diff:
                    # Remove all rows from the dataframes
                    acc.drop(acc.index, inplace=True)
                    gyr.drop(gyr.index, inplace=True)

                tam = min(acc.shape[0], gyr.shape[0])

                new_acc = acc[feature_acc].iloc[:tam]
                new_gyr = gyr[feature_gyr[1:]].iloc[:tam]

                # Concatenating the accelerometer and gyroscope dataframes
                df = pd.concat([new_acc, new_gyr], axis=1)
                # Adding the other columns
                df["user"] = user
                df["position"] = position
                df["activity code"] = activity

                # Drop samples with NaN
                if df.isnull().values.any():
                    cont += 1
                    print(
                        f"The dataframe contains {df.shape[0]} samples and {df.isnull().values.sum()} NaN values"
                    )
                    continue

                dfs.append(df)
    # Concatenating the dataframes
    df = pd.concat(dfs, ignore_index=True)
    df.reset_index(inplace=True, drop=True)
    print("We removed {} dataframes".format(cont))
    return df


def read_recodGaitV1(recod_gait_path: str, interpol: bool = False) -> pd.DataFrame:
    """Read the RecodGaitV1 dataset and return a DataFrame with the data (coming from all txt files)
    The returned dataframe has the following columns:
    - accel-x: Acceleration on the x axis
    - accel-y: Acceleration on the y axis
    - accel-z: Acceleration on the z axis
    - user: User code
    - session: Session number
    - index: Index of the sample coming from the txt
    This dataset has rotation vectors, but they are not used in this function

    Parameters
    ----------
    recod_gait_path : str
        Path to the RecodGaitV1 dataset
    interpol : bool, optional
        If True, the data will be interpolated, by default False

    Returns
    -------
    pd.DataFrame
        DataFrame with the data from the RecodGaitV1 dataset
    """

    recod_gait_path = Path(recod_gait_path)

    features_acc = {
        "accel-start-time": np.float32,
        "accel-x": np.float32,
        "accel-y": np.float32,
        "accel-z": np.float32,
    }

    dfs = []

    cont = 1
    for file in natsorted(recod_gait_path.glob("*.txt")):
        user = file.name.split("__")[0]
        session = file.name.split("__")[1]

        if "accelerometer" in file.name and "nw" not in file.name:
            # Reading the accelerometer data
            df = pd.read_csv(
                file,
                sep=",",
                header=None,
                names=features_acc.keys(),
                dtype=features_acc,
            )

            if interpol:
                time_acc = np.array(df["accel-start-time"])
                time_acc = (time_acc - time_acc[0]) / 1e9

                df_acc, new_time = interpolation(
                    df, ["accel-x", "accel-y", "accel-z"], time_acc
                )
                df_acc["accel-start-time"] = new_time

                df = df_acc.copy()

            else:
                df["accel-start-time"] = (
                    df["accel-start-time"] - df["accel-start-time"][0]
                ) / 1e9

            df["user"] = int(user)
            df["session"] = int(session[-1])
            df["index"] = cont
            cont += 1

            # Drop samples with NaN
            if df.isnull().values.any():
                print(
                    f"There are {df.isnull().values.sum()} NaN values in the file {file.name}"
                )

            df = df.dropna()

            dfs.append(df)

    df = pd.concat(dfs, axis=0)
    df.reset_index(drop=True, inplace=True)

    return df

def read_recodGaitV2(recod_gait_path: str) -> pd.DataFrame:
    """Read the RecodGaitV2 dataset and return a DataFrame with the data (coming from all txt files)
    The returned dataframe has the following columns:
    - accel-x: Acceleration on the x axis
    - accel-y: Acceleration on the y axis
    - accel-z: Acceleration on the z axis
    - user: User code
    - session: Session number
    - index: Index of the sample coming from the txt
    This dataset has rotation vectors, but they are not used in this function

    Parameters
    ----------
    recod_gait_path : str
        Path to the RecodGaitV2 dataset

    Returns
    -------
    pd.DataFrame
        DataFrame with the data from the RecodGaitV2 dataset
    """

    recod_gait_path = Path(recod_gait_path)

    features_acc = {
        "accel-start-time": np.float32,
        "accel-x": np.float32,
        "accel-y": np.float32,
        "accel-z": np.float32,
    }

    dfs = []
    index = 1
    for file in natsorted(recod_gait_path.glob("*.txt")):
        if "accelerometer" in file.name:
            file_name = file.name
            user = file.name.split("__")[0]
            session = file.name.split("__")[1]

            # Reading the accelerometer data
            df = pd.read_csv(
                file,
                sep=",",
                header=None,
                names=features_acc.keys(),
                dtype=features_acc,
            )

            time_acc = np.array(df["accel-start-time"])
            time_acc = (time_acc - time_acc[0]) / 1e9

            # We interpolate the accelerometer data to fix the sampling rate to 40 Hz
            df_acc, new_time = interpolation(
                df, ["accel-x", "accel-y", "accel-z"], time_acc
            )
            df_acc["accel-start-time"] = new_time

            df = df_acc.copy()

            # df['accel-start-time'] = (df['accel-start-time'] - df['accel-start-time'][0]) / 1e9

            df["user"] = int(user)
            df["session"] = int(session[-1])
            df["index"] = index
            index += 1

            # Drop samples with NaN
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)

            dfs.append(df)

    df = pd.concat(dfs, axis=0)
    df.reset_index(drop=True, inplace=True)

    return df


def read_GaitOpticalInertial(gait_optical_path: str) -> pd.DataFrame:
    """Read the GaitOpticalInertial dataset and return a DataFrame with the data (coming from all csv files)
    The returned dataframe has the following columns:
    - accel-x: Acceleration on the x axis
    - accel-y: Acceleration on the y axis
    - accel-z: Acceleration on the z axis
    - gyro-x: Angular velocity on the x axis
    - gyro-y: Angular velocity on the y axis
    - gyro-z: Angular velocity on the z axis
    - user: User code
    - session: Session day
    - Sample: Sample number
    - time: Time of the sample
    - index: Index of the sample coming from the csv

    Parameters
    ----------
    gait_optical_path : str
        Path to the GaitOpticalInertial dataset

    Returns
    -------
    pd.DataFrame
        DataFrame with the data from the GaitOpticalInertial dataset
    """

    gait_optical_path = Path(gait_optical_path)

    datas = []
    for folder in gait_optical_path.iterdir():
        for file in folder.iterdir():
            user = int(file.stem.split("_")[1][4:])

            # print(user, file.stem)
            if file.suffix == ".csv":
                data = pd.read_csv(file)
                day = int(file.stem.split("_")[2][3:])
                num_sample = int(file.stem.split("_")[3])

                data["user"] = user
                data["session"] = day
                data["Sample"] = num_sample
                tam = len(data)
                n_seconds = tam / 100
                data["time"] = np.linspace(0, n_seconds, tam)
                data["index"] = np.arange(tam)

                datas.append(data)

    return pd.concat(datas).reset_index(drop=True, inplace=False)


def read_umafall(umafall_path, interpollation=False):

    """Read the UMAFall dataset and return a DataFrame with the data (coming from all csv files)
    The returned dataframe has the following columns:
    - accel-x: Acceleration on the x axis
    - accel-y: Acceleration on the y axis
    - accel-z: Acceleration on the z axis
    - gyro-x: Angular velocity on the x axis
    - gyro-y: Angular velocity on the y axis
    - gyro-z: Angular velocity on the z axis
    - user: User code
    - activity code: Activity code
    - activity: Activity name
    - file_name: Name of the file
    - index: Index of the sample coming from the csv

    Parameters
    ----------
    umafall_path : str
        Path to the UMAFall dataset
    interpollation : bool, optional
        If True, the data will be interpolated, by default False

    Returns
    -------

    pd.DataFrame
        DataFrame with the data from the UMAFall dataset
    """

    dfs_acc, dfs_gyr, dfs_mag, dfs_smartphone = [], [], [], []

    for file in natsorted(os.listdir(umafall_path)):
        if not file.endswith(".csv"):
            continue
        else:
            # Start to read data from the line 41
            file_path = os.path.join(umafall_path, file)
            # print(file)
            df_acc, df_gyr, df_mag, df_smartphone = reader_umafall_file(file_path, interpollation=False)
            subject = file.split("_")[2]
            activity_type = file.split("_")[3].split(".")[0]
            activity = file.split("_")[4].split(".")[0]
            file_name = file

            (
                df_acc["user"],
                df_acc["activity code"],
                df_acc["activity"],
                df_acc["file_name"],
            ) = (int(subject), activity_type, activity, file_name)
            (
                df_gyr["user"],
                df_gyr["activity code"],
                df_gyr["activity"],
                df_gyr["file_name"],
            ) = (int(subject), activity_type, activity, file_name)
            (
                df_mag["user"],
                df_mag["activity code"],
                df_mag["activity"],
                df_mag["file_name"],
            ) = (int(subject), activity_type, activity, file_name)
            (
                df_smartphone["user"],
                df_smartphone["activity code"],
                df_smartphone["activity"],
                df_smartphone["file_name"],
            ) = (int(subject), activity_type, activity, file_name)

            dfs_acc.append(df_acc)
            dfs_gyr.append(df_gyr)
            dfs_mag.append(df_mag)
            dfs_smartphone.append(df_smartphone)

    # Concatenate all the dataframes into a single dataframe and convert the data type to numeric
    df_acc = pd.concat(dfs_acc, ignore_index=True)

    df_gyr = pd.concat(dfs_gyr, ignore_index=True)

    df_mag = pd.concat(dfs_mag, ignore_index=True)

    df_smartphone = pd.concat(dfs_smartphone, ignore_index=True)
    df_smartphone["user"] = df_smartphone["user"].astype(int)

    return df_smartphone


####################################################
# Functions to help when reading the datasets
####################################################


def getfiles(user, activity, workspace, root):
    """This function will get the files from the real world dataset and move them to the realworld2016_dataset_organized folder

    Parameters
    ----------
    user : str
        User code
    activity : str
        Activity code
    workspace : str
        Path to the RealWorld dataset organized
    root : str
        Path to the raw RealWorld dataset

    Returns
    -------
    None
    """

    folder = workspace / "realworld2016_dataset_organized"

    for sensor in ["acc", "gyr"]:
        file = root / user / f"data/{sensor}_{activity}_csv.zip"
        with ZipFile(file, "r") as zip:
            zip.extractall(workspace / "junk")

        for i in os.listdir(workspace / "junk"):
            if i.find("zip") > -1:
                file = workspace / "junk" / i
                with ZipFile(file, "r") as zip:
                    zip.extractall(workspace / "junk")

        for files in os.listdir(workspace / "junk"):
            if os.path.isfile(workspace / "junk" / files):
                if files.find(activity) > -1 and files.find("zip") < 0:
                    os.rename(workspace / "junk" / files, folder / user / files)
                else:
                    pass
                    # shutil.remove(workspace / "junk" / files)
        # Lets remove the junk folder
        shutil.rmtree(workspace / "junk")


def real_world_organize():
    """This function will organize the real world dataset in a friendly way, creating folders for each user and separating the accelerometer and gyroscope data
    in another folder. It is a good idea to run this function before reading the dataset because it will make the reading process easier.

    Returns
    -------
    workspace : str
        Path to the RealWorld dataset organized
    users : List[str]
        List of users that you want to read
    """

    # Path to organize the dataset and the root of the dataset
    workspace = Path("../data/processed/RealWorld")
    root = Path("../data/original/RealWorld/realworld2016_dataset")

    # List of users and activities
    users = natsorted(os.listdir(root))
    activities: List[str] = [
        "climbingdown",
        "climbingup",
        "jumping",
        "lying",
        "running",
        "sitting",
        "standing",
        "walking",
    ]
    SAC: List[str] = [
        "sitting",
        "standing",
        "walking",
        "climbingup",
        "climbingdown",
        "running",
    ]

    # Create a folder to unzip the files .zip if it doesn't exist
    if not os.path.isdir(workspace / "junk"):
        os.makedirs(workspace / "junk")
    os.path.isdir(workspace / "junk")
    # and the same folder to organize the unzipped files in a friendly way
    if not os.path.isdir(workspace / "realworld2016_dataset_organized"):
        os.mkdir(workspace / "realworld2016_dataset_organized")
    os.path.isdir(workspace / "realworld2016_dataset_organized")

    # Create a folder for each user
    for i in users:
        if not os.path.isdir(workspace / "realworld2016_dataset_organized" / i):
            os.mkdir(workspace / "realworld2016_dataset_organized" / i)

    # Get the files from the dataset and move them to the right folder
    for user in users:
        for activity in activities:
            getfiles(user, activity, workspace, root)
    # Create a folder for the accelerometer and gyroscope data for each user
    for user in users:
        if not os.path.isdir(
            workspace / "realworld2016_dataset_organized" / user / "acc"
        ):
            os.mkdir(workspace / "realworld2016_dataset_organized" / user / "acc")
        if not os.path.isdir(
            workspace / "realworld2016_dataset_organized" / user / "gyr"
        ):
            os.mkdir(workspace / "realworld2016_dataset_organized" / user / "gyr")
    # Move the accelerometer and gyroscope data to the right folder
    for user in users:
        for files in os.listdir(workspace / "realworld2016_dataset_organized" / user):
            if files.find("acc") > -1 and os.path.isfile(
                workspace / "realworld2016_dataset_organized" / user / files
            ):
                origin = workspace / "realworld2016_dataset_organized" / user / files
                destiny = (
                    workspace / "realworld2016_dataset_organized" / user / "acc" / files
                )
                os.rename(origin, destiny)
            if files.find("Gyr") > -1 and os.path.isfile(
                workspace / "realworld2016_dataset_organized" / user / files
            ):
                origin = workspace / "realworld2016_dataset_organized" / user / files
                destiny = (
                    workspace / "realworld2016_dataset_organized" / user / "gyr" / files
                )
                os.rename(origin, destiny)
    # Verify if all users have the same number of accelerometer and gyroscope files
    flag = 1
    for user in users:
        files_acc = os.listdir(
            workspace / "realworld2016_dataset_organized" / user / "acc"
        )
        files_gyr = os.listdir(
            workspace / "realworld2016_dataset_organized" / user / "gyr"
        )
        if len(files_acc) != len(files_gyr):
            flag = 0
            print(
                f"User {user} has {len(files_acc)} acc files and {len(files_gyr)} gyr files"
            )
            flag = -1
    if flag == 1:
        print("All users have the same number of acc and gyr files")

    return workspace, users


def interpolation(df: pd.DataFrame, signals: list, time: np.array) -> pd.DataFrame:
    """Interpolate the signals in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the signals.
    signals : list
        List of signals to interpolate.
    time : np.array
        Array with the timestamps of the signals.

    Returns
    -------
    pd.DataFrame
        Dataframe with the interpolated signals.
    """
    df = df.copy()
    series = []
    for signal in signals:
        serie = np.array(df[signal])

        interp = interpolate.interp1d(
            time, serie, kind="linear", fill_value="extrapolate"
        )
        new_time = np.arange(0, time[-1], 1 / 40)
        new_serie = interp(new_time)
        series.append(new_serie)

    return pd.DataFrame(np.array(series).T, columns=signals), new_time


def slerp_interpolation(
    df: pd.DataFrame, signals: list, time: np.array
) -> pd.DataFrame:
    """Interpolate the quaternions in the dataframe using slerp.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the signals.
    signals : list
        List of signals to interpolate.
    time : np.array
        Array with the timestamps of the signals.

    Returns
    -------
    pd.DataFrame
        Dataframe with the interpolated signals.
    """
    pass


def sanity_function(train_df, val_df, test_df):
    """This function will print some information about the datasets, such as the size of each dataset, the number of samples per user and activity, etc.
    And it will also check if all users have the same number of samples per activity in each dataset.

    Parameters
    ----------
    train_df : pd.DataFrame
        Train dataset
    val_df : pd.DataFrame
        Validation dataset
    test_df : pd.DataFrame
        Test dataset

    Returns
    -------
    None
    """

    train_size: int = train_df.shape[0]
    val_size: int = val_df.shape[0]
    test_size: int = test_df.shape[0]
    total: int = train_size + val_size + test_size

    # Print some information about the datasets
    print(f"Train size: {train_size} ({train_size/total*100:.2f}%)")
    print(f"Validation size: {val_size} ({val_size/total*100:.2f}%)")
    print(f"Test size: {test_size} ({test_size/total*100:.2f}%)")

    print(f"Train activities: {train_df['standard activity code'].unique()}")
    print(f"Validation activities: {val_df['standard activity code'].unique()}")
    print(f"Test activities: {test_df['standard activity code'].unique()}")

    dataframes: Dict[str, pd.DataFrame] = {
        "Train": train_df,
        "Validation": val_df,
        "Test": test_df,
    }
    # Check if all users have the same number of samples per activity in each dataset
    for name, df in dataframes.items():
        users = df["user"].unique()
        activities = df["standard activity code"].unique()

        tam = len(
            df[
                (df["user"] == users[0])
                & (df["standard activity code"] == activities[0])
            ]
        )
        flag = True
        for user in users:
            for activity in activities:
                if (
                    len(
                        df[
                            (df["user"] == user)
                            & (df["standard activity code"] == activity)
                        ]
                    )
                    != tam
                ):
                    flag = False
        if flag:
            pass

    users = train_df["user"].unique()
    activities = train_df["standard activity code"].unique()

def umafall_interpolation(df):
    """Interpolate the UMAFall dataset to fix the sampling rate to 20 Hz

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the UMAFall data
    
    Returns
    -------
    pd.DataFrame
        Dataframe with the interpolated data
    """

    # For each sensor type (0: accelerometer, 1: gyroscope, 2: magnetometer)
    columns = df.columns
    dfs = []

    for sensor, sub_df in df.groupby(["Sensor Type"]):
        fs = 200 if sensor[0] == 0 else 20
        tam = len(sub_df)

        # Interpolate the data
        time = np.arange(0, tam, 1)
        x = sub_df["X-Axis"].values
        y = sub_df["Y-Axis"].values
        z = sub_df["Z-Axis"].values

        f_x = interpolate.interp1d(time, x, kind="cubic")
        f_y = interpolate.interp1d(time, y, kind="cubic")
        f_z = interpolate.interp1d(time, z, kind="cubic")

        time_new = np.arange(0, 15, 1 / fs)
        x_new = f_x(time_new)
        y_new = f_y(time_new)
        z_new = f_z(time_new)

        # Update the DataFrame
        df_new = pd.DataFrame()
        # print(len(time_new), len(x_new), len(y_new), len(z_new))
        df_new["TimeStamp"] = np.array(time_new)
        df_new["X-Axis"] = np.array(x_new)
        df_new["Y-Axis"] = np.array(y_new)
        df_new["Z-Axis"] = np.array(z_new)
        df_new["Sensor Type"] = np.array([sensor] * len(time_new))
        df_new["Sensor ID"] = np.array([0] * len(time_new))

        dfs.append(df_new)

    return pd.concat(dfs)


def reader_umafall_file(file_path, interpollation=False):
    # Initialize lists to store metadata and data
    metadata = []
    data = []

    # Read the CSV file line by line
    with open(file_path, "r") as file:
        lines = file.readlines()
        is_metadata = True

        for line in lines:
            line = line.strip()

            if line:
                if line.startswith("%") and is_metadata:
                    metadata.append(line)
                else:
                    is_metadata = False
                    data.append(line)

    # Join the metadata lines to form a single string
    metadata_str = "\n".join(metadata)

    # Read the data lines as a DataFrame
    df = pd.read_csv(
        Path(file_path),
        comment="%",
        header=None,
        names=[
            "TimeStamp",
            "Sample No",
            "X-Axis",
            "Y-Axis",
            "Z-Axis",
            "Sensor Type",
            "Sensor ID",
        ],
        sep=";",
        engine="python",
        dtype={
            "TimeStamp": np.float32,
            "Sample No": np.int64,
            "X-Axis": np.float32,
            "Y-Axis": np.float32,
            "Z-Axis": np.float32,
            "Sensor Type": np.int64,
            "Sensor ID": np.int64,
        },
    )
    columns = ["X-Axis", "Y-Axis", "Z-Axis"]
    for column in columns:
        # If is there any value similar to 1.012.598.156.929.010, remove all the dots, except the first one
        data = []
        for value in df[column].values:
            # Verify if the value is a float
            try:
                float(value)
                # If is a float, append the value as float to the list
                value = float(value)
                data.append(value)
            except:
                # print (value)
                # If is not a float, remove the dots, except the first one
                value = value.replace(".", "")
                value = value[0] + "." + value[1:]
                # convert the value to float
                data.append(float(value))
        df[column] = data

        # Convert the values to float
        # df[column] = df[column].astype(float)

    if interpollation:
        # Interpolate the data
        df = umafall_interpolation(df)

    df_smartphone = df[df['Sensor ID'] == 0]
    # df_smartphone = df_smartphone.rename(
    #     columns={"X-Axis": "acc-x", "Y-Axis": "acc-y", "Z-Axis": "acc-z"}
    # )
    
    df_acc = df[(df["Sensor Type"] == 0)]
    # df_acc = df_acc.rename(
    #     columns={"X-Axis": "acc-x", "Y-Axis": "acc-y", "Z-Axis": "acc-z"}
    # )

    df_gyr = df[df["Sensor Type"] == 1]
    # df_gyr.drop(columns_to_remove, axis=1, inplace=True)
    # df_gyr = df_gyr.rename(
    #     columns={"X-Axis": "gyr-x", "Y-Axis": "gyr-y", "Z-Axis": "gyr-z"}
    # )

    df_mag = df[df["Sensor Type"] == 2]
    # df_mag.drop(columns_to_remove, axis=1, inplace=True)
    # df_mag = df_mag.rename(
    #     columns={"X-Axis": "mag-x", "Y-Axis": "mag-y", "Z-Axis": "mag-z"}
    # )

    # Set the column names based on the metadata
    column_names = metadata_str.split("\n")[-1].split("; ")

    return df_acc, df_gyr, df_mag, df_smartphone