import pandas as pd
from dataset_processor import BalanceToMinimumClass, BalanceToMinimumClassAndUser
import os
from pathlib import Path
from typing import Callable
import re
import shutil
from tqdm import tqdm
from itertools import product
import random


def post_process_hapt_dataset(
        input_path: Path,
        output_path: Path,
        balance_function: Callable = None
    ):
    data = pd.read_csv(input_path)

    users_in_train_df = pd.read_csv('../data/original/UCI/Train/subject_id_train.txt', header=None)
    users_in_train_ids = [int(user_id) for user_id in users_in_train_df[0].unique()]
    users_in_test_df = pd.read_csv('../data/original/UCI/Test/subject_id_test.txt', header=None)
    users_in_test_ids = [int(user_id) for user_id in users_in_test_df[0].unique()]

    balanced_data_per_user_and_class = []
    for user_id in users_in_train_ids:
        user_data = data[data['user'] == user_id]
        user_data_to_append = user_data.copy()
        if balance_function:
            user_data_to_append = balance_function(user_data_to_append)
        balanced_data_per_user_and_class.append(user_data_to_append)
    balanced_train_data_per_user_and_class = pd.concat(balanced_data_per_user_and_class)

    grouped_train_data = balanced_train_data_per_user_and_class.groupby(['user', 'standard activity code']).count()[['activity code']]
    users_in_val_ids = grouped_train_data.pivot_table(index='user', columns='standard activity code', values='activity code', fill_value=0).sort_values(9)[-3:].index
    
    balanced_data_per_user_and_class = []
    for user_id in users_in_test_ids:
        user_data = data[data['user'] == user_id]
        balanced_data_per_user_and_class.append(BalanceToMinimumClass()(user_data.copy()))
    balanced_test_data_per_user_and_class = pd.concat(balanced_data_per_user_and_class)
    
    train_data = balanced_train_data_per_user_and_class[~balanced_train_data_per_user_and_class['user'].isin(users_in_val_ids)]
    val_data = balanced_train_data_per_user_and_class[balanced_train_data_per_user_and_class['user'].isin(users_in_val_ids)]
    test_data = balanced_test_data_per_user_and_class

    train_data.to_csv(output_path / 'train.csv', index=False)
    val_data.to_csv(output_path / 'validation.csv', index=False)
    test_data.to_csv(output_path / 'test.csv', index=False)


# For RECODGAIT dataset
def generate_pairs_from_rg_partition(partition: str, data_dict: dict, positive_pairs_per_user: int=500):
    negative_pairs_per_user = positive_pairs_per_user * 5
    pairs = []
    partition_data_dict = data_dict[partition]
    users_ids = list(partition_data_dict.keys())
    for user_id in users_ids:
        unique_session_ids = list(partition_data_dict[user_id].keys())
        # Positive pairs
        for _ in range(positive_pairs_per_user):
            random_session_id_1, random_session_id_2 = random.sample(unique_session_ids, k=2)
            user_sample_index_1 = random.choice(partition_data_dict[user_id][random_session_id_1])
            user_sample_index_2 = random.choice(partition_data_dict[user_id][random_session_id_2])
            # Append the pair
            pairs.append({
                'sample-1-user-id': user_id,
                'sample-2-user-id': user_id,
                'sample-1-session-id': random_session_id_1,
                'sample-2-session-id': random_session_id_2,
                'sample-1-index': user_sample_index_1,
                'sample-2-index': user_sample_index_2,
                'label': 1
            })
        # Negative pairs
        other_users_ids = [other_user_id for other_user_id in users_ids if user_id != other_user_id]
        # print(f'User {user_id} - USERS {users_ids} - IMPOSTORS: {other_users_ids}')
        for _ in range(negative_pairs_per_user):
            # Original user sample selection
            user_session_id = 1 if partition == 'test' else random.choice(unique_session_ids)
            user_sample_index = random.choice(partition_data_dict[user_id][user_session_id])
            # Impostor user sample selection
            random_impostor_user_id = random.choice(list(other_users_ids))
            random_impostor_session_ids = list(partition_data_dict[random_impostor_user_id].keys())
            random_impostor_session_id = 3 if partition == 'test' else random.choice(random_impostor_session_ids)
            random_impostor_sample_index = random.choice(partition_data_dict[random_impostor_user_id][random_impostor_session_id])
            # Append the pair
            pairs.append({
                'sample-1-user-id': user_id,
                'sample-2-user-id': random_impostor_user_id,
                'sample-1-session-id': user_session_id,
                'sample-2-session-id': random_impostor_session_id,
                'sample-1-index': user_sample_index,
                'sample-2-index': random_impostor_sample_index,
                'label': 0
            })
    return pd.DataFrame(pairs)


# Mix the data based on defined pairs
def mix_data_according_to_pairs(pairs_data: pd.DataFrame, samples_data: pd.DataFrame, output_path: Path):
    sample1_indexes = pairs_data['sample-1-index'].tolist()
    sample2_indexes = pairs_data['sample-2-index'].tolist()
    
    preffixes = ['accel-x', 'accel-y', 'accel-z']

    filter_columns = [f'{preffix}-{i}' for preffix, i in product(preffixes, range(60))]
    additional_columns = ['user', 'session']
    
    filter_columns = filter_columns + additional_columns

    sample1_data = samples_data.iloc[sample1_indexes][filter_columns]
    sample2_data = samples_data.iloc[sample2_indexes][filter_columns]

    # Rename columns
    sample1_data.columns = [f'sample-1-{col}' for col in sample1_data.columns]
    sample2_data.columns = [f'sample-2-{col}' for col in sample2_data.columns]
    # Reset the indexes
    sample1_data.reset_index(drop=True, inplace=True)
    sample2_data.reset_index(drop=True, inplace=True)
    # Concatenate the data
    final_data = pd.concat([sample1_data, sample2_data, pairs_data], axis=1)
    final_data.to_csv(output_path, index=False)


def post_process_recodgait_dataset_daghar_like(input_path: Path, output_path: Path):
    # Read the input file
    data = pd.read_csv(input_path)
    # Preprocess the data
    data['user'] = data['user'].apply(lambda x: int(x))
    data['session'] = data['session'].apply(lambda x: int(x))
    # Group by user and create a column for every unique sessions
    data_sessions = data.groupby(['user', 'session']).size().unstack(fill_value=0)
    data_sessions['S-count'] = data_sessions.apply(lambda row: 5 - row.value_counts().get(0, 0), axis=1)
    # Separate the data in train, validation and test to identify the respective user ids
    train_user_ids = data_sessions[data_sessions['S-count'].isin([3,5])].index
    val_user_ids = data_sessions[data_sessions['S-count'] == 4].index
    test_user_ids = data_sessions[data_sessions['S-count'] == 2].index
    # Generate the index dictionary
    data_index_dict = {
        partition: {
            user_id: {
                session: [
                    int(val)
                    for val in data[(data['user'] == user_id)&(data['session'] == session)].index
                    ]
                for session in data[data['user'] == user_id]['session'].unique().tolist()
            }
            for user_id in user_ids
        }
        for partition, user_ids in zip(['train', 'validation', 'test'], [train_user_ids, val_user_ids, test_user_ids])
    }
    # Generate the index pairs dataframes
    train_index_pairs_df = generate_pairs_from_rg_partition('train', data_index_dict)
    val_index_pairs_df = generate_pairs_from_rg_partition('validation', data_index_dict)
    test_index_pairs_df = generate_pairs_from_rg_partition('test', data_index_dict)
    # Save the dataframes
    train_index_pairs_df.to_csv('RG_train_index_pairs.csv', index=False)
    val_index_pairs_df.to_csv('RG_val_index_pairs.csv', index=False)
    test_index_pairs_df.to_csv('RG_test_index_pairs.csv', index=False)
    mix_data_according_to_pairs(train_index_pairs_df, data, output_path / 'train.csv')
    mix_data_according_to_pairs(val_index_pairs_df, data, output_path / 'validation.csv')
    mix_data_according_to_pairs(test_index_pairs_df, data, output_path / 'test.csv')


def post_process_recodgait_dataset_concatenated_in_user_files(input_path: Path, output_path: Path):
    # Read the input file
    data = pd.read_csv(input_path)
    # Preprocess the data
    data['user'] = data['user'].apply(lambda x: int(x))
    data['session'] = data['session'].apply(lambda x: int(x))
    # Group by user and create a column for every unique sessions
    data_sessions = data.groupby(['user', 'session']).size().unstack(fill_value=0)
    data_sessions['S-count'] = data_sessions.apply(lambda row: 5 - row.value_counts().get(0, 0), axis=1)
    # Separate the data in train, validation and test to identify the respective user ids
    train_user_ids = data_sessions[data_sessions['S-count'].isin([3,5])].index
    val_user_ids = data_sessions[data_sessions['S-count'] == 4].index
    test_user_ids = data_sessions[data_sessions['S-count'] == 2].index
    # Separate the data per users
    train_data = data[data['user'].isin(train_user_ids)].sort_values(['user', 'session', 'window']).reset_index(drop=True)
    val_data = data[data['user'].isin(val_user_ids)].sort_values(['user', 'session', 'window']).reset_index(drop=True)
    test_data = data[data['user'].isin(test_user_ids)].sort_values(['user', 'session', 'window']).reset_index(drop=True)
    # Save the dataframes
    train_data.to_csv(output_path / 'train.csv', index=False)
    val_data.to_csv(output_path / 'validation.csv', index=False)
    test_data.to_csv(output_path / 'test.csv', index=False)



def linearize_dataframe(df, column_prefixes, maintain):
    # Initialize a dictionary to hold the linearized columns
    linearized_data = {prefix: [] for prefix in column_prefixes}
    for m in maintain:
        linearized_data[m] = []

    # Regular expression to match columns that start with the prefix and are followed by a number
    def get_columns_with_number(df, prefix):
        return [col for col in df.columns if re.match(f"^{prefix}-\d+$", col)]

    # Iterate over each row in the DataFrame
    for i, row in df.iterrows():
        # For each prefix, find how many columns correspond to it (prefix-N, where N is a number)
        for prefix in column_prefixes:
            cols = get_columns_with_number(df, prefix)
            # Extend the list with the corresponding values from the row
            linearized_data[prefix].extend([row[col] for col in cols])
        # Replicate the user column values for the number of new rows
        for m in maintain:
            linearized_data[m].extend([row[m]] * len(cols))

    # Create the new linearized DataFrame
    linearized_df = pd.DataFrame(linearized_data)

    # Return the linearized DataFrame
    return linearized_df


def create_dataset(
        root_path: Path,
        output_path: Path,
        label: str = "standard activity code",
        columns_to_maintain_in_linearize_dataframe=["user", "standard activity code"],
        column_prefixes = ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
    ):
    train_df = pd.read_csv(root_path / "train.csv")
    val_df = pd.read_csv(root_path / "validation.csv")
    test_df = pd.read_csv(root_path / "test.csv")

    # Linearize the dataframes
    print("Linearizing train dataframe...")
    train_df = linearize_dataframe(train_df, column_prefixes, columns_to_maintain_in_linearize_dataframe)
    print("Linearizing validation dataframe...")
    val_df = linearize_dataframe(val_df, column_prefixes, columns_to_maintain_in_linearize_dataframe)
    print("Linearizing test dataframe...")
    test_df = linearize_dataframe(test_df, column_prefixes, columns_to_maintain_in_linearize_dataframe)
    if label:
        train_df["activity code"] = train_df[label]
        val_df["activity code"] = val_df[label]
        test_df["activity code"] = test_df[label]
    
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        output_path_split = output_path / split
        output_path_split.mkdir(parents=True, exist_ok=True)
        for user_id, user_df in tqdm(df.groupby("user")):
            user_df.to_csv(output_path_split / f"{user_id}.csv", index=False)
        print(f"Saved {len(df)} samples to {output_path}")
    
    return train_df, val_df, test_df


def apply_custom_function_to_every_csv_file(input_path: Path, output_path: Path, custom_function: Callable):
    for file in input_path.glob("*.csv"):
        df = pd.read_csv(file)
        df = custom_function(df)
        df.to_csv(output_path / file.name, index=False)

def duplicate_accel_channels(df):
    # preffixes = ['accel-x', 'accel-y', 'accel-z']
    df['gyro-x'] = df['accel-x']
    df['gyro-y'] = df['accel-y']
    df['gyro-z'] = df['accel-z']
    return df

def __main__():
    print('Starting the dataset post-processing...')

    # For HAPT dataset
    print('Processing the HAPT dataset...')
    # General input path
    input_path = Path('../data/transitions/unbalanced/HAPT/standartized_unbalanced.csv')
    # First post-processing: balanced classes per user - for DOWNSTREAM task
    output_path = Path('../data/transitions/processed/HAPT_daghar_like')
    os.makedirs(output_path, exist_ok=True)
    balance_function = BalanceToMinimumClassAndUser()
    post_process_hapt_dataset(input_path, output_path, balance_function)
    # Second post-processing: unbalanced classes per user - for PRETRAIN task
    temporal_path = Path('../data/transitions/processed/temporal')
    os.makedirs(temporal_path, exist_ok=True)
    post_process_hapt_dataset(input_path, temporal_path, None)
    # Second post-processing: generating the final dataset
    create_dataset(temporal_path, Path('../data/transitions/processed/HAPT_concatenated_in_user_files'))

    # Removing the temporary files
    shutil.rmtree(temporal_path)
    shutil.rmtree(Path('../data/transitions/processed/HAPT_concatenated_in_user_files/test'))

    # For RECODGAIT
    print('Processing the RecodGait dataset...')
    # General input path
    input_path = Path('../data/authentication/unbalanced/RecodGait_v2/standartized_unbalanced.csv')
    # First post-processing: balanced classes per user - for DOWNSTREAM task
    output_path = Path('../data/authentication/processed/RG_daghar_like')
    os.makedirs(output_path, exist_ok=True)
    post_process_recodgait_dataset_daghar_like(input_path, output_path)
    # Second post-processing: unbalanced classes per user - for PRETRAIN task
    temporal_path = Path('../data/authentication/processed/temporal')
    os.makedirs(temporal_path, exist_ok=True)
    post_process_recodgait_dataset_concatenated_in_user_files(input_path, temporal_path)
    # Second post-processing: generating the final dataset
    create_dataset(temporal_path, Path('../data/authentication/processed/RG_concatenated_in_user_files'), label=None, columns_to_maintain_in_linearize_dataframe=["user"], column_prefixes=["accel-x", "accel-y", "accel-z"])
    # Third post-processing: generating a 6-channels copy for CPC
    output_path = Path('../data/authentication/processed/RG_concatenated_in_user_files_accel_duplicated')
    os.makedirs(output_path / 'train', exist_ok=True)
    os.makedirs(output_path / 'validation', exist_ok=True)

    apply_custom_function_to_every_csv_file(
        input_path=Path('../data/authentication/processed/RG_concatenated_in_user_files/train'),
        output_path=output_path / 'train',
        custom_function=duplicate_accel_channels
    )

    apply_custom_function_to_every_csv_file(
        input_path=Path('../data/authentication/processed/RG_concatenated_in_user_files/validation'),
        output_path=output_path / 'validation',
        custom_function=duplicate_accel_channels
    )

    # Removing the temporary files
    shutil.rmtree(temporal_path)
    shutil.rmtree(Path('../data/authentication/processed/RG_concatenated_in_user_files/test'))
    
    
if __name__ == '__main__':
    __main__()
