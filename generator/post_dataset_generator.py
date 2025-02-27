import pandas as pd
from dataset_processor import BalanceToMinimumClass, BalanceToMinimumClassAndUser
import os
from pathlib import Path
from typing import Callable
import re
import shutil
from tqdm import tqdm
from itertools import product


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

    os.makedirs(output_path, exist_ok=True)
    train_data.to_csv(output_path / 'train.csv', index=False)
    val_data.to_csv(output_path / 'validation.csv', index=False)
    test_data.to_csv(output_path / 'test.csv', index=False)
    return


def post_process_recodgait_dataset(input_path: Path, output_path: Path):
    # Read the input file
    data = pd.read_csv(input_path)
    # Preprocess the data
    data['user'] = data['user'].apply(lambda x: int(x))
    data['session'] = data['session'].apply(lambda x: int(x))
    # Group by user and create a column for every unique sessions
    data_sessions = data.groupby(['user', 'session']).size().unstack(fill_value=0)
    data_sessions['S-count'] = data_sessions.apply(lambda row: 5 - row.value_counts().get(0, 0), axis=1)
    # Separate the data in train, validation and test
    train_user_ids = data_sessions[data_sessions['S-count'].isin([3,5])].index
    val_user_ids = data_sessions[data_sessions['S-count'] == 4].index
    test_user_ids = data_sessions[data_sessions['S-count'] == 2].index
    train_df = data[data['user'].isin(train_user_ids)]
    val_df = data[data['user'].isin(val_user_ids)]
    test_df = data[data['user'].isin(test_user_ids)]
    

def create_positive_and_negative_pairs(data: pd.DataFrame):
    # Identify the users in the dataset
    user_ids = data['user'].unique()
    # Generate the pairs
    pairs = []
    for user_id1, user_id2 in tqdm(product(user_ids, repeat=2), desc='User pairs', total=len(user_ids)**2):
        session_ids_user1 = data[data['user'] == user_id1]['session'].unique()
        session_ids_user2 = data[data['user'] == user_id2]['session'].unique()
        for session_user1, session_user2 in product(session_ids_user1, session_ids_user2):
            if session_user1 == session_user2 and user_id1 == user_id2:
                continue
            session_data_user1 = data[(data['user'] == user_id1) & (data['session'] == session_user1)]
            session_data_user2 = data[(data['user'] == user_id2) & (data['session'] == session_user2)]
            pair_type = '-'
            if user_id1 == user_id2:
                pair_type = '+'
            for idx_session_user_1, idx_session_user_2 in product(session_data_user1.index, session_data_user2.index):
                pairs.append(((idx_session_user_1, idx_session_user_2), pair_type))
    # Separate the pairs in positive and negative
    positive_pairs = [pair[0] for pair in pairs if pair[1] == '+']
    negative_pairs = [pair[0] for pair in pairs if pair[1] == '-']
    # Generate the dataframes
    positive_data = generate_df_from_pairs(positive_pairs, data, 1)
    negative_data = generate_df_from_pairs(negative_pairs, data, 0)
    return pd.concat([positive_data, negative_data])

def generate_df_from_pairs(
        pairs,
        data: pd.DataFrame,
        label: int,
        preffixes = ['accel-x', 'accel-y', 'accel-z'],
        additional_columns = ['user', 'session']
    ):
    columns = [val for val in data.columns for preffix in preffixes if val.startswith(preffix)] + additional_columns
    # Extract the ids  
    pair_elem_1 = [val[0] for val in pairs]
    pair_elem_2 = [val[1] for val in pairs]
    # Extract the data from the pairs
    s1 = data.loc[pair_elem_1, columns].reset_index(drop=True)
    s2 = data.loc[pair_elem_2, columns].reset_index(drop=True)
    # Add a preffix to the columns
    s1 = s1.rename(lambda x: f'S1-{x}', axis=1)
    s2 = s2.rename(lambda x: f'S2-{x}', axis=1)
    # Concatenate the data and add the label
    result_df = pd.concat([s1, s2], axis=1)
    result_df['label'] = label
    return result_df

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


def create_dataset(root_path: Path, output_path: Path):
    train_df = pd.read_csv(root_path / "train.csv")
    val_df = pd.read_csv(root_path / "validation.csv")
    test_df = pd.read_csv(root_path / "test.csv")

    # Linearize the dataframes
    column_prefixes = [
        "accel-x",
        "accel-y",
        "accel-z",
        "gyro-x",
        "gyro-y",
        "gyro-z",
    ]
    
    train_df = linearize_dataframe(train_df, column_prefixes, ["user", "standard activity code"])
    val_df = linearize_dataframe(val_df, column_prefixes, ["user", "standard activity code"])
    test_df = linearize_dataframe(test_df, column_prefixes, ["user", "standard activity code"])
    train_df["activity code"] = train_df["standard activity code"]
    val_df["activity code"] = val_df["standard activity code"]
    test_df["activity code"] = test_df["standard activity code"]
    
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        output_path_split = output_path / split
        output_path_split.mkdir(parents=True, exist_ok=True)
        for user_id, user_df in df.groupby("user"):
            user_df.to_csv(output_path_split / f"{user_id}.csv", index=False)
        print(f"Saved {len(df)} samples to {output_path}")
    
    return train_df, val_df, test_df




def __main__():
    # General input path
    input_path = Path('../data/transitions/unbalanced/HAPT/standartized_unbalanced.csv')
    # First post-processing: balanced classes per user - for DOWNSTREAM task
    output_path = Path('../data/transitions/processed/HAPT_daghar_like')
    balance_function = BalanceToMinimumClassAndUser()
    post_process_hapt_dataset(input_path, output_path, balance_function)
    # Second post-processing: unbalanced classes per user - for PRETRAIN task
    output_path = Path('../data/transitions/processed/temporal')
    post_process_hapt_dataset(input_path, output_path, None)
    # Second post-processing: generating the final dataset
    create_dataset(output_path, Path('../data/transitions/processed/HAPT_concatenated_in_user_files'))

    # Removing the temporary files
    shutil.rmtree(output_path)
    shutil.rmtree(Path('../data/transitions/processed/HAPT_concatenated_in_user_files/test'))
    
if __name__ == '__main__':
    __main__()
