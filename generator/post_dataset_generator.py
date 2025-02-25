import pandas as pd
from dataset_processor import BalanceToMinimumClass, BalanceToMinimumClassAndUser
import os
from pathlib import Path
from typing import Callable
import re
import shutil


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
