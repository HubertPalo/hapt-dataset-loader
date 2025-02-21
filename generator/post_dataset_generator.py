import pandas as pd
from dataset_processor import BalanceToMinimumClass, BalanceToMinimumClassAndUser
import os
from pathlib import Path


def post_process_hapt_dataset(output_path: Path = Path('../data/transitions/balanced/HAPT/standartized_balanced')):
    data = pd.read_csv('../data/transitions/unbalanced/HAPT/standartized_unbalanced.csv')

    users_in_train_df = pd.read_csv('../data/original/UCI/Train/subject_id_train.txt', header=None)
    users_in_train_ids = [int(user_id) for user_id in users_in_train_df[0].unique()]
    users_in_test_df = pd.read_csv('../data/original/UCI/Test/subject_id_test.txt', header=None)
    users_in_test_ids = [int(user_id) for user_id in users_in_test_df[0].unique()]

    balanced_data_per_user_and_class = []
    for user_id in users_in_train_ids:
        user_data = data[data['user'] == user_id]
        balanced_data_per_user_and_class.append(BalanceToMinimumClass()(user_data.copy()))
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

def __main__():
    post_process_hapt_dataset()

if __name__ == '__main__':
    __main__()
