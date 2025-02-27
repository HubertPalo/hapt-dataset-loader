import random
import pandas as pd

def create_dummy_rg_df(users_count=5, max_sessions=5, samples_per_session=2):
    dummy_sample = []
    user_col = []
    session_col = []
    for u in range(users_count):
        total_sessions = random.randint(3,max_sessions)
        user_id = u+1
        for s in range(total_sessions):
            session_id = s+1
            for i in range(samples_per_session):
                val = f'{user_id}-{session_id}-{i+1}'
                dummy_sample.append(val)
                user_col.append(user_id)
                session_col.append(session_id)

    dummy_columns = [f'accel-{axis}-{i}' for axis in ['x', 'y', 'z'] for i in range(0, 5)]

    dummy_data = {
        col_id: dummy_sample
        for col_id in dummy_columns
    }
    dummy_df = pd.DataFrame(data=dummy_data, columns=dummy_columns)
    dummy_df['user'] = user_col
    dummy_df['session'] = session_col
    return dummy_df