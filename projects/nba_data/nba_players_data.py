import pandas as pd
import os
import requests

# Checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if 'nba2k-full.csv' not in os.listdir('../Data'):
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/nba2k-full.csv', 'wb').write(r.content)
    print('Loaded.')

data_path = "../Data/nba2k-full.csv"

# write your code here
def clean_data(path):
    df = pd.read_csv(path)
    df.b_day = pd.to_datetime(df.b_day, format='%m/%d/%y')
    df.draft_year = pd.to_datetime(df.draft_year, format='%Y')
    df.team.fillna('No Team', inplace=True)
    df.height = df.height.apply(lambda x: x.split('/')[-1].strip()).astype('float')
    df.weight = df.weight.apply(lambda x: x.split('/')[-1].split('kg.')[0].strip()).astype('float')
    df.salary = df.salary.apply(lambda x: x.replace('$', '')).astype('float')
    df.country = df.country.apply(lambda x: x if x == 'USA' else 'Not-USA')
    df.draft_round = df.draft_round.apply(lambda x: '0' if x == 'Undrafted' else x)

    return df


def feature_data(df):
    df.version = df.version.apply(lambda x: 2020 if x == 'NBA2k20' else 2021)
    df.version = pd.to_datetime(df.version, format='%Y')

    df['age'] = df.version.dt.year - df.b_day.dt.year
    df['experience'] = df.version.dt.year - df.draft_year.dt.year
    df['bmi'] = df[['weight', 'height']].apply(lambda x: x.weight / (x.height * x.height), axis=1)
    df.drop(['version', 'b_day', 'draft_year', 'weight', 'height'], inplace=True, axis=1)
    # Remove high cardinality columns
    for c in df.columns:
        if c not in ['bmi', 'salary'] and len(df[c].unique()) >= 50:
            df.drop(c, axis=1, inplace=True)

    return df

def multicol_data(df):
    return df.drop(['age'], axis=1)

def transform_data(df):
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np

    num_feat_df = df.select_dtypes('number').drop('salary', axis=1)  # numerical features
    cat_feat_df = df.select_dtypes('object')  # categorical features

    std_scaler = StandardScaler()
    transformed_num_feat = std_scaler.fit_transform(num_feat_df)

    onehot_encoder = OneHotEncoder(sparse_output=False)
    enc_cat_feat = onehot_encoder.fit_transform(cat_feat_df)

    f1 = num_feat_df.columns
    f2 = np.concatenate(onehot_encoder.categories_)
    F = np.concatenate((f1, f2))
    X = np.concatenate((transformed_num_feat, enc_cat_feat), axis=1)
    X = pd.DataFrame(X, columns=F)
    y = df.salary
    return X, y