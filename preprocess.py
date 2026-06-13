import argparse
import pandas as pd

# Convert each raw dataset into a common schema: user_id, item_id, click, time.
# User and item ids are factorized into contiguous integers; clicks are binary.


def _standardize(df):
    df['user_id'] = pd.factorize(df['user_id'])[0]
    df['item_id'] = pd.factorize(df['item_id'])[0]
    df['click'] = df['click'].astype(float)
    return df[['user_id', 'item_id', 'click', 'time']]


def prepare_yelp(src, dst):
    df = pd.read_csv(src)
    df.columns = ['user_id', 'item_id', 'click', 'time']
    _standardize(df).to_csv(dst, index=False)


def prepare_youku(src, dst):
    df = pd.read_csv(src)
    df.columns = ['user_id', 'item_id', 'click', 'time']
    _standardize(df).to_csv(dst, index=False)


def prepare_movielens(src, dst, threshold=4.0):
    # MovieLens ratings.csv with columns: userId, movieId, rating, timestamp.
    df = pd.read_csv(src)
    df = df.rename(columns={'userId': 'user_id', 'movieId': 'item_id', 'timestamp': 'time'})
    df['click'] = (df['rating'] >= threshold).astype(float)
    _standardize(df).to_csv(dst, index=False)


DATASETS = {'yelp': prepare_yelp, 'youku': prepare_youku, 'movielens': prepare_movielens}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    args = ap.parse_args()
    DATASETS[args.dataset](args.src, args.dst)
