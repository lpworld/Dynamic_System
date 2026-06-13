# Data

Run the preprocessing scripts to produce the files referenced by the configs
(`data/yelp.csv`, `data/movielens.csv`, `data/youku.csv`). The generated `.csv`
files are not tracked.

```
python preprocess.py --dataset yelp      --src yelp.txt           --dst data/yelp.csv
python preprocess.py --dataset youku     --src youku.txt          --dst data/youku.csv
python preprocess.py --dataset movielens --src ml-25m/ratings.csv --dst data/movielens.csv
```

MovieLens is available at https://grouplens.org/datasets/movielens/.
