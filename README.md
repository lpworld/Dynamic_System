# FDSR: Forward-looking Dynamical System Recommender

Implementation of FDSR, a recommender that models a user's evolving preference as
a trajectory in a latent embedding space. A personalized evolution function is
solved with a DeepONet to predict the next trajectory point, and items closest to
that point (exact nearest neighbor) are recommended. A forward-looking term derived
from a mixture density network accounts for how the current recommendation shifts
future preferences, balancing exploration and exploitation.

## Requirements

The model relies on `tf.contrib` and must be run on the TensorFlow 1.4 backend.

```
pip install -r requirements.txt
```

## Repository layout

| File | Role | Figure 2 component |
| --- | --- | --- |
| `model.py` | The FDSR model | whole pipeline |
| `model.py: seq_attention` | Self-attentive GRU trajectory encoder theta_u with Euclidean attention | trajectory representation |
| `model.py: mdn_head`, `mdn_nll` | Mixture density network; forward-looking term E[X_next] = sum_i alpha_i mu_i | future-expectation term |
| `model.py: spectral_norm`, `dense_sn` | Lipschitz-controlled evolution function f(X,t) | evolution function |
| `model.py: branch/trunk` | DeepONet solver for the trajectory point | dynamical-system solver |
| `model.py: distance / loss` | Euclidean retrieval and squared-hinge contrastive loss | recommendation |
| `utils.py` | Data batching, time-stratified folds, metrics | - |
| `train.py` | Offline experiments (Tables II-IV) | - |
| `simulate.py` | Long-term feedback-loop simulation (Section V) | - |
| `preprocess.py` | Raw to common-schema conversion for the three datasets | - |
| `configs/` | Per-dataset hyperparameters and random seeds | - |

## Data

Three datasets are used: Yelp, MovieLens, and Alibaba-Youku. Yelp and Youku are
provided as raw files (`yelp.txt`, `youku.txt`); MovieLens can be downloaded from
https://grouplens.org/datasets/movielens/. Preprocessing maps each dataset to the
common schema `user_id, item_id, click, time` with contiguous ids and binary clicks.

```
python preprocess.py --dataset yelp      --src yelp.txt              --dst data/yelp.csv
python preprocess.py --dataset youku     --src youku.txt             --dst data/youku.csv
python preprocess.py --dataset movielens --src ml-25m/ratings.csv    --dst data/movielens.csv
```

## Offline experiments

```
python train.py --config configs/yelp.yaml
python train.py --config configs/movielens.yaml
python train.py --config configs/youku.yaml
```

Each run reports AUC, HR@10, Coverage, Diversity, Gini, and Serendipity, averaged
over the time-stratified 5-fold split. All models score and rank the same labeled
candidate set using the organic positive/negative labels in the data; no negative
sampling is performed.

### Ablations

The spectral-normalization ablation is a single switch in the config:

```
spectral_normalize: false
```

## Baselines

The baselines are implemented under the shared protocol in `baselines/` and run
through a single entry point:

```
python -m baselines.run --model sasrec --config configs/yelp.yaml
```

| Group | Models |
| --- | --- |
| Factorization / CTR | `mf`, `bsa_mf`, `prop_mf`, `deepfm`, `din` |
| Sequential | `transrec`, `sasrec`, `bert4rec`, `mamba4rec` |
| Bandits / RL | `linucb`, `cofiba`, `reinforce`, `drn`, `rlur` |
| Exploration | `prefrec`, `deep_exploration`, `hac`, `resact` |
| Generative / LLM | `dreamrec`, `diffrec`, `tallrec` |

Every model scores and ranks the same labeled candidate set without negative
sampling and reports the same six metrics as FDSR. `tallrec` runs outside the
TensorFlow pipeline and requires `pip install torch transformers` and a base model
(`base_model='gpt2'` by default).

## Simulation

```
python simulate.py --users 400 --items 400 --iterations 20
```

## Notes on the implementation

- The future-expectation term is the closed-form mixture mean sum_i alpha_i mu_i, so no sampling
  over next trajectory points is required at training or inference time.
- Attention weights in the trajectory encoder are Euclidean distances between
  consecutive item embeddings, so steps where the user's interest shifts are
  weighted more heavily.
- Spectral normalization bounds the Lipschitz constant of the evolution function,
  matching the regularity assumptions of the dynamical system.
- Retrieval is exact nearest-neighbor search in the Euclidean embedding space.
