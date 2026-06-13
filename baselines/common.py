import os, sys, time, random
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (DataInput, build_sessions, time_stratified_folds,
                   _rank_auc, compute_gini, compute_diversity)

# Shared evaluation harness for the baseline models. Every baseline exposes the
# same interface so all methods are trained and scored under one protocol:
#
#   __init__(user_count, item_count, hidden_size, batch_size, **kwargs)
#   train(sess, uij, lr) -> float                       # batch loss (0.0 if not applicable)
#   test(sess, uij)      -> (labels, scores, users, items)   # higher score = more relevant
#   item_embeddings(sess) -> ndarray[item_count, H] or None  # for diversity / serendipity
#
# uij is the batch tuple produced by DataInput:
#   (user, item, last_item, hist, next_hist, click)
#
# All models score and rank the same labeled candidate set, using the organic
# positive/negative labels in the data without any negative sampling.


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def evaluate(sess, model, test_set, batch_size, item_count, top_k=10):
    item_emb = model.item_embeddings(sess)
    if item_emb is not None:
        item_emb = np.asarray(item_emb)

    arr_auc, per_user = [], {}
    for _, uij in DataInput(test_set, batch_size):
        labels, scores, users, items = model.test(sess, uij)
        hist = uij[3]
        for k in range(len(scores)):
            pos = labels[k] >= 0.5
            arr_auc.append([0 if pos else 1, 1 if pos else 0, scores[k]])
            per_user.setdefault(users[k], []).append((scores[k], items[k], pos, hist[k]))
    auc = _rank_auc(arr_auc)

    hits, rec_items, rec_emb, seren_dist, seren_acc = [], [], [], [], []
    for cand in per_user.values():
        ranked = sorted(cand, key=lambda r: r[0], reverse=True)
        if any(p for _, _, p, _ in cand):
            hits.append(1.0 if any(p for _, _, p, _ in ranked[:top_k]) else 0.0)
        for score, item, pos, h in ranked[:top_k]:
            rec_items.append(item)
            if item_emb is not None:
                rec_emb.append(item_emb[item])
                seren_dist.append(float(np.min(np.linalg.norm(item_emb[h] - item_emb[item], axis=1))))
                seren_acc.append(pos)

    hit_rate = float(np.mean(hits)) if hits else 0.0
    coverage = len(set(rec_items)) / item_count
    gini = compute_gini(rec_items)
    diversity = compute_diversity(rec_emb[:10]) / item_count if rec_emb else 0.0
    serendipity = 0.0
    if seren_dist:
        threshold = np.percentile(seren_dist, 50)
        serendipity = sum(1 for d, a in zip(seren_dist, seren_acc) if a and d > threshold) / len(seren_dist)
    return dict(auc=auc, hit_rate=hit_rate, coverage=coverage, diversity=diversity,
                gini=gini, serendipity=serendipity)


def run_fold(model_cls, cfg, train_df, test_df, user_count, item_count, model_kwargs):
    tf.reset_default_graph()
    set_seed(cfg['seed'])
    sl, bs = cfg['session_length'], cfg['batch_size']

    train_set = build_sessions(train_df, 'user_id', 'item_id', 'time', 'click', sl)
    test_set = build_sessions(test_df, 'user_id', 'item_id', 'time', 'click', sl)
    random.shuffle(train_set)
    random.shuffle(test_set)
    train_set = train_set[:len(train_set) // bs * bs]
    test_set = test_set[:len(test_set) // bs * bs]

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = model_cls(user_count, item_count, cfg['hidden_size'], bs, **model_kwargs)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for _ in range(cfg['epochs']):
            for _, uij in DataInput(train_set, bs):
                model.train(sess, uij, cfg['learning_rate'])
        return evaluate(sess, model, test_set, bs, item_count)


def run(model_cls, cfg, **model_kwargs):
    set_seed(cfg['seed'])
    data = pd.read_csv(cfg['data_path'])
    user_count = int(data['user_id'].max()) + 1
    item_count = int(data['item_id'].max()) + 1

    metrics = []
    for f, (train_df, test_df) in enumerate(time_stratified_folds(data, 'time', cfg['n_folds'])):
        res = run_fold(model_cls, cfg, train_df, test_df, user_count, item_count, model_kwargs)
        print('Fold %d  AUC %.4f  HR@10 %.4f  Coverage %.4f  Diversity %.4f  Gini %.4f  Serendipity %.4f'
              % (f, res['auc'], res['hit_rate'], res['coverage'], res['diversity'],
                 res['gini'], res['serendipity']))
        sys.stdout.flush()
        metrics.append(res)

    print('Mean across %d folds:' % cfg['n_folds'])
    for key in ['auc', 'hit_rate', 'coverage', 'diversity', 'gini', 'serendipity']:
        vals = [m[key] for m in metrics]
        print('  %-12s %.4f +/- %.4f' % (key, np.mean(vals), np.std(vals)))
