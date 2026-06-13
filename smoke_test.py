import importlib
import numpy as np
import pandas as pd
import tensorflow as tf
from model import FDSR
from utils import DataInput, build_sessions, evaluate, compute_coverage, compute_serendipity
from baselines import common
from baselines.run import REGISTRY

DATA = 'data/youku.csv'
N_USERS = 300
SL = 10
BS = 32
H = 64
TRAIN_BATCHES = 3


def load():
    data = pd.read_csv(DATA)
    data = data[data['user_id'] < N_USERS].copy()
    data['user_id'] = pd.factorize(data['user_id'])[0]
    data['item_id'] = pd.factorize(data['item_id'])[0]
    user_count = int(data['user_id'].max()) + 1
    item_count = int(data['item_id'].max()) + 1
    sessions = build_sessions(data, 'user_id', 'item_id', 'time', 'click', SL)
    sessions = sessions[:len(sessions) // BS * BS]
    cut = max(BS, len(sessions) // 5 // BS * BS)
    return sessions[cut:], sessions[:cut], user_count, item_count


def first_batches(sessions):
    out = []
    for n, (_, uij) in enumerate(DataInput(sessions, BS)):
        out.append(uij)
        if n + 1 >= TRAIN_BATCHES:
            break
    return out


def run_fdsr(train_set, test_set, uc, ic):
    tf.reset_default_graph()
    tf.set_random_seed(625)
    with tf.Session() as sess:
        model = FDSR(uc, ic, H, BS)
        sess.run(tf.global_variables_initializer())
        for uij in first_batches(train_set):
            model.train(sess, uij, 0.1)
        auc, hr, _ = evaluate(sess, model, test_set, BS)
        cov, div, gini = compute_coverage(sess, model, test_set, BS, ic)
        item_emb = model.get_item_emb(sess, list(range(ic)))
        ser = compute_serendipity(sess, model, test_set, BS, item_emb)
    return dict(auc=auc, hit_rate=hr, coverage=cov, gini=gini, serendipity=ser)


def run_baseline(model_cls, train_set, test_set, uc, ic):
    tf.reset_default_graph()
    tf.set_random_seed(625)
    with tf.Session() as sess:
        model = model_cls(uc, ic, H, BS)
        sess.run(tf.global_variables_initializer())
        for uij in first_batches(train_set):
            model.train(sess, uij, 0.1)
        return common.evaluate(sess, model, test_set, BS, ic)


def fmt(r):
    return ('auc=%.3f hr=%.3f cov=%.3f gini=%.3f ser=%.3f'
            % (r['auc'], r['hit_rate'], r['coverage'], r['gini'], r['serendipity']))


def main():
    train_set, test_set, uc, ic = load()
    print('users=%d items=%d train=%d test=%d' % (uc, ic, len(train_set), len(test_set)))
    rows = []

    try:
        rows.append(('FDSR', 'OK', fmt(run_fdsr(train_set, test_set, uc, ic))))
    except Exception as e:
        rows.append(('FDSR', 'FAIL', repr(e)[:200]))

    for name in sorted(REGISTRY):
        module_name, class_name, _ = REGISTRY[name]
        try:
            model_cls = getattr(importlib.import_module(module_name), class_name)
            rows.append((name, 'OK', fmt(run_baseline(model_cls, train_set, test_set, uc, ic))))
        except Exception as e:
            rows.append((name, 'FAIL', repr(e)[:200]))

    print('\n=== smoke results ===')
    for name, status, info in rows:
        print('%-16s %-4s %s' % (name, status, info))
    failed = [n for n, s, _ in rows if s == 'FAIL']
    print('\n%d/%d runnable' % (len(rows) - len(failed), len(rows)))
    if failed:
        print('failed: ' + ', '.join(failed))


if __name__ == '__main__':
    main()
