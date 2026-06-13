import sys, time, random, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from model import FDSR
from utils import (load_config, DataInput, build_sessions, time_stratified_folds,
                   evaluate, compute_coverage, compute_serendipity)
import warnings
warnings.filterwarnings('ignore')

# Offline experiments (Tables II-IV). This code targets the TensorFlow 1.4 backend.
tf.logging.set_verbosity(tf.logging.ERROR)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def run_fold(cfg, train_df, test_df, user_count, item_count):
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
        model = FDSR(user_count, item_count, cfg['hidden_size'], bs,
                     mdn_components=cfg['mdn_components'], branch_num=cfg['branch_num'],
                     spectral_normalize=cfg['spectral_normalize'],
                     mdn_weight=cfg['mdn_weight'], margin=cfg['margin'])
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        start = time.time()
        for _ in range(cfg['epochs']):
            loss_sum = 0.0
            for _, uij in DataInput(train_set, bs):
                loss_sum += model.train(sess, uij, cfg['learning_rate'])
            model.global_epoch_step_op.eval()

        auc, hit_rate, _ = evaluate(sess, model, test_set, bs)
        coverage, diversity, gini = compute_coverage(sess, model, test_set, bs, item_count)
        item_emb_table = model.get_item_emb(sess, list(range(item_count)))
        serendipity = compute_serendipity(sess, model, test_set, bs, item_emb_table)
        return dict(auc=auc, hit_rate=hit_rate, coverage=coverage, diversity=diversity,
                    gini=gini, serendipity=serendipity, seconds=time.time() - start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg['seed'])

    data = pd.read_csv(cfg['data_path'])
    user_count = int(data['user_id'].max()) + 1
    item_count = int(data['item_id'].max()) + 1

    metrics = []
    for f, (train_df, test_df) in enumerate(time_stratified_folds(data, 'time', cfg['n_folds'])):
        res = run_fold(cfg, train_df, test_df, user_count, item_count)
        print('Fold %d  AUC %.4f  HR@10 %.4f  Coverage %.4f  Diversity %.4f  Gini %.4f  Serendipity %.4f'
              % (f, res['auc'], res['hit_rate'], res['coverage'], res['diversity'],
                 res['gini'], res['serendipity']))
        sys.stdout.flush()
        metrics.append(res)

    print('Mean across %d folds:' % cfg['n_folds'])
    for key in ['auc', 'hit_rate', 'coverage', 'diversity', 'gini', 'serendipity']:
        vals = [m[key] for m in metrics]
        print('  %-12s %.4f +/- %.4f' % (key, np.mean(vals), np.std(vals)))


if __name__ == '__main__':
    main()
