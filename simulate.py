import sys, time, random, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from model import FDSR
from utils import DataInput, accuracy, recommendation, compute_gini
import warnings
warnings.filterwarnings('ignore')

# Long-term feedback-loop simulation (Section V). User responses are generated
# under an item-response-theory rating model. With --drift-std > 0 the model is
# extended to richer dynamics (Supplement III): the per-user trait random-walks
# and the per-item popularity follows a slow AR(1) drift across feedback loops,
# so user preferences and item popularity are non-stationary. This code targets
# TensorFlow 1.4.
tf.logging.set_verbosity(tf.logging.ERROR)


def build_synthetic(num_user, num_item, time_range, seed, return_traits=False):
    np.random.seed(seed)
    random.seed(seed)
    users, items, ratings = [], [], []
    for u in range(num_user):
        ru = np.random.normal(0.5, 1, 1)
        for i in range(num_item):
            ri = np.random.normal(0.5, 0.5, 1)
            eij = np.random.normal(0, 0.1, 1)
            r = ru + ri + eij
            ratings.append(int(max(min(r[0], 1), 0) > 0.5))
            users.append(u)
            items.append(i)
    times = np.random.randint(1, time_range, size=num_user * num_item)
    data = pd.DataFrame({'user_id': users, 'item_id': items, 'click': ratings, 'time': times})
    if return_traits:
        # Persistent per-user / per-item traits for the drift extension. Drawn
        # after the data above so the static (--drift-std 0) run is unchanged.
        ru_user = np.random.normal(0.5, 1.0, num_user)
        ri_item = np.random.normal(0.5, 0.5, num_item)
        return data.sort_values('time'), ru_user, ri_item
    return data.sort_values('time')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--users', type=int, default=400)
    ap.add_argument('--items', type=int, default=400)
    ap.add_argument('--iterations', type=int, default=20)
    ap.add_argument('--time-range', type=int, default=40)
    ap.add_argument('--seed', type=int, default=625)
    # Drift extension (Supplement III). 0.0 reproduces the original static run;
    # a modest value (e.g. 0.05) gives drifting prefs + non-stationary popularity.
    ap.add_argument('--drift-std', type=float, default=0.0,
                    help='per-loop random-walk std for user traits; '
                         'item popularity follows AR(1) with the same scale')
    args = ap.parse_args()

    batch_size = 20
    hidden_size = 128
    session_length = 10

    data, ru_user, ri_item = build_synthetic(args.users, args.items, args.time_range,
                                             args.seed, return_traits=True)
    validate = 4 * len(data) // 5
    train_data = data.iloc[:validate]
    test_data = data.iloc[validate:]

    train_set, test_set, gini_item_set = [], [], []
    for user in range(args.users):
        tu = train_data[train_data['user_id'] == user].reset_index(drop=True)
        items, clicks = list(tu['item_id']), list(tu['click'])
        if len(items) > session_length + 1:
            for i in range(len(items) - session_length - 1):
                train_set.append((user, items[i+session_length-1], items[i+session_length],
                                  items[i:i+session_length], items[i+1:i+session_length+1],
                                  float(clicks[i+session_length])))
    for user in range(args.users):
        su = list(test_data[test_data['user_id'] == user]['item_id'])
        if not su:
            continue
        su = random.choices(su, k=session_length + 1)
        gini_item_set += su
        test_set.append((user, su[session_length-1], su[session_length],
                         su[0:session_length], su[1:session_length+1], 1.0))
    train_set = train_set[:len(train_set) // batch_size * batch_size]
    test_set = test_set[:len(test_set) // batch_size * batch_size]
    i_table = list(range(args.items))

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = FDSR(args.users, args.items, hidden_size, batch_size)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        start = time.time()
        lr = 1.0
        for _ in range(args.iterations):
            random.shuffle(train_set)
            for _, uij in DataInput(train_set, batch_size):
                model.train(sess, uij, lr)
            item_emb_table = model.get_item_emb(sess, i_table)
            auc, _ = accuracy(sess, model, train_set, batch_size)
            _, embedding_table = accuracy(sess, model, test_set, batch_size)
            if args.drift_std > 0:
                # Non-stationary dynamics: random-walk the user traits and let
                # item popularity drift via AR(1), then label the new feedback
                # through the drifted item-response model rather than assuming
                # every recommendation is accepted.
                ru_user = ru_user + np.random.normal(0, args.drift_std, args.users)
                ri_item = 0.9 * ri_item + 0.1 * np.random.normal(0.5, 0.5, args.items)
                newtrain_set, new_gini_set = recommendation(
                    embedding_table, item_emb_table, ru_user, ri_item)
            else:
                newtrain_set, new_gini_set = recommendation(embedding_table, item_emb_table)
            train_set = train_set + newtrain_set
            gini_item_set = gini_item_set + new_gini_set
            gini = compute_gini(gini_item_set)
            print('Iter %d  AUC %.4f  Gini %.4f  (%.1fs)'
                  % (model.global_epoch_step.eval(), auc, gini, time.time() - start))
            sys.stdout.flush()
            model.global_epoch_step_op.eval()


if __name__ == '__main__':
    main()
