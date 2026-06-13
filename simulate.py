import sys, time, random, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from model import FDSR
from utils import DataInput, accuracy, recommendation, compute_gini
import warnings
warnings.filterwarnings('ignore')

# Long-term feedback-loop simulation (Section V). User responses are generated
# under an item-response-theory rating model. This code targets TensorFlow 1.4.
tf.logging.set_verbosity(tf.logging.ERROR)


def build_synthetic(num_user, num_item, time_range, seed):
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
    return data.sort_values('time')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--users', type=int, default=400)
    ap.add_argument('--items', type=int, default=400)
    ap.add_argument('--iterations', type=int, default=20)
    ap.add_argument('--time-range', type=int, default=40)
    ap.add_argument('--seed', type=int, default=625)
    args = ap.parse_args()

    batch_size = 20
    hidden_size = 128
    session_length = 10

    data = build_synthetic(args.users, args.items, args.time_range, args.seed)
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
