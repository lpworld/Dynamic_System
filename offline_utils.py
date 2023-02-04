import scipy
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.metrics import mean_squared_error, mean_absolute_error

batch_size = 32
margin = 0.1

class DataInput:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        records = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
        self.i += 1
        user, item, last_item, hist, next_hist, click = [], [], [], [], [], []
        for record in records:
            user.append(record[0])
            item.append(record[1])
            last_item.append(record[2])
            hist.append(record[3])
            next_hist.append(record[4])
            click.append(record[5])
        return self.i, (user, item, last_item, hist, next_hist, click)

def evaluate(sess, model, test_set):
    arr, arr_auc, hit, embedding_table = [], [], [], []
    userid = list(set([x[0] for x in test_set]))
    for _, uij in DataInput(test_set, batch_size):
        distance, label, user, item, embedding = model.test(sess, uij)
        distance = [float(x < margin) for x in distance]
        for index in range(len(distance)):
            embedding_table.append(embedding[index])
            arr.append([label[index], distance[index], user[index], item[index]])
            if label[index] >= 0.5:
                arr_auc.append([0, 1, distance[index]])
            else:
                arr_auc.append([1, 0, distance[index]])

    arr_auc = sorted(arr_auc, key=lambda d:d[2])
    auc, fp1, fp2, tp1, tp2 = 0.0, 0.0, 0.0, 0.0, 0.0
    for record in arr_auc:
        fp2 += record[0] # noclick
        tp2 += record[1] # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2
    # if all nonclick or click, disgard
    threshold = len(arr_auc) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        auc = -0.5
    if tp2 * fp2 > 0.0:  # normal auc
        auc = (1.0 - auc / (2.0 * tp2 * fp2))

    for user in userid:
        arr_user = [x for x in arr if x[2]==user and x[1]==1]
        if len(arr_user) > 0:
            hit.append(sum([x[0] for x in arr_user])/len(arr_user))
    hit_rate = np.mean(hit)

    return auc, hit_rate, embedding_table

def update_target_graph(primary_network='Primary_DQN', target_network='Target_DQN'):
    # Get the parameters of our Primary Network
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, primary_network)
    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, target_network)
    op_holder = []
    # Update our target_network parameters with DQNNetwork parameters
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def recommendation(embedding_table, item_emb_table):
    newtrain_set, new_gini_set = [], []
    tree = scipy.spatial.KDTree(item_emb_table)
    for user, user_embedding in enumerate(embedding_table):
        d, i  = tree.query(user_embedding,k=11)
        new_gini_set = new_gini_set + list(i)
        newtrain_set.append((user, i[9], i[10], i[0:10], i[1:11], 1.0))
    return newtrain_set, new_gini_set

def compute_gini(gini_item_set):
    item_dict = dict(Counter(gini_item_set))
    items = list(item_dict.values())
    items.sort()
    cum_wealths = np.cumsum(sorted(np.append(items, 0)))
    sum_wealths = cum_wealths[-1]
    xarray = np.array(range(0, len(cum_wealths))) / np.float(len(cum_wealths)-1)
    yarray = cum_wealths / sum_wealths
    B = np.trapz(yarray, x=xarray)
    A = 0.5 - B
    return A / (A+B)

def compute_diversity(train_set):
    embedding = [x[1] for x in train_set]
    diversity = []
    for i in range(len(train_set)):
        for j in range(len(train_set)):
            diversity.append(np.linalg.norm(embedding[i] - embedding[j]))
    return np.mean(diversity)


def compute_coverage(sess, model, test_set, item_count):
    rec_item, rec_embedding = [], []
    for _, uij in DataInput(test_set, batch_size):
        distance, label, user, item, embedding = model.test(sess, uij)
        distance = [float(x < margin) for x in distance]
        for index in range(len(distance)):
            if distance[index] > 0.5:
                rec_item.append(item[index])
                rec_embedding.append(embedding[index])
    diversity = compute_diversity(rec_embedding[:10])
    gini = compute_gini(rec_item)
    return len(set(rec_item))/item_count, diversity/item_count, gini