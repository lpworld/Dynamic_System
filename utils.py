import numpy as np
import scipy.spatial
from collections import Counter

try:
    import yaml
except ImportError:
    yaml = None


def load_config(path):
    if yaml is None:
        raise ImportError('pyyaml is required to read configuration files')
    with open(path) as f:
        return yaml.safe_load(f)


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


def build_sessions(frame, user_col, item_col, time_col, click_col, session_length):
    # Slide a window of `session_length` interactions over each user's
    # chronologically ordered history to form (history, next) records.
    records = []
    for user in frame[user_col].unique():
        sub = frame[frame[user_col] == user].sort_values(time_col).reset_index(drop=True)
        items = list(sub[item_col])
        clicks = list(sub[click_col])
        if len(items) <= session_length + 1:
            continue
        for i in range(len(items) - session_length - 1):
            records.append((
                user,
                items[i+session_length-1],
                items[i+session_length],
                items[i:i+session_length],
                items[i+1:i+session_length+1],
                float(clicks[i+session_length]),
            ))
    return records


def time_stratified_folds(frame, time_col, n_folds):
    # Chronological expanding-window cross-validation: each test fold lies
    # strictly after the interactions used to train it.
    frame = frame.sort_values(time_col).reset_index(drop=True)
    bounds = np.linspace(0, len(frame), n_folds + 2, dtype=int)
    for f in range(n_folds):
        train = frame.iloc[:bounds[f+1]].copy()
        test = frame.iloc[bounds[f+1]:bounds[f+2]].copy()
        yield train, test


def _rank_auc(arr_auc):
    arr_auc = sorted(arr_auc, key=lambda d: d[2])
    auc, fp1, fp2, tp1, tp2 = 0.0, 0.0, 0.0, 0.0, 0.0
    for record in arr_auc:
        fp2 += record[0]
        tp2 += record[1]
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2
    if tp2 == 0.0 or fp2 == 0.0:
        return -0.5
    return 1.0 - auc / (2.0 * tp2 * fp2)


def _hit_at_k(records, k):
    ranked = sorted(records, key=lambda r: r[0])[:k]
    return 1.0 if any(pos for _, pos in ranked) else 0.0


def evaluate(sess, model, test_set, batch_size):
    # AUC over the labeled candidate set (ranked by closeness to the predicted
    # trajectory point) and Hit Rate@10 over users with at least one positive.
    arr_auc, by_user, embedding_table = [], {}, []
    for _, uij in DataInput(test_set, batch_size):
        label, distance, user, item, embedding = model.test(sess, uij)
        for k in range(len(distance)):
            embedding_table.append(embedding[k])
            pos = label[k] >= 0.5
            arr_auc.append([0 if pos else 1, 1 if pos else 0, -distance[k]])
            by_user.setdefault(user[k], []).append((distance[k], pos))
    auc = _rank_auc(arr_auc)
    hits = [_hit_at_k(v, 10) for v in by_user.values() if any(p for _, p in v)]
    hit_rate = float(np.mean(hits)) if hits else 0.0
    return auc, hit_rate, embedding_table


def compute_diversity(embedding):
    diversity = []
    for i in range(len(embedding)):
        for j in range(len(embedding)):
            diversity.append(np.linalg.norm(embedding[i] - embedding[j]))
    return np.mean(diversity) if diversity else 0.0


def compute_gini(gini_item_set):
    item_dict = dict(Counter(gini_item_set))
    items = sorted(item_dict.values())
    cum_wealths = np.cumsum(sorted(np.append(items, 0)))
    sum_wealths = cum_wealths[-1]
    xarray = np.array(range(0, len(cum_wealths))) / float(len(cum_wealths) - 1)
    yarray = cum_wealths / sum_wealths
    B = np.trapz(yarray, x=xarray)
    A = 0.5 - B
    return A / (A + B)


def compute_coverage(sess, model, test_set, batch_size, item_count):
    rec_item, rec_embedding = [], []
    for _, uij in DataInput(test_set, batch_size):
        label, distance, user, item, embedding = model.test(sess, uij)
        for k in range(len(distance)):
            if distance[k] < model.margin:
                rec_item.append(item[k])
                rec_embedding.append(embedding[k])
    diversity = compute_diversity(rec_embedding[:10])
    gini = compute_gini(rec_item)
    return len(set(rec_item)) / item_count, diversity / item_count, gini


def compute_serendipity(sess, model, test_set, batch_size, item_emb_table, quantile=0.5):
    # Serendipity (Chen et al., WWW'19): fraction of recommended items that are
    # both accepted by the user and unexpected w.r.t. their history. The
    # unexpectedness of an item is its minimum distance to the user's history
    # embeddings, thresholded at a global quantile of those distances.
    item_emb_table = np.asarray(item_emb_table)
    distances, accepted = [], []
    for _, uij in DataInput(test_set, batch_size):
        label, distance, user, item, embedding = model.test(sess, uij)
        hist = uij[3]
        for k in range(len(distance)):
            if distance[k] >= model.margin:
                continue
            hist_emb = item_emb_table[hist[k]]
            rec_emb = item_emb_table[item[k]]
            distances.append(float(np.min(np.linalg.norm(hist_emb - rec_emb, axis=1))))
            accepted.append(label[k] >= 0.5)
    if not distances:
        return 0.0
    threshold = np.percentile(distances, quantile * 100)
    serendipitous = sum(1 for d, a in zip(distances, accepted) if a and d > threshold)
    return serendipitous / len(distances)


def accuracy(sess, model, test_set, batch_size):
    correct, embedding_table = [], []
    for _, uij in DataInput(test_set, batch_size):
        label, distance, user, item, embedding = model.test(sess, uij)
        for k in range(len(distance)):
            embedding_table.append(embedding[k])
            pred = 1.0 if distance[k] < model.margin else 0.0
            correct.append(1 if pred == label[k] else 0)
    return np.mean(correct), embedding_table


def recommendation(embedding_table, item_emb_table, ru=None, ri=None):
    # Generate the next round of feedback from the current recommendations.
    # By default every recommended item is treated as accepted (label 1.0). When
    # the drifted traits ru (per user) and ri (per item) are supplied, the label
    # is instead drawn from the item-response model under the current, non-
    # stationary preferences and item popularity (the drift extension).
    newtrain_set, new_gini_set = [], []
    tree = scipy.spatial.KDTree(item_emb_table)
    for user, user_embedding in enumerate(embedding_table):
        _, i = tree.query(user_embedding, k=11)
        new_gini_set = new_gini_set + list(i)
        if ru is not None and ri is not None:
            r = ru[user % len(ru)] + ri[i[10]] + np.random.normal(0, 0.1)
            click = float(int(max(min(r, 1.0), 0.0) > 0.5))
        else:
            click = 1.0
        newtrain_set.append((user, i[9], i[10], list(i[0:10]), list(i[1:11]), click))
    return newtrain_set, new_gini_set
