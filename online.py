import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import warnings
warnings.filterwarnings('ignore')
    
import sys, time, random
import tensorflow as tf
from simulation_model import Dynamic_System, DataInput, evaluate, recommendation, compute_gini
tf.logging.set_verbosity(tf.logging.ERROR)

### work with synthetic data
num_user = 400
num_item = 400
ratings, users, items = [], [], []
rating_matrix = np.zeros((num_user,num_item))
for u in range(num_user):
    ru  = np.random.normal(0.5,1,1)
    for i in range(num_item):
        ri = np.random.normal(0.5,0.5,1)
        eij = np.random.normal(0,0.1,1)
        r = ru+ri+eij
        rating = max(min(r[0],1),0)
        rating = int(rating>0.5)
        rating_matrix[u][i] = rating
        ratings.append(rating)
        users.append(u)
        items.append(i)
time_range = 40  ## split the ratings into 40 time range
times = np.random.randint(1,time_range,size=num_user*num_item)
data = pd.DataFrame({'user_id':users,'item_id':items,'click':ratings,'time':times})
data = data.sort_values(['time']) #simulate time records
validate = 4 * len(data) // 5
train_data = data.loc[:validate,]
test_data = data.loc[validate:,]

batch_size = 20
hidden_size = 128
train_set, test_set = [], []
gini_item_set = []

for user in range(num_user):
    train_user = train_data.loc[train_data['user_id']==user]
    length = len(train_user)
    train_user.index = range(length)
    if length > 11:
        for i in range(length-11):
            train_set.append((user, train_user.loc[i+9,'item_id'], train_user.loc[i+10,'item_id'], list(train_user.loc[i:i+9,'item_id']), list(train_user.loc[i+1:i+10,'item_id']), float(train_user.loc[i+10,'click'])))
for user in range(num_user):
    test_user = test_data.loc[test_data['user_id']==user]
    length = len(test_user)
    sub_item = list(test_user['item_id'])
    sub_item = random.choices(sub_item,k=11)
    gini_item_set = gini_item_set + sub_item
    test_set.append((user, sub_item[9], sub_item[10], sub_item[0:10], sub_item[1:11], 1.0))
train_set = train_set[:len(train_set)//batch_size*batch_size]
test_set = test_set[:len(test_set)//batch_size*batch_size]
i_table = list(range(num_item))

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    model = Dynamic_System(num_user, num_item, hidden_size, batch_size)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    sys.stdout.flush()
    lr = 1
    start_time = time.time()
    last_auc = 0.0
    
    for _ in range(20):
        item_embedding = []
        random.shuffle(train_set)
        epoch_size = round(len(train_set) / batch_size)
        for _, uij in DataInput(train_set, batch_size):
            loss = model.train(sess, uij, lr)    
        print('Epoch %d DONE\tCost time: %.2f' % (model.global_epoch_step.eval(), time.time()-start_time))
        item_emb_table = model.get_item_emb(sess, i_table)
        #Compute Gini Coefficients Here
        auc, _ = evaluate(sess, model, train_set, batch_size)
        _, embedding_table = evaluate(sess, model, test_set, batch_size)
        print('AUC: %.4f\t' % auc)
        #Update Training Set Here
        newtrain_set, new_gini_set = recommendation(embedding_table, item_emb_table)
        train_set = train_set + newtrain_set
        gini_item_set = gini_item_set + new_gini_set
        gini = compute_gini(gini_item_set)
        print('GINI: %.4f\t' % gini)
        sys.stdout.flush()
        model.global_epoch_step_op.eval()
