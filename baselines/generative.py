import tensorflow as tf
import numpy as np

MEMORY_WINDOW = 10
T_STEPS = 20


def beta_schedule(t_steps):
    # linear beta schedule, returns betas, alphas and cumulative alphas
    betas = np.linspace(1e-4, 0.02, t_steps).astype(np.float32)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas).astype(np.float32)
    return betas, alphas, alpha_bar


def timestep_embedding(t, dim):
    # sinusoidal embedding of the integer step t, t is [B] float
    half = dim // 2
    freq = tf.exp(-np.log(10000.0) * tf.range(half, dtype=tf.float32) / max(half - 1, 1))
    arg = tf.expand_dims(t, 1) * tf.expand_dims(freq, 0)
    emb = tf.concat([tf.sin(arg), tf.cos(arg)], axis=1)
    if dim % 2:
        emb = tf.pad(emb, [[0, 0], [0, 1]])
    return emb


def eps_theta(x_t, c, t_emb, hidden_size, name):
    # denoiser MLP predicting the noise from the noisy state, condition and step
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        h = tf.concat([x_t, c, t_emb], axis=1)
        h = tf.layers.dense(h, 2 * hidden_size, activation=tf.nn.relu, name='fc1')
        h = tf.layers.dense(h, hidden_size, activation=tf.nn.relu, name='fc2')
        eps = tf.layers.dense(h, hidden_size, name='out')
    return eps


class DreamRec(object):
    def __init__(self, user_count, item_count, hidden_size, batch_size,
                 drop_prob=0.1, guide_w=2.0, **kwargs):
        self.hidden_size = hidden_size
        self.i = tf.placeholder(tf.int32, [batch_size,])
        self.hist = tf.placeholder(tf.int32, [batch_size, MEMORY_WINDOW])
        self.label = tf.placeholder(tf.float32, [batch_size,])
        self.lr = tf.placeholder(tf.float64, [])
        self.u = tf.placeholder(tf.int32, [batch_size,])

        self.item_emb_w = tf.get_variable('item_emb_w', [item_count, hidden_size])
        h_emb = tf.nn.embedding_lookup(self.item_emb_w, self.hist)
        cand_emb = tf.nn.embedding_lookup(self.item_emb_w, self.i)

        # condition: mean-pooled history embedding
        c = tf.reduce_mean(h_emb, axis=1)

        betas, alphas, alpha_bar = beta_schedule(T_STEPS)
        ab = tf.constant(alpha_bar)

        # forward diffusion of the clicked candidate embedding
        x0 = cand_emb
        t = tf.random_uniform([batch_size], 0, T_STEPS, dtype=tf.int32)
        ab_t = tf.expand_dims(tf.gather(ab, t), 1)
        noise = tf.random_normal(tf.shape(x0))
        x_t = tf.sqrt(ab_t) * x0 + tf.sqrt(1.0 - ab_t) * noise

        # classifier-free guidance: drop the condition for a random subset
        keep = tf.cast(tf.random_uniform([batch_size, 1]) > drop_prob, tf.float32)
        c_train = keep * c
        t_emb = timestep_embedding(tf.cast(t, tf.float32), hidden_size)

        pred_noise = eps_theta(x_t, c_train, t_emb, hidden_size, 'denoiser')
        mask = tf.expand_dims(self.label, 1)
        sq = tf.reduce_sum(tf.square(pred_noise - noise) * mask, axis=1)
        self.loss = tf.reduce_sum(sq) / (tf.reduce_sum(self.label) + 1e-8)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # reverse sampling from Gaussian noise conditioned on c
        betas_t = tf.constant(betas)
        alphas_t = tf.constant(alphas)
        c_emb = tf.zeros_like(c)
        x = tf.random_normal([batch_size, hidden_size])
        for step in reversed(range(T_STEPS)):
            te = timestep_embedding(tf.fill([batch_size], float(step)), hidden_size)
            eps_c = eps_theta(x, c, te, hidden_size, 'denoiser')
            eps_u = eps_theta(x, c_emb, te, hidden_size, 'denoiser')
            eps = (1.0 + guide_w) * eps_c - guide_w * eps_u
            a = alphas_t[step]
            abar = ab[step]
            coef = (1.0 - a) / tf.sqrt(1.0 - abar)
            mean = (x - coef * eps) / tf.sqrt(a)
            if step > 0:
                z = tf.random_normal([batch_size, hidden_size])
                x = mean + tf.sqrt(betas_t[step]) * z
            else:
                x = mean
        p = x # [B, H] predicted target embedding

        self.score = -tf.norm(p - cand_emb, ord='euclidean', axis=1)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                self.u: uij[0], self.i: uij[1], self.hist: uij[3],
                self.label: uij[5], self.lr: lr})
        return loss

    def test(self, sess, uij):
        score = sess.run(self.score, feed_dict={
                self.u: uij[0], self.i: uij[1], self.hist: uij[3],
                self.label: uij[5]})
        return list(uij[5]), list(score), list(uij[0]), list(uij[1])

    def item_embeddings(self, sess):
        return sess.run(self.item_emb_w)


class DiffRec(object):
    def __init__(self, user_count, item_count, hidden_size, batch_size, **kwargs):
        self.hidden_size = hidden_size
        self.i = tf.placeholder(tf.int32, [batch_size,])
        self.hist = tf.placeholder(tf.int32, [batch_size, MEMORY_WINDOW])
        self.label = tf.placeholder(tf.float32, [batch_size,])
        self.lr = tf.placeholder(tf.float64, [])
        self.u = tf.placeholder(tf.int32, [batch_size,])

        self.item_emb_w = tf.get_variable('item_emb_w', [item_count, hidden_size])
        h_emb = tf.nn.embedding_lookup(self.item_emb_w, self.hist)
        cand_emb = tf.nn.embedding_lookup(self.item_emb_w, self.i)

        c = tf.reduce_mean(h_emb, axis=1)

        betas, alphas, alpha_bar = beta_schedule(T_STEPS)
        ab = tf.constant(alpha_bar)

        # target preference embedding: history mean blended with clicked candidate
        mask = tf.expand_dims(self.label, 1)
        x0 = 0.5 * c + 0.5 * mask * cand_emb

        t = tf.random_uniform([batch_size], 0, T_STEPS, dtype=tf.int32)
        ab_t = tf.expand_dims(tf.gather(ab, t), 1)
        noise = tf.random_normal(tf.shape(x0))
        x_t = tf.sqrt(ab_t) * x0 + tf.sqrt(1.0 - ab_t) * noise

        t_emb = timestep_embedding(tf.cast(t, tf.float32), hidden_size)
        pred_noise = eps_theta(x_t, c, t_emb, hidden_size, 'denoiser')
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(pred_noise - noise), axis=1))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        betas_t = tf.constant(betas)
        alphas_t = tf.constant(alphas)
        x = tf.random_normal([batch_size, hidden_size])
        for step in reversed(range(T_STEPS)):
            te = timestep_embedding(tf.fill([batch_size], float(step)), hidden_size)
            eps = eps_theta(x, c, te, hidden_size, 'denoiser')
            a = alphas_t[step]
            abar = ab[step]
            coef = (1.0 - a) / tf.sqrt(1.0 - abar)
            mean = (x - coef * eps) / tf.sqrt(a)
            if step > 0:
                z = tf.random_normal([batch_size, hidden_size])
                x = mean + tf.sqrt(betas_t[step]) * z
            else:
                x = mean
        p = x # [B, H] preference embedding

        self.score = tf.reduce_sum(p * cand_emb, axis=1)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                self.u: uij[0], self.i: uij[1], self.hist: uij[3],
                self.label: uij[5], self.lr: lr})
        return loss

    def test(self, sess, uij):
        score = sess.run(self.score, feed_dict={
                self.u: uij[0], self.i: uij[1], self.hist: uij[3],
                self.label: uij[5]})
        return list(uij[5]), list(score), list(uij[0]), list(uij[1])

    def item_embeddings(self, sess):
        return sess.run(self.item_emb_w)


# TALLRec runs outside the TF pipeline: it wraps a pretrained causal LM through
# HuggingFace transformers and ignores the TF session. Requires transformers.
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    torch = None


class TALLRec(object):
    def __init__(self, user_count, item_count, hidden_size, batch_size,
                 base_model='gpt2', max_len=128, ft_steps=1, **kwargs):
        self.base_model = base_model
        self.max_len = max_len
        self.ft_steps = ft_steps
        self.tok = None
        self.model = None
        self.optimizer = None

    def _build(self):
        if torch is None:
            raise RuntimeError(
                'TALLRec needs PyTorch and transformers. '
                'Run `pip install torch transformers` and supply a base model '
                "(e.g. base_model='gpt2').")
        if self.model is not None:
            return
        self.tok = AutoTokenizer.from_pretrained(self.base_model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        self.yes_id = self.tok.encode(' Yes')[-1]
        self.no_id = self.tok.encode(' No')[-1]

    def _prompt(self, hist_row, cand):
        items = [str(int(x)) for x in hist_row if int(x) != 0]
        return ('The user interacted with items: ' + ', '.join(items) +
                '. Will the user interact with item ' + str(int(cand)) +
                '? Answer Yes or No. Answer:')

    def train(self, sess, uij, lr):
        self._build()
        hist, cand, click = uij[3], uij[1], uij[5]
        prompts, targets = [], []
        for k in range(len(cand)):
            ans = ' Yes' if float(click[k]) >= 0.5 else ' No'
            prompts.append(self._prompt(hist[k], cand[k]))
            targets.append(prompts[-1] + ans)

        enc = self.tok(targets, return_tensors='pt', padding=True,
                       truncation=True, max_length=self.max_len).to(self.device)
        labels = enc['input_ids'].clone()
        labels[enc['attention_mask'] == 0] = -100

        self.model.train()
        total = 0.0
        for _ in range(self.ft_steps):
            out = self.model(input_ids=enc['input_ids'],
                             attention_mask=enc['attention_mask'], labels=labels)
            self.optimizer.zero_grad()
            out.loss.backward()
            self.optimizer.step()
            total += float(out.loss.item())
        return total / max(self.ft_steps, 1)

    def test(self, sess, uij):
        self._build()
        hist, cand = uij[3], uij[1]
        prompts = [self._prompt(hist[k], cand[k]) for k in range(len(cand))]
        enc = self.tok(prompts, return_tensors='pt', padding=True,
                       truncation=True, max_length=self.max_len).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids=enc['input_ids'],
                                attention_mask=enc['attention_mask']).logits
        last = enc['attention_mask'].sum(dim=1) - 1
        rows = torch.arange(logits.size(0))
        next_logits = logits[rows, last] # [B, V] logits for the answer token
        pair = next_logits[:, [self.yes_id, self.no_id]]
        prob_yes = torch.softmax(pair, dim=1)[:, 0]
        scores = prob_yes.cpu().numpy().tolist()
        return list(uij[5]), scores, list(uij[0]), list(uij[1])

    def item_embeddings(self, sess):
        return None
