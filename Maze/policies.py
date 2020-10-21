import numpy as np
import tensorflow as tf
from utils import fc, batch_to_seq, seq_to_batch, ortho_init, cat_entropy
from distributions import make_pdtype

class LocPolicy(object):
    def __init__(self, sess, ob_space, loc_space, ac_space, nbatch, nsteps, max_timesteps, reuse=False, seed=0):
        nenv = nbatch // nsteps
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            G = tf.placeholder(tf.float32, [nbatch, max_timesteps, loc_space])
            X = tf.placeholder(tf.float32, (nbatch, )+ob_space.shape)
            Y = tf.placeholder(tf.float32, [nbatch, loc_space])
            M = tf.placeholder(tf.float32, [nbatch])
            S = tf.placeholder(tf.float32, [nenv, 128])
            ys = batch_to_seq(Y, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)

            tf.set_random_seed(seed)
            self.embed_W = tf.get_variable("embed_w", [loc_space, 64], initializer=ortho_init(1.0, seed))
            self.embed_b = tf.get_variable("embed_b", [64,])
            self.wa = tf.get_variable("wa", [128, 128], initializer=ortho_init(1.0, seed))
            self.wb = tf.get_variable("wb", [128,])
            self.ua = tf.get_variable("ua", [128, 128], initializer=ortho_init(1.0, seed))
            self.ub = tf.get_variable("ub", [128,])
            self.va = tf.get_variable("va", [128])
            self.rnn = tf.nn.rnn_cell.GRUCell(128, kernel_initializer=ortho_init(1.0, seed))
            enc_hidden = tf.zeros((nbatch, 128))
            embed_G = tf.matmul(tf.reshape(G, (-1, loc_space)),self.embed_W)+self.embed_b
            embed_G = tf.reshape(embed_G, (nbatch, max_timesteps, -1))
            enc_output, _ = tf.nn.dynamic_rnn(cell=self.rnn, inputs=embed_G, dtype=tf.float32)
            gs = batch_to_seq(enc_output, nenv, nsteps)
            dec_hidden = S
            h = []
            for idx, (y, m, g) in enumerate(zip(ys, ms, gs)):
                dec_hidden = dec_hidden*(1-m)
                embed_y = tf.matmul(y,self.embed_W)+self.embed_b
                dec_output, dec_hidden = tf.nn.dynamic_rnn(cell=self.rnn, inputs=tf.expand_dims(embed_y,axis=1), initial_state=dec_hidden)

                tmp = tf.reshape(tf.matmul(tf.reshape(g, (-1, 128)), self.ua)+self.ub,(nenv, max_timesteps, 128))
                tmp = tf.tanh(tf.expand_dims(tf.matmul(dec_hidden, self.wa)+self.wb,axis=1) + tmp)
                score = tf.reduce_sum(tmp*tf.expand_dims(tf.expand_dims(self.va, axis=0), axis=1), axis=2, keepdims=True)
                attention_weights = tf.nn.softmax(score, axis=1)
                context_vector = attention_weights * g
                context_vector = tf.reduce_sum(context_vector, axis=1)
                x = tf.concat([context_vector, dec_hidden], axis=-1)
                h.append(x)
            h = seq_to_batch(h)
            vf = fc(h, 'v', 1, seed=seed)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, seed=seed, init_scale=0.01)
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv,128))

        def step(ob, loc, goal, state, mask):
            a, v, state, neglogp = sess.run([a0, vf, dec_hidden, neglogp0], {X:ob, Y:loc, G:goal, M:mask, S:state})
            return a, v, state, neglogp

        def value(ob, loc, goal, state, mask):
            return sess.run(vf, {X:ob, Y:loc, G:goal, M:mask, S:state})

        self.G = G
        self.X = X
        self.Y = Y
        self.S = S
        self.M = M
        self.vf = vf
        self.step = step
        self.value = value


class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            X = tf.placeholder(shape=(nbatch,) + ob_space.shape, dtype=tf.float32)
            activ = tf.tanh
            processed_x = tf.layers.flatten(X)
            pi_h1 = activ(fc(processed_x, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            vf_h1 = activ(fc(processed_x, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(vf_h2, 'vf', 1)[:,0]

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model')

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        def neg_log_prob(actions):
            return self.pd.neglogp(actions)

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
        self.neg_log_prob = neg_log_prob
        self.entropy = self.pd.entropy()

    def save(self, save_path):
        ps = getsess().run(self.var_list)
        joblib.dump(ps, save_path)

    def load(self, load_path):
        loaded_params = joblib.load(load_path)
        restores = []
        for p, loaded_p in zip(self.var_list, loaded_params):
            restores.append(p.assign(loaded_p))
        getsess().run(restores)

