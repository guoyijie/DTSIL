import os
import time
import numpy as np
import os.path as osp
import tensorflow as tf
import logger
from collections import deque
from utils import EpisodeStats
import copy
from traj_buffer import Buffer

class Model(object):
    def __init__(self, policy, ob_space, loc_space, ac_space, nbatch_act, nbatch_train, nbatch_sl_train, nsteps, ent_coef, vf_coef, max_grad_norm, seq_len, seed):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, loc_space, ac_space, nbatch_act, 1, seq_len+1, reuse=False, seed=seed)
        train_model = policy(sess, ob_space, loc_space, ac_space, nbatch_train, nsteps, seq_len+1, reuse=True, seed=seed)
        sl_train_model = policy(sess, ob_space, loc_space, ac_space, nbatch_sl_train, seq_len, seq_len+1, reuse=True, seed=seed)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)
        def train(lr, cliprange, obs, locs, goals, returns, masks, actions, values, neglogpacs, states):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, train_model.Y:locs, train_model.G:goals, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            td_map[train_model.S] = states
            td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        sl_A = tf.placeholder(tf.int32, [None])
        sl_M = tf.placeholder(tf.float32, [None])
        sl_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sl_train_model.pi, labels=sl_A)*(1-sl_M))/tf.reduce_sum(1-sl_M)
        prediction = tf.cast(tf.argmax(sl_train_model.pi, axis=1), tf.int32)
        decision = tf.equal(prediction, sl_A)
        sl_acc = tf.reduce_sum(tf.cast(decision, tf.float32)*(1-sl_M))/tf.reduce_sum(1-sl_M)
        sl_grads = tf.gradients(sl_loss, params)
        if max_grad_norm is not None:
            sl_grads, _sl_grad_norm = tf.clip_by_global_norm(sl_grads, max_grad_norm)
        _sl_train = trainer.apply_gradients(list(zip(sl_grads, params)))

        def sl_train(lr, obs, locs, goals, masks, actions, loc_states):
            td_map = {sl_train_model.X:obs, sl_train_model.Y:locs, sl_train_model.G:goals, sl_A:actions, LR:lr, sl_train_model.S:loc_states, sl_train_model.M:masks, sl_M:masks}
            return sess.run(
                [prediction, sl_acc, sl_loss, _sl_train],
                td_map
            )[:-1]

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.sl_train = sl_train
        self.train_model = train_model
        self.sl_train_model = sl_train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)

class Runner(object):

    def __init__(self, env, model, nsteps, gamma, lam, next_n, seq_len, ext_coef, int_coef, replay_buffer, seed):
        self.rng = np.random.RandomState(seed)
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]        

        self.lam = lam
        self.gamma = gamma
        self.next_n = next_n
        self.seq_len = seq_len
        self.ext_coef = ext_coef
        self.int_coef = int_coef

        self.replay_buffer = replay_buffer
        self.rnn_dones = copy.deepcopy(self.dones)

        demo = [[0,0,0] for _ in range(seq_len+1)]
        self.demos = [demo for _ in range(self.env.num_envs)]
        self.lengths = [self.seq_len+1 for _ in range(self.env.num_envs)]
        self.goal_strings = [None for _ in range(self.env.num_envs)]
        self.seq_idx = [0 for _ in range(self.env.num_envs)]
        self.goals = []
        for i in range(self.env.num_envs):
            self.goals.append([x[:-1] for x in self.demos[i][self.seq_idx[i]:self.seq_idx[i]+self.seq_len+1]])

        self.locs = [[0 for x in self.goals[0][0]] for _ in range(self.env.num_envs)]
        self.episode_rewards = [0 for _ in range(self.env.num_envs)]
        self.episode_positive_rewards = [0 for _ in range(self.env.num_envs)]
        self.last_ckpts = [0 for _ in range(self.env.num_envs)]
        self.recent_success_ratio = deque([], maxlen=40)
        self.running_episodes = [[] for _ in range(self.env.num_envs)]
        self.recent_imitation_rewards = deque([], maxlen=40)
        self.explore_episodes_mode = [True for _ in range(self.env.num_envs)]

    def run(self, K, p):
        mb_obs, mb_locs, mb_goals, mb_raw_rewards, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_rnn_dones = [],[],[],[],[],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.locs, self.goals, self.states, self.rnn_dones)
            mb_obs.append(self.obs.copy())
            mb_locs.append(copy.deepcopy(self.locs))
            mb_goals.append(copy.deepcopy(self.goals))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            mb_rnn_dones.append(self.rnn_dones)
            obs, raw_rewards, self.dones, infos = self.env.step(actions)
            locs = [info['loc'] for info in infos]
            self.rnn_dones = copy.deepcopy(self.dones)
            rewards = np.zeros_like(raw_rewards, dtype=np.float32)
            for i in range(self.env.num_envs):
                self.running_episodes[i].append((self.locs[i], actions[i], self.episode_positive_rewards[i], copy.deepcopy(self.obs[i]), self.episode_rewards[i]))
                self.episode_rewards[i] += raw_rewards[i]
                self.episode_positive_rewards[i] += max(raw_rewards[i], 0)
                for j in range(min(self.last_ckpts[i]+self.next_n, self.lengths[i]-1), self.last_ckpts[i], -1):
                    if(np.max(np.abs(np.array(locs[i]+[self.episode_positive_rewards[i]])-np.array(self.demos[i][j])))==0):
                        self.last_ckpts[i] = j
                        rewards[i] = self.int_coef+raw_rewards[i]*self.ext_coef
                        break
                if self.last_ckpts[i]>=self.seq_idx[i]+self.seq_len:
                    if len(self.demos[i][self.last_ckpts[i]:]) >= self.seq_len+1:
                        self.seq_idx[i] = self.last_ckpts[i]
                        self.goals[i] = [x[:-1] for x in self.demos[i][self.seq_idx[i]:self.seq_idx[i]+self.seq_len+1]]
                        self.rnn_dones[i] = True
                self.locs[i] = locs[i]
                self.obs[i] = obs[i]
                if self.dones[i]:
                    self.running_episodes[i].append((self.locs[i], 4, self.episode_positive_rewards[i], copy.deepcopy(self.obs[i]), self.episode_rewards[i]))
                    self.replay_buffer.add_episode(copy.deepcopy(self.running_episodes[i]))
                    self.running_episodes[i] = []
                    if not self.explore_episodes_mode[i]:
                        self.recent_imitation_rewards.append(self.episode_rewards[i])
                    self.episode_rewards[i] = 0
                    self.episode_positive_rewards[i] = 0
                    self.locs[i] = [0 for x in self.goals[0][0]]
                    self.recent_success_ratio.append(self.last_ckpts[i]/(self.lengths[i]-1))
                    self.last_ckpts[i] = 0
                    tmp = self.rng.uniform()
                    self.explore_episodes_mode[i] = (tmp>=p)
                    if self.explore_episodes_mode[i]:
                        demo, length, goal_string = self.replay_buffer.sample_goals(1, self.seq_len)
                        self.demos[i] = demo[0]
                        self.lengths[i] = length[0]
                        self.goal_strings[i] = goal_string[0]
                    else:
                        self.demos[i], self.lengths[i], self.goal_strings[i] = self.replay_buffer.get_best_episode(K, self.seq_len)
                    self.seq_idx[i] = 0
                    self.goals[i] = [x[:-1] for x in self.demos[i][self.seq_idx[i]:self.seq_idx[i]+self.seq_len+1]]
            mb_raw_rewards.append(raw_rewards)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_locs = np.asarray(mb_locs, dtype=np.float32)
        mb_goals = np.asarray(mb_goals, dtype=np.float32)
        mb_raw_rewards = np.asarray(mb_raw_rewards, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_rnn_dones = np.asarray(mb_rnn_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.locs, self.goals, self.states, self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_locs, mb_goals, mb_raw_rewards, mb_rewards, mb_returns, mb_dones, mb_rnn_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states)


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def sl_train(model, replay_buffer, nslupdates, nenvs=32, seq_len=50, envsperbatch=8, lr=1e-4):
    if len(replay_buffer.replay_buffer) > nenvs:
        envinds = np.arange(nenvs)
        flatinds = np.arange(nenvs * seq_len).reshape(nenvs, seq_len)
        slloss = []
        slacc = []
        for _ in range(nslupdates):
            obs, locs, goals, masks, actions = replay_buffer.sample_sl_data(nenvs=nenvs, seq_len=seq_len)
            for start in range(0, nenvs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mbflatinds = flatinds[mbenvinds].ravel()
                slices = (arr[mbflatinds] for arr in (obs, locs, goals, masks, actions))
                mblocstates = np.zeros((envsperbatch, 128))
                prediction, acc, loss = model.sl_train(lr, *slices, mblocstates)
                slloss.append(loss)
                slacc.append(acc)
        return np.mean(slacc), np.mean(slloss)
    else:
        return np.float('nan'), np.float('nan')

def learn(seed, policy, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            nminibatches=4, noptepochs=4, cliprange=0.1,
            next_n=10, nslupdates=10, seq_len=10,
            ext_coef=1, int_coef=0.1, K=10):

    rng = np.random.RandomState(seed)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    loc_space = 2
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    nbatch_sl_train = nenvs * seq_len // nminibatches

    make_model = lambda : Model(policy=policy, ob_space=ob_space, loc_space=loc_space,
                    ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nbatch_sl_train=nbatch_sl_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, seq_len=seq_len, seed=seed)
    model = make_model()

    replay_buffer = Buffer(max_size=1000, seed=seed)
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, next_n=next_n, seq_len=seq_len, int_coef=int_coef, ext_coef=ext_coef, replay_buffer=replay_buffer, seed=seed)
    episode_raw_stats = EpisodeStats(nsteps, nenvs)
    episode_stats = EpisodeStats(nsteps, nenvs)
    tfirststart = time.time()
    nupdates = total_timesteps//nbatch
    sl_acc = 0
    p = 0
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        p = update*nbatch/(total_timesteps*0.875)
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        obs, locs, goals, raw_rewards, rewards, returns, masks, rnn_masks, actions, values, neglogpacs, states = runner.run(K, p)
        episode_raw_stats.feed(raw_rewards, masks)
        episode_stats.feed(rewards, masks)
        mblossvals = []
        assert nenvs % nminibatches == 0
        envsperbatch = nenvs // nminibatches
        envinds = np.arange(nenvs)
        flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
        envsperbatch = nbatch_train // nsteps
        for _ in range(noptepochs):
            rng.shuffle(envinds)
            for start in range(0, nenvs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mbflatinds = flatinds[mbenvinds].ravel()
                slices = (arr[mbflatinds] for arr in (obs, locs, goals, returns, rnn_masks, actions, values, neglogpacs))
                mbstates = states[mbenvinds]
                mblossvals.append(model.train(lr, cliprange, *slices, mbstates))

        if nslupdates > 0 and sl_acc < 0.75:
            sl_acc, sl_loss = sl_train(model, replay_buffer, nslupdates=nslupdates, seq_len=seq_len, nenvs=nenvs, envsperbatch=envsperbatch, lr=lr)
        elif nslupdates > 0:
            sl_acc, sl_loss = sl_train(model, replay_buffer, nslupdates=1, seq_len=seq_len, nenvs=nenvs, envsperbatch=envsperbatch, lr=lr)

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        logger.logkv("serial_timesteps", update*nsteps)
        logger.logkv("nupdates", update)
        logger.logkv("total_timesteps", update*nbatch)
        logger.logkv("fps", fps)
        logger.logkv('episode_raw_reward', episode_raw_stats.mean_reward())
        logger.logkv('imitation_episode_reward', np.mean(runner.recent_imitation_rewards))
        logger.logkv('episode_reward', episode_stats.mean_reward())
        logger.logkv('episode_success_ratio', np.mean(runner.recent_success_ratio))
        logger.logkv('time_elapsed', tnow - tfirststart)
        if nslupdates > 0:
            logger.logkv('sl_loss', sl_loss)
            logger.logkv('sl_acc', sl_acc)
        logger.logkv('replay_buffer_num', replay_buffer.num_episodes())
        logger.logkv('replay_buffer_best', replay_buffer.max_reward())
        if noptepochs > 0:
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
        logger.dumpkvs()
        print(logger.get_dir())
    env.close()
    return model


