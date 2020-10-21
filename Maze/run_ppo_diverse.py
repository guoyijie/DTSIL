import sys
import logger
from subproc_vec_env import SubprocVecEnv
from policies import LocPolicy
import multiprocessing
import tensorflow as tf
import numpy as np
import glob

def train(env_id, num_timesteps, seed, num_env, gamma=0.99, ent_coef=0.01, nepochs=4, lr=2.5e-4, next_n=10, seq_len=10, nslupdates=10, K=1):

    ncpu = multiprocessing.cpu_count()
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True 
    tf.Session(config=config).__enter__()

    def make_env(rank):
        def _thunk():
            import maze
            env = maze.MazeEnv(config=open('config/'+env_id+'.xml'))
            return env
        return _thunk
    env = SubprocVecEnv([make_env(i) for i in range(num_env)])

    from ppo_diverse import learn
    policy = LocPolicy
    learn(policy=policy, env=env, nsteps=128, nminibatches=4,
        lam=0.95, gamma=gamma, noptepochs=nepochs,
        ent_coef=ent_coef,
        lr = lr,
        cliprange = 0.1,
        total_timesteps=int(num_timesteps * 1.1),
        next_n=next_n, seq_len=seq_len,
        nslupdates=nslupdates, 
        K=K, seed=seed)

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env-id', type=str, default='map')
    parser.add_argument('--num-env', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(40e6))
    parser.add_argument('--next-n', type=int, default=10)
    parser.add_argument('--nslupdates', type=int, default=10)
    parser.add_argument('--nepochs', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=10)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--log', type=str, default='result/tmp2')
    args = parser.parse_args()
    logger.configure(args.log)
    train(args.env_id, num_timesteps=args.num_timesteps, seed=args.seed,
        num_env=args.num_env, gamma=args.gamma, ent_coef=args.ent_coef,
        next_n=args.next_n, nslupdates=args.nslupdates,
        nepochs=args.nepochs, seq_len=args.seq_len, K=args.K)

if __name__ == '__main__':
    main()


