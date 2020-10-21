from collections import deque
import numpy as np
import logger
import copy
import math

class Buffer(object):
    def __init__(self, max_size, seed):
        self.rng = np.random.RandomState(seed)
        self.max_size = max_size
        self.replay_buffer = dict() 
        self.score = dict()
        self.best_episode_string = None

    def list2str(self, l):
        string = ''
        for i in l[:-1]:
            string += '%0.1f'%i
            string += ','
        string += '%02d'%l[-1]
        return string

    def str2list(self, s):
        return list(map(int, s.split(',')))
        
    def num_episodes(self):
        return len(self.replay_buffer.keys())

    def max_reward(self):
        if self.best_episode_string is None:
            return -1
        else:
            return self.replay_buffer[self.best_episode_string][-1][4]       

    def visitation_score(self, score):
        x = pow(1/(score+1e-6), 0.5)+1e-6
        return x


    def add_episode(self, episode):
        for i in range(2,len(episode)+1):
            string = self.list2str(episode[i-1][0]+[episode[i-1][2]])
            if string in self.replay_buffer:
                self.score[string] += 1
                if episode[i-1][4]>self.replay_buffer[string][-1][4] or (episode[i-1][4]==self.replay_buffer[string][-1][4] and i < len(self.replay_buffer[string])):
                    print('replace episode', string)
                    self.replay_buffer[string] = episode[:i]
            else:
                print('add episode', string)
                self.replay_buffer[string] = episode[:i]
                self.score[string] = 1
                if episode[i-1][4] >= self.max_reward():
                    self.best_episode_string = string
        assert(list(self.replay_buffer.keys())==list(self.score.keys()))

    def sample_goals(self, nenv, seq_len):
        p = np.array([self.visitation_score(x) for k, x in list(self.score.items())])
        p = p/sum(p)
        goal_idx = self.rng.choice(list(self.score.keys()), size=nenv, p=p)
        goals = []
        lengths = []
        for i in goal_idx:
            goal = [x[0]+[x[2]] for x in self.replay_buffer[i]]
            lengths.append(len(goal))
            print('sample goals',i, 'length', len(goal))
            for _ in range(len(goal), seq_len*math.ceil(len(goal)/seq_len)+1):
                goal.append([0 for _ in goal[-1]])
            goals.append(goal)
            assert(i in self.score)
            assert(i in self.replay_buffer)
        return goals, lengths, goal_idx
    
    def get_best_episode(self, K, seq_len):
        values = [v[-1][2]*10000+len(v) for k,v in self.replay_buffer.items()]
        strings = [k for k,v in self.replay_buffer.items()]
        sorted_values = sorted(values, reverse=True)
        goal_strings = []
        for i in range(len(values)):
            if values[i] >= sorted_values[K] and values[i] >= 0.5*np.max(values):
                goal_strings.append(strings[i])
        goal_string = self.rng.choice(goal_strings)
        goal = [x[0]+[x[2]] for x in self.replay_buffer[goal_string]]
        length = len(goal)
        print('sample best episode', goal_string)
        for _ in range(len(goal), seq_len*math.ceil(len(goal)/seq_len)+1):
            goal.append([0 for _ in goal[-1]])
        return goal, length, goal_string
 
    def process_episode(self, episode, seq_len):
        ob = [x[3] for x in episode[:-1]]
        for _ in range(len(episode)-1, seq_len):
            ob.append(np.zeros_like(ob[-1]))
        loc =[x[0] for x in episode[:-1]]
        for _ in range(len(episode)-1, seq_len):
            loc.append([0 for _ in loc[-1]])
        goal_tmp = [x[0] for x in episode]
        for _ in range(len(episode), seq_len+1):
            goal_tmp.append([0 for _ in goal_tmp[-1]])
        goal = [goal_tmp for _ in range(seq_len)]

        mask = [0 for _ in range(len(episode)-1)]+[1 for _ in range(len(episode)-1,seq_len)]
        action = [x[1] for x in episode[:-1]]
        for _ in range(len(episode)-1, seq_len):
            action.append(4)
        return ob, loc, goal, mask, action

    def sample_sl_data(self, seq_len=2, nenvs=32):
        batch_size = (seq_len+1) * nenvs
        obs, locs, goals, masks, actions = [],[],[],[],[]
        for _ in range(nenvs):
            idx = self.rng.choice(list(self.replay_buffer.keys()), size=1)[0]
            demo = self.replay_buffer[idx]
            if len(demo) < seq_len+1:
                start = 0
            else:
                start = self.rng.choice(np.arange(0, len(demo)-seq_len))
            ob, loc, goal, mask, action = self.process_episode(demo[start:start+seq_len+1], seq_len)
            obs.extend(ob)
            locs.extend(loc)
            goals.extend(goal)
            masks.extend(mask)
            actions.extend(action)
        obs = np.array(obs)
        locs = np.array(locs)
        goals = np.array(goals)
        masks = np.array(masks)
        actions = np.array(actions) 
        return obs, locs, goals, masks, actions

    def save(self, save_path):
        np.save(save_path, {'replay_buffer':self.replay_buffer, 'score':self.score,
                            'best_episode_string':self.best_episode_string})

    def load(self, load_path):
        tmp = np.load(load_path, allow_pickle=True)
        tmp = tmp.item()
        self.replay_buffer = tmp['replay_buffer']
        self.score = tmp['score']
        self.best_episode_string = tmp['best_episode_string']
