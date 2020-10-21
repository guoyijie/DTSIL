from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import gym
import gym.spaces
import copy
from bs4 import BeautifulSoup

BLOCK=0
AGENT=1
APPLE=2
TREASURE=3
ROCK=4
DX = [0, 1, 0, -1, 0]
DY = [-1, 0, 1, 0, 0]

COLOR = [
        [44, 42, 60], # block
        [105, 105, 105], # agent
        [250, 128, 114], #apple
        [255, 255, 0], #treasure
        [25,25,112], #rock
        ]


def generate_maze(csv_file):
    raw_csv = np.genfromtxt(csv_file, delimiter=',')
    h = raw_csv.shape[0]
    w = raw_csv.shape[1]
    maze_tensor = np.zeros((h, w, len(COLOR)))
    obj_pos = [[] for _ in range(len(COLOR))]
    for y in range(raw_csv.shape[0]):
        for x in range(raw_csv.shape[1]):
            if raw_csv[y][x] > 0:
                obj_idx = int(raw_csv[y][x]-1)
                maze_tensor[y][x][obj_idx] = 1
                if obj_idx is not BLOCK:
                    obj_pos[obj_idx].append([y, x])

    return maze_tensor, obj_pos

def visualize_maze(maze, img_size=80):
    my = maze.shape[0]
    mx = maze.shape[1]
    colors = np.array(COLOR, np.uint8)
    num_channel = maze.shape[2]
    vis_maze = np.matmul(maze, colors[:num_channel])
    vis_maze = vis_maze.astype(np.uint8)
    for i in range(vis_maze.shape[0]):
        for j in range(vis_maze.shape[1]):
            if maze[i][j].sum() == 0.0:
                vis_maze[i][j][:] = int(255)
    image = Image.fromarray(vis_maze)
    return image.resize((int(float(img_size) * mx / my), img_size), Image.NEAREST)


def visualize_mazes(maze, img_size=80):
    if maze.ndim == 3:
        return visualize_maze(maze, img_size=img_size)
    elif maze.ndim == 4:
        n = maze.shape[0]
        size = maze.shape[1]
        dim = maze.shape[-1]
        concat_m = maze.transpose((1,0,2,3)).reshape((size, n * size, dim))
        return visualize_maze(concat_m, img_size=img_size)
    else:
        raise ValueError("maze should be 3d or 4d tensor")


class Maze(object):
    def __init__(self, csv_file, img_size):
        self.csv = csv_file
        self.img_size = img_size
        self.reset()

    def reset(self):
        self.maze, self.obj_pos = generate_maze(self.csv)
        self.h = self.maze.shape[0]
        self.w = self.maze.shape[1]

        self.agent_pos = self.obj_pos[AGENT][0]
        self.update_object_state()

    def is_reachable(self, y, x):
        if x < 0 or x >= self.w or y < 0 and y >= self.h:
            return False
        if self.maze[y][x][BLOCK] == 1:
            return False
        return True

    def move_agent(self, direction):
        y = self.agent_pos[0] + DY[direction]
        x = self.agent_pos[1] + DX[direction]
        if not self.is_reachable(y, x):
            return False
        self.maze[self.agent_pos[0]][self.agent_pos[1]][AGENT] = 0
        self.maze[y][x][AGENT] = 1
        self.obj_pos[AGENT][0] = [y,x]
        self.agent_pos = [y, x]
        return True


    def is_object_reached(self, obj_idx):
        if self.maze.shape[2] <= obj_idx:
            return -1, False
        for i, [y,x] in enumerate(self.obj_pos[obj_idx]):
            if (self.agent_pos[0]==y and self.agent_pos[1]==x):
                return i, self.maze[self.agent_pos[0]][self.agent_pos[1]][obj_idx]==1
        return -1, False

    def update_object_state(self):
        self.obj_state = []
        for [y, x] in self.obj_pos[TREASURE]:
            self.obj_state.append(self.maze[y][x][TREASURE])
        for [y, x] in self.obj_pos[APPLE]:
            self.obj_state.append(self.maze[y][x][APPLE])
        #for [y, x] in self.obj_pos[ROCK]:
        #    self.obj_state.append(self.maze[y][x][ROCK])

    def remove_object(self, y, x, obj_idx):
        removed = self.maze[y][x][obj_idx] == 1
        self.maze[y][x][obj_idx] = 0
        self.update_object_state()
        return removed

    def state(self):
        #x, y, apple, treasure, rock
        state = [float(self.agent_pos[1]/self.maze.shape[1]), float(self.agent_pos[0]/self.maze.shape[0])]
        return state + self.obj_state

    def visualize(self):
        return visualize_maze(self.maze, self.img_size)

class MazeEnv(object):
    def __init__(self, config=""):
        self.config = BeautifulSoup(config, "lxml")
        # map
        self.csv = self.config.maze["csv"]
        self.max_step = int(self.config.maze["time"])

        if 'type' not in self.config.maze:
            self.dynamic_type='deterministic'

        if self.config.maze.has_attr('type') and self.config.maze['type'] == 'random_initial':
            self.dynamic_type='random_initial'

        # reward
        self.rewards = [[] for _ in range(len(COLOR))]
        self.rewards[TREASURE] = float(self.config.reward["treasure"])
        self.rewards[APPLE] = float(self.config.reward["apple"])
        self.rewards[ROCK] = float(self.config.reward["rock"])
        # meta
        self.meta_remaining_time = self.config.meta["remaining_time"]

        self.img_size = 80
        self.action_space = gym.spaces.discrete.Discrete(5)
        self.reset()
        self.observation_space = gym.spaces.box.Box(0, 1, self.state().shape)

        self.reward_range = [self.rewards[ROCK],self.rewards[TREASURE]]
        self.metadata = {'remaining_time': self.meta_remaining_time}
        self.spec = None
 

    def state(self):
        return np.array(self.maze.state())

    def reset(self):
        self.t = 0
        self.terminated = False
        self.maze = Maze(self.csv, self.img_size)
        if self.dynamic_type=='random_initial':
            for _ in range(3):
                a = np.random.choice([1,2,4])
                self.step(a)
        self.t = 0
        self.terminated = False
        return self.state()

    def visualize(self):
        return self.maze.visualize()

    def visualize_trajectory(self, trajectory):
        w, h, _ = self.maze.maze.shape
        ob = self.maze.visualize()
        plt.imshow(ob)
        for i in range(len(trajectory)-1):
            x = trajectory[i][0] * self.img_size/w
            y = trajectory[i][1] * self.img_size/h
            dx = (trajectory[i+1][0]-trajectory[i][0]) * self.img_size/w
            dy = (trajectory[i+1][1]-trajectory[i][0]) * self.img_size/h
            plt.arrow(x, y, dx, dy)
        plt.show()
        plt.close()

    def step(self, act):
        assert self.action_space.contains(act), "invalid action: %d" % act
        assert not self.terminated, "episode is terminated"

        self.maze.move_agent(act)
        self.maze.update_object_state()
        reward = 0

        treasure_idx, treasure_reached = self.maze.is_object_reached(TREASURE)
        if treasure_reached:
            reward = self.rewards[TREASURE]
            self.maze.remove_object(self.maze.agent_pos[0], self.maze.agent_pos[1], TREASURE)
            self.terminated = True

        apple_idx, apple_reached = self.maze.is_object_reached(APPLE)
        if apple_reached:
            reward = self.rewards[APPLE]
            self.maze.remove_object(self.maze.agent_pos[0], self.maze.agent_pos[1], APPLE)

        rock_idx, rock_reached = self.maze.is_object_reached(ROCK)
        if rock_reached:
            reward = self.rewards[ROCK]
            self.maze.remove_object(self.maze.agent_pos[0], self.maze.agent_pos[1], ROCK)

        self.t += 1
        if self.t >= self.max_step:
            self.terminated = True

        return self.state(), reward, self.terminated, {'loc':list(self.state()[:2])}
