import random
import shelve
import requests
import gym
import time
import numpy as np

OUT_OF_BOUNDS = -1
GROUND = 0
WATER = 1
TREE = 2
GRASS = 3
RABBIT = 4
GRASS_CONSTANT = 0.03

MAX_POINTS = 100
MAX_TIMESTEPS = 1000
SERVER = 'http://192.168.1.160:5000/'

class World:

    def __init__(self, size=100, _map=None):
        if _map:
            self.map = _map
            self.size = len(self.map)
        else:
            self.size = size
            self.map = World.empty(self.size)
            self.generate()

        self.timestep = 0

    def __repr__(self):
        if not self.map: return 'Not generated'
        return '\n'.join([''.join([str(c) for c in row]) for row in self.map])

    def step(self):
        # generate all the grass
        self.generate_grass()
        self.timestep += 1

    def generate_grass(self):
        # add in the grass beside the grass
        # delete some grass
        for y in range(self.size):
            for x in range(self.size):
                if self.map[x][y] == GRASS:
                    if x > 0:
                        if self.map[x-1][y] == GROUND:
                            if random.random() < GRASS_CONSTANT:
                                self.map[x-1][y] = GRASS
                    if y > 0:
                        if self.map[x][y - 1] == GROUND:
                            if random.random() < GRASS_CONSTANT:
                                self.map[x][y - 1] = GRASS
                    if x < self.size - 1:
                        if self.map[x + 1][y] == GROUND:
                            if random.random() < GRASS_CONSTANT:
                                self.map[x + 1][y] = GRASS
                    if y < self.size - 1:
                        if self.map[x][y + 1] == GROUND:
                            if random.random() < GRASS_CONSTANT:
                                self.map[x][y + 1] = GRASS
                    if random.random() < GRASS_CONSTANT:
                        self.map[x][y] = GROUND
                    x += 1

    @staticmethod
    def empty(size):
        return [[GROUND for i in range(size)] for j in range(size)]

    def generate(self):
        self.map = World.empty(self.size)
        # Generate water
        for i in range(self.size):
            for j in range(self.size):
                if random.random() < 0.02:
                    for x in range(i-1, i+2):
                        if x < 0: continue
                        if x >= self.size: continue
                        for y in range(j-1, j+2):
                            if y < 0: continue
                            if y >= self.size: continue
                            self.map[x][y] = WATER

        # generate trees, generate grass
        for i in range(self.size):
            for j in range(self.size):
                if self.map[i][j] > 0: continue
                if random.random() < 0.002:
                    self.map[i][j] = TREE
                elif random.random() < 0.05:
                    self.map[i][j] = GRASS

    def save(self, filename='world-map'):
        d = shelve.open(filename) # open -- file may get suffix added by low-level
        d['map'] = self.map
        d.close()

    @classmethod
    def load(cls, filename='world-map'):
        d = shelve.open(filename)
        new_world = World(_map=d['map'])
        d.close()
        return new_world

    def find_empty_pos(self):
        while True:
            x = int(random.random() * self.size)
            y = int(random.random() * self.size)
            if self.map[y][x] == 0:
                return (x, y)

    def send(self, key, rabbit):
        response = requests.post(url=f'{SERVER}data', json={
            'state': self.map,
            'step': self.timestep,
            'key': key,
            'rabbit_pos': rabbit.pos
        })
        print(response)

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
EAT = 4
DRINK = 5

class Rabbit(gym.Env):

    def __init__(self, x=0, y=0):
        self.hunger = 0
        self.thirst = 0
        self.timestep = 0
        self.world = World()

        self.pos = dict(
            x=x,
            y=y
        )
        self.depth = 3
        # actions are:
        # move left/right/up/down/eat/drink
        self.action_space = gym.spaces.Discrete(5)
        # observations are what the rabbit sees in depth and its hunger
        self.low = np.array(
            [OUT_OF_BOUNDS] * ((self.depth * 2 + 1) ** 2 - 1) + [0]
        )
        # print(self.low)
        self.high = np.array(
            [RABBIT] * ((self.depth * 2 + 1) ** 2 - 1) + [MAX_POINTS]
        )
        # print(self.high)
        self.observation_space = gym.spaces.Box(self.low, self.high, dtype=np.float32)

    def __repr__(self):
        return f'Rabbit: thirst {self.thirst}, hunger {self.hunger}, position {self.pos["x"]},{self.pos["y"]}'

    def get_state(self):
        state = []
        for y in range(self.pos['y'] - self.depth, self.pos['y'] + self.depth + 1):
            state.append([])
            for x in range(self.pos['x'] - self.depth, self.pos['x'] + self.depth + 1):
                if x == self.pos['x'] and y == self.pos['y']: continue
                try:
                    state[-1].append(self.world.map[y][x])
                except Exception as e:
                    state[-1].append(OUT_OF_BOUNDS)
        return [c for row in state for c in row] + [
            self.hunger,
            # self.thirst
        ]

    def step(self, action):
        self.hunger += 1
        self.thirst += 1
        if action == UP:
            if self.pos['y'] != 0:
                if self.world.map[self.pos['y'] - 1][self.pos['x']] in [GROUND, GRASS]:
                    self.pos['y'] -= 1
        elif action == DOWN:
            if self.pos['y'] != self.world.size - 1:
                if self.world.map[self.pos['y'] + 1][self.pos['x']] in [GROUND, GRASS]:
                    self.pos['y'] += 1
        elif action == LEFT:
            if self.pos['x'] != 0:
                if self.world.map[self.pos['y']][self.pos['x'] - 1] in [GROUND, GRASS]:
                    self.pos['x'] -= 1
        elif action == RIGHT:
            if self.pos['x'] != self.world.size - 1:
                if self.world.map[self.pos['y']][self.pos['x'] + 1] in [GROUND, GRASS]:
                    self.pos['x'] += 1
        elif action == EAT:
            if self.world.map[self.pos['y']][self.pos['x']] == GRASS:
                self.hunger = 0

        # elif action == DRINK:
        #     if self.pos['y'] != 0 and self.world.map[self.pos['y'] - 1][self.pos['x']] == WATER:
        #         self.thirst = 0
        #     elif self.pos['y'] != self.world.size - 1 and self.world.map[self.pos['y'] + 1][self.pos['x']] == WATER:
        #         self.thirst = 0
        #     elif self.pos['x'] != 0 and self.world.map[self.pos['y']][self.pos['x'] - 1] == WATER:
        #         self.thirst = 0
        #     elif self.pos['x'] != self.world.size - 1 and self.world.map[self.pos['y']][self.pos['x'] + 1] == WATER:
        #         self.thirst = 0
        else:
            raise Exception("Not a valid action")



        state = self.get_state()
        reward = 1
        done = self.hunger >= MAX_POINTS or self.timestep >= MAX_TIMESTEPS # self.thirst >= MAX_POINTS or
        self.timestep += 1
        # update the world
        self.world.step()

        return state, reward, done, {}

    def render(self, mode='random'):
        self.world.send(mode, self)

    def reset(self):
        self.hunger = 0
        self.thirst = 0
        self.world = World()
        (x, y) = self.world.find_empty_pos()
        self.pos = dict(
            x=x,
            y=y
        )
        return self.get_state()

if __name__ == '__main__':
    import ray
    from ray import tune
    from ray.rllib.utils import try_import_tf
    from ray.tune import grid_search
    from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
    ray.init()

    ppo_config = DEFAULT_CONFIG.copy()
    ppo_config['num_gpus'] = 0
    ppo_config['tf_session_args']['device_count']['GPU'] = 0
    ppo_config['num_workers'] = 2
    ppo_config['num_sgd_iter'] = 30
    ppo_config['sgd_minibatch_size'] = 128
    ppo_config['model']['fcnet_hiddens'] = [32]

    rabbit = Rabbit()

    state = rabbit.reset()


    done = False
    cumulative_reward = 0

    i = 0
    while not done:
        action = rabbit.action_space.sample()
        state, reward, done, results = rabbit.step(action)
        cumulative_reward += reward
        if i < 100:
            time.sleep(0.1)
            rabbit.render()
        i += 1
        print('timestep: ', rabbit.timestep)

    print('Random: ', cumulative_reward)

    trainer = PPOTrainer(ppo_config, Rabbit);
    for i in range(20):
        print("Training iteration %d" % (i + 1))
        result = trainer.train()
        if i % 10 == 9:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)

    done = False
    cumulative_reward = 0
    state = rabbit.reset()

    i = 0
    print('Training...')
    while not done:
        action = trainer.compute_action(state)
        state, reward, done, results = rabbit.step(action)
        cumulative_reward += reward
        if i < 100:
            time.sleep(0.1)
            rabbit.render('trained')
        i += 1
        print('timestep: ', rabbit.timestep)

    print('Trained: ', cumulative_reward)
