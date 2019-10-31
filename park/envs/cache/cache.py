import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import heapq

from park import core, logger
from park.param import config
from park.utils import seeding
from park.envs.cache.trace_loader import load_traces
import gym
from gym.spaces import Discrete
from gym.spaces import Box
import inspect

accept = 1
reject = 0


class TraceSrc(object):
    '''
    Tracesrc is the Trace Loader

    @param trace: The file name of the trace file
    @param cache_size: The fixed size of the whole cache
    @param load_trace: The list of trace data. The item could be gotten by using load_trace.iloc[self.idx]
    @param n_request: length of the trace
    @param min_values, max values: Used for the restricted of the value space
    @param req: The obj_time of the object
    '''

    def __init__(self, trace, cache_size):
        self.trace = trace
        self.cache_size = cache_size
        self.load_trace = load_traces(self.trace, self.cache_size, 0)
        self.n_request = len(self.load_trace)
        print('Number of requests:', self.n_request)
        self.cache_size = cache_size

        #Important: min_values and max_values are for the state [obj_size, cache_size_remain, last_requested_time]
        #These are different values than the ones in the trace
        #self.min_values = np.asarray([1, 0, 0])
        #self.max_values = np.asarray([self.cache_size, self.cache_size, max(self.load_trace[0])])
        self.min_values = np.asarray([0]*(self.cache_size+1))
        self.max_values = np.asarray([max(self.load_trace[0])]*(self.cache_size+1))
        self.req = 0

    def reset(self, random):
        if self.trace == 'test':
            self.load_trace = load_traces(self.trace, self.cache_size, random)
        print('Number of requests:', self.n_request)
        self.n_request = len(self.load_trace)
        #elf.min_values = np.asarray([1, 0, 0])
        #self.max_values = np.asarray([self.cache_size, self.cache_size, max(self.load_trace[0])])
        self.min_values = np.asarray([0]*(self.cache_size+1))
        self.max_values = np.asarray([max(self.load_trace[0])]*(self.cache_size+1))
        self.req = 0

    def step(self):
        #  Obs is: (obj_time, obj_id)
        #  print("req id in trace step:", self.req)''
        obs = self.load_trace.iloc[self.req].values
        obs[1] = obs[1] % 30
        #print(obs)
        self.req += 1
        done = self.req >= self.n_request
        return obs, done

    def next(self):
        obs = self.load_trace.iloc[self.req].values
        obs[1] = obs[1] % 30
        done = (self.req + 1) >= self.n_request
        return obs, done

    def trace_done(self):
        return done 

class CacheSim(object):
    def __init__(self, cache_size, policy, action_space, state_space):
        # invariant
        '''
        This is the simulater for the cache.
        @param cache_size
        @param policy: Not implement yet. Maybe we should instead put this part in the action
        @param action_space: The restriction for action_space. For the cache admission agent, it is [0, 1]: 0 is for reject and 1 is for admit
        @param req: It is the idx for the object requiration
        @param non_cache: It is the list for those requiration that aren't cached, have obj_id and last req time
        @param cache: It is the list for those requiration that are cached, have obj id and last req time
        @param count_ohr: ohr is (sigma hit) / req
        @param count_bhr: ohr is (sigma object_size * hit) / sigma object_size
        @param size_all: size_all is sigma object_size
        '''

        self.cache_size = cache_size
        self.policy = policy
        self.action_space = action_space
        self.observation_space = state_space
        self.req = 0
        #self.non_cache = defaultdict(int)
        #self.cache = defaultdict(int)  # requested items with caching
        #self.non_cache = {}
        self.cache = []
        self.cache_pq = []
        self.cache_remain = self.cache_size
        self.last_req_time_dict = {} #obj_id -> last_req
        #self.rec_dict = {} #obj_id -> freq
        self.position_map = {} #obj_id -> cache_position
        self.count_ohr = 0
        #self.count_bhr = 0
        self.size_all = 0

    def reset(self):
        self.req = 0
        #self.non_cache = defaultdict(int)
        #self.cache = defaultdict(int)
        self.cache = []
        #self.non_cache = {}
        self.cache_pq = []
        self.cache_remain = self.cache_size
        self.last_req_time_dict = {}
        #self.rec_dict = {}
        self.position_map = {}
        self.count_ohr = 0
        #self.count_bhr = 0
        self.size_all = 0

    def fill_cache(self, cache_size, unique_objects):
        self.cache = np.random.randint(1, high=unique_objects, size = cache_size)
        #print(self.cache)
        for idx, obj_id in enumerate(self.cache):
            self.position_map[obj_id] = idx
            self.last_req_time_dict[obj_id] = 0


    def step(self, action, obj):
        req = self.req
        # print(self.req)
        cache_size_online_remain = self.cache_remain
        discard_obj_if_admit = []
        obj_time, obj_id = obj[0], obj[1]

        # Initialize the last request time
        if obj_id in self.last_req_time_dict:
            self.last_req_time_dict[obj_id] = req
        else:
            self.last_req_time_dict[obj_id] = 0

        #Initialize object frequency
        #self.freq_dict[obj_id] = self.freq_dict.get(obj_id, 0) + 1

        cost = 0

        #If hit
        if obj_id in self.position_map:
            self.count_ohr += 1
            hit = 1
        else:
            #If not hit
            if action == 0:
                #Don't admit
                hit = 0
            else:
                #Admit and evict at position i
                evict_position = action - 1
                evict_obj_id = self.cache[evict_position]
                self.cache[evict_position] = obj_id
                del self.position_map[evict_obj_id]
                self.position_map[obj_id] = evict_position
                hit = 0


        ohr = float(self.count_ohr / (req + 1))
        reward = hit

        self.req += 1
        self.cache_remain = cache_size_online_remain

        
        #info = [self.count_bhr, self.size_all, float(float(self.count_bhr) / float(self.size_all))]
        return reward

    def next_hit(self, obj):
        #returns true if obj is hit
        return obj[1] in self.position_map

    def get_state(self, obj=[0, 0]):
        '''
        Return the state of the object,  recency of object for each object in cache
        '''
        obj_time, obj_id = obj[0], obj[1]
        state = []
        assert(obj_time >= 0)
        state.append(obj_time - self.last_req_time_dict.get(obj_id, 0))
        for obj_id in self.cache:
            #print(self.last_req_time_dict)
            #print(obj_time)
            assert(obj_time >= self.last_req_time_dict[obj_id])
            state.append(obj_time - self.last_req_time_dict[obj_id])

        return state

class CacheEnv(gym.Env):
    """
    Cache description.

    * STATE *
        The state is represented as a vector:
        [request object size, 
         cache remaining size, 
         time of last request to the same object]

    * ACTIONS *
    TODO: should be fixed here, there should be both 
        Whether the cache accept the incoming request, represented as an
        integer in [0, 1].

    * REWARD * (BHR)
        Cost of previous step (object size) * hit

    * REFERENCE *
    """
    def __init__(self, seed=42):
        self.seed(seed)
        #self.cache_size = config.cache_size
        self.cache_size = config.cache_size

        # load trace, attach initial online feature values
        self.src = TraceSrc(trace=config.cache_trace, cache_size=self.cache_size)

        # set up the state and action space
        #TODO: Modify the action space size to C+1
        self.action_space = Discrete(self.cache_size+1)
        self.observation_space = Box(self.src.min_values, \
                                      self.src.max_values, \
                                      dtype=np.float32)

        # cache simulator
        self.sim = CacheSim(cache_size=self.cache_size, \
                            policy='lru', \
                            action_space=self.action_space, \
                            state_space=self.observation_space)

        self.sim.fill_cache(self.cache_size, self.src.max_values[0])

        # reset environment (generate new jobs)
        #self.reset(1, 2)

    def reset(self, low=1, high=1001):
        print('env.reset called')
        #print('caller name:', inspect.stack()[1][3])
        new_trace = self.np_random.randint(low, high)
        print(new_trace)
        self.src.reset(new_trace)
        self.sim.reset()
        self.sim.fill_cache(self.cache_size, self.src.max_values[0])
        if config.cache_trace == 'test':
            print("New Env Start", new_trace)
        elif config.cache_trace == 'real':
            print("New Env Start Real")
        #print(self.sim.get_state())
        return self.sim.get_state()

    def seed(self, seed):
        self.np_random = seeding.np_random(seed)

    def step(self, action):
        # 0 <= action < num_servers
        global accept
        assert self.action_space.contains(action)
        state, done = self.src.step()
        reward = self.sim.step(action, state)
        obj, done = self.src.next()
        while self.sim.next_hit(obj):
            state, done = self.src.step()
            hit_reward = self.sim.step(accept, state)
            reward += hit_reward
            if done is True:
                break
            obj, done = self.src.next()

        obs = self.sim.get_state(obj)
        #print(obs)
        info = {}
        return obs, reward, done, info


    def render(self, mode='human', close=False):
        pass
