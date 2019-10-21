import multiprocessing
import multiprocessing.pool
from envs.aslaug_v1 import AslaugEnv
from policies.aslaug_policy_v0 import AslaugPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
import numpy as np
import random

hyperparams_ranges = {
    "nminibatches": [4, 8, 16, 32],
    "n_steps": [16, 32, 64, 128, 256, 512, 1024, 2048],
    "gamma": [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],
    "learning_rate": (1e-5, 1),
    "ent_coef": (0.00000001, 0.01),
    "cliprange": [0.1, 0.2, 0.3, 0.4],
    "noptepochs": [1, 5, 10, 20, 30, 50],
    "lam": [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]
}


n_individuals = 4
n_best = 2
n_die = 1
steps = int(10e3)
childhood = 0.5
generations = 100
p_mutation = 0.05
p_crossover = 0.25
r_crossover = [0, 0.5]


def create_envs(n_envs):
    return SubprocVecEnv([lambda: AslaugEnv() for i in range(n_envs)])


def create_model(**kwargs):
    return PPO2(env=create_envs(4), policy=AslaugPolicy, tensorboard_log="tb_logs_evo", cliprange_vf=-1, **kwargs)


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)


class Evaluater:
    def __init__(self, params, model_fn):
        self.model = model_fn(**params)

        self.rew_sum = 0.0
        self.rew_sum_n = 0
        self.c_step = 0
        self.print_counter = 0

    def eval(self, tot_steps, childhood):
        self.tot_steps = tot_steps
        self.childhood = childhood
        self.model.learn(tot_steps, callback=self.callback)
        if self.rew_sum_n == 0:
            print("Error, rewsum is zero")
            return -1e6
        self.avg_rew = self.rew_sum / self.rew_sum_n
        return self.avg_rew

    def callback(self, _locals, _globals):
        self.c_step += self.model.n_batch
        if self.print_counter*1000 <= self.c_step:
            self.print_counter += 1
            print("{}/{}".format(self.c_step, self.tot_steps))
        if self.c_step/self.tot_steps >= self.childhood:
            self.rew_sum += sum(_locals["self"].episode_reward)/len(_locals["self"].episode_reward)
            self.rew_sum_n += 1


def evaluate_params(params, model_fn, tot_steps, childhood):
    ev = Evaluater(params, model_fn)
    val = ev.eval(tot_steps, childhood)
    del ev
    return val


class Individual:
    def __init__(self, hyperparams_ranges, p_mutation=0.05, p_crossover=0.5, r_crossover=[0, 0.5], params=None):
        self.hyperparams_ranges = hyperparams_ranges
        self.p_crossover = p_crossover
        self.r_crossover = r_crossover
        self.p_mutation = p_mutation
        if params is None:
            self.params = {}
            for key in hyperparams_ranges:
                self.params[key] = self.select_key_randomly(key)
        else:
            self.params = params

    def mutate(self):
        for key in self.hyperparams_ranges:
            if np.random.random() <= self.p_mutation:
                self.params[key] = self.select_key_randomly(key)

    def select_key_randomly(self, key):
        ranges = self.hyperparams_ranges[key]
        if type(ranges) is tuple:
            return np.random.uniform(ranges[0], ranges[1])
        else:
            return random.sample(ranges, 1)[0]

    def pair(self, indiv):
        for key in self.hyperparams_ranges:
            if np.random.random() >= self.p_crossover:
                break
            ranges = self.hyperparams_ranges[key]
            mag = np.random.uniform(self.r_crossover[0], self.r_crossover[1])
            if type(ranges) is tuple:
                self.params[key] = mag*self.params[key] + (1-mag)*indiv.params[key]
            else:
                idx_1 = self.hyperparams_ranges[key].index(self.params[key])
                idx_2 = indiv.hyperparams_ranges[key].index(indiv.params[key])
                idx = int(round(mag*idx_1 + (1-mag)*idx_2))
                self.params[key] = self.hyperparams_ranges[key][idx]


for generation in range(generations):
    print("Starting generation {}.".format(generation))
    indiv = [Individual(hyperparams_ranges, p_mutation, p_crossover, r_crossover) for i in range(n_individuals)]
    args = [(ind.params, create_model, steps, childhood) for ind in indiv]
    print(args)
    with MyPool(n_individuals) as p:
        scores = p.starmap(evaluate_params, args)
        scores = np.array(scores)
    best_scores_idx = scores.argsort()[-n_best:][::-1]
    print("Best score until now: {}.".format(np.max(scores)))
    print("Best parameters: {}".format(indiv[np.argmax(scores)].params))
    # Pair
    for idx in best_scores_idx:
        idx2 = np.random.choice(best_scores_idx)
        indiv[idx].pair(indiv[idx2])

    # Kill
    worst_scores_idx = (-1*scores).argsort()[-n_die:][::-1]
    for idx in worst_scores_idx:
        idx2 = np.random.choice(best_scores_idx)
        indiv[idx] = Individual(indiv[idx].hyperparams_ranges, indiv[idx].p_mutation, indiv[idx].p_crossover, indiv[idx].r_crossover, params=indiv[idx2].params)

    # Mutate
    for ind in indiv:
        ind.mutate()
