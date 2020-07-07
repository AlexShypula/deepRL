import gym
from gym.wrappers import Monitor
gym.logger.set_level(40)
import numpy as np
from collections import deque
import torch
torch.manual_seed(42)
from DQN.atari_wrappers import wrap_deepmind
from time import sleep


class Simulation():
    def __init__(self, environment = "CartPole-v0", save_every = 5):
        env = gym.make(environment)
        self.env = Monitor(env, './video', video_callable=lambda episode_no: episode_no % save_every == 0, force=True)
        if environment == "Pong-v0":
            self.env = wrap_deepmind(env, frame_stack=True, scale = True)
        self.environment = environment
        #self.env.seed(0)
    def reset(self):
        observation = self.env.reset()
        if self.environment == "Pong-v0":
            observation = torch.from_numpy(np.stack(observation)).transpose_(0,2).transpose_(1,2).float().unsqueeze(0)
        else:
            observation = torch.from_numpy(observation).float().unsqueeze(0)
        return observation
    def step(self, action):
        observation, reward, is_done, info = self.env.step(action)
        if self.environment == "Pong-v0":
            observation = torch.from_numpy(np.stack(observation)).transpose_(0,2).transpose_(1,2).float().unsqueeze(0)
        else:
            observation = torch.from_numpy(observation).float().unsqueeze(0)
        return observation, reward, is_done, info
    def render(self):
        self.env.render()
    def close(self):
        self.env.close()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class NetworkParamSampler:
    def __init__(self, network, params_to_sample=200, cache_size=100, monitor_var=False):
        self.monitor_var = monitor_var
        if self.monitor_var:
            self.running_var_stats = []

        self.variance_deque = deque(maxlen=cache_size)

        self.n_params = count_parameters(network)
        self.sample_prob = params_to_sample / self.n_params
        self.sample_indices = []
        for param in network.parameters():
            indices = (torch.rand_like(param) <= self.sample_prob).nonzero()
            self.sample_indices.append(indices)

    def sample(self, network):
        param_sample = []
        for param, indices in zip(network.parameters(), self.sample_indices):
            if param.requires_grad:
                for i in indices:
                    param_sample.append(param.grad[tuple(i)].item())
        self.variance_deque.append(np.array(param_sample))
        if self.monitor_var and len(self.variance_deque) > 1:
            self.running_var_stats.append(self.get_running_var())

    def get_running_std(self):
        if len(self.variance_deque) > 1:
            stacked = np.stack(self.variance_deque, axis=0)
            std_arr = np.std(stacked, axis=0)
            return std_arr.mean()
        else:
            return 0

    def get_running_var(self):
        if len(self.variance_deque) > 1:
            stacked = np.stack(self.variance_deque, axis=0)
            var_arr = np.var(stacked, axis=0)
            return var_arr.mean()
        else:
            return 0


from multiprocessing import Process, Event
from time import sleep




def render_q_net(q_net, env, max_t, end_render_flag: Event):
    with torch.no_grad():
        q_net.eval()
        q_net.cpu()
        while not end_render_flag.is_set():
            state = env.reset()
            for _ in range(max_t):
                action = q_net.act(state.cpu(), epsilon=0)
                next_state, reward, is_finished, _ = env.step(action)
                env.render()
                if is_finished:
                    sleep(5)
                    break
        env.close()
#
# end_proc_flag = Event()
# p = Process(target=render_q_net, args=(self, end_proc_flag))
# p.start()
#
# end_proc_flag.set()
# p.join()


def printer(integer: int, end_proc_flag: Event):
    print("inside inner loop with: ", integer)
    while not end_proc_flag.is_set():
        print("repeating now on: ", integer)
        sleep(0.5)

#
# for i in range(5):
#     print("outer loop: ", i)
#     end_proc_flag = Event()
#     p = Process(target=printer, args=(i, end_proc_flag))
#     p.start()
#     sleep(1)
#     end_proc_flag.set()
#     p.join()
