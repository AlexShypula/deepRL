import gym
gym.logger.set_level(40)
import numpy as np
from collections import deque

import torch
torch.manual_seed(42)
import torch.nn as nn
from tqdm import tqdm, trange
from copy import deepcopy
import random
from DQN.utils import Simulation, NetworkParamSampler, render_q_net
from torch.utils.tensorboard import SummaryWriter
from itertools import islice
from multiprocessing import Process, Event
from copy import deepcopy


class DQN_Trainer:
    def __init__(self, q_net,
                 optimizer,
                 double_q_flag=True,
                 max_episodes=4000,
                 max_steps = 50000,
                 learning_starts=2000,
                 online_update_every = 1,
                 target_update_every = 5000,
                 percent_annealing = 0.1,
                 max_t=200,
                 replay_buffer_size = 10000,
                 batch_size=16,
                 gamma = 0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 polyak_factor = 0.999,
                 update_polyak_flag=False,
                 update_tgt_net_steps = 10000,
                 device = "cpu",
                 environmnet = "CartPole-v0",
                 loss = "mse",
                 early_stop_score = float("inf"),
                 metrics_buffer_size=500,
                 scores_buffer_size=100,
                 network_params_to_sample = 200,
                 logdir = "tensorboard",
                 **kwargs
                 ):
        self.update_no = 0
        self.episode_no = 0
        self.q_net = q_net
        self.target_q_net = deepcopy(q_net)
        self.target_q_net.eval()
        self.optimizer = optimizer
        self.double_q_flag = double_q_flag
        self.learning_starts = learning_starts
        self.online_update_every = online_update_every
        self.target_update_every = target_update_every
        assert epsilon_start >= epsilon_end, "epsilon should not increase over time"
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.n_steps_annealing = percent_annealing * self.max_steps
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_step_size = (epsilon_start - epsilon_end) / self.n_steps_annealing
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        #assert metrics_buffer_size >= max_t, "metrics buffer must be larger or equal to max episode length"
        self.metrics_buffer_size = metrics_buffer_size
        self.scores_buffer_size = scores_buffer_size
        self.max_t = max_t
        self.polyak_factor = polyak_factor
        self.update_polyak_flag = update_polyak_flag
        self.update_tgt_net_steps = update_tgt_net_steps
        self.device = device
        self.early_stop_score = early_stop_score
        self.network_params_to_sample = network_params_to_sample
        self.end_proc_flag = None

        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        self.rewards_buffer = deque(maxlen=self.max_t)
        self.q_value_buffer = deque(maxlen=self.metrics_buffer_size)
        self.scores_buffer = deque(maxlen=self.scores_buffer_size)

        if loss == "mse":
            self.loss = nn.MSELoss()
        elif loss == "huber":
            self.loss = nn.SmoothL1Loss()
        elif loss == "mae":
            self.loss = nn.L1Loss()
        else:
            Exception(f"loss should be mse, huber, or mae, it is: {loss}")

        self.q_net_sampler = NetworkParamSampler(network=self.q_net,
                                                 params_to_sample=self.network_params_to_sample,
                                                 cache_size=self.metrics_buffer_size,
                                                 monitor_var=False)
        self.tgt_net_sampler = NetworkParamSampler(network=self.target_q_net,
                                                   params_to_sample=self.network_params_to_sample,
                                                   cache_size=self.metrics_buffer_size,
                                                   monitor_var=False)
        self.env = Simulation(environmnet)
        self.writer = SummaryWriter(logdir)

    def _update_q_net(self, batch_size = None, clip_grad = True):

        batch_size = batch_size if batch_size else self.batch_size
        if len(self.replay_buffer) < batch_size:
            batch_size = len(self.replay_buffer)
        self.q_net.train()
        sample = random.sample(self.replay_buffer, batch_size)

        states = torch.cat([observation.get("state") for observation in sample], dim = 0).to(self.device)
        next_states = torch.cat([observation.get("next_state") for observation in sample], dim = 0).to(self.device)
        actions = torch.tensor([observation.get("action") for observation in sample]).unsqueeze(1).to(self.device)
        rewards = torch.tensor([observation.get("reward") for observation in sample]).unsqueeze(1).float().to(self.device)

        is_finished_mask = (torch.tensor([observation.get("is_finished") for observation in sample])==False).byte().unsqueeze(1).to(self.device)

        q_fitted = self.q_net(states)
        # in double q learning use the online net to predict argmax and then use the target net to evaluate the value
        # see section double q learning on pg.2 here: https://arxiv.org/pdf/1509.06461.pdf
        if self.double_q_flag:
            self.q_net.eval()
            with torch.no_grad():
                argmax_locs = self.q_net(next_states).argmax(dim=1).unsqueeze(1)
                q_next_fitted = self.target_q_net(next_states)
                q_next_fitted = q_next_fitted.gather(dim=1, index = argmax_locs) # will be b x 1 tensor
            self.q_net.train()
        else:
            q_next_fitted = self.target_q_net(next_states).max(dim=1)[0].unsqueeze(1)
        # max returns a tuple of the max and the indices of the max values
        avg_q_val = q_fitted.max(dim=1)[0].mean().item()
        avg_tgt_val = q_next_fitted.mean().item()

        q_fitted_for_actions = q_fitted.gather(dim=1, index = actions)
        q_approx = rewards + self.gamma * (q_next_fitted * is_finished_mask) # mask off any is_finished

        loss = self.loss(q_fitted_for_actions, q_approx.detach())
        self.optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            for param in self.q_net.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self._update_target_q_net()
        self._step_epsilon()

        return loss.detach(), avg_q_val, avg_tgt_val

    def _step_epsilon(self):

        if self.update_no < self.n_steps_annealing:
            self.epsilon-=self.epsilon_step_size

    def _step_log(self, loss, q_value, tgt_value, is_finished):

        self.q_value_buffer.append(q_value)
        traling_q_val = np.mean(self.q_value_buffer)

        buffer_size = len(self.rewards_buffer)
        rewards_buf = islice(self.rewards_buffer, buffer_size - self.metrics_buffer_size, buffer_size)
        trailing_reward = np.mean(list(rewards_buf))

        self.q_net_sampler.sample(self.q_net)
        # self.tgt_net_sampler.sample(self.target_q_net)
        q_net_grad_var = self.q_net_sampler.get_running_var()
        # tgt_net_grad_var = self.tgt_net_sampler.get_running_var()

        self.writer.add_scalar('Train/loss', loss, self.update_no)
        self.writer.add_scalar('Train/q_value', q_value, self.update_no)
        self.writer.add_scalar('Train/tgt_value', tgt_value, self.update_no)

        self.writer.add_scalar('Train/trailing_q_value', traling_q_val, self.update_no)
        self.writer.add_scalar('Train/trailing_reward', trailing_reward, self.update_no)

        self.writer.add_scalar('Train/q_net_grad_var', q_net_grad_var, self.update_no)
        self.writer.add_scalar('Train/epsilon', self.epsilon, self.update_no)
        # self.writer.add_scalar('Train/tgt_net_grad_var', tgt_net_grad_var, self.update_no)

        self.update_no += 1
        if is_finished:
            self.episode_no += 1

    def _validation_episode(self):
        state = self.env.reset()
        R = 0
        with torch.no_grad():
            for i in range(1, self.max_t+1):
                self.env.render()
                action = self.q_net.act(state.to(self.device), epsilon=1e-9)
                state, reward, is_finished, _ = self.env.step(action)
                R+=reward
                if is_finished:
                    break
        return R

    def _process_episode(self, episode_length: int, print_every = 100):

        buffer_size = len(self.rewards_buffer)
        R = sum(islice(self.rewards_buffer, buffer_size-episode_length, buffer_size))
        self.scores_buffer.append(R)

        self.writer.add_scalar('Episode/score', R, self.episode_no)
        self.writer.add_scalar('Episode/trailing_score', np.mean(self.scores_buffer), self.episode_no)


        if self.episode_no % print_every == 0:
            R = self._validation_episode()
            self.writer.add_scalar('Validation/score', R, self.episode_no // print_every)
            print(f"Episode {self.episode_no}\t Validation Score: {R:.2f}")
            print(f"Episode {self.episode_no}\t Avg Score: {np.mean(self.scores_buffer):.2f}")


            # if self.end_proc_flag:
            #     self.end_proc_flag.set()
            #     self.p.join()
            # self.end_proc_flag = Event()
            # self.p = Process(target=render_q_net, args=(self.q_net, self.env, self.max_t, self.end_proc_flag))
            # self.p.start()


        if np.mean(self.scores_buffer) >= self.early_stop_score or self.episode_no > self.max_episodes:
            print(f"Environment solved in {self.episode_no - 1} with the avg score of last"
                  f"{self.scores_buffer_size} episodes as {np.mean(self.scores_buffer)}")
            return True
        else:
            return False

    def _update_target_q_net(self):
        if self.update_polyak_flag:
            for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(
                    (1 - self.polyak_factor) * param.data + target_param.data * (self.polyak_factor))

        else:
            if (self.update_no + 1) % self.target_update_every == 0:
                for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                    target_param.data.copy_(param.data)


    def train(self, print_every = 100):

        state = self.env.reset()
        t=1
        for step in trange(self.max_steps*self.online_update_every + self.learning_starts, position=0, smoothing=0):
            action = self.q_net.act(state.to(self.device), epsilon=self.epsilon)
            next_state, reward, is_finished, _ = self.env.step(action)
            self.rewards_buffer.append(reward)
            self.replay_buffer.append({"state": state,
                                       "action": action,
                                       "reward": reward, #if not is_finished else -reward,
                                       "next_state": next_state,
                                       "is_finished": is_finished})
            if step > self.learning_starts and step % self.online_update_every == 0:
                # update q_net, target_q_net, and epsilon
                loss, avg_q_val, avg_tgt_val = self._update_q_net()
                # logs to tensorboard and increments update_no, episode_no
                self._step_log(loss, avg_q_val, avg_tgt_val, is_finished)
                if is_finished or t > self.max_t:
                    early_stop = self._process_episode(episode_length=t, print_every=print_every)
                    if early_stop:
                        break

                if self.episode_no % 3 == 0:
                    self.env.render()

            if is_finished or t > self.max_t:
                #early_stop = self._process_episode(episode_length=t, print_every=print_every)
                state = self.env.reset()
                t = 1
            else:
                state = next_state
                t+=1

        self.env.close()





