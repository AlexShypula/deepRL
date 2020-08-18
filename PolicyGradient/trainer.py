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
import torch.nn as nn
from multiprocessing import Process, Event
from copy import deepcopy
import numpy as np
from torch.optim import Optimizer
from model import ActorCritic


class EpisodeRunner:
    def __init__(self, environment_name: str, max_episode_len: int, method: str, gamma: float, reward_to_go: bool,
                 device: str, critic_loss: nn.Module = None):
        self.max_episode_len = max_episode_len
        self.method = method
        self.state_rewards = []
        self.state_log_probs = []
        self.episode_entropy = 0
        if self.method in ("actor_critic" or "critic_baseline"):
            self.state_values = []
            self.value_states = True
        else:
            self.value_states = False
        self.gamma = gamma
        self.reward_to_go = reward_to_go
        self.device = device
        self.critic_loss = critic_loss
        self.env = Simulation(environment_name)

    def _reset_buffers(self):
        self.state_rewards = []
        self.state_log_probs = []
        if self.value_states:
            self.state_values = []
        self.episode_entropy = 0


    def run_episode(self, actor_critic, render_flag = False, baseline = None, return_grads = False):
        state = self.env.reset()
        for t in range(1,self.episode_len+1):
            state, is_finished = self._step(state, actor_critic)
            if render_flag: # render the environment to view / watch
                self.env.render()
            if is_finished:
                break
        policy_loss, critic_loss, score, avg_value, entropy = self._process_episode(actor_critic, state, is_finished,
                                                                                    t, beta_entropy,  baseline)
        if return_grads:
            grads, policy_loss, critic_loss = self._backprop_and_detach_loss(actor_critic, policy_loss, critic_loss)
        else:
            grads = None
        return policy_loss, critic_loss, score, avg_value, grads, entropy

    def _backprop_and_detach_loss(self, actor_critic, policy_loss, critic_loss):
        policy_loss.backward()
        policy_loss = policy_loss.detach().item()
        if self.value_states:
            critic_loss.backward()
            critic_loss = critic_loss.detach().item()
        grads = [param.grad for param in actor_critic.parameters()]
        return grads, policy_loss, critic_loss

    def _step(self, state, actor_critic):
        # act
        action, log_prob, entropy = actor_critic.act()
        self.state_log_probs.append(log_prob)
        self.episode_entropy+=entropy
        if self.value_states:
            # critique
            state_value = actor_critic.value(state)
            self.state_values.append(state_value)
        # step
        next_state, reward, is_finished, _ = self.env.step(action)
        self.state_rewards.append(reward)
        return next_state, is_finished

    def _process_episode(self, actor_critic, state, is_finished: bool, t: int, beta_entropy: float, baseline = None):
        discounts = [self.gamma ** i for i in range(len(self.state_rewards) + 1)]
        discounted_rewards = [discount * reward for discount, reward in zip(discounts, self.state_rewards)]
        if self.reward_to_go:
            score = sum(discounted_rewards)
            returns = np.cumsum(discounted_rewards[::-1])[::-1].copy() / np.array(discounts)
        else:
            score = sum(discounted_rewards)
            returns = [score] * len(discounted_rewards)
        # if actor critic use the
        if self.method == "actor_critic":
            values = [v.detach() for v in self.state_values]
            last_state = actor_critic(state).detach() if not is_finished else 0 # V(T) = 0 if finished
            next_state_values = values[1:] + [last_state]
            advantages = (np.array(self.state_rewards) + self.gamma * np.array(next_state_values)) - np.array(values)
        elif self.method == "critic_baseline":
            values = [v.detach() for v in self.state_values]
            advantages = returns - np.array(values)
        else:
            baseline = baseline if baseline else 0
            advantages = returns - baseline
        entropy = self.episode_entropy / t
        # we multiply log probs by -1 because we're doing gradient descent in pytorch

        policy_loss = torch.cat([-log_prob * A for log_prob, A in
                                 zip(self.state_log_probs, advantages)]).sum()
        # we subtract entropy because we want to maximize entropy or alternatively minimize negative entropy
        policy_loss-=(entropy*beta_entropy)
        if self.value_states:
            avg_value = np.mean(values)
            critic_loss = self.critic_loss(values, returns)
        else:
            avg_value = None
            critic_loss = None

        self._reset_buffers()

        return policy_loss.to(self.device), critic_loss.to(self.device), score, avg_value, entropy.detach().item()

class PolicyGradientTrainer:
    def __init__(self, actor_critic: ActorCritic, n_updates: int, gamma: float, method: str, batch_size: int,
                 policy_optimizer: Optimizer, critic_optimizer: Optimizer = None, render_every: int = None,
                 parallelize_batch: bool = False, buffer_stats_len=100):
        self.actor_critic = actor_critic
        self.policy_optimizer = policy_optimizer
        self.critic_optimizer = critic_optimizer
        self.render_every = render_every
        self.buffer_stats_len = buffer_stats_len
        self.score_buffer = deque(maxlen=self.buffer_stats_len)
        self.method = method
        if self.method in ("actor_critic" or "critic_baseline"):
            self.value_states = True
        else:
            self.value_states = False

    def _train_batch(self, episode_runner: EpisodeRunner, render_flag: bool, batch_size: int = None):
        batch_size = batch_size if batch_size else self.batch_size
        batch_trainer = self._parallel_train_batch if self.parallelize else self._iterative_train_batch
        return batch_trainer(episode_runner, render_flag, batch_size)

    def _iterative_train_batch(self, episode_runner: EpisodeRunner, render_flag: bool, batch_size: int):
        batch_critic_loss = batch_policy_loss = batch_entropy = batch_score = 0
        for i in range(batch_size):
            policy_loss, critic_loss, score, avg_value, entropy = episode_runner.run_episode(self.actor_critic, render_flag)
            batch_policy_loss+=policy_loss
            batch_critic_loss+=critic_loss if critic_loss else 0
            batch_score += score
            batch_entropy += entropy
        policy_loss/=batch_size
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()
        batch_policy_loss = batch_policy_loss.detach().item()
        if self.value_states:
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()
            batch_critic_loss = batch_critic_loss.detach().item()
        batch_score/=batch_size
        batch_entropy/=batch_size
        return batch_policy_loss, batch_critic_loss, batch_score, batch_entropy

    def _parallel_train_batch(self, episode_runner: EpisodeRunner, batch_size: int):
        raise NotImplementedError

    def _log_batch(self, policy_loss, critic_loss, batch_score, batch_entropy, update_no):

        if self.score_buffer:
            self.score_buffer.append(batch_score)
            trailing_score = np.mean(self.score_buffer)
            self.writer.add_scalar('Train/trailing_score', trailing_score, update_no)

        if self.value_states:
            self.writer.add_scalar('Train/critic_losss', critic_loss, update_no)

        self.writer.add_scalar('Train/policy_loss', policy_loss, update_no)
        self.writer.add_scalar('Train/batch_score', batch_score, update_no)
        self.writer.add_scalar('Train/batch_entropy', batch_entropy, update_no)


        # self.writer.add_scalar('Train/tgt_net_grad_var', tgt_net_grad_var, self.update_no)


    def train(self, eposode_runner: EpisodeRunner, n_updates: int = None, batch_size = None):
        n_updates = n_updates if n_updates else self.n_updates
        for update_no in trange(1, n_updates+1):
            render_flag = self.render_every != None and (update_no % self.render_every) == 0
            policy_loss, critic_loss, batch_score, batch_entropy = self._train_batch(episode_runner=eposode_runner,
                                                                      render_flag=render_flag, batch_size = batch_size)
            self._log_batch(policy_loss, critic_loss, batch_score, batch_entropy, update_no)



