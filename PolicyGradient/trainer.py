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
    def __init__(self, max_episode_len: int, method: str, gamma: float, reward_to_go: bool, device: str,
                 critic_loss: nn.Module = None):
        self.max_episode_len = max_episode_len
        self.method = method
        self.state_rewards = []
        self.state_log_probs = []
        if self.method in ("actor_critic" or "critic_baseline"):
            self.state_values = []
            self.value_states = True
        else:
            self.value_states = False
        self.gamma = gamma
        self.reward_to_go = reward_to_go
        self.device = device
        self.critic_loss = critic_loss

    def _reset_buffers(self):
        self.state_rewards = []
        self.state_log_probs = []
        if self.value_states:
            self.state_values = []


    def run_episode(self, actor_critic, env, baseline = None, return_grads = False):
        state = env.reset()
        for t in range(1,self.episode_len+1):
            state, is_finished = self._step(state, actor_critic, env)
            if is_finished:
                break
        policy_loss, critic_loss, score, avg_value = self._process_episode(baseline)
        if return_grads:
            policy_loss.backward()
            critic_loss.backward()
            grads = [param.grad for param in actor_critic.parameters()]
        else:
            grads = None
        return policy_loss, critic_loss, score, avg_value, grads


    def _step(self, state, actor_critic, env):
        # act
        action, log_prob = actor_critic.act()
        self.state_log_probs.append(log_prob)
        if self.value_states:
            # critique
            state_value = actor_critic.value(state)
            self.state_values.append(state_value)
        # step
        next_state, reward, is_finished, _ = env.step(action)
        self.state_rewards.append(reward)
        return next_state, is_finished

    def _process_episode(self, baseline = None):
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
            next_state_values = values[1:] + [0] # V(T) = 0]
            advantages = np.array(values) - (np.array(self.state_rewards) + self.gamma * np.array(next_state_values))
        elif self.method == "critic_baseline":
            values = [v.detach() for v in self.state_values]
            advantages = returns - np.array(values)
        else:
            baseline = baseline if baseline else 0
            advantages = returns - baseline
        # we multiply log probs by -1 because we're doing gradient descent in pytorch
        policy_loss = torch.cat([-log_prob * A for log_prob, A in
                                 zip(self.state_log_probs, advantages)]).sum().to(self.device)
        if self.value_states:
            avg_value = np.mean(values)
            critic_loss = self.critic_loss(values, returns)
        else:
            avg_value = None
            critic_loss = None

        self._reset_buffers()

        return policy_loss, critic_loss, score, avg_value

class PolicyGradientTrainer:
    def __init__(self, actor_critic: ActorCritic, policy_optimizer: Optimizer,
                 critic_optimizer: Optimizer , n_updates: int, gamma: float, method: str, batch_size: int,
                 parallelize_batch: bool = False, buffer_stats_len=100):
        self.buffer_stats_len = buffer_stats_len
        self.score_buffer = deque(maxlen=self.buffer_stats_len)

    def _train_batch(self, batch_size: int = None):
        batch_size = batch_size if batch_size else self.batch_size
        batch_trainer = self._parallel_train_batch if self.parallelize else self._iterative_train_batch
        batch_loss, batch_score = batch_trainer(batch_size)

    def _iterative_train_batch(self, batch_size: int):
        batch_loss = batch_score = 0
        for _ in range(batch_size):
            olicy_loss, critic_loss, score, avg_value = self.process_episode()
            batch_loss += loss
            batch_score += batch_score
        batch_loss/=batch_size
        batch_score/=batch_size
        batch_loss.batkward()
        self.actor_optimizer.step()
        if self.method in ("actor_critic", "baseline_critic"):
            self.critic_optimizer.step()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        return batch_loss.detach(), batch_score

    def _parallel_train_batch(self, batch_size: int):
        raise NotImplementedError

    def _log_batch(self, batch_loss, batch_score, update_no):

        trailing_score = np.mean(self.score_buffer) if len(self.score_buffer) > 0 else batch_score

        self.writer.add_scalar('Train/batch_loss', batch_loss, update_no)
        self.writer.add_scalar('Train/batch_score', batch_score, update_no)
        self.writer.add_scalar('Train/trailing_score', trailing_score, update_no)


        # self.writer.add_scalar('Train/tgt_net_grad_var', tgt_net_grad_var, self.update_no)

        self.update_no += 1
        if is_finished:
            self.episode_no += 1


    def train(self, n_updates: int = None, batch_size = None):
        n_updates = n_updates if n_updates else self.n_updates
        for update_no in trange(1, n_updates+1):
            self._train_batch(batch_size = batch_size)






#
# def reinforce(policy,
#               optimizer,
#               sampler,
#               n_episodes=4000,
#               max_t=1000,
#               gamma=1.0,
#               print_every=100,
#               delta=0.05,
#               baseline_flag=False,
#               norm=False,
#               divide_by_return=False,
#               causality=False):
#     print(f"baseline f {baseline_flag}, norm {norm}, divide {divide_by_return}, causality {causality}")
#     scores_deque = deque(maxlen=100)
#     E_sq = 1
#     running_baseline = 0
#     scores = []
#     plt_string = "PolicyGradient"
#     if baseline_flag:
#         plt_string = plt_string + " w/Baseline"
#     if norm:
#         plt_string = plt_string + " normalized by gradients"
#     if divide_by_return:
#         plt_string = plt_string + " normalized by reward"
#     for i_episode in range(1, n_episodes + 1):
#         saved_log_probs = []
#         rewards = []
#         state = env.reset()
#         for t in range(max_t):
#             action, log_prob = policy.act(state)
#             saved_log_probs.append(log_prob)
#             state, reward, is_finished, _ = env.step(action)
#             rewards.append(reward)
#             if is_finished:
#                 break
#         scores_deque.append(sum(rewards))
#         scores.append(sum(rewards))
#         discounts = [gamma ** i for i in range(len(rewards) + 1)]
#         rewards = [discount * reward for discount, reward in
#                    zip(discounts, rewards)]
#
#         if causality:
#             returns = np.cumsum(rewards[::-1])[::-1].copy()
#             avg_R = np.mean(rewards)
#         else:
#             avg_R = sum(rewards)
#             returns = [avg_R] * len(rewards)
#
#         if baseline_flag:
#             if norm:
#                 baseline = running_baseline / E_sq
#             else:
#                 baseline = running_baseline
#             returns = [R - baseline for R in returns]
#
#         if divide_by_return:
#             returns = [R / avg_R for R in returns]
#
#         policy_loss = torch.cat([-log_prob * R for
#                                  log_prob, R in zip(saved_log_probs, returns)]).sum().to(device)
#         optimizer.zero_grad()
#         policy_loss.backward()
#         optimizer.step()
#         sampler.sample_policy(policy)
#         if norm:
#             grad_sq = get_model_grad_sq(policy)
#             E_sq *= (1 - delta)
#             E_sq += delta * grad_sq
#         if baseline_flag:
#             running_baseline *= (1 - delta)
#             if norm:
#                 running_baseline += (delta * avg_R * grad_sq)
#             else:
#                 running_baseline += (delta * avg_R)
#         if i_episode % print_every == 0:
#             print(f"Episode {i_episode}\t Avg Score: {np.mean(scores_deque):.2f}")
#         if np.mean(scores_deque) >= 197.0:
#             print(f"Environment solved in {i_episode - 1} with the avg score as {np.mean(scores_deque)}")
#             break
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     plt.plot(np.arange(1, len(sampler.running_var_stats) + 1), sampler.running_var_stats)
#
#     plt.ylabel('Variance')
#     plt.xlabel('Episode #')
#     plt.title(f"Variance for {plt_string}")
#     plt.show()
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     plt.plot(np.arange(1, len(scores) + 1), scores)
#     plt.ylabel('Score')
#     plt.xlabel('Episode #')
#     plt.title(f"Score for {plt_string}")
#     plt.show()
#
#     return i_episode



#
# def reinforce(n_episodes=4000,
#               max_t=1000,
#               gamma=1.0,
#               print_every=100,
#               delta=0.05):
#     scores_deque = deque(maxlen=100)
#     #     rewards_dequeue = deque(maxlen = 100)
#     baseline = 0
#     scores = []
#     for i_episode in range(1, n_episodes + 1):
#         saved_log_probs = []
#         rewards = []
#         state = env.reset()
#         for t in range(max_t):
#             #             breakpoint()
#             action, log_prob = policy.act(state)
#             saved_log_probs.append(log_prob)
#             state, reward, is_finished, _ = env.step(action)
#             rewards.append(reward)
#             if is_finished:
#                 break
#         #         rewards_dequeue.append(rewards)
#         scores_deque.append(sum(rewards))
#         scores.append(sum(rewards))
#         #         if i_episode > 100:
#
#         #             avg_rewards = np.mean(pad_sequences(rewards_dequeue, padding = "post"),
#         #                                   axis = 0)
#         #             diff = len(rewards) - len(avg_rewards)
#         #             if diff > 0:
#         #                 avg_rewards = np.pad(avg_rewards, pad_width=(0,diff), mode = "constant")
#         #             rewards = [r-a for r, a in zip(avg_rewards, rewards)]
#
#         discounts = [gamma ** i for i in range(len(rewards) + 1)]
#         discounted_rewards = [discount * reward for discount, reward in
#                               zip(discounts, rewards)]
#         R = sum(discounted_rewards)
#         effective_R = R - baseline
#         policy_loss = []
#         #         for log_prob in saved_log_probs:
#         #             policy_loss.append(-log_prob * R)
#         #         policy_loss = torch.cat(policy_loss).sum()
#         policy_loss = torch.cat([-log_prob * effective_R for
#                                  log_prob in saved_log_probs]).sum().to(device)
#         optimizer.zero_grad()
#         policy_loss.backward()
#         optimizer.step()
#         baseline *= (1 - delta)
#         baseline += (delta * R)
#         if i_episode % print_every == 0:
#             print(f"Episode {i_episode}\t Avg Score: {np.mean(scores_deque):.2f}")
#         if np.mean(scores_deque) >= 197.0:
#             print(f"Environment solved in {i_episode - 1} with the avg score as {np.mean(scores_deque)}")
#             break
#     return scores


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





