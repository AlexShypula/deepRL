from model import ActorCritic
from trainer import EpisodeRunner, PolicyGradientTrainer
import torch.optim as optim
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser

@dataclass
class ParseOptions:
	n_updates: int = field(metadata=dict(args=["-n_updates"]))
	max_t: int = field(metadata=dict(args=["-max_t"]), default=200)
	buffer_stats_len: int = field(metadata=dict(args=["-replay_buf_size"]), default=100)
	batch_size: int = field(metadata=dict(args=["-batch_size"]), default=8)
	method: str = field(metadata=dict(args=["-method"]), default="actor_critic")
	actor_learning_rate: float = field(metadata=dict(args=["-lr"]), default=1e-3)
	critic_learning_rate: float = field(metadata=dict(args=["-lr"]), default=1e-2)
	gamma: float = field(metadata=dict(args=["-gamma", "--discount_factor"]), default=0.99)
	device: str = field(metadata=dict(args=["-device"]), default= "cpu")
	environment: str = field(metadata=dict(args=["-environment"]), default="CartPole-v0")
	critic_loss: str = field(metadata=dict(args=["-loss"]), default="huber")
	optimizer: str = field(metadata=dict(args=["-optimizer"]), default="adam")
	early_stop_score: float = field(metadata=dict(args=["-early_stop_score"]), default=float("inf"))
	network_params_to_sample: int = field(metadata=dict(args=["-params_to_sample"]), default=200)
	render_every: int = field(metadata=dict(args=["-print_every"]), default=50)
	state_dim: int = field(metadata=dict(args=["-state_dim"]), default=4)
	hidden_dim: int = field(metadata=dict(args=["-hidden_dim"]), default=32)
	policy_out_dim: int = field(metadata=dict(args=["-policy_out_dim"]), default=2)
	critic_out_dim: int = field(metadata=dict(args=["-critic_out_dim"]), default=1)


if __name__ == "__main__":
    parser = ArgumentParser(ParseOptions)
    print(parser.parse_args())
    args = parser.parse_args()
	'''device, state_dim = 4, hidden_dim = 32, policy_out_dim = 2, critic_out_dim = 1)'''
	actor_critic = ActorCritic(device=args.device, state_dim=args.state_dim, hidden_dim=args.hidden_dim,
							   policy_out_dim=args.policy_out_dim, critic_out_dim=args.critic_out_dim)
	episode_runnder = EpisodeRunner(environment_name=args.environment, max_episode_len=args.max_t,
									method= args.method, gamma=args.gamma, reward_to_go=True, device=args.device,
									critic_loss = args.critic_loss)
	'''class EpisodeRunner:
    def __init__(self, environment_name: str, max_episode_len: int, method: str, gamma: float, reward_to_go: bool,
                 device: str, critic_loss: nn.Module = None):'''

	'''PolicyGradientTrainer:
    def __init__(self, actor_critic: ActorCritic, n_updates: int, gamma: float, method: str, batch_size: int,
                 policy_optimizer: Optimizer, critic_optimizer: Optimizer = None, render_every: int = None,
                 parallelize_batch: bool = False, buffer_stats_len=100):'''


	# q_net_cartpole = DQN_MLP(hidden_dim=256)
	# optimizer_cartpole = optim.Adam(q_net_cartpole.parameters(),
	# 					   lr=1e-3,
	# 					   betas=(0.9, 0.999),
	# 					   eps=1e-08,
	# 					   weight_decay=0,
	# 					   amsgrad=False)
	#
	# trainer_cartpole = DQN_Trainer(q_net=q_net_cartpole,
	# 					  optimizer = optimizer,
	# 					  double_q_flag=False,
	# 					  max_episodes=1e9,
	# 					  max_steps=70000,
	# 					  learning_starts=2000,
	# 					  online_update_every=1,
	# 					  target_update_every=5000,
	# 					  percent_annealing=0.1,
	# 					  max_t=200,
	# 					  replay_buffer_size=10000,
	# 					  batch_size=32,
	# 					  gamma=0.99,
	# 					  epsilon_start=1.0,
	# 					  epsilon_end=0.01,
	# 					  update_polyak_flag=False,
	# 					  update_tgt_net_steps=5000, #TODO FIX THIS
	# 					  environmnet="CartPole-v0",
	# 					  loss="huber",
	# 					  early_stop_score=195,
	# 					  metrics_buffer_size=200,
	# 					  scores_buffer_size=100)
	# trainer_cartpole.train(print_every=100)
	q_net = DQN_CNN(hidden_dim=256, out_dim=6)

	optimizer = optim.Adam(q_net.parameters(),
						   lr=1e-4,
						   betas=(0.9, 0.999),
						   eps=1e-08,
						   weight_decay=0,
						   amsgrad=False)

	trainer = DQN_Trainer(q_net=q_net,
						  optimizer=optimizer,
						  double_q_flag=True,  # False,
						  max_episodes=1e9,
						  max_steps=1e7,
						  learning_starts=2000,  # 10000,
						  online_update_every=4,
						  target_update_every=5000,
						  percent_annealing=0.1,
						  max_t=10000,
						  replay_buffer_size=10000,
						  batch_size=32,
						  gamma=0.99,
						  epsilon_start=0.01,  # 1.0,
						  epsilon_end=0.01,
						  update_polyak_flag=False,
						  update_tgt_net_steps=5000,  # TODO FIX THIS
						  environmnet="Pong-v0",
						  loss="huber",
						  early_stop_score=2000,
						  metrics_buffer_size=100,
						  scores_buffer_size=100)

	trainer.train(print_every=5)
