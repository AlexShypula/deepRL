from DQN.model import DQN_MLP
from DQN.trainer import DQN_Trainer
import torch.optim as optim
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser


@dataclass
class ParseOptions:
	n_episodes: int = field(metadata=dict(args=["-n_episodes"]), default=4000)
	max_t: int = field(metadata=dict(args=["-max_t"]), default=200)
	replay_buffer_size: int = field(metadata=dict(args=["-replay_buf_size"]), default=500)
	batch_size: int = field(metadata=dict(args=["-batch_size"]), default=16)
	learning_rate: float = field(metadata=dict(args=["-lr"]), default=1e-3)
	gamma: float = field(metadata=dict(args=["-gamma", "--discount_factor"]), default=0.99)
	epsilon_start: float = field(metadata=dict(args=["-epsilon_start"]), default=1.0)
	epsilon_end: float = field(metadata=dict(args=["-epsilon_end" ]), default=0.1)
	polyak_factor: float = field(metadata=dict(args=["-polyak_factor"]), default=0.999)
	device: str = field(metadata=dict(args=["-device"]), default= "cpu")
	environment: str = field(metadata=dict(args=["-environment"]), default="CartPole-v0")
	loss: str = field(metadata=dict(args=["-loss"]), default="mse")
	optimizer: str = field(metadata=dict(args=["-optimizer"]), default="adam")
	early_stop_score: float = field(metadata=dict(args=["-early_stop_score"]), default=float("inf"))
	metrics_buffer_size: int = field(metadata=dict(args=["-metrics_buf_size"]), default=500)
	scores_buffer_size: int = field(metadata=dict(args=["-scores_buf_size"]), default=100)
	network_params_to_sample: int = field(metadata=dict(args=["-params_to_sample"]), default=200)
	print_every: int = field(metadata=dict(args=["-print_every"]), default=100)

	state_dim: int = field(metadata=dict(args=["-state_dim"]), default=4)
	hidden_dim: int = field(metadata=dict(args=["-hidden_dim"]), default=32)
	out_dim: int = field(metadata=dict(args=["-out_dim"]), default=2)

def run_dqn_cartpole(state_dim, hidden_dim, out_dim, optimizer, learning_rate, print_every, **kwargs):
	q_net = DQN_MLP(state_dim = state_dim, hidden_dim = hidden_dim, out_dim = out_dim)
	optimizer = optim.Adam(q_net.parameters(),
						   lr=learning_rate,
						   betas=(0.9, 0.999),
						   eps=1e-08,
						   weight_decay=0,
						   amsgrad=False)
	trainer = DQN_Trainer(q_net = q_net, optimizer = optimizer, **kwargs)
	trainer.train(print_every=print_every)


if __name__ == "__main__":
    # parser = ArgumentParser(ParseOptions)
    # print(parser.parse_args())
    # args = parser.parse_args()
    # run_dqn_cartpole(**vars(args))
	q_net = DQN_MLP(hidden_dim=64)
	optimizer = optim.Adam(q_net.parameters(),
						   lr=1e-3,
						   betas=(0.9, 0.999),
						   eps=1e-08,
						   weight_decay=0,
						   amsgrad=False)

	trainer = DQN_Trainer(q_net=q_net,
						  polyak_factor=0.999,
						  batch_size=32,
						  replay_buffer_size=10000,
						  optimizer=optimizer,
						  metrics_buffer_size=400,
						  n_episodes_annealing=2000)
	trainer.train(print_every=100)
