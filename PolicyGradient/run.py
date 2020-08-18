from PolicyGradient.model import ActorCritic
from PolicyGradient.trainer import EpisodeRunner, PolicyGradientTrainer
import torch.optim as optim
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
import webbrowser
import subprocess
import time

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
	logdir: str = field(metadata=dict(args=["-log-dir"]), default=None)


if __name__ == "__main__":
	parser = ArgumentParser(ParseOptions)
	print(parser.parse_args())
	args = parser.parse_args()
	actor_critic = ActorCritic(device=args.device, method=args.method, state_dim=args.state_dim,
							   hidden_dim=args.hidden_dim, policy_out_dim=args.policy_out_dim,
							   critic_out_dim=args.critic_out_dim)
	if args.optimizer == "adam":
		policy_optimizer = optim.Adam(actor_critic.policy.parameters(), lr = args.actor_learning_rate)
		if args.method.lower() in ("actor_critic", "critic_baseline"):
			critic_optimizer = optim.Adam(actor_critic.critic.parameters(), lr = args.critic_learning_rate)
		else:
			critic_optimizer = None
	else:
		raise NotImplementedError

	episode_runner = EpisodeRunner(environment_name=args.environment, max_episode_len=args.max_t,
									method= args.method, gamma=args.gamma, reward_to_go=True, device=args.device,
									critic_loss = args.critic_loss)

	pg_trainer = PolicyGradientTrainer(actor_critic=actor_critic, n_updates=args.n_updates, gamma=args.gamma,
									   method=args.method, batch_size=args.batch_size, policy_optimizer=policy_optimizer,
									   critic_optimizer=critic_optimizer, render_every=args.render_every,
									   buffer_stats_len=args.buffer_stats_len)

	time.sleep(1)
	subprocess.Popen(["tensorboard", "--logdir", args.log_dir])
	time.sleep(1)
	webbrowser.open("127.0.0.1:6006")
	pg_trainer.train(episode_runner=episode_runner, n_updates = args.n_updates, batch_size=args.batch_size)




