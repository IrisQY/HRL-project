import os
from experiment import DQN, EnsembleDQN, BootDQN
import argparse
# python main.py --mode BootDQN --num_episodes 2000 --max_timesteps 1000
# python main.py --mode DQN --num_episodes 2000 --max_timesteps 1000
# python main.py --mode EnsembleDQN --num_episodes 2000 --max_timesteps 1000

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='BootDQN')

parser.add_argument('--number_seeds', type=int, default= 3)
parser.add_argument('--hidden_idx', type=int, default= 0)
parser.add_argument('--learning_rate', type=float, default=5e-4)
parser.add_argument('--buffer_size', type=int, default=50000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_batches', type=int, default=100)
parser.add_argument('--starts_learning', type=int, default=5000)
parser.add_argument('--final_epsilon', type=float, default=0.02)
parser.add_argument('--discount', type=float, default=0.99)
parser.add_argument('--target_freq', type=int, default=10)

parser.add_argument('--num_episodes', type=int, default=1000)
parser.add_argument('--max_timesteps', type=int, default=500)
parser.add_argument('--verbose', type=bool, default=True)
parser.add_argument('--print_every', type=int, default=50)

parser.add_argument('--num_ensemble', type=int, default=10)

config = parser.parse_args()

if config.mode == 'DQN':
    DQN(config)
elif config.mode == 'EnsembleDQN':
    EnsembleDQN(config)
elif config.mode == 'BootDQN':
    BootDQN(config)
