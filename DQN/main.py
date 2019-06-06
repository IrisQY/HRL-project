from experiment import DQN, RLSVI
import argparse
# python main.py --mode DQN 
# python main.py --mode RLSVI --prior_variance 5
# python main.py --mode RLSVI --prior_variance 1

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='DQN')

parser.add_argument('--prior_variance', type=float, default=1.0)
parser.add_argument('--noise_variance', type=float, default=0.1)

parser.add_argument('--num_episodes', type=int, default=500)
parser.add_argument('--train_iterations', type=int, default=100)
parser.add_argument('--eval_iterations', type=int, default=10)


parser.add_argument('--learning_rate', type=float, default=0.00025)
parser.add_argument('--buffer_size', type=int, default=500000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_batches', type=int, default=150)
parser.add_argument('--starts_learning', type=int, default=5000)
parser.add_argument('--final_epsilon', type=float, default=0.1)
parser.add_argument('--discount', type=float, default=0.99)
parser.add_argument('--target_freq', type=int, default=10)

parser.add_argument('--print_every', type=int, default=5)

parser.add_argument('--num_ensemble', type=int, default=10)

config = parser.parse_args()

if config.mode == 'DQN':
    DQN(config)
elif config.mode == 'RLSVI':
	RLSVI(config)

