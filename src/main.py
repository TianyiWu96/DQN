"""
Main file for running the deep Q-network.
"""

import os
from argparse import ArgumentParser, ArgumentTypeError
from src.players.pong_player import PongPlayer
from src.players.half_pong_player import HalfPongPlayer
from src.players.tetris_player import TetrisPlayer

def main():
    games = ("pong", "half_pong", "tetris")
    args = parse_args(games)

    # Make data directories.
    for path in (args.data_path, args.weight_save_path, args.log_path):
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
    
    players = {
        "pong": PongPlayer,
        "half_pong": HalfPongPlayer,
        "tetris": TetrisPlayer
    }

    player = players[args.game]

    # Construct player and start playing.
    player(weight_save_path=args.weight_save_path,
        weight_restore_path=args.weight_restore_path,
        log_path=args.log_path,
        weight_save_frequency=args.weight_save_frequency,
        log_frequency=args.log_frequency,
        update_frequency=args.update_frequency,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        burn_in_duration=args.burn_in_duration,
        exploration_duration=args.exploration_duration,
        exploration_end_rate=args.exploration_end_rate,
        replay_max_size=args.replay_max_size,
        discount_rate=args.discount_rate,
        action_repeat=args.action_repeat,
        state_frames=args.state_frames,
        frame_height=args.frame_height,
        frame_width=args.frame_width,
        dueling=args.dueling,
        pooling=args.pooling,
        training=args.training).start()

def parse_args(games):
    parser = ArgumentParser()
    parser.add_argument("--game",
        help="which game to play: " + str(games),
        default="pong")
    parser.add_argument("--data_path", 
        help="path for data like images, logs, weights", 
        default="./src/data/")
    parser.add_argument("--weight_save_path", 
        help="path for weight saving", 
        default="./src/data/model_weights/")
    parser.add_argument("--weight_restore_path", 
        help="path for weights to restore",
        default=False)
    parser.add_argument("--log_path",
        help="path for log file", 
        default="./src/data/log.txt")
    parser.add_argument("--weight_save_frequency", 
        help="weight save frequency: int",
        type=int,
        default=int(1e5))
    parser.add_argument("--log_frequency", 
        help="log frequency: int",
        type=int,
        default=int(1e3))
    parser.add_argument("--update_frequency", 
        help="update frequency: int",
        type=int,
        default=4)
    parser.add_argument("--batch_size", 
        help="batch size: int",
        type=int,
        default=32)
    parser.add_argument("--learning_rate", 
        help="learning rate: float [0, 1]",
        type=float,
        default=1e-6)
    parser.add_argument("--burn_in_duration", 
        help="start training after this many frames: int",
        type=int,
        default=int(5e4))
    parser.add_argument("--exploration_duration", 
        help="number of updates to perform before exploration decay is done: int",
        type=int,
        default=int(2e6))
    parser.add_argument("--exploration_end_rate", 
        help="end exploration rate: float [0, 1]",
        type=float,
        default=.05)
    parser.add_argument("--replay_max_size", 
        help="maximum replay memory size: int",
        type=int,
        default=int(1e5))
    parser.add_argument("--discount_rate", 
        help="discount_rate: float [0, 1]",
        type=float,
        default=.9)
    parser.add_argument("--action_repeat", 
        help="number of frame to repeat each action for: int",
        type=int,
        default=1)
    parser.add_argument("--state_frames", 
        help="number of frames in each state: int",
        type=int,
        default=4)
    parser.add_argument("--frame_height", 
        help="frame heigh: int",
        type=int,
        default=84)
    parser.add_argument("--frame_width", 
        help="frame width: int",
        type=int,
        default=84)
    parser.add_argument("--dueling", 
        help="use dueling architecture: bool",
        type=str2bool,
        default=False)
    parser.add_argument("--pooling", 
        help="use pooling architecture: bool",
        type=str2bool,
        default=False)
    parser.add_argument("--training", 
        help="train network: bool",
        type=str2bool,
        default=True)
    return parser.parse_args()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    main()
    