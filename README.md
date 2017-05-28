Deep Q-network for game playing using PyGamePlayer.

### Requirements

python3, tensorflow-gpu, scipy, numpy, pygame


### Usage
```
usage: python3 src/main.py [-h] [--game GAME] [--data_path DATA_PATH]
               [--weight_save_path WEIGHT_SAVE_PATH]
               [--weight_restore_path WEIGHT_RESTORE_PATH]
               [--log_path LOG_PATH]
               [--weight_save_frequency WEIGHT_SAVE_FREQUENCY]
               [--log_frequency LOG_FREQUENCY]
               [--update_frequency UPDATE_FREQUENCY] [--batch_size BATCH_SIZE]
               [--learning_rate LEARNING_RATE]
               [--burn_in_duration BURN_IN_DURATION]
               [--exploration_duration EXPLORATION_DURATION]
               [--exploration_end_rate EXPLORATION_END_RATE]
               [--replay_max_size REPLAY_MAX_SIZE]
               [--discount_rate DISCOUNT_RATE] [--action_repeat ACTION_REPEAT]
               [--state_frames STATE_FRAMES] [--frame_height FRAME_HEIGHT]
               [--frame_width FRAME_WIDTH] [--dueling DUELING]
               [--pooling POOLING] [--training TRAINING]
```

optional arguments:
```
  -h, --help            show this help message and exit
  --game GAME           which game to play: ('pong', 'half_pong', 'tetris')
  --data_path DATA_PATH
                        path for data like images, logs, weights
  --weight_save_path WEIGHT_SAVE_PATH
                        path for weight saving
  --weight_restore_path WEIGHT_RESTORE_PATH
                        path for weights to restore
  --log_path LOG_PATH   path for log file
  --weight_save_frequency WEIGHT_SAVE_FREQUENCY
                        weight save frequency: int
  --log_frequency LOG_FREQUENCY
                        log frequency: int
  --update_frequency UPDATE_FREQUENCY
                        update frequency: int
  --batch_size BATCH_SIZE
                        batch size: int
  --learning_rate LEARNING_RATE
                        learning rate: float [0, 1]
  --burn_in_duration BURN_IN_DURATION
                        start training after this many frames: int
  --exploration_duration EXPLORATION_DURATION
                        number of updates to perform before exploration decay
                        is done: int
  --exploration_end_rate EXPLORATION_END_RATE
                        end exploration rate: float [0, 1]
  --replay_max_size REPLAY_MAX_SIZE
                        maximum replay memory size: int
  --discount_rate DISCOUNT_RATE
                        discount_rate: float [0, 1]
  --action_repeat ACTION_REPEAT
                        number of frame to repeat each action for: int
  --state_frames STATE_FRAMES
                        number of frames in each state: int
  --frame_height FRAME_HEIGHT
                        frame heigh: int
  --frame_width FRAME_WIDTH
                        frame width: int
  --dueling DUELING     use dueling architecture: bool
  --pooling POOLING     use pooling architecture: bool
  --training TRAINING   train network: bool
```
