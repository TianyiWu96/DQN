"""Interfaces for Deep Q-Network."""
from collections import deque
import random
import os
from DQN.qnet import QNet
import numpy as np
from scipy.misc import imresize
import tensorflow as tf


class DeepQLearner(object):
    """Provides wrapper around TensorFlow for Deep Q-Network."""
    def __init__(self, 
        actions, 
        weight_save_path,
        weight_restore_path,
        log_path,
        weight_save_frequency,
        update_frequency,
        log_frequency,
        batch_size,
        learning_rate,
        exploration_duration,
        exploration_end_rate,
        replay_start_size,
        replay_max_size,
        discount,
        action_repeat,
        state_frames,
        frame_height,
        frame_width,
        dueling,
        pooling,
        training):
        """Intializes the TensorFlow graph.

        Args:
            actions: List of viable actions learner can make. (Must be PyGame constants.)
            checkpoint_path: File path to store saved weights.
            save: If true, will save weights regularly.
            restore: If true, will restore weights right away from checkpoint_path.
        """

        # Initialize state variables.
        self.actions = actions
        self.net = QNet(len(actions), learning_rate)
        self.iteration = -1
        self.actions_taken = 0
        self.repeating_action_rewards = 0

        # Handle network save/restore.
        self.weight_save_path = weight_save_path
        self.weight_restore_path = weight_restore_path
        self.log_path = log_path
        if self.weight_restore_path: 
            try: self.__restore() 
            except: pass
        if self.log_path: 
            open(self.log_path, 'w').close()

        # Save training parameters.
        self.weight_save_frequency = weight_save_frequency
        self.update_frequency = update_frequency
        self.log_frequency = log_frequency
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.exploration_duration = exploration_duration
        self.exploration_rate = 1.
        self.exploration_end_rate = exploration_end_rate
        self.replay_start_size = replay_start_size
        self.replay_max_size = replay_max_size
        self.discount = discount
        self.action_repeat = action_repeat
        self.state_frames = state_frames
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.dueling = dueling
        self.pooling = pooling
        self.training = training

        # Store all previous transitions in a deque to allow for efficient
        # popping from the front and to allow for size management.
        #
        # Transitions are dictionaries of the form below.
        #     {
        #         'state_in': The Q-network input at this point in time.
        #         'action': The action index (indices) taken at this frame.
        #         'reward': The reward from this action.
        #         'terminal': True if the action led to a terminal state.
        #         'state_out': The state resulting from the given action.
        #     }
        self.transitions = deque(maxlen=self.replay_max_size)

    def __normalize_frame(self, frame):
        """Normalizes the screen array to be 84x84x1, with floating point values in
        the range [0, 1].

        Args:
            frame: The pixel values from the screen.

        Returns:
            An 84x84x1 floating point numpy array.
        """
        return np.reshape(
            np.mean(imresize(frame, (self.frame_height, self.frame_width)), axis=2),
            (self.frame_height, self.frame_width, 1))

    def __preprocess(self, frame):
        """Resize image, pool across color channels, and normalize pixels.

        Args:
            frame: The frame to process.

        Returns:
            The preprocessed frame.
        """
        proc_frame = self.__normalize_frame(frame)
        if not len(self.transitions):
            return np.repeat(proc_frame, self.state_frames, axis=2)
        else:
            return np.concatenate(
                (proc_frame, self.transitions[-1]['state_in'][:, :, -(self.state_frames-1):]),
                axis=2)

    def __remember_transition(self, pre_frame, action, terminal):
        """Returns the transition dictionary for the given data. Defer recording the
        reward and resulting state until they are observed.

        Args:
            pre_frame: The frame at the current time.
            action: The index of the action(s) taken at current time.
            terminal: True if the action at current time led to episode termination.
        """
        self.transitions.append({
            'state_in': pre_frame,
            'action': self.actions.index(action),
            'terminal': terminal
        })

    def __observe_result(self, resulting_state, reward):
        """Records the resulting state and reward from the previous action.
        Clips reward as necessary.

        Args:
            resulting_state: The (preprocessed) state resulting from the previous action.
            reward: The reward from the previous transition.
        """
        if not len(self.transitions):
            return
        self.transitions[-1]['reward'] = reward
        self.transitions[-1]['state_out'] = resulting_state

    def __is_burning_in(self):
        """Returns true if the network is still burning in (observing transitions)."""
        return len(self.transitions) < self.replay_start_size

    def do_explore(self):
        """Returns true if a random action should be taken, false otherwise.
        Decays the exploration rate if the final exploration frame has not been reached.
        """
        if not self.__is_burning_in() and self.exploration_rate > EXPLORATION_END_RATE:
            # TODO: This is an ugly fix. Find the source of this problem.
            self.exploration_rate = max(
                self.exploration_rate - self.exploration_reduction,
                EXPLORATION_END_RATE)
        return random.random() < self.exploration_rate or self.__is_burning_in()

    def __best_action(self, frame):
        """Returns the best action to perform.

        Args:
            frame: The current (preprocessed) frame.
        """
        return self.actions[np.argmax(self.net.compute_q(frame))]

    def __random_action(self):
        """Returns a random action to perform."""
        return self.actions[int(random.random() * len(self.actions))]

    def __compute_target_reward(self, trans):
        """Computes the target reward for the given transition.

        Args:
            trans: The transition for which to compute the target reward.

        Returns:
            The target reward.
        """
        target_reward = trans['reward']
        if not trans['terminal']:
            target_reward += DISCOUNT * np.amax(self.net.compute_q(trans['state_out']))
        return target_reward

    def step(self, frame, reward, terminal, score_ratio=None):
        """Steps the training algorithm given the current frame and previous reward.
        Assumes that the reward is a consequence of the previous action.

        Args:
            frame: Current game frame.
            reward: Reward value from previous action.
            terminal: True if the previous action was termnial.

        Returns:
            The next action to perform.
        """
        self.iteration += 1

        # Log if necessary.
        if self.iteration % self.log_frequency == 0:
            self.__log_status(score_ratio)

        # Repeat previous action for some number of iterations.
        # If we ARE repeating an action, we pretend that we did not see
        # this frame and just keep doing what we're doing.
        if self.iteration % self.action_repeat != 0:
            self.repeating_action_rewards += reward
            return [self.transitions[-1]['action']]

        # Observe the previous reward.
        proc_frame = self.__preprocess(frame)
        self.__observe_result(proc_frame, self.repeating_action_rewards)

        # Save network if necessary before updating.
        if self.weight_save_path and self.iteration % self.weight_save_frequency == 0 and self.iteration > 0:
            self.__save()

        # If not burning in, update the network.
        if not self.__is_burning_in() and self.actions_taken % self.update_frequency == 0 and len(self.transitions) >= self.replay_start_size:
            # Update network from the previous action.
            minibatch = random.sample(self.transitions, self.batch_size)
            batch_frames = [trans['state_in'] for trans in minibatch]
            batch_actions = [trans['action'] for trans in minibatch]
            batch_targets = [self.__compute_target_reward(trans) for trans in minibatch]
            self.net.update(batch_frames, batch_actions, batch_targets)

        # Select the next action.
        action = self.__random_action() if self.do_explore() else self.__best_action(proc_frame)
        self.actions_taken += 1

        # Remember the action and the input frames, reward to be observed later.
        self.__remember_transition(proc_frame, action, terminal)

        # Reset rewards counter for each group of 4 frames.
        self.repeating_action_rewards = 0

        return [action]

    def __log_status(self, score_ratio=None):
        """Print the current status of the Q-DQN."""
        print('        Iteration: %d' % self.iteration)

        if self.__is_burning_in() or len(self.transitions) < self.replay_max_size:
            print('        Replay capacity: %d (burn in %s)' % (len(self.transitions), 'not done' if self.__is_burning_in() else 'done'))

        if self.exploration_rate > self.exploration_end_rate and not self.__is_burning_in():
            print('        Exploration rate: %0.9f (%s annealing)' % (self.exploration_rate, 'not' if self.__is_burning_in() else 'still'))

        # If we're using the network, print a sample of the output.
        if not self.__is_burning_in():
            print('        Sample Q output:', self.net.compute_q(self.transitions[-1]['state_in']))

        if score_ratio:
            print('        Score ratio: %0.9f' % score_ratio)
            
        print('===============================================================')
        open(self.log_path, "a").write(str(score_ratio) + '\n')

    def __save(self):
        """Save the current network parameters in the checkpoint path.

        Args:
            checkpoint_path: Path to store checkpoint files.
            iteration: The current iteration of the algorithm.
        """
        if not os.path.exists(os.path.dirname(self.checkpoint_path)):
            os.makedirs(os.path.dirname(self.checkpoint_path))
        self.net.saver.save(self.net.sess, self.checkpoint_path, global_step=self.iteration)

    def __restore(self):
        """Restore the network from the checkpoint path.

        Args:
            checkpoint_path: Path from which to restore weights.
        """
        if not os.path.exists(self.checkpoint_path):
            raise Exception('No such checkpoint path %s!' % self.checkpoint_path)
        model_path = tf.train.get_checkpoint_state(self.checkpoint_path).model_checkpoint_path
        self.iteration = int(model_path[(model_path.rfind('-')+1):]) - 1
        # set exploration rate
        self.exploration_rate = max(EXPLORATION_END_RATE, EXPLORATION_START_RATE - self.exploration_reduction * self.iteration / 4)
        self.net.saver.restore(self.net.sess, model_path)
        print("Network weights, exploration rate, and iteration number restored!")
