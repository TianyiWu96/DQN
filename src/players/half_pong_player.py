"""Implementation of PyGamePlayer for Half Pong."""
from src.players.pygame_player import PyGamePlayer
from pygame.constants import K_DOWN, K_UP, K_UNKNOWN
from src.qlearn import DeepQLearner

ACTIONS = [K_DOWN, K_UNKNOWN, K_UP]

class HalfPongPlayer(PyGamePlayer):
    """Simple implementation of PyGamePlayer for Half Pong."""
    def __init__(self, 
        weight_save_path,
        weight_restore_path,
        log_path,
        weight_save_frequency,
        log_frequency,
        update_frequency,
        batch_size,
        learning_rate,
        burn_in_duration,
        exploration_duration,
        exploration_end_rate,
        replay_max_size,
        discount_rate,
        action_repeat,
        state_frames,
        frame_height,
        frame_width,
        dueling,
        pooling,
        training, 
        force_game_fps=8, 
        run_real_time=False):

        super(HalfPongPlayer, self).__init__(
            force_game_fps=force_game_fps,
            run_real_time=run_real_time)
        self.log_frequency = log_frequency
        self.last_hit_count = 0
        self.last_miss_count = 0
        self.starting_hit_count = 0
        self.starting_miss_count = 0
        self.score_ratio = None
        
        self.dql = DeepQLearner(ACTIONS, 
            weight_save_path=weight_save_path,
            weight_restore_path=weight_restore_path,
            log_path=log_path,
            weight_save_frequency=weight_save_frequency,
            log_frequency=log_frequency,
            update_frequency=update_frequency,
            batch_size=batch_size,
            learning_rate=learning_rate,
            burn_in_duration=burn_in_duration,
            exploration_duration=exploration_duration,
            exploration_end_rate=exploration_end_rate,
            replay_max_size=replay_max_size,
            discount_rate=discount_rate,
            action_repeat=action_repeat,
            state_frames=state_frames,
            frame_height=frame_height,
            frame_width=frame_width,
            dueling=dueling,
            pooling=pooling,
            training=training)

    def get_keys_pressed(self, screen_array, reward, terminal):
        """Returns the keys to press at the given timestep. See parent class function."""
        return self.dql.step(screen_array, reward, terminal, self.score_ratio)

    def get_feedback(self):
        """Returns the feedback for the current state of the game. In this case, just returns
        the difference in the learner's score minus the difference in the other player's score.
        See parent class function.
        """
        # import must be done here because otherwise importing would cause the game to start playing
        from src.games.half_pong import hit_count, miss_count

        # get the difference in score between this and the last run
        score_change = (hit_count - self.last_hit_count) - (miss_count - self.last_miss_count)
        self.last_miss_count = miss_count
        self.last_hit_count = hit_count

        if self.last_miss_count % self.log_frequency == 0:
            self.starting_miss_count = self.last_miss_count
            self.starting_hit_count = self.last_hit_count
        
        self.score_ratio = float((hit_count - self.starting_hit_count) / (miss_count - self.starting_miss_count + 1))

        return float(score_change), score_change == -1

    def start(self):
        super(HalfPongPlayer, self).start()
        import src.games.half_pong
