"""Simple Pong Player for testing Deep Q logic."""
from players.pygame_player import PyGamePlayer
from pygame.constants import K_DOWN, K_UP, K_UNKNOWN
from qlearn import DeepQLearner



# Possible actions for Pong. Last one is equivalent to "do nothing."
ACTIONS = [K_DOWN, K_UNKNOWN, K_UP]

# Either 'HITS' or 'SCORES'.
SCORING_FUNCTION = 'HITS'

class PongPlayer(PyGamePlayer):
    """Implementation of PyGamePlayer for Pong."""
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
        """
        Example class for playing Pong
        """
        super(PongPlayer, self).__init__(
            force_game_fps=force_game_fps,
            run_real_time=run_real_time)
        self.last_bar1_score = 0.0
        self.last_bar2_score = 0.0
        self.last_hit_count = 0
        self.last_miss_count = 0
        self.starting_hit_count = 0
        self.starting_miss_count = 0
        self.score_ratio = None
        self.log_frequency = log_frequency

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

    def get_keys_pressed(self, screen_array, feedback, terminal):
        return self.dql.step(screen_array, feedback, terminal, self.score_ratio)

    def get_feedback(self):
        # import must be done here because otherwise importing would cause the game to start playing
        if SCORING_FUNCTION == 'SCORES':
            from games.pong import bar1_score, bar2_score

            # get the difference in score between this and the last run
            score_change = (bar1_score - self.last_bar1_score) - (bar2_score - self.last_bar2_score)
            self.last_bar1_score = bar1_score
            self.last_bar2_score = bar2_score

        elif SCORING_FUNCTION == 'HITS':
            from games.pong import hit_count, miss_count

            # get the difference in score between this and the last run
            score_change = (hit_count - self.last_hit_count) - (miss_count - self.last_miss_count)
            self.last_miss_count = miss_count
            self.last_hit_count = hit_count

            if self.last_miss_count % self.log_frequency == 0:
                self.starting_miss_count = self.last_miss_count
                self.starting_hit_count = self.last_hit_count
            
            self.score_ratio = float((hit_count - self.starting_hit_count) / (miss_count - self.starting_miss_count + 1))

        return float(score_change), score_change != 0

    def start(self):
        super(PongPlayer, self).start()
        import games.pong
