import pygame.constants as pgc
from players.pygame_player import PyGamePlayer, function_intercept
import games.tetris
from qlearn import DeepQLearner



ACTIONS = [pgc.K_UNKNOWN, pgc.K_RIGHT, pgc.K_LEFT, pgc.K_DOWN, pgc.K_UP]

class TetrisPlayer(PyGamePlayer):
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
        super(TetrisPlayer, self).__init__(force_game_fps=10, run_real_time=False)
        self._new_reward = 0.0
        self._terminal = False
        
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
            discoun_rate=discount_rate,
            action_repeat=action_repeat,
            state_frames=state_frames,
            frame_height=frame_height,
            frame_width=frame_width,
            dueling=dueling,
            pooling=pooling,
            training=training)

        def add_removed_lines_to_reward(lines_removed, *args, **kwargs):
            self._new_reward += lines_removed    
            return lines_removed

        def check_for_game_over(ret, text):
            if text == 'Game Over':
            	self._terminal = True

        # to get the reward we will intercept the removeCompleteLines method and store what it returns
        games.tetris.removeCompleteLines = function_intercept(games.tetris.removeCompleteLines,
                                                              add_removed_lines_to_reward)
        # find out if we have had a game over
        games.tetris.showTextScreen = function_intercept(games.tetris.showTextScreen,
                                                         check_for_game_over)

    def get_keys_pressed(self, screen_array, feedback, terminal):
        if self._terminal :
        	self._terminal = False
        	return [pgc.K_SPACE]

        return  self.dql.step(screen_array, feedback, terminal)
        
        
    def get_feedback(self):
    	if self._terminal :
            from games.tetris import blankSpaces
            terminal = self._terminal
            
            # found the follwoing reward/penalty strategy in a paper. coeff is taken from the paper. Should play around with it a little
            return float(.35*blankSpaces), terminal
            
    	
    	temp = self._new_reward
    	self._new_reward = 0.0
    	self.lines_removed = False
    	terminal = self._terminal
    	return temp*temp, terminal


    def start(self):
        super(TetrisPlayer, self).start()
        games.tetris.main()
