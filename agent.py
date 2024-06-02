import torch
import random
import numpy
from Snake_game_AI import snake_game
from collections import deque
import numpy as np
import math
import pygame
import matplotlib.pyplot as plt
from IPython import display
from model import *
from plots import plot_live_results
import cv2


class Agent:
    def __init__(self):

        self.max_memory = 100_000
        self.batch_size = 1000
        self.learning_rate = 0.00075  # og 0.001
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.95  # discount rate
        self.memory = deque(maxlen=self.max_memory)  # popleft()
        self.model = Linear_QNet(12, 256, 3)
        self.trainer = QTrainer(self.model, lr=self.learning_rate, gamma=self.gamma)

    @staticmethod
    def get_blocks_that_hit_w_itself_in_future(head_pos, movement_direction, body_blocks):
        """
        get all the posible direction blocks for the next two frames. used in collision for itself
        :param head_pos: head position
        :param movement_direction: movement direction, so it doestn count his own neck
        :return: blocks for the hit
        """
        x_search_range = []
        y_search_range = []

        # list of blocks displacement in each direction
        # we need to  avoid the direction that is oposite of the direction of movement
        if movement_direction[0] != 1:
            x_search_range.extend([-20, -10])

        if movement_direction[1] != 1:
            y_search_range.extend([20, 10])

        if movement_direction[2] != 1:
            x_search_range.extend([20, 10])

        if movement_direction[3] != 1:
            y_search_range.extend([-20, -10])

        search_blocks_x = [[head_pos[0] - offset, head_pos[1]] for offset in x_search_range]
        search_blocks_y = [[head_pos[0], head_pos[1] - offset] for offset in y_search_range]

        search_blocks_total = search_blocks_x + search_blocks_y

        warning_blocks = [blocks for blocks in search_blocks_total if blocks in body_blocks]

        return warning_blocks

    @staticmethod
    def direction_transform(x):
        """ auxiliar function to transform a magnitude direction to unitary"""
        if x != 0:
            return math.copysign(1, x)

        elif x == 0:
            return 0

    @staticmethod
    def unitary_dir_tuple_to_4x1_mov_vector(xy_vectors):
        """ unit direction tuple transform to 4x1 direction vector """
        v = [0, 0, 0, 0]
        for i, vector in enumerate(xy_vectors):
            if vector[0] == 1:
                v[2] = 1

            if vector[0] == -1:
                v[0] = 1

            if vector[1] == 1:
                v[3] = 1

            if vector[1] == -1:
                v[1] = 1

        return v

    @staticmethod
    def get_final_vector_from_warning_blocks(head_pos, warning_blocks):
        """
        get the main 4x1 vector of all warning direction that produce a collision with itself in the next two moves
        :param head_pos: head pos
        :param warning_blocks: warning blocks of the body that will collision
        :return: vector 4,1 warning [LEFT UP RIGHT DOWN]
        """

        tuple_xy_warning_blocks = [[Agent.direction_transform(x - head_pos[0]),
                                    Agent.direction_transform(y - head_pos[1])] for x, y in warning_blocks]

        return Agent.unitary_dir_tuple_to_4x1_mov_vector(tuple_xy_warning_blocks)

    @staticmethod
    def cancel_direction_movement(array_1, array_2):
        """
        check if the direction of movement cancel with the next move,
        snake cant make a move that its oposite to its current one.
        :param X1,Y1: next move
        :param X2, Y2: current direction of movement
        :return: treated value for this situation
        """
        x1 = array_1[0]
        y1 = array_1[1]
        x2 = array_2[0]
        y2 = array_2[1]

        result_x = 0
        result_y = 0

        if x1 + x2 > 0 or x1 + x2 < 0:
            result_x = x1

        if y1 + y2 > 0 or y1 + y2 < 0:
            result_y = y1

        return np.asarray([result_x, result_y])

    @staticmethod
    def check_space_in_body(target_space, body_array):
        """
        check if the space defined by target space is in the location of the snake's body
        :param target_space: target space in the screen
        :param body_array: array with body parts, must exclude the head
        :return: bool, if the space its inside the snake body
        """
        results = []
        for row in body_array:
            eq_bool = np.array_equal(target_space, row)
            results.append(eq_bool)

        return any(results)

    def check_if_food_visible(self, game):
        """
        Calculate a straigth line between two points (x1, y1) [head in this case] and (x2, y2) [food], and check for collision with other points [body parts].
        :return: If the food is visible or not
        """

        x1 = game.snake_head[0]
        y1 = game.snake_head[1]
        x2 = game.foodx
        y2 = game.foodx
        points = game.snake_list[:-1]

        def on_segment(p, q, r):
            if (max(p[0], r[0]) >= q[0] >= min(p[0], r[0]) and
                    max(p[1], r[1]) >= q[1] >= min(p[1], r[1])):
                return True
            return False

        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0
            return 1 if val > 0 else -1

        def do_intersect(p1, q1, p2, q2):
            o1 = orientation(p1, q1, p2)
            o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1)
            o4 = orientation(p2, q2, q1)

            if o1 != o2 and o3 != o4:
                return True
            if o1 == 0 and on_segment(p1, p2, q1):
                return True
            if o2 == 0 and on_segment(p1, q2, q1):
                return True
            if o3 == 0 and on_segment(p2, p1, q2):
                return True
            if o4 == 0 and on_segment(p2, q1, q2):
                return True
            return False

        for i in range(len(points)):
            if do_intersect((x1, y1), (x2, y2), points[i], points[(i + 1) % len(points)]):
                return True
        return False

    def get_game_state(self, game):
        """
        Get the state of the game, what is going on in the screen
        DIRECTIONS BASED ON --> [LEFT UP RIGHT DOWN] 1  X 4

        :param game: game class
        :return:
                [DIRECTION OF MOVEMENT 1 X 4 ARRAY,
                FOOD DIRECTION 1 X 4 ARRAY,
                DANGER FROM WALL 1 X 4 ARRAY,
                DANGER FROM ITSELF 1 X 4 ARRAY]
        """
        snake_direction_state = [game.x1_change < 0,
                                 game.y1_change < 0,
                                 game.x1_change > 0,
                                 game.y1_change > 0]

        if not self.check_if_food_visible(game=game):

            food_direction_state = [game.foodx - game.snake_head[0] < 0,
                                    game.foody - game.snake_head[1] < 0,
                                    game.foodx - game.snake_head[0] > 0,
                                    game.foody - game.snake_head[1] > 0]

        else:
            food_direction_state = [0, 0, 0, 0]

        wall_danger_direction_state = [game.snake_head[0] - game.snake_block <= 0,
                                       game.snake_head[1] - game.snake_block <= 0,
                                       game.snake_head[0] + game.snake_block >= game.width,
                                       game.snake_head[1] + game.snake_block >= game.height]

        snake_body_danger_direction_state = [Agent.check_space_in_body(game.snake_head +
                                                                       Agent.cancel_direction_movement(
                                                                           np.asarray([xx, yy]),
                                                                           np.asarray(
                                                                               [game.x1_change, game.y1_change])),
                                                                       np.asarray(game.snake_list[:-1])) for xx, yy in
                                             [(-10, 0), (0, -10), (10, 0), (0, 10)]]

        collision_danger_direction_state = [x or y for x, y in zip(wall_danger_direction_state,
                                                                   snake_body_danger_direction_state)]

        state = snake_direction_state + food_direction_state + collision_danger_direction_state

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """
        list to remember actions and his game results
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """
        checks if memory length is greater than batch size. then makes a mini sample and train the network
        """
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, game_over_state):
        """ train with a single step information of current frame"""
        self.trainer.train_step(state, action, reward, next_state, game_over_state)

    def get_action(self, state):
        self.epsilon = 200 - self.n_games  # OLD 80
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def predict_action(self, state):
        """
        function only for inference
        :param state:
        :return:
        """
        final_move = [0, 0, 0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1
        return final_move


def run_agent(mode="manual", speed_game=25, starting_len_snake=1, visual_game=True):
    """
    run the agent in a game
    """
    game_over_state = None
    n_food_current_game = 0
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent_class = Agent()
    game = snake_game(speed_game=speed_game, starting_len=starting_len_snake, visual_game=visual_game)
    if mode == "predict":
        agent_class.model.load()
        game.dificulty = 1

    while not game.game_over:

        # # WHEN PLAYED WITH RL_snake TRAINING
        if mode == "train":

            # initial state of game
            state_of_game_OLD = agent_class.get_game_state(game)

            # get an action to move the snake
            action = agent_class.get_action(state_of_game_OLD)

            # get the results of that actions
            if visual_game:
                reward, game_over_state, n_food_current_game = game.game_frame(action=action)

            else:
                reward, game_over_state, n_food_current_game = game.game_frame_no_visual(action=action)

            # state of game after action
            state_of_game_NEW = agent_class.get_game_state(game)

            #  current frame training
            agent_class.train_short_memory(state_of_game_OLD, action, reward, state_of_game_NEW, game_over_state)

            # save the results to later train
            agent_class.remember(state_of_game_OLD, action, reward, state_of_game_NEW, game_over_state)

        # WHEN PLAYED MANUAL
        elif mode == "manual":

            action = game.create_action_from_keyboard()
            # get the results of that actions
            reward, game_over_state, n_food_current_game = game.game_frame(action=action)

        # WHEN PLAYED BY THE TRAINED MODEL
        elif mode == "predict":

            state_of_game = agent_class.get_game_state(game)
            action = agent_class.predict_action(state_of_game)
            reward, game_over_state, n_food_current_game = game.game_frame(action=action)

        if game_over_state:

            game.reset_game()

            if mode == "train":

                agent_class.n_games += 1
                agent_class.train_long_memory()

                if n_food_current_game > record:
                    game.dificulty += 0.05
                    record = n_food_current_game
                    agent_class.model.save()

                plot_scores.append(n_food_current_game)
                total_score += n_food_current_game
                mean_score = total_score / agent_class.n_games
                plot_mean_scores.append(mean_score)
                plot_live_results(plot_scores)

        # 'Q' key stops everything
        ## if q pressed in plot window
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            cv2.destroyAllWindows()
            exit(1)

        if mode == "train" and visual_game:
            ## if q pressed in game window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game.game_over = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        # quit game with Q key
                        game.game_over = True

    pygame.quit()


if __name__ == "__main__":
    run_agent(mode="train", speed_game=50, starting_len_snake=15, visual_game=True)
