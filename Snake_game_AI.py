import numpy as np
import pygame
import random

class snake_game:
    """
    pygame snake game with reinforcment learning traning
    Use arrow keys to move
    Press R to restart
        when you lose:
        - Press C to play again
        - Press Q to quiit game
    """
    def __init__(self, speed_game=25, starting_len=1, visual_game=True):

        self.remaining_frames = None
        self.foodx = None
        self.foody = None
        self.x1 = None
        self.y1 = None
        self.x1_change = None
        self.y1_change = None
        self.snake_list = []
        self.length_of_snake = None
        self.game_over = False

        # Set up screen
        self.width, self.height = 600, 600

        # Initialize Pygame
        if visual_game:
            pygame.init()

            # Set up screen
            self.screen_size = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("RL_snake Project Snake Neozro")

            # Fonts
            centauri_font = "Centauri.otf"
            self.text_font_size = 10
            self.font_style = pygame.font.Font(centauri_font, self.text_font_size)

        # HEAD
        self.snake_head = [self.width / 2, self.height / 2]

        # Colors
        self.color_fondo = (38, 56, 43)
        self.color_fondo_game_over = (0, 0, 0)
        self.color_comida = (255, 0, 255)
        self.color_snake = (0, 255, 71)
        self.color_texto = (0, 251, 255)
        self.NEON_PURPLE = (255, 0, 255)

        # Snake attributes
        self.starting_length_snake_init = starting_len
        self.starting_length_snake = starting_len
        self.dificulty = 0.10
        self.snake_block = 10
        self.snake_speed = speed_game
        self.limit_frames_per_food = 175

        self.n_frames_current_game=0
        self.n_food_current_game=0

        self.reset_game()

    def draw_snake(self):
        """
        funcion que genera el cuerpo de la serpiente basado en los elementos de la lista
        """
        # print(self.snake_list)
        for x in self.snake_list:
            pygame.draw.rect(self.screen_size, self.color_snake, [x[0], x[1], self.snake_block, self.snake_block])

    @staticmethod
    def gradient_color(value, limit_value):
        """
        return rgb gradient from green to red based on value and limit value
        :param value: current value
        :param limit_value: max value equivalent to red color
        :return: rgb colors
        """
        normalized_value = 1 - (value / limit_value)

        if normalized_value < 0.5:
            r = int((0.5-normalized_value) * 255)
            g = 255
            b = 0

        else:
            r = 255
            g = max(0,int((1-normalized_value)/0.5 * 255))
            b = 0

        return r, g, b

    def show_scores(self):
        """ plot diferent relevant scores on left corner of screen """


        # frames score

        custom_color = snake_game.gradient_color( value = self.remaining_frames,
                                                  limit_value = self.limit_frames_per_food * self.length_of_snake)

        text_surface_frames = self.font_style.render(str(self.n_frames_current_game),
                                              True,
                                                     custom_color)

        self.screen_size.blit(text_surface_frames, (5, 5))

        # food score
        food_score = self.font_style.render(str(self.n_food_current_game),
                                                     True,
                                                     (255, 43, 244))

        self.screen_size.blit(food_score, (5, 20))

    def new_random_food(self):
        """
        get random position for food
        """

        dificulty_clamped = min(1,self.dificulty) # NOQA

        self.foodx = round(random.randrange(int(self.width/2 * (1-dificulty_clamped)),
                                           int(self.width/2 + self.width/2 * dificulty_clamped) - self.snake_block) / 10.0) * 10.0

        self.foody = round(random.randrange(int(self.height/2 * (1-dificulty_clamped)),
                                            int(self.height/2 + self.height/2 * dificulty_clamped) - self.snake_block) / 10.0) * 10.0


    def draw_food(self):
        """
        draw on screen the food
        """
        pygame.draw.rect(self.screen_size, self.color_comida, [self.foodx, self.foody, self.snake_block, self.snake_block])

    def reset_game(self):
        """
        reset all the main variables so the game can start again
        """

        self.starting_length_snake = random.randint(1, self.starting_length_snake_init)
        self.n_frames_current_game=0
        self.n_food_current_game=0

        self.new_random_food()
        
        # Initial position
        self.x1 = self.width / 2
        self.y1 = self.height / 2

        random_initial_movements = [[10,0],[-10,0],[0,10],[0,-10]]

        # Initial movement
        idx = random.randint(1,3)
        self.x1_change = random_initial_movements[idx][0]
        self.y1_change = random_initial_movements[idx][1]

        self.snake_list = []
        self.snake_head = [self.width / 2, self.height / 2]
        self.length_of_snake = self.starting_length_snake

    
    def create_action_from_keyboard(self):
        """
        get the action from the keyboard.
        in the training the action is decided by the algorithm
        """
        # default action, keep moving straight
        input_action = np.array([0,1,0])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_q:
                    # quit game with Q key
                    self.game_over = True

                if event.key == pygame.K_LEFT:
                    input_action = np.array([1,0,0])

                elif event.key == pygame.K_RIGHT:
                    input_action = np.array([0,0,1])

        return input_action

    def move_snake_from_action(self, action):
        """
        move the snake based on the action
        :param action: array with 3 elements with diferents options
        [010]-> keep moving straight
        [100]-> left
        [001]-> right
        """
        signo_mov = 1

        if np.array_equal(action, np.array([0,1,0])):
            # straight doest effect the snake and it keeps its course in the same direction
            return

        # when moving the left turn of the snake make the opposite of the action of right
        if np.array_equal(action, np.array([1,0,0])):
            signo_mov = -1

        # Movement
        if self.x1_change == -self.snake_block:
            self.x1_change = 0
            self.y1_change = -self.snake_block * signo_mov

        elif self.x1_change == self.snake_block:
            self.x1_change = 0
            self.y1_change = self.snake_block * signo_mov

        elif self.y1_change == -self.snake_block:
            self.x1_change = self.snake_block * signo_mov
            self.y1_change = 0

        elif self.y1_change == self.snake_block:
            self.x1_change = -self.snake_block * signo_mov
            self.y1_change = 0

    def check_collision_boundary(self):
        """
         in each frame of the game, it checks if the snake hit with boundaries
        :return: bolean if the snake hit with boundaries
        """

        bool_collision_boundaries = self.x1 >= self.width or self.x1 < 0 or self.y1 >= self.height or self.y1 < 0

        return bool_collision_boundaries

    def check_collision_with_itself(self):
        """
        in each frame of the game, it checks if the snake hit with itself
        :return: bolean if the snake hit with itself
        """
        if self.snake_head in self.snake_list[:-1]:
            return True
        else: return False


    def check_countdown_elapsed_frames_finished(self):
        """ in each frame of the game, it checks if 75 frames has passed since the last food was eaten """

        self.remaining_frames = self.limit_frames_per_food * (self.n_food_current_game+1) - self.n_frames_current_game

        if self.remaining_frames < 0 :
            return True
        else: return False


    def generate_extended_body_snake(self):
        """
        if needed you can create a snake at the start of the desire len
        :return: new list of body parts
        """
        # result_body = [self.snake_head]
        result_body = []
        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        for i in range(len(self.snake_list)-1):
            new_x = head_x - (i + 1) * - self.x1_change
            new_y = head_y - (i + 1) * - self.y1_change

            result_body.insert(0,[new_x, new_y])

        result_body.append([head_x + self.x1_change,
                            head_y + self.y1_change])
        return result_body
    def move_snake_body(self):
        """ update the snake body based on the movement of the head """


        # calculate the new head pos
        self.x1 += self.x1_change
        self.y1 += self.y1_change

        if  self.snake_list:
            self.snake_head = [self.x1, self.y1]
            self.snake_list.append(self.snake_head)
        else:
            self.snake_list = self.generate_extended_body_snake()


        # delete last element in list
        if len(self.snake_list) > self.length_of_snake:
            del self.snake_list[0]

    def check_if_food_eaten(self):
        """ check if the food was eaten in every frame"""
        """ check if the food was eaten in every frame"""
        if self.x1 == self.foodx and self.y1 == self.foody:
            return True
        else: return False

    def game_frame(self, action):
        """
        This function draw single frame of the game and process the inputs and results for each of them.
        :param action: action to be taken by the agent
        :return: reward: reward obtained by the action in the current frame
                 game_over_curent_game: if the game is over in the current frame
                 self.n_food_current_game: comulative number of food eaten in the current game
        """
        # fill/clean canvas
        reward = 0
        game_over_curent_game = False
        self.screen_size.fill(self.color_fondo)

        self.move_snake_from_action(action)

        if self.check_collision_boundary():
            reward = -10
            game_over_curent_game = True

        self.move_snake_body()

        if self.check_collision_with_itself():
            reward = -10
            game_over_curent_game = True


        if self.check_countdown_elapsed_frames_finished():
            reward = -10
            game_over_curent_game = True

        self.draw_snake()

        self.show_scores()
        pygame.display.update()

        # Check if snake ate the food
        if self.check_if_food_eaten():
            reward = +10
            self.new_random_food()
            self.length_of_snake += 1
            self.n_food_current_game+=1

        self.draw_food()

        self.n_frames_current_game+=1


        pygame.display.update()
        pygame.time.Clock().tick(self.snake_speed)

        return reward, game_over_curent_game, self.n_food_current_game

    def game_frame_no_visual(self, action):
        """
        This function draw single frame of the game and process the inputs and results for each of them.
        :param action: action to be taken by the agent
        :return: reward: reward obtained by the action in the current frame
                 game_over_curent_game: if the game is over in the current frame
                 self.n_food_current_game: comulative number of food eaten in the current game
        """
        # fill/clean canvas
        reward = 0
        game_over_curent_game = False

        self.move_snake_from_action(action)

        if self.check_collision_boundary():
            reward = -10
            game_over_curent_game = True

        self.move_snake_body()

        if self.check_collision_with_itself():

            reward = -10
            game_over_curent_game = True

        if self.check_countdown_elapsed_frames_finished():
            reward = -10
            game_over_curent_game = True

        if self.check_if_food_eaten():
            reward = +10
            self.new_random_food()
            self.length_of_snake += 1
            self.n_food_current_game+=1

        self.n_frames_current_game+=1

        return reward, game_over_curent_game, self.n_food_current_game

    @staticmethod
    def find_closest_list(target, list_of_lists):
        import math
        closest_list = None
        min_distance = float('inf')

        for lst in list_of_lists:
            distance = math.sqrt(sum((lst[i] - target[i])**2 for i in range(len(target))))
            if distance < min_distance:
                min_distance = distance
                closest_list = lst

        return closest_list
    # def game_state_TEST(self, game):
    #
    #     print("GAME STATE --> ",agent().get_game_state(game))

    def run_game(self):
        """ run every frame in a loop until game over """

        while not self.game_over:

            action  = self. create_action_from_keyboard()
            reward, game_over_state, n_food_current_game = self.game_frame(action=action)
            # print(f"reward: {reward}, game_over_state: {game_over_state}, n_food_current_game: {n_food_current_game}")

            # self.game_state_TEST()

            if game_over_state:
                print("RESET")
                self.reset_game()

        pygame.quit()
