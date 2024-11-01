import pygame
import neat
import os
import random
import math
import pickle
import sys
import time
import tempfile
import configparser

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE

# Colors
WHITE = (255, 255, 255)
GREY = (200, 200, 200)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

FONT = pygame.font.SysFont("arial", 20)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class Snake:
    def __init__(self):
        self.positions = [((GRID_WIDTH // 2), (GRID_HEIGHT // 2))]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.length = 1

    def get_head_position(self):
        return self.positions[0]

    def turn(self, direction):
        if (direction[0] * -1, direction[1] * -1) == self.direction:
            return
        else:
            self.direction = direction

    def move(self):
        cur = self.get_head_position()
        x, y = self.direction
        new = ((cur[0] + x) % GRID_WIDTH, (cur[1] + y) % GRID_HEIGHT)
        if new in self.positions[2:]:
            return False
        self.positions.insert(0, new)
        if len(self.positions) > self.length:
            self.positions.pop()
        return True

    def reset(self):
        self.positions = [((GRID_WIDTH // 2), (GRID_HEIGHT // 2))]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.length = 1

class Food:
    def __init__(self, snake_positions):
        self.position = (0, 0)
        self.randomize_position(snake_positions)

    def randomize_position(self, snake_positions):
        while True:
            self.position = (random.randint(0, GRID_WIDTH - 1),
                           random.randint(0, GRID_HEIGHT - 1))
            if self.position not in snake_positions:
                break

class SnakeGame:
    def __init__(self, render=True):
        self.snake = Snake()
        self.food = Food(self.snake.positions)
        self.score = 0
        self.render = render
        if self.render:
            self.display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Snake NEAT")
            self.clock = pygame.time.Clock()

    def reset(self):
        self.snake.reset()
        self.food.randomize_position(self.snake.positions)
        self.score = 0

    def step(self, action):
        if action == 1:
            self.snake.turn(self.left_turn())
        elif action == 2:
            self.snake.turn(self.right_turn())

        alive = self.snake.move()
        if not alive:
            return False, self.score

        head_pos = self.snake.get_head_position()
        if head_pos == self.food.position:
            self.snake.length += 1
            self.score += 1
            self.food.randomize_position(self.snake.positions)

        return True, self.score

    def left_turn(self):
        x, y = self.snake.direction
        return (-y, x)

    def right_turn(self):
        x, y = self.snake.direction
        return (y, -x)

    def get_inputs(self):
        inputs = []
        head_x, head_y = self.snake.get_head_position()
        food_x, food_y = self.food.position

        inputs.append((food_x - head_x) / GRID_WIDTH)
        inputs.append((food_y - head_y) / GRID_HEIGHT)

        directions = [UP, DOWN, LEFT, RIGHT]
        for d in directions:
            inputs.append(1.0 if self.snake.direction == d else 0.0)

        obstacle_directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]

        for dir in obstacle_directions:
            distance = 0
            found_body = 0
            found_wall = 0
            x, y = head_x, head_y
            while True:
                x += dir[0]
                y += dir[1]
                distance += 1
                if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
                    found_wall = 1
                    break
                if (x, y) in self.snake.positions:
                    found_body = 1
                    break
                if distance >= 10:
                    break

            inputs.append(min(distance / 10.0, 1.0))
            inputs.append(float(found_body))

        inputs.append(self.snake.length / (GRID_WIDTH * GRID_HEIGHT))
        return inputs

    def render_game(self):
        if not self.render:
            return
        self.display.fill(WHITE)

        for x in range(0, SCREEN_WIDTH, GRID_SIZE):
            pygame.draw.line(self.display, GREY, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            pygame.draw.line(self.display, GREY, (0, y), (SCREEN_WIDTH, y))

        for pos in self.snake.positions:
            rect = pygame.Rect(pos[0]*GRID_SIZE, pos[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(self.display, GREEN, rect)

        food_rect = pygame.Rect(self.food.position[0]*GRID_SIZE,
                              self.food.position[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(self.display, RED, food_rect)

        score_text = FONT.render(f"Score: {self.score}", True, BLACK)
        self.display.blit(score_text, (5, 5))

        pygame.display.flip()
        self.clock.tick(60)

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = SnakeGame(render=False)
    
    max_steps = 500
    steps_without_food = 0
    
    for _ in range(max_steps):
        if steps_without_food > 50:  # Prevent snake from going in circles
            break
            
        inputs = game.get_inputs()
        output = net.activate(inputs)
        action = output.index(max(output))
        
        alive, score = game.step(action)
        
        if not alive:
            break
            
        if score > game.score:
            steps_without_food = 0
        else:
            steps_without_food += 1
    
    return game.score

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def visualize_winner(winner, config):
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    game = SnakeGame(render=True)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        inputs = game.get_inputs()
        output = net.activate(inputs)
        action = output.index(max(output))

        alive, score = game.step(action)

        if not alive:
            print(f"Game Over! Score: {score}")
            game.reset()
            time.sleep(1)

        game.render_game()

    pygame.quit()
    sys.exit()

def run():
    config_str = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 1000
pop_size             = 150
reset_on_extinction  = False

[DefaultGenome]
activation_default      = tanh
activation_mutate_rate = 0.0
activation_options     = tanh relu

aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

conn_add_prob           = 0.5
conn_delete_prob        = 0.5

enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = full

node_add_prob           = 0.2
node_delete_prob        = 0.2

num_hidden              = 0
num_inputs              = 23
num_outputs             = 3

response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation      = 15
species_elitism     = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""

    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_config:
        temp_config.write(config_str)
        temp_config_path = temp_config.name

    try:
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            temp_config_path
        )

        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        winner = p.run(eval_genomes, 50)

        with open("best_genome.pkl", "wb") as f:
            pickle.dump(winner, f)

        print('\nBest genome:\n{!s}'.format(winner))
        visualize_winner(winner, config)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

if __name__ == "__main__":
    run()