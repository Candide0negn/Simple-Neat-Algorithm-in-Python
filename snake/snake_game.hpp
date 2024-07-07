#pragma once

#include <SFML/Graphics.hpp>
#include <deque>
#include <functional>
#include <cmath>
#include <algorithm>
#include "neat.hpp"

enum class Action {
    DoNothing,
    RotateLeft,
    RotateRight,
};

enum class GameResult {
    Running,
    GameOver,
    Winner,
};

enum class Direction {
    Up,
    Right,
    Down,
    Left,
};

struct Coordinates {
    int row;
    int col;

    bool operator==(const Coordinates& other) const {
        return row == other.row && col == other.col;
    }
};

struct Snake {
    std::deque<Coordinates> body;

    Coordinates head() const { return body.front(); }
};

class SnakeEngine {
public:
    SnakeEngine(int width, int height, bool allow_through_walls)
        : width(width), height(height), allow_through_walls(allow_through_walls), rng(std::random_device{}()) {
        reset();
    }

    void reset() {
        snake.body.clear();
        snake.body.push_front({height / 2, width / 2});
        current_direction = Direction::Right;
        score = 0;
        generate_food();
    }

    GameResult process(Action action) {
        current_direction = update_direction(current_direction, action);
        Coordinates new_head = get_next_head(snake.head(), current_direction);

        auto tail = snake.body.back();
        snake.body.pop_back();

        if (hits_wall(new_head) || hits_snake_body(new_head)) {
            return GameResult::GameOver;
        }

        if (new_head == food) {
            score++;

            if (snake.body.size() == height * width - 1) {
                return GameResult::Winner;
            }

            generate_food();
            snake.body.push_back(tail);
        }

        snake.body.push_front(new_head);
        return GameResult::Running;
    }

    const Snake& get_snake() const { return snake; }
    Coordinates get_food() const { return food; }
    int get_width() const { return width; }
    int get_height() const { return height; }
    int get_score() const { return score; }
    Direction get_current_direction() const { return current_direction; }

    bool hits_wall(const Coordinates& c) const {
        if (allow_through_walls) {
            return false;
        }
        return c.row < 0 || c.row >= height || c.col < 0 || c.col >= width;
    }

    bool hits_snake_body(const Coordinates& c) const {
        return std::find(snake.body.begin() + 1, snake.body.end(), c) != snake.body.end();
    }

private:
    Snake snake;
    Coordinates food;
    bool allow_through_walls;
    int width;
    int height;
    int score;
    Direction current_direction;
    Rng rng;

    void generate_food() {
        do {
            food.row = rng.next_int(height);
            food.col = rng.next_int(width);
        } while (hits_snake_body(food));
    }

    Direction update_direction(Direction current, Action action) {
        switch (action) {
            case Action::RotateLeft:
                return static_cast<Direction>((static_cast<int>(current) + 3) % 4);
            case Action::RotateRight:
                return static_cast<Direction>((static_cast<int>(current) + 1) % 4);
            default:
                return current;
        }
    }

    Coordinates get_next_head(const Coordinates& head, Direction direction) {
        Coordinates next = head;
        switch (direction) {
            case Direction::Up:    next.row--; break;
            case Direction::Right: next.col++; break;
            case Direction::Down:  next.row++; break;
            case Direction::Left:  next.col--; break;
        }
        if (allow_through_walls) {
            next.row = (next.row + height) % height;
            next.col = (next.col + width) % width;
        }
        return next;
    }
};

struct GameRendererConfig {
    sf::Vector2f grid_size;
    sf::Vector2f grid_offset;
    sf::Color stroke_color;
    sf::Color background_color;
    sf::Color snake_color;
    sf::Color food_color;
};

class GameRenderer {
public:
    GameRenderer(sf::RenderWindow& window, const SnakeEngine& engine, GameRendererConfig config)
        : window(window), engine(engine), config(config) {
        field_size = sf::Vector2f(config.grid_size.x / engine.get_width(),
                                  config.grid_size.y / engine.get_height());
    }

    void draw() {
        draw_grid();
        draw_snake();
        draw_food();
    }

private:
    void draw_grid() {
        sf::RectangleShape stroke(config.grid_size);
        stroke.setPosition(config.grid_offset);
        stroke.setFillColor(config.stroke_color);
        window.draw(stroke);

        for (int i = 0; i < engine.get_height(); i++) {
            for (int j = 0; j < engine.get_width(); j++) {
                draw_field({i, j}, config.background_color);
            }
        }
    }

    void draw_snake() {
        const auto& snake = engine.get_snake();
        for (size_t i = 0; i < snake.body.size(); i++) {
            auto segment = snake.body[i];
            draw_field(segment, config.snake_color);
        }
    }

    void draw_food() {
        draw_field(engine.get_food(), config.food_color);
    }

    void draw_field(Coordinates c, sf::Color color) {
        sf::RectangleShape shape(field_size);
        shape.setPosition(to_position(c));
        shape.setFillColor(color);
        window.draw(shape);
    }

    sf::Vector2f to_position(Coordinates c) {
        return sf::Vector2f(config.grid_offset.x + c.col * field_size.x,
                            config.grid_offset.y + c.row * field_size.y);
    }

    sf::RenderWindow& window;
    const SnakeEngine& engine;
    GameRendererConfig config;
    sf::Vector2f field_size;
};

class Controller {
public:
    virtual ~Controller() = default;
    virtual void on_event(sf::Event&) {}
    virtual Action get_action() = 0;
};

class KeyboardController : public Controller {
public:
    void on_event(sf::Event& event) override {
        if (event.type == sf::Event::KeyPressed) {
            if (event.key.code == sf::Keyboard::Right) {
                next_action = Action::RotateRight;
            } else if (event.key.code == sf::Keyboard::Left) {
                next_action = Action::RotateLeft;
            }
        }
    }

    Action get_action() override {
        auto action = next_action;
        next_action = Action::DoNothing;
        return action;
    }

private:
    Action next_action = Action::DoNothing;
};

class AIController : public Controller {
public:
    AIController(const FeedForwardNeuralNetwork& network, const SnakeEngine& engine) 
        : network(network), engine(engine) {}

    Action get_action() override {
        std::vector<double> inputs = get_network_inputs();
        std::vector<double> outputs = network.activate(inputs);
        return interpret_network_outputs(outputs);
    }

private:
    std::vector<double> get_network_inputs() {
        std::vector<double> inputs;
        const auto& snake = engine.get_snake();
        const auto& food = engine.get_food();
        Direction current_direction = engine.get_current_direction();

        // Normalize coordinates
        double norm_head_x = static_cast<double>(snake.head().col) / engine.get_width();
        double norm_head_y = static_cast<double>(snake.head().row) / engine.get_height();
        double norm_food_x = static_cast<double>(food.col) / engine.get_width();
        double norm_food_y = static_cast<double>(food.row) / engine.get_height();

        // Distance to food (2 inputs)
        inputs.push_back(norm_food_x - norm_head_x);
        inputs.push_back(norm_food_y - norm_head_y);

        // Snake direction (4 inputs)
        for (int i = 0; i < 4; ++i) {
            inputs.push_back(i == static_cast<int>(current_direction) ? 1.0 : 0.0);
        }

        // Obstacles in 8 directions (16 inputs)
        std::vector<std::pair<int, int>> directions = {
            {-1, 0}, {-1, 1}, {0, 1}, {1, 1},
            {1, 0}, {1, -1}, {0, -1}, {-1, -1}
        };

        for (const auto& dir : directions) {
            Coordinates check = snake.head();
            double distance = 0;
            bool found_body = false;
            bool found_wall = false;

            while (!found_body && !found_wall) {
                check.row += dir.first;
                check.col += dir.second;
                distance += 1.0;

                if (engine.hits_wall(check)) {
                    found_wall = true;
                } else if (engine.hits_snake_body(check)) {
                    found_body = true;
                }
            }

            inputs.push_back(1.0 / distance); // Closer obstacles have higher values
            inputs.push_back(found_body ? 1.0 : 0.0); // Is the obstacle a body part?
        }

        // Length of snake (normalized) (1 input)
        inputs.push_back(static_cast<double>(snake.body.size()) / (engine.get_width() * engine.get_height()));

        // Snake's current length (1 input)
        inputs.push_back(static_cast<double>(snake.body.size()) / (engine.get_width() * engine.get_height()));

        std::cout << "Number of inputs: " << inputs.size() << std::endl;

        return inputs;
        
    }

    Action interpret_network_outputs(const std::vector<double>& outputs) {
        assert(outputs.size() == 3);
        int max_index = std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
        switch (max_index) {
            case 0: return Action::DoNothing;
            case 1: return Action::RotateLeft;
            case 2: return Action::RotateRight;
            default: return Action::DoNothing; // Should never happen
        }
    }

    const FeedForwardNeuralNetwork& network;
    const SnakeEngine& engine;
};

std::unique_ptr<Controller> make_controller(const std::string& controller_type, const FeedForwardNeuralNetwork* network = nullptr, const SnakeEngine* engine = nullptr) {
    if (controller_type == "player") {
        return std::make_unique<KeyboardController>();
    } else if (controller_type == "ai" && network != nullptr && engine != nullptr) {
        return std::make_unique<AIController>(*network, *engine);
    }
    throw std::runtime_error("Invalid controller type or missing network/engine for AI controller");
}

class Ticker {
public:
    Ticker(int fps) : interval(sf::seconds(1.0f / fps)) {}

    bool tick() {
        if (clock.getElapsedTime() >= interval) {
            clock.restart();
            return true;
        }
        return false;
    }

private:
    sf::Clock clock;
    sf::Time interval;
};