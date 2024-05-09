#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <deque>
#include <unordered_map>
#include <optional>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <random>
#include <variant>
#include <functional>
#include <fstream>

// Neuroevolution Code

struct RNG {
    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<double> dist_01{0.0, 1.0};
    std::uniform_int_distribution<int> dist_int{0, 100};
    
    double choose(double probability, double a, double b) {
        return dist_01(rng) < probability ? a : b;
    }
    
    bool choose(double probability, bool a, bool b) {
        return dist_01(rng) < probability ? a : b;
    }
    
    template <typename T>
    T choose_random(const std::vector<T>& vec) {
        std::uniform_int_distribution<size_t> dist(0, vec.size() - 1);
        return vec[dist(rng)];
    }

    template<typename Iter>
    Iter choose_random(Iter start, Iter end) {
        std::uniform_int_distribution<size_t> dist(0, std::distance(start, end) - 1);
        std::advance(start, dist(rng));
        return start;
    }

    double next_gaussian(double mean, double stdev) {
        std::normal_distribution<double> dist(mean, stdev);
        return dist(rng);
    }

    int next_int(int max) {
        std::uniform_int_distribution<int> dist(0, max);
        return dist(rng);
    }

    std::function<double(double)> choose(double probability, const std::function<double(double)>& a, const std::function<double(double)>& b) {
        return dist_01(rng) < probability ? a : b;
    }

    Activation choose(double probability, const Activation& a, const Activation& b) {
        if (dist_01(rng) < probability) {
            return a;
        } else {
            return b;
        }
    }
};

RNG rng;

using ActivationFn = std::function<double(double)>;

struct Activation {
    std::variant<std::monostate, ActivationFn> fn;
};

// Link and Neuron Gene Definitions
struct LinkId {
    int input_id;
    int output_id;
    
    bool operator==(const LinkId& other) const {
        return input_id == other.input_id && output_id == other.output_id;
    }
};

namespace std {
    template <>
    struct hash<LinkId> {
        std::size_t operator()(const LinkId& link) const noexcept {
            return std::hash<int>()(link.input_id) ^ std::hash<int>()(link.output_id);
        }
    };
}

struct NeuronGene {
    int neuron_id;
    double bias;
    Activation activation;
};

struct LinkGene {
    LinkId link_id;
    double weight;
    bool is_enabled;
};

struct Genome {
    int genome_id;
    int num_inputs;
    int num_outputs;
    std::vector<NeuronGene> neurons;
    std::vector<LinkGene> links;

    void add_neuron(const NeuronGene& neuron) {
        neurons.push_back(neuron);
    }

    void add_link(const LinkGene& link) {
        links.push_back(link);
    }

    std::optional<NeuronGene> find_neuron(int neuron_id) const {
        for (const auto& neuron : neurons) {
            if (neuron.neuron_id == neuron_id) {
                return neuron;
            }
        }
        return std::nullopt;
    }

    std::optional<LinkGene> find_link(const LinkId& link_id) const {
        for (const auto& link : links) {
            if (link.link_id == link_id) {
                return link;
            }
        }
        return std::nullopt;
    }

    std::vector<int> make_input_ids() const {
        std::vector<int> input_ids(num_inputs);
        for (int i = 0; i < num_inputs; ++i) {
            input_ids[i] = -i - 1;
        }
        return input_ids;
    }

    std::vector<int> make_output_ids() const {
        std::vector<int> output_ids(num_outputs);
        for (int i = 0; i < num_outputs; ++i) {
            output_ids[i] = i;
        }
        return output_ids;
    }
};

NeuronGene crossover_neuron(const NeuronGene& a, const NeuronGene& b) {
    assert(a.neuron_id == b.neuron_id);
    int neuron_id = a.neuron_id;
    double bias = rng.choose(0.5, a.bias, b.bias);
    Activation activation;
    if (std::holds_alternative<ActivationFn>(a.activation.fn) && std::holds_alternative<ActivationFn>(b.activation.fn)) {
        ActivationFn fn = rng.choose(0.5, std::get<ActivationFn>(a.activation.fn), std::get<ActivationFn>(b.activation.fn));
        activation.fn = fn;
    } else {
        activation = rng.choose(0.5, a.activation, b.activation);
    }
    return {neuron_id, bias, activation};
}

LinkGene crossover_link(const LinkGene& a, const LinkGene& b) {
    assert(a.link_id == b.link_id);
    LinkId link_id = a.link_id;
    double weight = rng.choose(0.5, a.weight, b.weight);
    bool is_enabled = rng.choose(0.5, a.is_enabled, b.is_enabled);
    return {link_id, weight, is_enabled};
}

struct Individual {
    Genome genome;
    double fitness;
};

Genome crossover(const Individual& dominant, const Individual& recessive) {
    Genome offspring{dominant.genome.genome_id, dominant.genome.num_inputs, dominant.genome.num_outputs};

    for (const auto& dominant_neuron : dominant.genome.neurons) {
        int neuron_id = dominant_neuron.neuron_id;
        std::optional<NeuronGene> recessive_neuron = recessive.genome.find_neuron(neuron_id);
        if (!recessive_neuron) {
            offspring.add_neuron(dominant_neuron);
        } else {
            offspring.add_neuron(crossover_neuron(dominant_neuron, *recessive_neuron));
        }
    }

    for (const auto& dominant_link : dominant.genome.links) {
        LinkId link_id = dominant_link.link_id;
        std::optional<LinkGene> recessive_link = recessive.genome.find_link(link_id);
        if (!recessive_link) {
            offspring.add_link(dominant_link);
        } else {
            offspring.add_link(crossover_link(dominant_link, *recessive_link));
        }
    }

    return offspring;
}

void mutate_add_link(Genome& genome) {
    auto choose_random_input_or_hidden = [&](const std::vector<NeuronGene>& neurons) {
        std::vector<int> ids;
        for (const auto& neuron : neurons) {
            ids.push_back(neuron.neuron_id);
        }
        return rng.choose_random(ids);
    };

    auto choose_random_output_or_hidden = [&](const std::vector<NeuronGene>& neurons) {
        std::vector<int> ids;
        for (const auto& neuron : neurons) {
            ids.push_back(neuron.neuron_id);
        }
        return rng.choose_random(ids);
    };

    int input_id = choose_random_input_or_hidden(genome.neurons);
    int output_id = choose_random_output_or_hidden(genome.neurons);
    LinkId link_id{input_id, output_id};

    if (auto existing_link = genome.find_link(link_id)) {
        existing_link->is_enabled = true;
        return;
    }

    auto would_create_cycle = [&](const std::vector<LinkGene>& links, int input, int output) {
        std::unordered_map<int, std::vector<int>> adjacency_list;
        for (const auto& link : links) {
            adjacency_list[link.link_id.input_id].push_back(link.link_id.output_id);
        }

        std::function<bool(int, int)> dfs = [&](int current, int target) {
            if (current == target) return true;
            for (int neighbor : adjacency_list[current]) {
                if (dfs(neighbor, target)) return true;
            }
            return false;
        };

        return dfs(input, output);
    };

    if (would_create_cycle(genome.links, input_id, output_id)) {
        return;
    }

    LinkGene new_link{link_id, rng.next_gaussian(0.0, 1.0), true};
    genome.add_link(new_link);
}

void mutate_remove_link(Genome& genome) {
    if (genome.links.empty()) {
        return;
    }
    int index = rng.next_int(genome.links.size() - 1);
    genome.links.erase(genome.links.begin() + index);
}

void mutate_add_neuron(Genome& genome) {
    if (genome.links.empty()) {
        return;
    }

    int index = rng.next_int(genome.links.size() - 1);
    auto& link_to_split = genome.links[index];
    link_to_split.is_enabled = false;

    NeuronGene new_neuron = {static_cast<int>(genome.neurons.size()), rng.next_gaussian(0.0, 1.0), Activation{}};
    genome.add_neuron(new_neuron);

    LinkId link_id = link_to_split.link_id;
    double weight = link_to_split.weight;

    genome.add_link({{link_id.input_id, new_neuron.neuron_id}, 1.0, true});
    genome.add_link({{new_neuron.neuron_id, link_id.output_id}, weight, true});
}

void mutate_remove_neuron(Genome& genome) {
    if (genome.neurons.empty()) {
        return;
    }

    int index = rng.next_int(genome.neurons.size() - 1);
    int neuron_id = genome.neurons[index].neuron_id;

    genome.neurons.erase(genome.neurons.begin() + index);

    auto link_it = genome.links.begin();
    while (link_it != genome.links.end()) {
        if (link_it->link_id.input_id == neuron_id || link_it->link_id.output_id == neuron_id) {
            link_it = genome.links.erase(link_it);
        } else {
            ++link_it;
        }
    }
}

struct DoubleConfig {
    double init_mean = 0.0;
    double init_stdev = 1.0;
    double min = -20.0;
    double max = 20.0;
    double mutation_rate = 0.2;
    double mutate_power = 1.2;
    double replace_rate = 0.05;
    int population_size = 150;
    int num_inputs = 24;
    int num_outputs = 4;
    double survival_threshold = 0.2;
};

DoubleConfig config;

double clamp(double x, double min, double max) {
    return std::min(max, std::max(min, x));
}

double clamp(double x) {
    return clamp(x, config.min, config.max);
}

double new_value() {
    return clamp(rng.next_gaussian(config.init_mean, config.init_stdev));
}

double mutate_delta(double value) {
    double delta = rng.next_gaussian(0.0, config.mutate_power);
    return clamp(value + delta);
}

class Population {
public:
    Population(const DoubleConfig& neat_config, RNG& rng)
        : config(neat_config), rng(rng), next_genome_id(0), next_individual_id(0) {
        for (int i = 0; i < neat_config.population_size; ++i) {
            individuals.push_back({new_genome(), kFitnessNotComputed});
        }
    }

    template<typename FitnessFn>
    Individual run(FitnessFn compute_fitness, int num_generations) {
        for (int i = 0; i < num_generations; ++i) {
            compute_fitness(individuals.begin(), individuals.end());
            update_best();
            individuals = reproduce();
        }
        return best;
    }

private:
    const double kFitnessNotComputed = -1.0;

    Genome new_genome() {
        Genome genome{next_genome_id++, config.num_inputs, config.num_outputs};

        for (int neuron_id = 0; neuron_id < config.num_outputs; ++neuron_id) {
            genome.add_neuron(new_neuron(neuron_id));
        }

        for (int i = 0; i < config.num_inputs; ++i) {
            int input_id = -i - 1;
            for (int output_id = 0; output_id < config.num_outputs; ++output_id) {
                genome.add_link(new_link(input_id, output_id));
            }
        }
        return genome;
    }

    NeuronGene new_neuron(int neuron_id) {
        return {neuron_id, new_value(), Activation{}};
    }

    LinkGene new_link(int input_id, int output_id) {
        LinkId link_id{input_id, output_id};
        return {link_id, new_value(), true};
    }

    void update_best() {
        best = *std::max_element(individuals.begin(), individuals.end(), [](const Individual& a, const Individual& b) {
            return a.fitness < b.fitness;
        });
    }

    std::vector<Individual> reproduce() {
    auto old_members = sort_individuals_by_fitness(individuals);
    int reproduction_cutoff = static_cast<int>(std::ceil(config.survival_threshold * old_members.size()));
    std::vector<Individual> new_generation;
    int spawn_size = config.population_size;
    while (spawn_size-- > 0) {
        const auto& p1 = *rng.choose_random(old_members.begin(), old_members.begin() + reproduction_cutoff);
        const auto& p2 = *rng.choose_random(old_members.begin(), old_members.begin() + reproduction_cutoff);
        Genome offspring_genome = crossover(p1, p2);
        mutate(offspring_genome);
        new_generation.push_back({offspring_genome, kFitnessNotComputed});
    }
    return new_generation;
    }


    std::vector<Individual> sort_individuals_by_fitness(std::vector<Individual>& individuals) {
        std::sort(individuals.begin(), individuals.end(), [](const Individual& a, const Individual& b) {
            return a.fitness > b.fitness;
        });
        return individuals;
    }

    void mutate(Genome& genome) {
        if (rng.dist_01(rng.rng) < config.mutation_rate) {
            mutate_add_link(genome);
        }
        if (rng.dist_01(rng.rng) < config.mutation_rate) {
            mutate_remove_link(genome);
        }
        if (rng.dist_01(rng.rng) < config.mutation_rate) {
            mutate_add_neuron(genome);
        }
        if (rng.dist_01(rng.rng) < config.mutation_rate) {
            mutate_remove_neuron(genome);
        }
    }

    const DoubleConfig& config;
    RNG& rng;
    int next_genome_id;
    int next_individual_id;
    Individual best;
    std::vector<Individual> individuals;
};

// Neuron and Neural Network Definitions
struct NeuronInput {
    int input_id;
    double weight;
};

struct Neuron {
    ActivationFn activation;
    double bias;
    std::vector<NeuronInput> inputs;
    int neuron_id;
};

class FeedForwardNeuralNetwork {
public:
    FeedForwardNeuralNetwork(std::vector<int> input_ids, std::vector<int> output_ids, std::vector<Neuron> neurons)
        : input_ids(std::move(input_ids)), output_ids(std::move(output_ids)), neurons(std::move(neurons)) {}

    std::vector<double> activate(const std::vector<double>& inputs) const {
        assert(inputs.size() == input_ids.size());

        std::unordered_map<int, double> values;
        for (size_t i = 0; i < inputs.size(); ++i) {
            int input_id = input_ids[i];
            values[input_id] = inputs[i];
        }

        for (const auto& neuron : neurons) {
            double value = 0.0;
            for (const NeuronInput& input : neuron.inputs) {
                value += values.at(input.input_id) * input.weight;
            }
            value += neuron.bias;
            if (neuron.activation) {
                value = neuron.activation(value);
            }
            values[neuron.neuron_id] = value;
        }

        std::vector<double> outputs;
        for (int output_id : output_ids) {
            outputs.push_back(values.at(output_id));
        }
        return outputs;
    }

private:
    std::vector<int> input_ids;
    std::vector<int> output_ids;
    std::vector<Neuron> neurons;
};

FeedForwardNeuralNetwork create_from_genome(const Genome& genome) {
    std::vector<int> inputs = genome.make_input_ids();
    std::vector<int> outputs = genome.make_output_ids();

    std::vector<Neuron> neurons;
    for (const auto& neuron_gene : genome.neurons) {
        std::vector<NeuronInput> neuron_inputs;
        for (const auto& link : genome.links) {
            if (neuron_gene.neuron_id == link.link_id.output_id) {
                neuron_inputs.push_back(NeuronInput{link.link_id.input_id, link.weight});
            }
        }
        neurons.push_back(Neuron{
            std::get<ActivationFn>(neuron_gene.activation.fn),
            neuron_gene.bias,
            neuron_inputs,
            neuron_gene.neuron_id
        });
    }

    return FeedForwardNeuralNetwork(inputs, outputs, neurons);
}

// Snake Game Definitions
enum class Action {
    DoNothing,
    RotateLeft,
    RotateRight,
};

enum class Result {
    Running,
    GameOver,
    Winner,
};

enum class Direction {
    Up,
    Down,
    Left,
    Right,
};

struct Coordinates {
    int row;
    int column;

    bool operator==(const Coordinates& other) const {
        return row == other.row && column == other.column;
    }
};

namespace std {
    template <>
    struct hash<Coordinates> {
        std::size_t operator()(const Coordinates& c) const noexcept {
            return std::hash<int>()(c.row) ^ std::hash<int>()(c.column);
        }
    };
}

struct Snake {
    std::deque<Coordinates> body;

    Coordinates head() const { return body.front(); }
    Coordinates tail() const { return body.back(); }

    bool contains(const Coordinates& c) const {
        return std::find(body.begin(), body.end(), c) != body.end();
    }

    void grow(const Coordinates& new_head) {
        body.push_front(new_head);
    }

    void move(const Coordinates& new_head) {
        body.push_front(new_head);
        body.pop_back();
    }
};

class SnakeEngine {
public:
    int get_width() const { return width; }
    int get_height() const { return height; }
    SnakeEngine(int width, int height, bool allow_through_walls = false)
        : width(width), height(height), allow_through_walls(allow_through_walls), score(0), current_direction(Direction::Right) {
        reset();
    }

    Result process(Action action) {
        update_direction(action);
        Coordinates new_head = get_next_head(snake.head(), current_direction);

        if (hits_wall(new_head) || snake.contains(new_head)) {
            return Result::GameOver;
        }

        if (new_head == food) {
            score++;
            if (snake.body.size() == width * height) {
                return Result::Winner;
            }
            generate_food();
            snake.grow(new_head);
        } else {
            snake.move(new_head);
        }

        return Result::Running;
    }

    void reset() {
        snake.body = {{Coordinates{height / 2, width / 2}}};
        score = 0;
        current_direction = Direction::Right;
        generate_food();
    }

    int get_score() const { return score; }

    const Snake& get_snake() const { return snake; }
    const Coordinates& get_food() const { return food; }
    Direction get_direction() const { return current_direction; }

private:
    Snake snake;
    Coordinates food;
    bool allow_through_walls;
    int width;
    int height;
    int score;
    Direction current_direction;

    void generate_food() {
        do {
            food.row = rng.next_int(height - 1);
            food.column = rng.next_int(width - 1);
        } while (snake.contains(food));
    }

    Coordinates get_next_head(const Coordinates& head, Direction direction) const {
        Coordinates new_head = head;
        switch (direction) {
            case Direction::Up: new_head.row--; break;
            case Direction::Down: new_head.row++; break;
            case Direction::Left: new_head.column--; break;
            case Direction::Right: new_head.column++; break;
        }

        if (allow_through_walls) {
            new_head.row = (new_head.row + height) % height;
            new_head.column = (new_head.column + width) % width;
        }

        return new_head;
    }

    bool hits_wall(const Coordinates& c) const {
        return !allow_through_walls && (c.row < 0 || c.row >= height || c.column < 0 || c.column >= width);
    }

    void update_direction(Action action) {
        switch (action) {
            case Action::RotateLeft:
                current_direction = static_cast<Direction>((static_cast<int>(current_direction) + 3) % 4);
                break;
            case Action::RotateRight:
                current_direction = static_cast<Direction>((static_cast<int>(current_direction) + 1) % 4);
                break;
            default:
                break;
        }
    }
};

// Creating the neural network controlled snake agent
Action map_nn_output_to_action(const std::vector<double>& nn_output) {
    auto max_it = std::max_element(nn_output.begin(), nn_output.end());
    int max_idx = std::distance(nn_output.begin(), max_it);
    switch (max_idx) {
        case 0: return Action::DoNothing;
        case 1: return Action::RotateLeft;
        case 2: return Action::RotateRight;
        default: return Action::DoNothing;
    }
}



double evaluate_fitness(const FeedForwardNeuralNetwork& nn, SnakeEngine& snake_engine) {
    snake_engine.reset();
    Result result = Result::Running;
    int steps_without_food = 0;
    int max_steps_without_food = 100;

    while (result == Result::Running && steps_without_food < max_steps_without_food) {
        std::vector<double> inputs = generate_inputs(snake_engine);
        
        // Predict the next action using the neural network
        std::vector<double> nn_output = nn.activate(inputs);
        Action action = map_nn_output_to_action(nn_output);

        // Execute the action on the snake engine
        result = snake_engine.process(action);

        // Count steps without food
        if (result == Result::Running) {
            steps_without_food++;
        } else {
            steps_without_food = 0;
        }
    }

    // Fitness is the score plus a small penalty for excess steps without food
    return static_cast<double>(snake_engine.get_score()) - 0.1 * static_cast<double>(steps_without_food);
}

std::vector<double> generate_inputs(SnakeEngine& snake_engine) {
    const Snake& snake = snake_engine.get_snake();
    const Coordinates& head = snake.head();
    const Coordinates& food = snake_engine.get_food();
    Direction direction = snake_engine.get_direction();

    // Inputs: normalized distances to walls, food, and snake body
    auto normalize = [&](int val, int max_val) {
        return static_cast<double>(val) / static_cast<double>(max_val);
    };

    std::vector<double> inputs;

    // Add distances to walls
    inputs.push_back(normalize(head.row, snake_engine.get_height()));
    inputs.push_back(normalize(snake_engine.get_height() - head.row, snake_engine.get_height()));
    inputs.push_back(normalize(head.column, snake_engine.get_width()));
    inputs.push_back(normalize(snake_engine.get_width() - head.column, snake_engine.get_width()));

    // Add distances to food
    inputs.push_back(normalize(food.row - head.row, snake_engine.get_height()));
    inputs.push_back(normalize(food.column - head.column, snake_engine.get_width()));

    // Add current direction to the inputs
    inputs.push_back(direction == Direction::Up ? 1.0 : 0.0);
    inputs.push_back(direction == Direction::Right ? 1.0 : 0.0);
    inputs.push_back(direction == Direction::Down ? 1.0 : 0.0);
    inputs.push_back(direction == Direction::Left ? 1.0 : 0.0);

    // Add nearby snake body information
    std::unordered_map<Coordinates, double> distances;
    for (const auto& part : snake.body) {
        int row_diff = part.row - head.row;
        int col_diff = part.column - head.column;

        if (row_diff == 0 && col_diff == 0) continue;

        if (row_diff == 0 && col_diff > 0) {
            distances[Coordinates{0, 1}] = normalize(col_diff, snake_engine.get_width());
        } else if (row_diff == 0 && col_diff < 0) {
            distances[Coordinates{0, -1}] = normalize(-col_diff, snake_engine.get_width());
        } else if (col_diff == 0 && row_diff > 0) {
            distances[Coordinates{1, 0}] = normalize(row_diff, snake_engine.get_height());
        } else if (col_diff == 0 && row_diff < 0) {
            distances[Coordinates{-1, 0}] = normalize(-row_diff, snake_engine.get_height());
        }
    }

    // Default values if no body part nearby
    inputs.push_back(distances[Coordinates{-1, 0}]);
    inputs.push_back(distances[Coordinates{1, 0}]);
    inputs.push_back(distances[Coordinates{0, -1}]);
    inputs.push_back(distances[Coordinates{0, 1}]);

    return inputs;
}

// Fitness computation function for the entire population
void compute_fitness(std::vector<Individual>::iterator begin, std::vector<Individual>::iterator end, SnakeEngine& snake_engine) {
    for (auto it = begin; it != end; ++it) {
        FeedForwardNeuralNetwork nn = create_from_genome(it->genome);
        it->fitness = evaluate_fitness(nn, snake_engine);
    }
}

// SFML-based Snake Game UI
class SnakeUI {
public:
    SnakeUI(int width, int height, int cell_size)
        : width(width), height(height), cell_size(cell_size), window(sf::VideoMode(width * cell_size, height * cell_size), "Neuroevolution Snake") {
        window.setFramerateLimit(60);
        snake_shape.setSize(sf::Vector2f(static_cast<float>(cell_size), static_cast<float>(cell_size)));
        snake_shape.setFillColor(sf::Color::Green);
        food_shape.setSize(sf::Vector2f(static_cast<float>(cell_size), static_cast<float>(cell_size)));
        food_shape.setFillColor(sf::Color::Red);
    }

    void draw(const Snake& snake, const Coordinates& food) {
        window.clear();

        for (const auto& part : snake.body) {
            snake_shape.setPosition(static_cast<float>(part.column * cell_size), static_cast<float>(part.row * cell_size));
            window.draw(snake_shape);
        }

        food_shape.setPosition(static_cast<float>(food.column * cell_size), static_cast<float>(food.row * cell_size));
        window.draw(food_shape);

        window.display();
    }

    bool is_open() const {
        return window.isOpen();
    }

    void process_events() {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }
    }

private:
    int width;
    int height;
    int cell_size;
    sf::RenderWindow window;
    sf::RectangleShape snake_shape;
    sf::RectangleShape food_shape;
};

// Function to visualize the best individual from the population
        
void visualize_best_individual(const Individual& best, SnakeEngine& snake_engine, SnakeUI& snake_ui) {
    FeedForwardNeuralNetwork nn = create_from_genome(best.genome);
    snake_engine.reset();
    Result result = Result::Running;
    int steps_without_food = 0;
    int max_steps_without_food = 100;

    while (result == Result::Running && snake_ui.is_open() && steps_without_food < max_steps_without_food) {
        std::vector<double> inputs = generate_inputs(snake_engine);

        // Predict the next action using the neural network
        std::vector<double> nn_output = nn.activate(inputs);
        Action action = map_nn_output_to_action(nn_output);

        // Execute the action on the snake engine
        result = snake_engine.process(action);

        // Count steps without food
        if (result == Result::Running) {
            steps_without_food++;
        } else {
            steps_without_food = 0;
        }

        // Draw the game
        snake_ui.process_events();
        snake_ui.draw(snake_engine.get_snake(), snake_engine.get_food());

        // Introduce a small delay to make the visualization easier to follow
        sf::sleep(sf::milliseconds(50));
    }
}

int main() {
    DoubleConfig neat_config;
    neat_config.population_size = 150;
    neat_config.num_inputs = 24;
    neat_config.num_outputs = 4;
    neat_config.survival_threshold = 0.2;
    neat_config.mutation_rate = 0.2;

    RNG rng;
    Population population(neat_config, rng);

    SnakeEngine snake_engine(20, 20, false);
    SnakeUI snake_ui(20, 20, 20);

    auto compute_fitness_lambda = [&](auto begin, auto end) {
        compute_fitness(begin, end, snake_engine);
    };

    int num_generations = 100;
    Individual best = population.run(compute_fitness_lambda, num_generations);

    std::cout << "Best fitness: " << best.fitness << std::endl;

    // Visualize the best individual
    visualize_best_individual(best, snake_engine, snake_ui);

    return 0;
}