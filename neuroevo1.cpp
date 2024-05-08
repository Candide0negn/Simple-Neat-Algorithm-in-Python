#include <iostream>
#include <vector>
#include <unordered_map>
#include <optional>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <random>
#include <variant>

// Placeholder for your random number generator
struct RNG {
    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<double> dist_01{0.0, 1.0};

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

    double next_gaussian(double mean, double stdev) {
        std::normal_distribution<double> dist(mean, stdev);
        return dist(rng);
    }
};

RNG rng;

// Activation functions
using ActivationFn = std::function<double(double)>;

struct Activation {
    std::variant<std::monostate, ActivationFn> fn;
};

// Unique identifier for a link between neurons
struct LinkId {
    int input_id;
    int output_id;
    bool operator==(const LinkId& other) const {
        return input_id == other.input_id && output_id == other.output_id;
    }
};

// Hash function for LinkId to be used in unordered_map
struct LinkIdHash {
    std::size_t operator()(const LinkId& link) const noexcept {
        return std::hash<int>()(link.input_id) ^ std::hash<int>()(link.output_id);
    }
};

// Struct representing a neuron gene
struct NeuronGene {
    int neuron_id;
    double bias;
    Activation activation;
};

// Struct representing a link gene
struct LinkGene {
    LinkId link_id;
    double weight;
    bool is_enabled;
};

// Struct representing a genome
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

// Function to crossover two neuron genes
NeuronGene crossover_neuron(const NeuronGene& a, const NeuronGene& b) {
    assert(a.neuron_id == b.neuron_id);
    int neuron_id = a.neuron_id;
    double bias = rng.choose(0.5, a.bias, b.bias);
    Activation activation = rng.choose(0.5, a.activation, b.activation);
    return {neuron_id, bias, activation};
}

// Function to crossover two link genes
LinkGene crossover_link(const LinkGene& a, const LinkGene& b) {
    assert(a.link_id == b.link_id);
    LinkId link_id = a.link_id;
    double weight = rng.choose(0.5, a.weight, b.weight);
    bool is_enabled = rng.choose(0.5, a.is_enabled, b.is_enabled);
    return {link_id, weight, is_enabled};
}

// Struct representing an individual in the population
struct Individual {
    Genome genome;
    double fitness;
};

// Function to crossover two individuals' genomes
Genome crossover(const Individual& dominant, const Individual& recessive) {
    Genome offspring{dominant.genome.genome_id, dominant.genome.num_inputs, dominant.genome.num_outputs};

    // Inherit neuron genes
    for (const auto& dominant_neuron : dominant.genome.neurons) {
        int neuron_id = dominant_neuron.neuron_id;
        std::optional<NeuronGene> recessive_neuron = recessive.genome.find_neuron(neuron_id);
        if (!recessive_neuron) {
            offspring.add_neuron(dominant_neuron);
        } else {
            offspring.add_neuron(crossover_neuron(dominant_neuron, *recessive_neuron));
        }
    }

    // Inherit link genes
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

// Function to mutate a genome by adding a link
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

    // Don't duplicate links
    auto existing_link = genome.find_link(link_id);
    if (existing_link) {
        // At least enable it
        existing_link->is_enabled = true;
        return;
    }

    auto would_create_cycle = [&](const std::vector<LinkGene>& links, int input, int output) {
        return false; // Placeholder for an actual cycle detection algorithm
    };

    if (would_create_cycle(genome.links, input_id, output_id)) {
        return;
    }

    LinkGene new_link{link_id, rng.next_gaussian(0.0, 1.0), true};
    genome.add_link(new_link);
}

// Function to mutate a genome by removing a link
void mutate_remove_link(Genome& genome) {
    if (genome.links.empty()) {
        return;
    }
    auto to_remove_it = std::find_if(genome.links.begin(), genome.links.end(), [&](const LinkGene& link) {
        return rng.choose_random(genome.links) == link;
    });
    if (to_remove_it != genome.links.end()) {
        genome.links.erase(to_remove_it);
    }
}

// Function to mutate a genome by adding a neuron
void mutate_add_neuron(Genome& genome) {
    if (genome.links.empty()) {
        return;
    }

    auto& link_to_split = rng.choose_random(genome.links);
    link_to_split.is_enabled = false;

    NeuronGene new_neuron = {static_cast<int>(genome.neurons.size()), rng.next_gaussian(0.0, 1.0), Activation{}};
    genome.add_neuron(new_neuron);

    LinkId link_id = link_to_split.link_id;
    double weight = link_to_split.weight;

    genome.add_link({{link_id.input_id, new_neuron.neuron_id}, 1.0, true});
    genome.add_link({{new_neuron.neuron_id, link_id.output_id}, weight, true});
}

// Function to mutate a genome by removing a neuron
void mutate_remove_neuron(Genome& genome) {
    if (genome.neurons.empty()) {
        return;
    }

    auto choose_random_hidden = [&](const std::vector<NeuronGene>& neurons) {
        std::vector<int> hidden_ids;
        for (const auto& neuron : neurons) {
            if (neuron.neuron_id >= genome.num_inputs && neuron.neuron_id < genome.num_inputs + genome.num_outputs) {
                hidden_ids.push_back(neuron.neuron_id);
            }
        }
        return rng.choose_random(hidden_ids);
    };

    auto neuron_it = std::find_if(genome.neurons.begin(), genome.neurons.end(), [&](const NeuronGene& neuron) {
        return neuron.neuron_id == choose_random_hidden(genome.neurons);
    });

    if (neuron_it != genome.neurons.end()) {
        auto& links = genome.links;
        links.erase(std::remove_if(links.begin(), links.end(), [neuron_it](const LinkGene& link) {
            return link.link_id.input_id == neuron_it->neuron_id || link.link_id.output_id == neuron_it->neuron_id;
        }), links.end());

        genome.neurons.erase(neuron_it);
    }
}

// Configuration structure for double values
struct DoubleConfig {
    double init_mean = 0.0;
    double init_stdev = 1.0;
    double min = -20.0;
    double max = 20.0;
    double mutation_rate = 0.2;
    double mutate_power = 1.2;
    double replace_rate = 0.05;
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

// Population class to manage a population of individuals
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

        // Fully connected feed-forward
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
            const auto& p1 = *rng.choose_random(old_members, reproduction_cutoff);
            const auto& p2 = *rng.choose_random(old_members, reproduction_cutoff);
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

// Struct representing a neuron's input connection
struct NeuronInput {
    int input_id;
    double weight;
};

// Struct representing a neuron in a neural network
struct Neuron {
    ActivationFn activation;
    double bias;
    std::vector<NeuronInput> inputs;
};

// Class representing a feed-forward neural network
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

        for (int output_id : output_ids) {
            values[output_id] = 0.0;
        }

        for (const auto& neuron : neurons) {
            double value = 0.0;
            for (const NeuronInput& input : neuron.inputs) {
                assert(values.find(input.input_id) != values.end());
                value += values[input.input_id] * input.weight;
            }
            value += neuron.bias;
            value = neuron.activation(value);
            values[neuron.neuron_id] = value;
        }

        std::vector<double> outputs;
        for (int output_id : output_ids) {
            assert(values.find(output_id) != values.end());
            outputs.push_back(values[output_id]);
        }
        return outputs;
    }

private:
    std::vector<int> input_ids;
    std::vector<int> output_ids;
    std::vector<Neuron> neurons;
};

// Function to create a feed-forward neural network from a genome
FeedForwardNeuralNetwork create_from_genome(const Genome& genome) {
    std::vector<int> inputs = genome.make_input_ids();
    std::vector<int> outputs = genome.make_output_ids();

    auto feed_forward_layers = [&](const std::vector<int>& input_ids, const std::vector<int>& output_ids, const std::vector<LinkGene>& links) {
        std::vector<std::vector<int>> layers;
        std::unordered_map<int, int> layer_map;

        for (int input_id : input_ids) {
            layer_map[input_id] = 0;
        }

        int layer_idx = 1;
        std::vector<int> current_layer = output_ids;
        while (!current_layer.empty()) {
            layers.push_back(current_layer);
            std::vector<int> next_layer;

            for (int neuron_id : current_layer) {
                for (const auto& link : links) {
                    if (link.link_id.output_id == neuron_id && layer_map.find(link.link_id.input_id) == layer_map.end()) {
                        layer_map[link.link_id.input_id] = layer_idx;
                        next_layer.push_back(link.link_id.input_id);
                    }
                }
            }

            current_layer = next_layer;
            ++layer_idx;
        }

        std::reverse(layers.begin(), layers.end());
        return layers;
    };

    std::vector<std::vector<int>> layers = feed_forward_layers(inputs, outputs, genome.links);

    std::vector<Neuron> neurons;
    for (const auto& layer : layers) {
        for (int neuron_id : layer) {
            std::vector<NeuronInput> neuron_inputs;
            for (const auto& link : genome.links) {
                if (neuron_id == link.link_id.output_id) {
                    neuron_inputs.emplace_back(NeuronInput{link.link_id.input_id, link.weight});
                }
            }

            std::optional<NeuronGene> neuron_gene_opt = genome.find_neuron(neuron_id);
            assert(neuron_gene_opt.has_value());

            neurons.emplace_back(Neuron{std::get<ActivationFn>(neuron_gene_opt->activation.fn), neuron_gene_opt->bias, std::move(neuron_inputs)});
        }
    }

    return FeedForwardNeuralNetwork{std::move(inputs), std::move(outputs), std::move(neurons)};
}

// Main function
int main(int argc, char** argv) {
    DoubleConfig neat_config;
    neat_config.population_size = 150;
    neat_config.num_inputs = 24;
    neat_config.num_outputs = 4;
    neat_config.survival_threshold = 0.2;
    neat_config.mutation_rate = 0.2;

    RNG rng;
    Population population{neat_config, rng};

    auto compute_fitness = [](auto begin, auto end) {
    for (auto it = begin; it != end; ++it) {
        // Placeholder for fitness computation logic
        it->fitness = rng.next_gaussian(0.0, 1.0);
    }
};

int num_generations = 100;
std::string winner_filename = "winner_genome.txt";

auto winner = population.run(compute_fitness, num_generations);

// Function to save the winner genome to a file (as an example)
void save(const Genome& genome, const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Unable to open file for saving." << std::endl;
        return;
    }

    file << "Genome ID: " << genome.genome_id << '\n';
    file << "Num Inputs: " << genome.num_inputs << '\n';
    file << "Num Outputs: " << genome.num_outputs << '\n';

    file << "Neurons:\n";
    for (const auto& neuron : genome.neurons) {
        file << "  Neuron ID: " << neuron.neuron_id << ", Bias: " << neuron.bias << '\n';
    }

    file << "Links:\n";
    for (const auto& link : genome.links) {
        file << "  Link: " << link.link_id.input_id << " -> " << link.link_id.output_id
             << ", Weight: " << link.weight << ", Enabled: " << link.is_enabled << '\n';
    }

    file.close();
}

save(winner.genome, winner_filename);

return 0;
}