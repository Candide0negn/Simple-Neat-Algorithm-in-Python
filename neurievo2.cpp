#include <vector>
#include <cassert>
#include <algorithm>
#include <optional>
#include <cmath>
#include <random>
#include <unordered_map>
#include <variant>

// Forward declarations for custom types
struct NeuronGene;
struct LinkGene;
struct Genome;
class RNG;

// Activation function type using std::variant for flexibility
using ActivationFn = std::variant<std::function<double(double)>>;

// Activation functions example
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Random Number Generator Wrapper
class RNG {
public:
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;

    RNG() : gen(std::random_device()()), dis(0.0, 1.0) {}

    double random() {
        return dis(gen);
    }

    int random_int(int min, int max) {
        std::uniform_int_distribution<> dist(min, max);
        return dist(gen);
    }

    double choose(double probability, double option1, double option2) {
        return random() < probability ? option1 : option2;
    }

    template<typename T>
    T& choose_random(std::vector<T>& items) {
        return items[random_int(0, items.size() - 1)];
    }

    double next_gaussian(double mean, double stddev) {
        std::normal_distribution<> dist(mean, stddev);
        return dist(gen);
    }
};

// NeuronGene structure
struct NeuronGene {
    int neuron_id;
    double bias;
    ActivationFn activation;

    NeuronGene(int id, double b, ActivationFn act) : neuron_id(id), bias(b), activation(act) {}
};

// LinkId structure to uniquely identify links between neurons
struct LinkId {
    int input_id;
    int output_id;

    LinkId(int in, int out) : input_id(in), output_id(out) {}

    bool operator==(const LinkId& other) const {
        return input_id == other.input_id && output_id == other.output_id;
    }
};

// LinkGene structure
struct LinkGene {
    LinkId link_id;
    double weight;
    bool is_enabled;

    LinkGene(LinkId id, double w, bool enabled) : link_id(id), weight(w), is_enabled(enabled) {}
};

// Genome structure containing all the genes
struct Genome {
    int genome_id;
    int num_inputs;
    int num_outputs;
    std::vector<NeuronGene> neurons;
    std::vector<LinkGene> links;

    Genome(int id, int inputs, int outputs) : genome_id(id), num_inputs(inputs), num_outputs(outputs) {}

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
};

// Crossover functions for neurons and links
NeuronGene crossover_neuron(const NeuronGene& a, const NeuronGene& b, RNG& rng) {
    assert(a.neuron_id == b.neuron_id);
    int neuron_id = a.neuron_id;
    double bias = rng.choose(0.5, a.bias, b.bias);
    ActivationFn activation = rng.choose(0.5, a.activation, b.activation);
    return NeuronGene(neuron_id, bias, activation);
}

LinkGene crossover_link(const LinkGene& a, const LinkGene& b, RNG& rng) {
    assert(a.link_id == b.link_id);
    LinkId link_id = a.link_id;
    double weight = rng.choose(0.5, a.weight, b.weight);
    bool is_enabled = rng.choose(0.5, a.is_enabled, b.is_enabled);
    return LinkGene(link_id, weight, is_enabled);
}

// Individual structure for the population
struct Individual {
    Genome genome;
    double fitness;

    Individual(Genome g, double f) : genome(g), fitness(f) {}
};

// Crossover for genomes
Genome crossover(const Individual& dominant, const Individual& recessive, RNG& rng) {
    Genome offspring(dominant.genome.genome_id + 1, dominant.genome.num_inputs, dominant.genome.num_outputs);
    
    // Inherit neuron genes
    for (const auto& dominant_neuron : dominant.genome.neurons) {
        int neuron_id = dominant_neuron.neuron_id;
        std::optional<NeuronGene> recessive_neuron = recessive.genome.find_neuron(neuron_id);
        if (!recessive_neuron) {
            offspring.add_neuron(dominant_neuron);
        } else {
            offspring.add_neuron(crossover_neuron(dominant_neuron, *recessive_neuron, rng));
        }
    }

    // Inherit link genes
    for (const auto& dominant_link : dominant.genome.links) {
        LinkId link_id = dominant_link.link_id;
        std::optional<LinkGene> recessive_link = recessive.genome.find_link(link_id);
        if (!recessive_link) {
            offspring.add_link(dominant_link);
        } else {
            offspring.add_link(crossover_link(dominant_link, *recessive_link, rng));
        }
    }

    return offspring;
}

// Mutation functions
void mutate_add_link(Genome& genome, RNG& rng) {
    if (genome.neurons.empty()) return;

    int input_id = rng.random_int(0, genome.num_inputs - 1);
    int output_id = rng.random_int(genome.num_inputs, genome.num_inputs + genome.num_outputs - 1);

    LinkId link_id(input_id, output_id);

    auto existing_link = genome.find_link(link_id);
    if (existing_link) {
        existing_link->is_enabled = true;
        return;
    }

    if (would_create_cycle(genome.links, input_id, output_id)) {
        return;
    }

    LinkGene new_link(link_id, rng.next_gaussian(0.0, 1.0), true);
    genome.add_link(new_link);
}

void mutate_remove_link(Genome& genome, RNG& rng) {
    if (genome.links.empty()) {
        return;
    }

    int index = rng.random_int(0, genome.links.size() - 1);
    genome.links.erase(genome.links.begin() + index);
}

void mutate_add_neuron(Genome& genome, RNG& rng) {
    if (genome.links.empty()) {
        return;
    }

    LinkGene& link_to_split = rng.choose_random(genome.links);
    link_to_split.is_enabled = false;

    int new_neuron_id = genome.neurons.size();
    NeuronGene new_neuron(new_neuron_id, rng.next_gaussian(0.0, 1.0), sigmoid);
    genome.add_neuron(new_neuron);

    LinkId link_id1(link_to_split.link_id.input_id, new_neuron_id);
    LinkGene new_link1(link_id1, 1.0, true);
    genome.add_link(new_link1);

    LinkId link_id2(new_neuron_id, link_to_split.link_id.output_id);
    LinkGene new_link2(link_id2, link_to_split.weight, true);
    genome.add_link(new_link2);
}

void mutate_remove_neuron(Genome& genome, RNG& rng) {
    if (genome.neurons.size() <= (genome.num_inputs + genome.num_outputs)) {
        return;
    }

    int index = rng.random_int(genome.num_inputs, genome.neurons.size() - 1);
    int neuron_id = genome.neurons[index].neuron_id;

    auto& links = genome.links;
    links.erase(std::remove_if(links.begin(), links.end(),
                               [neuron_id](const LinkGene& link) {
                                   return link.link_id.input_id == neuron_id || link.link_id.output_id == neuron_id;
                               }),
                links.end());

    genome.neurons.erase(genome.neurons.begin() + index);
}

// Configuration and utility functions
struct DoubleConfig {
    double init_mean = 0.0;
    double init_stdev = 1.0;
    double min = -20.0;
    double max = 20.0;
    double mutation_rate = 0.2;
    double mutate_power = 1.2;
    double replace_rate = 0.05;
};

double clamp(double x, const DoubleConfig& config) {
    return std::min(config.max, std::max(config.min, x));
}

// Main function to simulate the genetic algorithm
int main() {
    RNG rng;
    DoubleConfig config;

    // Example usage
    NeuronGene neuron1(1, 0.5, sigmoid);
    NeuronGene neuron2(2, -0.3, sigmoid);
    LinkGene link1(LinkId(1, 2), 0.9, true);

    std::vector<NeuronGene> neurons = {neuron1, neuron2};
    std::vector<LinkGene> links = {link1};

    Genome parent1(1, 1, 1);
    parent1.add_neuron(neuron1);
    parent1.add_neuron(neuron2);
    parent1.add_link(link1);

    Genome parent2(2, 1, 1);
    parent2.add_neuron(neuron1);
    parent2.add_neuron(NeuronGene(2, 0.1, sigmoid));
    parent2.add_link(LinkGene(LinkId(1, 2), -0.5, true));

    Individual mom(parent1, 0.9);
    Individual dad(parent2, 0.8);

    Genome offspring = crossover(mom, dad, rng);
    std::cout << "Offspring Genome ID: " << offspring.genome_id << "\n";
    std::cout << "Neurons in offspring:\n";
    for (const auto& neuron : offspring.neurons) {
        std::cout << "Neuron ID: " << neuron.neuron_id << ", Bias: " << neuron.bias << "\n";
    }
    std::cout << "Links in offspring:\n";
    for (const auto& link : offspring.links) {
        std::cout << "Link from " << link.link_id.input_id << " to " << link.link_id.output_id
                  << ", Weight: " << link.weight << ", Enabled: " << link.is_enabled << "\n";
    }

    // Mutations
    mutate_add_neuron(offspring, rng);
    mutate_add_link(offspring, rng);
    mutate_remove_neuron(offspring, rng);
    mutate_remove_link(offspring, rng);

    return 0;
}

// Function implementations

bool would_create_cycle(const std::vector<LinkGene>& links, int input_id, int output_id) {
    std::unordered_map<int, std::vector<int>> graph;
    for (const auto& link : links) {
        if (link.is_enabled) {
            graph[link.link_id.input_id].push_back(link.link_id.output_id);
        }
    }

    // Depth-first search (DFS) to detect a cycle
    std::unordered_set<int> visited;
    std::function<bool(int, int)> dfs = [&](int current, int target) -> bool {
        if (current == target) return true;
        if (visited.count(current)) return false;
        visited.insert(current);

        for (int neighbor : graph[current]) {
            if (dfs(neighbor, target)) return true;
        }
        return false;
    };

    return dfs(input_id, output_id);
}

// A simple utility function to create a new neuron gene
NeuronGene new_neuron(int neuron_id) {
    static RNG rng;
    return NeuronGene(neuron_id, rng.next_gaussian(0.0, 1.0), sigmoid);
}

// A utility function to create a new link gene
LinkGene new_link(int input_id, int output_id) {
    static RNG rng;
    return LinkGene(LinkId(input_id, output_id), rng.next_gaussian(0.0, 1.0), true);
}

// Population class to manage the evolutionary process
class Population {
public:
    NeatConfig config;
    RNG& rng;
    std::vector<Individual> individuals;

    Population(NeatConfig config, RNG& rng) : config(config), rng(rng) {
        for (int i = 0; i < config.population_size; i++) {
            individuals.push_back(Individual(new_genome(i), DoubleConfig().init_mean));
        }
    }

    Genome new_genome(int id) {
        Genome genome(id, config.num_inputs, config.num_outputs);
        // Fully connect inputs to outputs for initial population
        for (int i = 0; i < config.num_inputs; i++) {
            for (int j = 0; j < config.num_outputs; j++) {
                genome.add_link(new_link(i, config.num_inputs + j));
            }
        }
        return genome;
    }

    void compute_fitness() {
        for (auto& individual : individuals) {
            individual.fitness = evaluate(individual.genome); // User-defined
        }
    }

    Individual& select_parent() {
        std::partial_sum(individuals.begin(), individuals.end(), individuals.begin(),
                         [](const Individual& a, const Individual& b) {
                             return Individual(a.genome, a.fitness + b.fitness);
                         });
        double total_fitness = individuals.back().fitness;
        double slice = rng.random() * total_fitness;
        auto it = std::lower_bound(individuals.begin(), individuals.end(), slice,
                                   [](const Individual& ind, double value) {
                                       return ind.fitness < value;
                                   });
        return *it;
    }

    std::vector<Individual> reproduce() {
        std::vector<Individual> new_generation;
        while (new_generation.size() < individuals.size()) {
            Individual& p1 = select_parent();
            Individual& p2 = select_parent();
            Genome offspring_genome = crossover(p1, p2, rng);
            mutate_add_link(offspring_genome, rng);
            mutate_add_neuron(offspring_genome, rng);
            new_generation.emplace_back(offspring_genome, 0); // Fitness will be computed
        }
        return new_generation;
    }

    void evolve() {
        for (int generation = 0; generation < config.num_generations; ++generation) {
            compute_fitness();
            individuals = reproduce();
        }
    }
};

// Define a simple fitness function for demonstration
double evaluate(const Genome& genome) {
    // A very simple example fitness function
    return genome.neurons.size() * 10.0 + genome.links.size();
}

// Define a simple NeatConfig for the population
struct NeatConfig {
    int population_size = 100;
    int num_generations = 50;
    int num_inputs = 3;
    int num_outputs = 2;
};

int main() {
    RNG rng;
    NeatConfig config;
    Population population(config, rng);

    population.evolve();

    // Print the best individual's genome
    auto best = std::max_element(population.individuals.begin(), population.individuals.end(),
                                 [](const Individual& a, const Individual& b) {
                                     return a.fitness < b.fitness;
                                 });
    std::cout << "Best Individual Fitness: " << best->fitness << "\n";
    std::cout << "Neurons: " << best->genome.neurons.size() << "\n";
    std::cout << "Links: " << best->genome.links.size() << "\n";

    return 0;
}