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

