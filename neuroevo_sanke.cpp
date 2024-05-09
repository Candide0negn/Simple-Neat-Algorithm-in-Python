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
    std::uniform_int_distribution<int> dist_int;
    
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

    int next_int(int max) {
        std::uniform_int_distribution<int> dist(0, max);
        return dist(rng);
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

struct LinkIdHash {
    std::size_t operator()(const LinkId& link) const noexcept {
        return std::hash<int>()(link.input_id) ^ std::hash<int>()(link.output_id);
    }
};

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
    Activation activation = rng.choose(0.5, a.activation, b.activation);
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

    auto existing_link = genome.find_link(link_id);
    if (existing_link) {
        existing_link->is_enabled = true;
        return;
    }

    auto would_create_cycle = [&](const std::vector<LinkGene>& links, int input, int output) {
        return false;
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
    auto to_remove_it = std::find_if(genome.links.begin(), genome.links.end(), [&](const LinkGene& link) {
        return rng.choose_random(genome.links) == link;
    });
    if (to_remove_it != genome.links.end()) {
        genome.links.erase(to_remove_it);
    }
}

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

    if (neuron_it

    