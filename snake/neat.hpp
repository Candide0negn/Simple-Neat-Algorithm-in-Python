#pragma once

#include <vector>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <algorithm>
#include <cmath>
#include <random>
#include <cassert>
#include <iostream> 
#include <memory>
#include <functional>
#include <stdexcept>
#include <fstream>

class Rng {
public:
    Rng(unsigned seed) : gen(seed) {}

    double next_double() {
        return dist(gen);
    }

    int next_int(int max) {
        return std::uniform_int_distribution<>(0, max - 1)(gen);
    }

    double next_gaussian(double mean, double stdev) {
        return std::normal_distribution<>(mean, stdev)(gen);
    }

    bool choose(double probability) {
        return next_double() < probability;
    }

    template<typename T>
    T choose_random(const std::vector<T>& vec) {
        return vec[next_int(vec.size())];
    }

    template<typename Iter>
    Iter choose_random(Iter begin, Iter end) {
        std::uniform_int_distribution<> dist(0, std::distance(begin, end) - 1);
        std::advance(begin, dist(gen));
        return begin;
    }

private:
    std::mt19937 gen;
    std::uniform_real_distribution<> dist;
};

enum class Activation {
    Identity,
    Sigmoid,
    Tanh,
    ReLU
};

struct ActivationFn {
    double operator()(Activation activation, double x) const {
        switch (activation) {
            case Activation::Identity: return x;
            case Activation::Sigmoid: return 1.0 / (1.0 + std::exp(-x));
            case Activation::Tanh: return std::tanh(x);
            case Activation::ReLU: return std::max(0.0, x);
        }
        return x; // Should never reach here
    }
};

struct NeuronGene {
    int neuron_id;
    double bias;
    Activation activation;
};

struct LinkId {
    int input_id;
    int output_id;

    bool operator==(const LinkId& other) const {
        return input_id == other.input_id && output_id == other.output_id;
    }
};

namespace std {
    template<>
    struct hash<LinkId> {
        size_t operator()(const LinkId& id) const {
            return hash<int>()(id.input_id) ^ hash<int>()(id.output_id);
        }
    };
}

struct LinkGene {
    LinkId link_id;
    double weight;
    bool is_enabled;
};

class Genome {
public:
    Genome(int genome_id, int num_inputs, int num_outputs)
        : genome_id(genome_id), num_inputs(num_inputs), num_outputs(num_outputs) {}

    void add_neuron(const NeuronGene& neuron) {
        neurons.push_back(neuron);
    }

    void add_link(const LinkGene& link) {
        links.push_back(link);
    }

    std::optional<NeuronGene> find_neuron(int neuron_id) const {
        auto it = std::find_if(neurons.begin(), neurons.end(),
            [neuron_id](const NeuronGene& n) { return n.neuron_id == neuron_id; });
        if (it != neurons.end()) {
            return *it;
        }
        return std::nullopt;
    }

    std::optional<LinkGene> find_link(const LinkId& link_id) const {
        auto it = std::find_if(links.begin(), links.end(),
            [&link_id](const LinkGene& l) { return l.link_id == link_id; });
        if (it != links.end()) {
            return *it;
        }
        return std::nullopt;
    }

    int get_num_inputs() const { return num_inputs; }
    int get_num_outputs() const { return num_outputs; }
    const std::vector<NeuronGene>& get_neurons() const { return neurons; }
    std::vector<NeuronGene>& get_neurons() { return neurons; }
    const std::vector<LinkGene>& get_links() const { return links; }
    std::vector<LinkGene>& get_links() { return links; }

private:
    int genome_id;
    int num_inputs;
    int num_outputs;
    std::vector<NeuronGene> neurons;
    std::vector<LinkGene> links;
};

struct Individual {
    Genome genome;
    double fitness;

    Individual() : genome(0, 0, 0), fitness(0.0) {}
    Individual(Genome g, double f) : genome(std::move(g)), fitness(f) {}
};

class GenomeIndexer {
public:
    int next() { return counter++; }
private:
    int counter = 0;
};

class NeuronMutator {
public:
    NeuronGene new_neuron(int neuron_id, Rng& rng) {
        return {neuron_id, rng.next_gaussian(0, 1), Activation::Sigmoid};
    }
};

class LinkMutator {
public:
    LinkGene new_link(int input_id, int output_id, Rng& rng) {
        return {{input_id, output_id}, rng.next_gaussian(0, 1), true};
    }
};

NeuronGene crossover_neuron(const NeuronGene &a, const NeuronGene &b, Rng &rng) {
    assert(a.neuron_id == b.neuron_id);
    int neuron_id = a.neuron_id;
    double bias = rng.choose(0.5) ? a.bias : b.bias;
    Activation activation = rng.choose(0.5) ? a.activation : b.activation;
    return {neuron_id, bias, activation};
}

LinkGene crossover_link(const LinkGene &a, const LinkGene &b, Rng &rng) {
    assert(a.link_id == b.link_id);
    LinkId link_id = a.link_id;
    double weight = rng.choose(0.5) ? a.weight : b.weight;
    bool is_enabled = rng.choose(0.5) ? a.is_enabled : b.is_enabled;
    return {link_id, weight, is_enabled};
}

Genome crossover(const Individual &dominant, const Individual &recessive, GenomeIndexer &indexer, Rng &rng) {
    Genome offspring(indexer.next(), dominant.genome.get_num_inputs(), dominant.genome.get_num_outputs());
    
    for (const auto &dominant_neuron : dominant.genome.get_neurons()) {
        int neuron_id = dominant_neuron.neuron_id;
        auto recessive_neuron = recessive.genome.find_neuron(neuron_id);
        if (!recessive_neuron) {
            offspring.add_neuron(dominant_neuron);
        } else {
            offspring.add_neuron(crossover_neuron(dominant_neuron, *recessive_neuron, rng));
        }
    }

    for (const auto &dominant_link : dominant.genome.get_links()) {
        LinkId link_id = dominant_link.link_id;
        auto recessive_link = recessive.genome.find_link(link_id);
        if (!recessive_link) {
            offspring.add_link(dominant_link);
        } else {
            offspring.add_link(crossover_link(dominant_link, *recessive_link, rng));
        }
    }
    
    return offspring;
}

bool would_create_cycle(const std::vector<LinkGene>& links, int input_id, int output_id) {
    std::unordered_map<int, std::vector<int>> adjacency_list;
    for (const auto& link : links) {
        adjacency_list[link.link_id.input_id].push_back(link.link_id.output_id);
    }
    adjacency_list[input_id].push_back(output_id);
    
    std::unordered_set<int> visited;
    std::function<bool(int, std::unordered_set<int>&)> dfs = [&](int node, std::unordered_set<int>& path) -> bool {
        if (path.find(node) != path.end()) return true;
        if (visited.find(node) != visited.end()) return false;
        
        visited.insert(node);
        path.insert(node);
        
        for (int neighbor : adjacency_list[node]) {
            if (dfs(neighbor, path)) return true;
        }
        
        path.erase(node);
        return false;
    };

    std::unordered_set<int> path;
    return dfs(input_id, path);
}

void mutate_add_link(Genome &genome, Rng &rng, LinkMutator &link_mutator) {
    auto choose_random_input_or_hidden = [&](const std::vector<NeuronGene>& neurons) {
        std::vector<int> valid_ids;
        for (const auto& neuron : neurons) {
            if (neuron.neuron_id < 0 || neuron.neuron_id >= genome.get_num_outputs()) {
                valid_ids.push_back(neuron.neuron_id);
            }
        }
        return valid_ids[rng.next_int(valid_ids.size())];
    };

    auto choose_random_output_or_hidden = [&](const std::vector<NeuronGene>& neurons) {
        std::vector<int> valid_ids;
        for (const auto& neuron : neurons) {
            if (neuron.neuron_id >= 0) {
                valid_ids.push_back(neuron.neuron_id);
            }
        }
        return valid_ids[rng.next_int(valid_ids.size())];
    };

    int input_id = choose_random_input_or_hidden(genome.get_neurons());
    int output_id = choose_random_output_or_hidden(genome.get_neurons());
    LinkId link_id{input_id, output_id};

    auto existing_link = genome.find_link(link_id);
    if (existing_link) {
        existing_link->is_enabled = true;
        return;
    }

    if (would_create_cycle(genome.get_links(), input_id, output_id)) {
        return;
    }

    LinkGene new_link = link_mutator.new_link(input_id, output_id, rng);
    genome.add_link(new_link);
}

void mutate_remove_link(Genome &genome, Rng &rng) {
    if (genome.get_links().empty()) {
        return;
    }

    auto to_remove_it = rng.choose_random(genome.get_links().begin(), genome.get_links().end());
    genome.get_links().erase(to_remove_it);
}

void mutate_add_neuron(Genome &genome, Rng &rng, NeuronMutator &neuron_mutator, LinkMutator &link_mutator) {
    if (genome.get_links().empty()) {
        return;
    }

    auto& links = genome.get_links();
    auto link_it = rng.choose_random(links.begin(), links.end());
    link_it->is_enabled = false;

    NeuronGene new_neuron = neuron_mutator.new_neuron(genome.get_neurons().size(), rng);
    genome.add_neuron(new_neuron);

    LinkId link_id = link_it->link_id;
    double weight = link_it->weight;

    genome.add_link(link_mutator.new_link(link_id.input_id, new_neuron.neuron_id, rng));
    genome.add_link(link_mutator.new_link(new_neuron.neuron_id, link_id.output_id, rng));
}

void mutate_remove_neuron(Genome &genome, Rng &rng) {
    auto num_hidden = genome.get_neurons().size() - genome.get_num_inputs() - genome.get_num_outputs();
    if (num_hidden == 0) {
        return;
    }

    auto choose_random_hidden = [&](const std::vector<NeuronGene>& neurons) {
        return neurons.begin() + genome.get_num_inputs() + rng.next_int(num_hidden);
    };

    auto neuron_it = choose_random_hidden(genome.get_neurons());

    auto &links = genome.get_links();
    links.erase(
        std::remove_if(links.begin(), links.end(),
                       [&neuron_it](const LinkGene &link) {
                           return link.link_id.input_id == neuron_it->neuron_id || 
                                  link.link_id.output_id == neuron_it->neuron_id;
                       }),
        links.end());

    genome.get_neurons().erase(neuron_it);
}

struct DoubleConfig {
    double init_mean = 0.0;
    double init_stdev = 1.0; 
    double min = -20.0;
    double max = 20.0;
    double mutation_rate = 0.2;
    double mutate_power = 1.2;
    double replace_rate = 0.05;
};

class DoubleMutator {
public:
    DoubleMutator(const DoubleConfig& config, Rng& rng) : config(config), rng(rng) {}

    double new_value() {
        return clamp(rng.next_gaussian(config.init_mean, config.init_stdev));
    }

    double mutate_delta(double value) {
        double delta = clamp(rng.next_gaussian(0.0, config.mutate_power));
        return clamp(value + delta);
    }

private:
    double clamp(double x) const {
        return std::min(config.max, std::max(config.min, x));
    }

    DoubleConfig config;
    Rng& rng;
};

struct NeatConfig {
    int population_size;
    double survival_threshold;
    int num_inputs;
    int num_outputs;
    double weight_mutation_rate;
    double weight_perturbation_chance;
    double new_node_mutation_rate;
    double new_link_mutation_rate;
};

class Population {
public:
    Population(NeatConfig config, Rng &rng) 
        : config(config), rng(rng), 
          double_mutator(DoubleConfig(), rng),
          neuron_mutator(),
          link_mutator() {
        for (int i = 0; i < config.population_size; i++) {
            individuals.push_back({new_genome(), 0.0});
        }
    }

    template<typename FitnessFn>
    Individual run(FitnessFn compute_fitness, int num_generations) {
        for (int i = 0; i < num_generations; i++) {
            compute_fitness(individuals.begin(), individuals.end());
            update_best();
            individuals = reproduce();
        }
        return best_individual;
    }

    const Individual& get_best_individual() const {
        return best_individual;
    }

private:
    Genome new_genome() {
        Genome genome(genome_indexer.next(), config.num_inputs, config.num_outputs);
        for (int neuron_id = 0; neuron_id < config.num_outputs; neuron_id++) {
            genome.add_neuron(neuron_mutator.new_neuron(neuron_id, rng));
        }

        // Fully connected direct feed-forward
        for (int i = 0; i < config.num_inputs; i++) {
            int input_id = -i - 1;
            for (int output_id = 0; output_id < config.num_outputs; output_id++) {
                genome.add_link(link_mutator.new_link(input_id, output_id, rng));
            }
        }
        return genome;
    }

    std::vector<Individual> reproduce() {
        auto old_members = sort_individuals_by_fitness(individuals);
        int reproduction_cutoff = std::ceil(config.survival_threshold * old_members.size());
    
        std::vector<Individual> new_generation;
        int spawn_size = config.population_size;
        while (spawn_size-- > 0) {
            const auto& p1 = *rng.choose_random(old_members.begin(), old_members.begin() + reproduction_cutoff);
            const auto& p2 = *rng.choose_random(old_members.begin(), old_members.begin() + reproduction_cutoff);
            Genome offspring = crossover(p1, p2, genome_indexer, rng);
            mutate(offspring);
            new_generation.push_back({std::move(offspring), 0.0});
        }
        return new_generation;
    }

    void update_best() {
        auto it = std::max_element(individuals.begin(), individuals.end(),
            [](const Individual& a, const Individual& b) { return a.fitness < b.fitness; });
        if (it != individuals.end() && it->fitness > best_individual.fitness) {
            best_individual = *it;
        }
    }

    void mutate(Genome& genome) {
        if (rng.choose(config.new_link_mutation_rate)) mutate_add_link(genome, rng, link_mutator);
        if (rng.choose(0.05)) mutate_remove_link(genome, rng);
        if (rng.choose(config.new_node_mutation_rate)) mutate_add_neuron(genome, rng, neuron_mutator, link_mutator);
        if (rng.choose(0.01)) mutate_remove_neuron(genome, rng);
        
        // Mutate weights
        for (auto& link : genome.get_links()) {
            if (rng.choose(config.weight_mutation_rate)) {
                if (rng.choose(config.weight_perturbation_chance)) {
                    link.weight = double_mutator.mutate_delta(link.weight);
                } else {
                    link.weight = double_mutator.new_value();
                }
            }
        }
    }

    std::vector<Individual> sort_individuals_by_fitness(const std::vector<Individual>& individuals) {
        std::vector<Individual> sorted = individuals;
        std::sort(sorted.begin(), sorted.end(),
            [](const Individual& a, const Individual& b) { return a.fitness > b.fitness; });
        return sorted;
    }

    NeatConfig config;
    Rng& rng;
    std::vector<Individual> individuals;
    Individual best_individual;
    GenomeIndexer genome_indexer;
    NeuronMutator neuron_mutator;
    LinkMutator link_mutator;
    DoubleMutator double_mutator;
};

struct NeuronInput {
    int input_id;
    double weight;
};

struct Neuron {
    int neuron_id;
    Activation activation;
    double bias;
    std::vector<NeuronInput> inputs;
};

class FeedForwardNeuralNetwork {
public:
    FeedForwardNeuralNetwork(const std::vector<int>& input_ids,
                             const std::vector<int>& output_ids,
                             const std::vector<Neuron>& neurons)
        : input_ids(input_ids), output_ids(output_ids), neurons(neurons) {}

    std::vector<double> activate(const std::vector<double>& inputs) const {
        std::cout << "Expected inputs: " << input_ids.size() << ", Received inputs: " << inputs.size() << std::endl;
        assert(inputs.size() == input_ids.size());

        std::unordered_map<int, double> values;
        for (size_t i = 0; i < input_ids.size(); i++) {
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
            value = ActivationFn()(neuron.activation, value);
            values[neuron.neuron_id] = value;
        }

        std::vector<double> outputs;
        for (int output_id : output_ids) {
            assert(values.find(output_id) != values.end());
            outputs.push_back(values[output_id]);
        }
        return outputs;
    }

    const std::vector<int>& get_input_ids() const { return input_ids; }
    const std::vector<int>& get_output_ids() const { return output_ids; }
    const std::vector<Neuron>& get_neurons() const { return neurons; }

private:
    std::vector<int> input_ids;
    std::vector<int> output_ids;
    std::vector<Neuron> neurons;
};

std::vector<std::vector<int>> feed_forward_layers(const std::vector<int>& inputs,
                                                  const std::vector<int>& outputs,
                                                  const std::vector<LinkGene>& links) {
    std::unordered_map<int, std::vector<int>> dependencies;
    std::unordered_set<int> all_nodes;

    for (const auto& link : links) {
        dependencies[link.link_id.output_id].push_back(link.link_id.input_id);
        all_nodes.insert(link.link_id.input_id);
        all_nodes.insert(link.link_id.output_id);
    }

    std::vector<std::vector<int>> layers;
    std::unordered_set<int> placed_nodes(inputs.begin(), inputs.end());
    layers.push_back(inputs);

    while (placed_nodes.size() < all_nodes.size()) {
        std::vector<int> current_layer;
        for (int node : all_nodes) {
            if (placed_nodes.count(node) == 0) {
                bool all_dependencies_placed = true;
                for (int dep : dependencies[node]) {
                    if (placed_nodes.count(dep) == 0) {
                        all_dependencies_placed = false;
                        break;
                    }
                }
                if (all_dependencies_placed) {
                    current_layer.push_back(node);
                }
            }
        }
        for (int node : current_layer) {
            placed_nodes.insert(node);
        }
        layers.push_back(current_layer);
    }

    return layers;
}

FeedForwardNeuralNetwork create_from_genome(const Genome& genome) {
    std::vector<int> inputs;
    for (int i = 0; i < genome.get_num_inputs(); ++i) {
        inputs.push_back(-i - 1);  // Assuming input IDs are negative
    }
    std::vector<int> outputs;
    for (int i = 0; i < genome.get_num_outputs(); ++i) {
        outputs.push_back(i);  // Assuming output IDs start from 0
    }
    std::vector<std::vector<int>> layers = feed_forward_layers(inputs, outputs, genome.get_links());

    std::vector<Neuron> neurons;
    for (const auto& layer : layers) {
        for (int neuron_id : layer) {
            if (neuron_id >= 0) {  // Skip input neurons
                std::vector<NeuronInput> neuron_inputs;
                for (const auto& link : genome.get_links()) {
                    if (neuron_id == link.link_id.output_id && link.is_enabled) {
                        neuron_inputs.emplace_back(NeuronInput{link.link_id.input_id, link.weight});
                    }
                }
                auto neuron_gene_opt = genome.find_neuron(neuron_id);
                assert(neuron_gene_opt.has_value());
                neurons.emplace_back(Neuron{neuron_id, neuron_gene_opt->activation, neuron_gene_opt->bias, std::move(neuron_inputs)});
            }
        }
    }
    return FeedForwardNeuralNetwork(std::move(inputs), std::move(outputs), std::move(neurons));
}

void save_best_network(const FeedForwardNeuralNetwork& network, const std::string& filename) {
    std::ofstream file(filename);
    
    // Save input and output IDs
    file << network.get_input_ids().size() << " ";
    for (int id : network.get_input_ids()) {
        file << id << " ";
    }
    file << network.get_output_ids().size() << " ";
    for (int id : network.get_output_ids()) {
        file << id << " ";
    }
    
    // Save neurons
    const auto& neurons = network.get_neurons();
    file << neurons.size() << "\n";
    for (const auto& neuron : neurons) {
        file << neuron.neuron_id << " " << static_cast<int>(neuron.activation) << " " << neuron.bias << " " << neuron.inputs.size() << " ";
        for (const auto& input : neuron.inputs) {
            file << input.input_id << " " << input.weight << " ";
        }
        file << "\n";
    }
    
    file.close();
}

FeedForwardNeuralNetwork load_best_network(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    std::vector<int> input_ids, output_ids;
    std::vector<Neuron> neurons;
    
    int size;
    file >> size;
    input_ids.resize(size);
    for (int& id : input_ids) {
        file >> id;
    }
    
    file >> size;
    output_ids.resize(size);
    for (int& id : output_ids) {
        file >> id;
    }
    
    file >> size;
    neurons.resize(size);
    for (auto& neuron : neurons) {
        int activation;
        file >> neuron.neuron_id >> activation >> neuron.bias;
        neuron.activation = static_cast<Activation>(activation);
        
        int input_size;
        file >> input_size;
        neuron.inputs.resize(input_size);
        for (auto& input : neuron.inputs) {
            file >> input.input_id >> input.weight;
        }
    }
    
    file.close();
    return FeedForwardNeuralNetwork(input_ids, output_ids, neurons);
}