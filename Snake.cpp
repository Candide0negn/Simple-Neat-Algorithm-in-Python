struct NeuronGene {
    int neuron_id;
    double bias;
    Activation activation;
}

struct LinkId {
    int input_id;
    int output_id;
}

struct LinkGene {
    LinkId link_id;
    double weight;
    bool is_enabled;
}

struct Genome {
    int genome_id;
    int num_inputs;
    int num_outputs;
    std::vector<NeuronGene> neurons;
    std::vector<LinkGene> links;
}

struct Individual {
    Genome genome;
    double fitness;
}

NeuronGene crossover_neuron(const NeuronGene &a, const NeuronGene &b) {
    assert(a.neuron_id == b.neuron_id);
    int neuron_id = a.neuron_id;
    double bias = rng.choose(0.5, a.bias, b.bias);
    Activation activation = rng.choose(0.5, a.activation, b.activation);
    return {neuron_id, bias, activation};
}

LinkGene crossover_neuron(const LinkGene &a, const LinkGene &b) {
    assert(a.link_id == b.link_id);
    LinkId link_id = a.link_id;
    double weight = rng.choose(0.5, a.weight, b.weight);
    bool is_enabled = rng.choose(0.5, a.is_enabled, b.is_enabled);
    return {link_id, weight, is_enabled};
}

Genome crossover(const Individual &dominant, const Individual &recessive) {
    Genome offspring{m_genome_indexer.next(), dominant.genome.num_inputs(),
                    dominant.genome.num_outputs()};
    
    //Inherit neuron genes.
    for (const auto &dominant_neuron : dominant.genome.neurons()) {
        int neuron_id = dominant_neuron.neuron_id;
        std::optional<NeuronGene> recessive_neuron = 
            recessive.genome.find_neuron(neuron_id);
        if (!recessive_neuron) {
            offspring.add_neuron(dominant_neuron);
        } else {
            offspring.add_neuron(
                crossover_neuron(dominant_neuron, *recessive_neuron));
        }
    }
}
