struct NeuronGene {
    int neuron_id;
    double bias;
    Activation activation;
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
    std::vector<NeuronGene> neurons
    std::vector<LinkGene> links
}

NeuronGene crossover_neuron(const NeuronGene &a, const NeuronGene &b) {
    assert(a.neuron_id == b.neuron_id);
    int neuron_id = a.neuron_id;
    double bias = rng.choose(0.5, a.bias, b.bias);
    Activation activation = rng(0.5, a.activation, b.activation);
    return {neuron_id, bias, activation};
}

LinkGene crossover_link(const LinkGene &a, const LinkGene &b){
    assert(a.link_id == b.link_id);
    LinkId link_id = a.link_id;
    double weight = rng.choose(0.5, a.weight, b.weight);
    bool is_enabled = rng(0.5, a.is_enabled, b.is_enabled);
    return {link_id, weight, is_enabled};
}

struct Individual{
    Genome genome;
    double fitness;
}

Genome crossover(const Individual &dominant, const Individual &recessive){
    Genome offsping{m_genome_indexer.next(), dominant.genome.num_inputs(),
                    dominant.genome.num_outputs()}
    //Inherit neuron genes
    for (const auto &dominant_neuron : dominant.genome.neurons()) {
        int neuron_id = dominant_neuron.neuron_id;
        std::optional<NeuronGene> recessive_neuron =
            recessive.genome.find_neuron(neuron_id);
        if(!recessive_neuron){
            offsping.add_neuron(dominant_neuron);
        }else{
            offsping.add_neuron(
                crossover_neuron(dominant_neuron, *recessive_neuron)
            );
        }
    }

    // Inherit link genes
    for (const auto &dominant_link : dominant.genome.links()) {
        LinkId link_id dominant_link.link_id;
        std::optional<LinkGene> recessive_link =
            recessive.genome.find_link(link_id);
        if(!recessive_link){
            offsping.add_link(dominant_link);
        }else{
            offsping.add_link(
                crossover_link(dominant_link, *recessive_link)
            );
        }
    } 
    return offsping;
}

void mutate_add_link(Genome &genome){
    int input_id = choose_random_input_or_hidden(genome.neurons());
    int output_id = choose_random_output_or_hidden(genome.neurons());
    LinkId link_id{input_id, output_id};

    //don't duplicate links
    auto existing_link = genome.find_link(link_id);
    if(existing_link){
        //At least enable it
        existing_link->is_enabled = true;
        return;
    }
    if (would_create_cycle(genome.links(), input_id, output_id)){
        return;
    }

    LinkGene new_link = link_mutator.new_value(input_id, output_id);
    genome.add_link(new_link);
}

void mutate_remove_link(Genome &genome){
    if(genome.links().empty()){
        return;
    }
    auto to_remove_it = rng.choose_random(genome.links());
    genome.links().erase(to_remove_it);
}

void mutate_add_neuron(Genome &genome){
    if(genome.links().empty()){
        // Neurons are added by splitting Links
        // If there are no links then we can't add neurons
        return;
    }

    LinkGene &link_to_split = rng.choose_random(genome_links());
    link_to_split.is_enabled = false;

    NeuronGene new_neuron = neuron_mutator.new_neuron();
    genome.add_neuron(new_neuron);

    LinkId link_id = link_to_split.link_id;
    double weight = link_to_split.weight;

    genome.add_link(
        LinkGene{{link_id.input_id, new_neuron.neuron_id}, 1.0, true}
    );
    genome.add_link(
        LinkGene{{new_neuron.neuron_id, link_id.output_id}}, weight, true
    ); 
}


