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

void mutate_remove_neuron(Genome &genome) {
    if (genome.num_hidden() == 0){
        return;
    }

    const auto &neurons = genome.neurons();
    auto neuron_it = choose_random_hidden(neurons);

    //Delete the associated links
    auto &links = genome.links();
    link.erase(
        std::remove_if(links.begin(), links.end(),
                        [neuron_it](const LinkGene &link) {
                            return link.has_neuron(*neuron_it);
                        }),
        links.end());

    //Delete the neuron
    genome.neurons().erase(neuron_it);
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

double new_value(){
    return clamp(rng.next_gaussian(
        config.init_mean, config.init_stdev));
}

double mutate_delta(double value){
    double delta = clamp(
        rng.next_gaussian(0.0, config.mutate_power));
}

double clamp(double x) const{
    return std::min(config.max, std::max(config.min, x));
}

int main(int argc, char** argv){
    //Define configurations
    Population population{neat_config, rng};
    ComputeFitnessFn compute_fitness{rng};
    auto winner = population.run(compute_fitness, num_generations);
    save(winner.genome, winner_filename);
    return 0;
}

class Population {
public:
    Population(NeatConfig config, rng &rng) : config{config}, rng{rng} {
        for (int i = 0; i < config.population_size; i++) {
            individuals.push_back({new_genome(), kFitnessNotComputed});
        }
    }
private:
    Genome new_genome(){
        Genome genome{next_genome_id(), num_inputs, num_outputs};
        for(int neuron_id = 0, neuron_id < num_outputs; neuron_id++){
            genome.add_neuron(new_neuron(neuron_id));
        }

        //fully connected feed-forward
        for(int i = 0; i < num_inputs; i++){
            int input_id = -i - 1;
            for(int output_id = 0; output_id < num_outputs; output_id++){
                genome.add_link(new_link(input_id, output_id));
            }
        }
        return genome;
    }
};

template<typename FitnessFn>
Individual run(FitnessFn compute_fitness, int num_generations){
    for( int i = 0; i < num_generations; i++){
        compute_fitness(individuals.begin(), individuals.end());
        update_best();
        individuals = reproduce();
    }
}

std::vector<Individual> reproduce(){
    auto old_members = sort_individuals_by_fitness(individuals);
    int reproduction_cutoff = std::ceil(
        config.survival_threshold * old_members.size());

    std::vector<Individual> new_generation;
    int spawn_size = population_size;
    while (spawn_size-- > = 0){
        const auto& p1 = *rng.choose_random(old_members, reproduction_cutoff);
        const auto& p2 = *rng.choose_random(old_members, reproduction_cutoff);
        Genome offsping = crossover(*p1. *p2);
        mutate(offsping);
        new_generation.push_back(std::move(offsping));
    }
    return new_generation;
}