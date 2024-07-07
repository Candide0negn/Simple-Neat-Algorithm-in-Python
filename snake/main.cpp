#include "neat.hpp"
#include "snake_game.hpp"
#include <SFML/Graphics.hpp>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <chrono>

const std::string BEST_NETWORK_FILENAME = "best_network.txt";

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./snake_game <player | ai | train>" << std::endl;
        return -1;
    }

    std::string mode = argv[1];

    if (mode == "train") {
        NeatConfig neat_config{
            .population_size = 100,
            .survival_threshold = 0.2,
            .num_inputs = 24,
            .num_outputs = 3,
            .weight_mutation_rate = 0.8,
            .weight_perturbation_chance = 0.9,
            .new_node_mutation_rate = 0.03,
            .new_link_mutation_rate = 0.05
        };

        Rng rng(std::random_device{}());
        Population population(neat_config, rng);

        sf::RenderWindow window(sf::VideoMode(800, 800), "Snake Game - Training");
        window.setFramerateLimit(60);

        GameRendererConfig renderer_config{
            .grid_size = sf::Vector2f(760, 760),
            .grid_offset = sf::Vector2f(20, 20),
            .stroke_color = sf::Color::Black,
            .background_color = sf::Color::White,
            .snake_color = sf::Color::Green,
            .food_color = sf::Color::Red
        };

        auto compute_fitness = [&window, &renderer_config](auto begin, auto end) {
            for (auto it = begin; it != end; ++it) {
                SnakeEngine engine(20, 20, false);
                GameRenderer renderer(window, engine, renderer_config);
                FeedForwardNeuralNetwork network = create_from_genome(it->genome);
                AIController controller(network, engine);

                int max_steps = 1000;
                double fitness = 0;

                for (int step = 0; step < max_steps; ++step) {
                    Action action = controller.get_action();
                    GameResult result = engine.process(action);
                    
                    window.clear(sf::Color::White);
                    renderer.draw();
                    window.display();

                    if (result != GameResult::Running) {
                        break;
                    }
                    fitness += engine.get_score() + step * 0.01; // Reward survival time slightly

                    std::this_thread::sleep_for(std::chrono::milliseconds(10));

                    sf::Event event;
                    while (window.pollEvent(event)) {
                        if (event.type == sf::Event::Closed) {
                            window.close();
                            return;
                        }
                    }
                }

                it->fitness = fitness;
                std::cout << "Individual fitness: " << fitness << std::endl;
            }
        };

        int num_generations = 100;
        for (int gen = 0; gen < num_generations; ++gen) {
            std::cout << "Generation " << gen + 1 << "/" << num_generations << std::endl;
            auto winner = population.run(compute_fitness, 1); // Run for 1 generation at a time
            std::cout << "Generation " << gen + 1 << "/" << num_generations << " complete. Best fitness: " << winner.fitness << std::endl;
            
            if (!window.isOpen()) {
                break; // Stop training if the window is closed
            }
        }

        save_best_network(create_from_genome(population.get_best_individual().genome), BEST_NETWORK_FILENAME);
        std::cout << "Training complete. Best fitness: " << population.get_best_individual().fitness << std::endl;
        return 0;
    }

    sf::RenderWindow window(sf::VideoMode(800, 800), "Snake Game");
    window.setFramerateLimit(60);

    SnakeEngine engine(20, 20, false);
    
    GameRendererConfig renderer_config{
        .grid_size = sf::Vector2f(760, 760),
        .grid_offset = sf::Vector2f(20, 20),
        .stroke_color = sf::Color::Black,
        .background_color = sf::Color::White,
        .snake_color = sf::Color::Green,
        .food_color = sf::Color::Red
    };
    
    GameRenderer renderer(window, engine, renderer_config);

    std::unique_ptr<Controller> controller;
    if (mode == "ai") {
        try {
            FeedForwardNeuralNetwork best_network = load_best_network(BEST_NETWORK_FILENAME);
            controller = make_controller("ai", &best_network, &engine);
        } catch (const std::runtime_error& e) {
            std::cerr << "Error loading best network: " << e.what() << std::endl;
            std::cerr << "Falling back to player mode." << std::endl;
            controller = make_controller("player");
        }
    } else if (mode == "player") {
        controller = make_controller("player");
    } else {
        std::cout << "Invalid mode. Use 'player', 'ai', or 'train'." << std::endl;
        return -1;
    }

    Ticker ticker(10);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            } else {
                controller->on_event(event);
            }
        }

        if (ticker.tick()) {
            GameResult result = engine.process(controller->get_action());
            if (result != GameResult::Running) {
                if (result == GameResult::GameOver) {
                    std::cout << "Game Over! Score: " << engine.get_score() << std::endl;
                } else {
                    std::cout << "You Win! Score: " << engine.get_score() << std::endl;
                }
                engine.reset();
            }
        }

        window.clear(sf::Color::White);
        renderer.draw();
        window.display();
    }

    return 0;
}
