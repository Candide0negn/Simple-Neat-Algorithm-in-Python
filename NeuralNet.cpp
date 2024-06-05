#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include "Matrix.cpp" // Assuming the Matrix class is in this file

class NeuralNet {
public:
    int iNodes, hNodes, oNodes, hLayers;
    std::vector<Matrix> weights;

    NeuralNet(int input, int hidden, int output, int hiddenLayers) 
        : iNodes(input), hNodes(hidden), oNodes(output), hLayers(hiddenLayers), weights(hiddenLayers + 1) {
        
        weights[0] = Matrix(hNodes, iNodes + 1);
        for (int i = 1; i < hLayers; ++i) {
            weights[i] = Matrix(hNodes, hNodes + 1);
        }
        weights[hLayers] = Matrix(oNodes, hNodes + 1);

        for (auto& w : weights) {
            w.randomize();
        }
    }

    void mutate(float mr) {
        for (auto& w : weights) {
            w.mutate(mr);
        }
    }

    std::vector<float> output(const std::vector<float>& inputsArr) {
        Matrix inputs = weights[0].singleColumnMatrixFromArray(inputsArr);
        Matrix curr_bias = inputs.addBias();

        for (int i = 0; i < hLayers; ++i) {
            Matrix hidden_ip = weights[i].dot(curr_bias);
            Matrix hidden_op = hidden_ip.activate();
            curr_bias = hidden_op.addBias();
        }

        Matrix output_ip = weights[hLayers].dot(curr_bias);
        Matrix output = output_ip.activate();

        return output.toArray();
    }

    NeuralNet crossover(const NeuralNet& partner) {
        NeuralNet child(iNodes, hNodes, oNodes, hLayers);
        for (size_t i = 0; i < weights.size(); ++i) {
            child.weights[i] = weights[i].crossover(partner.weights[i]);
        }
        return child;
    }

    NeuralNet clone() const {
        NeuralNet clone(iNodes, hNodes, oNodes, hLayers);
        for (size_t i = 0; i < weights.size(); ++i) {
            clone.weights[i] = weights[i].clone();
        }
        return clone;
    }

    void load(const std::vector<Matrix>& weight) {
        weights = weight;
    }

    std::vector<Matrix> pull() const {
        return weights;
    }

    void show(sf::RenderWindow& window, float x, float y, float w, float h, const std::vector<float>& vision, const std::vector<float>& decision) {
        float space = 5;
        float nSize = (h - (space * (iNodes - 2))) / iNodes;
        float nSpace = (w - (weights.size() * nSize)) / weights.size();
        float hBuff = (h - (space * (hNodes - 1)) - (nSize * hNodes)) / 2;
        float oBuff = (h - (space * (oNodes - 1)) - (nSize * oNodes)) / 2;

        int maxIndex = std::max_element(decision.begin(), decision.end()) - decision.begin();

        int lc = 0;  // Layer Count

        // DRAW NODES
        for (int i = 0; i < iNodes; ++i) {  // DRAW INPUTS
            sf::CircleShape node(nSize / 2);
            node.setPosition(x, y + (i * (nSize + space)));
            node.setFillColor(vision[i] != 0 ? sf::Color::Green : sf::Color::White);
            node.setOutlineThickness(1);
            node.setOutlineColor(sf::Color::Black);
            window.draw(node);

            sf::Text text;
            text.setCharacterSize(nSize / 2);
            text.setString(std::to_string(i));
            text.setPosition(x + nSize / 4, y + (i * (nSize + space)) + nSize / 4);
            text.setFillColor(sf::Color::Black);
            text.setStyle(sf::Text::Bold);
            window.draw(text);
        }

        lc++;

        for (int a = 0; a < hLayers; ++a) {
            for (int i = 0; i < hNodes; ++i) {  // DRAW HIDDEN
                sf::CircleShape node(nSize / 2);
                node.setPosition(x + (lc * nSize) + (lc * nSpace), y + hBuff + (i * (nSize + space)));
                node.setFillColor(sf::Color::White);
                node.setOutlineThickness(1);
                node.setOutlineColor(sf::Color::Black);
                window.draw(node);
            }
            lc++;
        }

        for (int i = 0; i < oNodes; ++i) {  // DRAW OUTPUTS
            sf::CircleShape node(nSize / 2);
            node.setPosition(x + (lc * nSpace) + (lc * nSize), y + oBuff + (i * (nSize + space)));
            node.setFillColor(i == maxIndex ? sf::Color::Green : sf::Color::White);
            node.setOutlineThickness(1);
            node.setOutlineColor(sf::Color::Black);
            window.draw(node);
        }

        lc = 1;

        // DRAW WEIGHTS
        for (int i = 0; i < weights[0].rows; ++i) {  // INPUT TO HIDDEN
            for (int j = 0; j < weights[0].cols - 1; ++j) {
                sf::Vertex line[] =
                {
                    sf::Vertex(sf::Vector2f(x + nSize, y + (nSize / 2) + (j * (space + nSize))), weights[0].matrix[i][j] < 0 ? sf::Color::Red : sf::Color::Blue),
                    sf::Vertex(sf::Vector2f(x + nSize + nSpace, y + hBuff + (nSize / 2) + (i * (space + nSize))), weights[0].matrix[i][j] < 0 ? sf::Color::Red : sf::Color::Blue)
                };
                window.draw(line, 2, sf::Lines);
            }
        }

        lc++;

        for (int a = 1; a < hLayers; ++a) {
            for (int i = 0; i < weights[a].rows; ++i) {  // HIDDEN TO HIDDEN
                for (int j = 0; j < weights[a].cols - 1; ++j) {
                    sf::Vertex line[] =
                    {
                        sf::Vertex(sf::Vector2f(x + (lc * nSize) + ((lc - 1) * nSpace), y + hBuff + (nSize / 2) + (j * (space + nSize))), weights[a].matrix[i][j] < 0 ? sf::Color::Red : sf::Color::Blue),
                        sf::Vertex(sf::Vector2f(x + (lc * nSize) + (lc * nSpace), y + hBuff + (nSize / 2) + (i * (space + nSize))), weights[a].matrix[i][j] < 0 ? sf::Color::Red : sf::Color::Blue)
                    };
                    window.draw(line, 2, sf::Lines);
                }
            }
            lc++;
        }

        for (int i = 0; i < weights[hLayers].rows; ++i) {  // HIDDEN TO OUTPUT
            for (int j = 0; j < weights[hLayers].cols - 1; ++j) {
                sf::Vertex line[] =
                {
                    sf::Vertex(sf::Vector2f(x + (lc * nSize) + ((lc - 1) * nSpace), y + hBuff + (nSize / 2) + (j * (space + nSize))), weights[hLayers].matrix[i][j] < 0 ? sf::Color::Red : sf::Color::Blue),
                    sf::Vertex(sf::Vector2f(x + (lc * nSize) + (lc * nSpace), y + oBuff + (nSize / 2) + (i * (space + nSize))), weights[hLayers].matrix[i][j] < 0 ? sf::Color::Red : sf::Color::Blue)
                };
                window.draw(line, 2, sf::Lines);
            }
        }

        // Draw output labels
        sf::Text text;
        text.setCharacterSize(15);
        text.setStyle(sf::Text::Bold);
        text.setFillColor(sf::Color::Black);
        text.setString("U");
        text.setPosition(x + (lc * nSize) + (lc * nSpace) + nSize / 2, y + oBuff + (nSize / 2));
        window.draw(text);

        text.setString("D");
        text.setPosition(x + (lc * nSize) + (lc * nSpace) + nSize / 2, y + oBuff + space + nSize + (nSize / 2));
        window.draw(text);

        text.setString("L");
        text.setPosition(x + (lc * nSize) + (lc * nSpace) + nSize / 2, y + oBuff + (2 * space) + (2 * nSize) + (nSize / 2));
        window.draw(text);

        text.setString("R");
        text.setPosition(x + (lc * nSize) + (lc * nSpace) + nSize / 2, y + oBuff + (3 * space) + (3 * nSize) + (nSize / 2));
        window.draw(text);
    }
};

int main() {
    sf::RenderWindow window(sf::VideoMode(800, 800), "Neural Network Visualization");

    NeuralNet neuralNet(5, 4, 3, 2); // Example initialization
    std::vector<float> vision = {0.1f, 0.5f, 0.0f, 0.2f, 0.7f}; // Example vision
    std::vector<float> decision = {0.3f, 0.4f, 0.8f}; // Example decision

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear(sf::Color::White);
        neuralNet.show(window, 50, 50, 700, 700, vision, decision);
        window.display();
    }

    return 0;
}
