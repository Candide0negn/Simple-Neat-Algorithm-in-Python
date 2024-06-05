#ifndef EVOLUTIONGRAPH_H
#define EVOLUTIONGRAPH_H

#include <SFML/Graphics.hpp>
#include <vector>
#include <string>

class EvolutionGraph {
public:
    EvolutionGraph();
    void run();

private:
    sf::RenderWindow window;
    std::vector<int> evolution;

    void setup();
    void draw();
    void drawAxes();
    void drawLabels();
    void drawGraph();
};

#endif // EVOLUTIONGRAPH_H
