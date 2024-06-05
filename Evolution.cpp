#include <SFML/Graphics.hpp>
#include <vector>

class EvolutionGraph {
public:
    EvolutionGraph() : window(sf::VideoMode(900, 600), "Evolution Graph") {
        window.setFramerateLimit(30);
        setup();
    }

    void run() {
        while (window.isOpen()) {
            sf::Event event;
            while (window.pollEvent(event)) {
                if (event.type == sf::Event::Closed)
                    window.close();
            }
            draw();
        }
    }

private:
    sf::RenderWindow window;
    std::vector<int> evolution = {0, 20, 50, 90, 130, 160, 200}; // Sample data

    void setup() {
        // Initial setup if needed
    }

    void draw() {
        window.clear(sf::Color(150, 150, 150));
        drawAxes();
        drawLabels();
        drawGraph();
        window.display();
    }

    void drawAxes() {
        sf::Vertex xAxis[] =
        {
            sf::Vertex(sf::Vector2f(50, 0)),
            sf::Vertex(sf::Vector2f(50, window.getSize().y - 50))
        };

        sf::Vertex yAxis[] =
        {
            sf::Vertex(sf::Vector2f(50, window.getSize().y - 50)),
            sf::Vertex(sf::Vector2f(window.getSize().x, window.getSize().y - 50))
        };

        window.draw(xAxis, 2, sf::Lines);
        window.draw(yAxis, 2, sf::Lines);
    }

    void drawLabels() {
        sf::Font font;
        if (!font.loadFromFile("arial.ttf")) {
            // Handle error
        }

        sf::Text genLabel("Generation", font, 15);
        genLabel.setFillColor(sf::Color::Black);
        genLabel.setPosition(window.getSize().x / 2, window.getSize().y - 10);
        window.draw(genLabel);

        sf::Text scoreLabel("Score", font, 15);
        scoreLabel.setFillColor(sf::Color::Black);
        scoreLabel.setPosition(10, window.getSize().y / 2);
        scoreLabel.setRotation(90);
        window.draw(scoreLabel);

        float x = 50;
        float y = window.getSize().y - 35;
        float xbuff = (window.getSize().x - 50) / 51.0f;
        float ybuff = (window.getSize().y - 50) / 200.0f;

        for (int i = 0; i <= 50; i++) {
            sf::Text label(std::to_string(i), font, 10);
            label.setFillColor(sf::Color::Black);
            label.setPosition(x, y);
            window.draw(label);
            x += xbuff;
        }

        x = 35;
        y = window.getSize().y - 50;
        float ydif = ybuff * 10.0f;
        for (int i = 0; i < 200; i += 10) {
            sf::Text label(std::to_string(i), font, 10);
            label.setFillColor(sf::Color::Black);
            label.setPosition(x, y);
            window.draw(label);

            sf::Vertex line[] =
            {
                sf::Vertex(sf::Vector2f(50, y)),
                sf::Vertex(sf::Vector2f(window.getSize().x, y))
            };
            window.draw(line, 2, sf::Lines);

            y -= ydif;
        }
    }

    void drawGraph() {
        float xbuff = (window.getSize().x - 50) / 51.0f;
        float ybuff = (window.getSize().y - 50) / 200.0f;
        int score = 0;

        sf::VertexArray lines(sf::LinesStrip, evolution.size());

        for (size_t i = 0; i < evolution.size(); ++i) {
            int newscore = evolution[i];
            lines[i] = sf::Vertex(sf::Vector2f(50 + (i * xbuff), window.getSize().y - 50 - (newscore * ybuff)), sf::Color::Red);
            score = newscore;
        }

        window.draw(lines);
    }
};

int main() {
    EvolutionGraph graph;
    graph.run();
    return 0;
}
