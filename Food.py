#include <SFML/Graphics.hpp>
#include <cmath>
#include <random>

const int SIZE = 20;

class PVector {
public:
    float x, y;

    PVector(float x = 0, float y = 0) : x(x), y(y) {}
};

class Food {
public:
    PVector pos;

    Food() {
        int x = 400 + SIZE + static_cast<int>(random(38)) * SIZE;
        int y = SIZE + static_cast<int>(random(38)) * SIZE;
        pos = PVector(x, y);
    }

    void show(sf::RenderWindow& window) {
        sf::RectangleShape rectangle(sf::Vector2f(SIZE, SIZE));
        rectangle.setPosition(pos.x, pos.y);
        rectangle.setFillColor(sf::Color(255, 0, 0));
        rectangle.setOutlineColor(sf::Color(0, 0, 0));
        rectangle.setOutlineThickness(1);
        window.draw(rectangle);
    }

    Food clone() {
        Food clone;
        clone.pos.x = pos.x;
        clone.pos.y = pos.y;
        return clone;
    }

private:
    float random(int max) {
        static std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, max);
        return dist(rng);
    }
};

int main() {
    sf::RenderWindow window(sf::VideoMode(800, 800), "Food Example");
    Food food;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear(sf::Color::White);
        food.show(window);
        window.display();
    }

    return 0;
}
