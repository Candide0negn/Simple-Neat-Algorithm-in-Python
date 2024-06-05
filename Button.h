#ifndef BUTTON_H
#define BUTTON_H

#include <SFML/Graphics.hpp>
#include <string>

class Button {
public:
    float X, Y, W, H;
    std::string text;

    Button(float x, float y, float w, float h, const std::string& t) 
        : X(x), Y(y), W(w), H(h), text(t) {}

    bool collide(float x, float y) const {
        return (x >= X - W / 2 && x <= X + W / 2 && y >= Y - H / 2 && y <= Y + H / 2);
    }

    void show(sf::RenderWindow& window) const {
        // Create the button rectangle
        sf::RectangleShape rectangle(sf::Vector2f(W, H));
        rectangle.setPosition(X, Y);
        rectangle.setFillColor(sf::Color::White);
        rectangle.setOutlineThickness(1);
        rectangle.setOutlineColor(sf::Color::Black);
        rectangle.setOrigin(W / 2, H / 2);
        window.draw(rectangle);

        // Load a font
        sf::Font font;
        if (!font.loadFromFile("arial.ttf")) { // Make sure you have an "arial.ttf" file in the same directory or provide a correct path
            // Handle error
        }

        // Create the text
        sf::Text buttonText(text, font, 22);
        buttonText.setFillColor(sf::Color::Black);
        buttonText.setPosition(X, Y - 3);
        buttonText.setOrigin(buttonText.getLocalBounds().width / 2, buttonText.getLocalBounds().height / 2);
        window.draw(buttonText);
    }
};

#endif // BUTTON_H
