#include <iostream>
#include <vector>
#include <SFML/Graphics.hpp>

const int SIZE = 20;
const int hidden_nodes = 16;
const int hidden_layers = 2;
const int fps = 100;

int highscore = 0;

float mutationRate = 0.05f;
float defaultmutation = mutationRate;

bool humanPlaying = false;
bool replayBest = true;
bool seeVision = false;
bool modelLoaded = false;

sf::Font font;

std::vector<int> evolution;

Button graphButton;
Button loadButton;
Button saveButton;
Button increaseMut;
Button decreaseMut;

EvolutionGraph graph;

Snake snake;
Snake model;

Population pop;

void setup() {
    if (!font.loadFromFile("agencyfb-bold.ttf")) {
        std::cerr << "Error loading font" << std::endl;
        return;
    }

    graphButton = Button(349, 15, 100, 30, "Graph");
    loadButton = Button(249, 15, 100, 30, "Load");
    saveButton = Button(149, 15, 100, 30, "Save");
    increaseMut = Button(340, 85, 20, 20, "+");
    decreaseMut = Button(365, 85, 20, 20, "-");

    sf::RenderWindow window(sf::VideoMode(1200, 800), "Snake AI");
    window.setFramerateLimit(fps);

    if (humanPlaying) {
        snake = Snake();
    } else {
        pop = Population(2000);
    }

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();

            if (event.type == sf::Event::MouseButtonPressed) {
                mousePressed(event.mouseButton.x, event.mouseButton.y);
            }

            if (event.type == sf::Event::KeyPressed) {
                keyPressed(event.key.code);
            }
        }

        window.clear();
        draw(window);
        window.display();
    }
}

void draw(sf::RenderWindow &window) {
    sf::RectangleShape rect(sf::Vector2f(1200 - 400 - 40, 800 - 40));
    rect.setPosition(400 + SIZE, SIZE);
    rect.setOutlineThickness(1);
    rect.setOutlineColor(sf::Color::White);
    rect.setFillColor(sf::Color::Transparent);
    window.draw(rect);

    sf::Text text;
    text.setFont(font);
    text.setCharacterSize(20);
    text.setFillColor(sf::Color::White);

    if (humanPlaying) {
        snake.move();
        snake.show(window);

        text.setString("SCORE : " + std::to_string(snake.score));
        text.setPosition(500, 50);
        window.draw(text);

        if (snake.dead) {
            snake = Snake();
        }
    } else {
        if (!modelLoaded) {
            if (pop.done()) {
                highscore = pop.bestSnake.score;
                pop.calculateFitness();
                pop.naturalSelection();
            } else {
                pop.update();
                pop.show(window);
            }

            text.setCharacterSize(25);
            text.setString("GEN : " + std::to_string(pop.gen));
            text.setPosition(120, 60);
            window.draw(text);

            text.setString("MUTATION RATE : " + std::to_string(mutationRate * 100) + "%");
            text.setPosition(120, 90);
            window.draw(text);

            text.setString("SCORE : " + std::to_string(pop.bestSnake.score));
            text.setPosition(120, 800 - 45);
            window.draw(text);

            text.setString("HIGHSCORE : " + std::to_string(highscore));
            text.setPosition(120, 800 - 15);
            window.draw(text);

            increaseMut.show(window);
            decreaseMut.show(window);
        } else {
            model.look();
            model.think();
            model.move();
            model.show(window);
            model.brain.show(window, 0, 0, 360, 790, model.vision, model.decision);

            if (model.dead) {
                Snake newmodel = Snake();
                newmodel.brain = model.brain.clone();
                model = newmodel;
            }

            text.setCharacterSize(25);
            text.setString("SCORE : " + std::to_string(model.score));
            text.setPosition(120, 800 - 45);
            window.draw(text);
        }

        text.setCharacterSize(18);
        text.setFillColor(sf::Color::Red);
        text.setString("RED < 0");
        text.setPosition(120, 800 - 75);
        window.draw(text);

        text.setFillColor(sf::Color::Blue);
        text.setString("BLUE > 0");
        text.setPosition(200, 800 - 75);
        window.draw(text);

        graphButton.show(window);
        loadButton.show(window);
        saveButton.show(window);
    }
}

void mousePressed(int mouseX, int mouseY) {
    if (graphButton.collide(mouseX, mouseY)) {
        graph = EvolutionGraph();
    }
    if (loadButton.collide(mouseX, mouseY)) {
        selectInput("Load Snake Model", fileSelectedIn);
    }
    if (saveButton.collide(mouseX, mouseY)) {
        selectOutput("Save Snake Model", fileSelectedOut);
    }
    if (increaseMut.collide(mouseX, mouseY)) {
        mutationRate *= 2;
        defaultmutation = mutationRate;
    }
    if (decreaseMut.collide(mouseX, mouseY)) {
        mutationRate /= 2;
        defaultmutation = mutationRate;
    }
}

void keyPressed(int key) {
    if (humanPlaying) {
        switch (key) {
            case sf::Keyboard::Up:
                snake.moveUp();
                break;
            case sf::Keyboard::Down:
                snake.moveDown();
                break;
            case sf::Keyboard::Left:
                snake.moveLeft();
                break;
            case sf::Keyboard::Right:
                snake.moveRight();
                break;
        }
    }
}

int main() {
    setup();
    return 0;
}
