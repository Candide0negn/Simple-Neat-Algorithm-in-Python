enum class Action {
    DoNothing,
    RotateLeft,
    RotateRight,
};

enum class Result {
    Running,
    GameOver,
    Winner,
};

struct Coordinates {
    int row;
    int column;
};

struct Snake{
    std::deque<Coordinates> body;

    Coordinates head() const {return body.front();}
};

class SnakeEngine {
public:
    GameResult process(Action action) {
        current_direction = update_direction(current_direction, action);
        Coordinates new_head = get_next_head(snake, current_direction); 

        //Remove tail before checking for collision
        auto tail = snake.body.back();
        snake.body.pop_back();

        if ( hits_wall(new_head) || hits_snake_body(new_head)) {
            return GameResult::GameOver;
        }

        if (new_head == food){
            score++;

            if (snake.body.size() == height * width){
                return GameResult::Winner;
            }

            generate_food();
            //Keep tail as snake grows longer
            snake.body.push_back(tail);
        }

        snake.body.push_front(new_head);
        return GameResult::Running;  
    }

private:
    Snake snake;
    Coordinates food;
    bool allow_through_walls;
    int width;
    int height;
    int score;
    Direction current_direction;

    void generate_food(){
        do {
            food.row = rng.next_int(height - 1);
            food.column = rng.next_int(width - 1);
        } while (hits_snake_body(food));
    }

    bool hits_wall(const Coordinates &c) const { ... }
    bool hits_snake_body(const Coordinates &c) { ... }

};