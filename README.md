The classic game of "Snake" provides a perfect scenario to explore the fundamental concepts of reinforcement learning. In this simple yet addictive game, players control a snake that grows each time it consumes an object, but they lose if the snake collides with itself or the edges of the board. This basic dynamic presents challenges that are easily translatable to the realm of reinforcement learning.

To run the model you need to execute the agent file
```python
if __name__ == "__main__":
    run_agent(mode="train", speed_game=50, starting_len_snake=15, visual_game=True)

```

![alt-text](https://github.com/NEOZRO/snake_game_reinforcement_learning/blob/main/img/snake_gameplay.gif)

Check out my full article in
https://neozro.github.io/project_2_snake_RL.html
