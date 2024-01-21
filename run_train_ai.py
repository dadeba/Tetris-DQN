import os
import gym
import gym_tetris
import json
import time

from statistics import mean, median
from gym_tetris.ai.QNetwork import QNetwork

def main():
    l = []
    st = time.time()
    discount  = 0.998
    eps_decay = 0.9995
    nepisodes = 4500
    weight_path = f"./model_dis{discount}_eps{eps_decay}_nep{nepisodes}.h5"

    if os.path.exists(weight_path):
        print(f"{weight_path} extis! quit")
        exit(-1)
    
    env = gym.make("tetris-v1", action_mode=1)
    network = QNetwork(discount = discount, epsilon = 1.0, epsilon_min=0.0001, epsilon_decay=eps_decay, weight_path = weight_path)
    network.load()
    
    running = True
    total_games = 0
    total_steps = 0
    total_reward = 0.0
    total_score  = 0.0
    while running:
        steps, rewards, scores = network.train(env, episodes=25)
        total_games += len(scores)
        total_steps += steps
        total_reward += sum(rewards)
        total_score  += sum(scores)
        network.save()
        print("==================")
        print("* Total Games: ", total_games)
        print("* Total Steps: ", total_steps)
        print("* Epsilon: ", network.epsilon)
        print("*")
        print("* Average: ", sum(rewards) / len(rewards), "/", sum(scores) / len(scores))
        print("* Median: ", median(rewards), "/", median(scores))
        print("* Mean: ", mean(rewards), "/", mean(scores))
        print("* Min: ", min(rewards), "/", min(scores))
        print("* Max: ", max(rewards), "/", max(scores))
        print("==================")

        log = {}
        log["total_games"]  = total_games
        log["total_steps"]  = total_steps
        log["total_reward"] = total_reward
        log["total_score"]  = total_score
        log["epsilon"]      = network.epsilon
        log["elapsed_time"] = time.time() - st
        log["reward_ave"]    = sum(rewards)/len(rewards)
        log["score_ave"]     = sum(scores)/len(scores)
        log["reward_median"] = median(rewards)
        log["score_median"]  = median(scores)
        log["reward_min"]    = min(rewards)
        log["score_min"]     = min(scores)
        log["reward_max"]    = max(rewards)
        log["score_max"]     = max(scores)

        l.append(log)
        
        if total_games >= nepisodes:
            break
        
    env.close()

    with open(f"train_log_{discount}_{eps_decay}.json",'w', encoding="utf-8") as f:
        json.dump(l,f,indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
