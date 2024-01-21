import os
import gym
import json
import pygame
import time
import gym_tetris

from gym_tetris.ai.QNetwork import QNetwork

neval = 100
discount  = 0.999
eps_decay = 0.9995
weight_path = f"./model_d{discount}_e{eps_decay}.h5"

discount  = 0.998
eps_decay = 0.9995
nepisodes = 4500
weight_path = f"./model_dis{discount}_eps{eps_decay}_nep{nepisodes}.h5"

if not os.path.exists(weight_path):
    print(f"no {weight_path} extis! quit")
    exit(-1)

def main():
    l = []
    st = time.time()
    env = gym.make("tetris-v1", action_mode=1)
    network = QNetwork(discount=1, epsilon=0, epsilon_min=0, epsilon_decay=0, weight_path=weight_path)
    network.load()

    obs = env.reset()
    running = True
    display = True

    ngame = 0
    total_score = 0.0
    total_step  = 0.0
    while running:
        action, state = network.act(obs)
        obs, reward, done, info = env.step(action)

        total_score += reward
        total_step  += 1
        
        if display:
            env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    display = not display
        if done:
            obs = env.reset()
            ngame += 1

        if ngame >= neval:
            break
            
    env.close()
    print(f"{weight_path}: score {total_score/nepisodes}  step {total_step/nepisodes}")

    log = {}
    log["total_episodes"] = ngame
    log["total_steps"]    = total_step
    log["total_score"]    = total_score
    log["elapsed_time"]   = time.time() - st
    log["score_ave"]      = total_score/nepisodes
    log["steps_ave"]      = total_step/nepisodes
    log["discount"]       = discount
    log["eps_decay"]      = eps_decay

    l.append(log)
    with open(f"play_log_dis{discount}_eps{eps_decay}_nep{nepisodes}.json",'w', encoding="utf-8") as f:
        json.dump(l,f,indent=2, ensure_ascii=False)


    
if __name__ == '__main__':
    main()
