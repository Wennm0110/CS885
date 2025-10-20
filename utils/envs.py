# 檔案名稱: utils/envs.py
import gymnasium as gym
import numpy as np
import random
from copy import deepcopy

# Play an episode according to a given policy
# env: environment
# policy: function(env, state)
# render: whether to render the episode or not (default - False)
def play_episode(env, policy, render=False):
    states, actions, rewards = [], [], []
    
    # 【修正 1】正確處理 reset() 的回傳值
    # env.reset() 現在回傳 (obs, info)，我們只需要 obs
    obs, info = env.reset()
    states.append(obs)
    
    done = False
    if render: env.render()
    
    while not done:
        # states[-1] 現在是正確的 obs (NumPy array)
        action = policy(env, states[-1])
        actions.append(action)
        
        # 【修正 2】正確處理 step() 的回傳值
        # env.step() 現在回傳 5 個值
        # done 的條件是 terminated 或 truncated
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if render: env.render()
        states.append(next_obs)
        rewards.append(reward)
        
    return states, actions, rewards

# Play an episode according to a given policy and add 
# to a replay buffer
# env: environment
# policy: function(env, state)
def play_episode_rb(env, policy, buf):
    states, actions, rewards = [], [], []
    
    # 【修正 1】正確處理 reset() 的回傳值
    obs, info = env.reset()
    
    done = False
    while not done:
        action = policy(env, obs)
        
        # 【修正 2】正確處理 step() 的回傳值
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 將 (s, a, r, s', done) 加入 buffer
        buf.add(obs, action, reward, next_obs, done)
        
        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        
        # 更新觀測值以進行下一步
        obs = next_obs
        
    return states, actions, rewards