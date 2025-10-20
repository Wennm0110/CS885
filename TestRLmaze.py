import numpy as np
import MDP
import RL
import matplotlib.pyplot as plt
from tqdm import tqdm

''' Construct a simple maze MDP '''
# Transition function: |A| x |S| x |S'| array
T = np.zeros([4,17,17])
a = 0.8; b = 0.1;
T[0,0,0]=a+b; T[0,0,1]=b; T[0,1,0]=b; T[0,1,1]=a; T[0,1,2]=b; T[0,2,1]=b; T[0,2,2]=a; T[0,2,3]=b; T[0,3,2]=b; T[0,3,3]=a+b;
T[0,4,4]=b; T[0,4,0]=a; T[0,4,5]=b; T[0,5,4]=b; T[0,5,1]=a; T[0,5,6]=b; T[0,6,5]=b; T[0,6,2]=a; T[0,6,7]=b; T[0,7,6]=b; T[0,7,3]=a; T[0,7,7]=b;
T[0,8,8]=b; T[0,8,4]=a; T[0,8,9]=b; T[0,9,8]=b; T[0,9,5]=a; T[0,9,10]=b; T[0,10,9]=b; T[0,10,6]=a; T[0,10,11]=b; T[0,11,10]=b; T[0,11,7]=a; T[0,11,11]=b;
T[0,12,12]=b; T[0,12,8]=a; T[0,12,13]=b; T[0,13,12]=b; T[0,13,9]=a; T[0,13,14]=b; T[0,14,16]=1; T[0,15,11]=a; T[0,15,14]=b; T[0,15,15]=b; T[0,16,16]=1;
T[1,0,0]=b; T[1,0,4]=a; T[1,0,1]=b; T[1,1,0]=b; T[1,1,5]=a; T[1,1,2]=b; T[1,2,1]=b; T[1,2,6]=a; T[1,2,3]=b; T[1,3,2]=b; T[1,3,7]=a; T[1,3,3]=b;
T[1,4,4]=b; T[1,4,8]=a; T[1,4,5]=b; T[1,5,4]=b; T[1,5,9]=a; T[1,5,6]=b; T[1,6,5]=b; T[1,6,10]=a; T[1,6,7]=b; T[1,7,6]=b; T[1,7,11]=a; T[1,7,7]=b;
T[1,8,8]=b; T[1,8,12]=a; T[1,8,9]=b; T[1,9,8]=b; T[1,9,13]=a; T[1,9,10]=b; T[1,10,9]=b; T[1,10,14]=a; T[1,10,11]=b; T[1,11,10]=b; T[1,11,15]=a; T[1,11,11]=b;
T[1,12,12]=a+b; T[1,12,13]=b; T[1,13,12]=b; T[1,13,13]=a; T[1,13,14]=b; T[1,14,16]=1; T[1,15,14]=b; T[1,15,15]=a+b; T[1,16,16]=1;
T[2,0,0]=a+b; T[2,0,4]=b; T[2,1,1]=b; T[2,1,0]=a; T[2,1,5]=b; T[2,2,2]=b; T[2,2,1]=a; T[2,2,6]=b; T[2,3,3]=b; T[2,3,2]=a; T[2,3,7]=b;
T[2,4,0]=b; T[2,4,4]=a; T[2,4,8]=b; T[2,5,1]=b; T[2,5,4]=a; T[2,5,9]=b; T[2,6,2]=b; T[2,6,5]=a; T[2,6,10]=b; T[2,7,3]=b; T[2,7,6]=a; T[2,7,11]=b;
T[2,8,4]=b; T[2,8,8]=a; T[2,8,12]=b; T[2,9,5]=b; T[2,9,8]=a; T[2,9,13]=b; T[2,10,6]=b; T[2,10,9]=a; T[2,10,14]=b; T[2,11,7]=b; T[2,11,10]=a; T[2,11,15]=b;
T[2,12,8]=b; T[2,12,12]=a+b; T[2,13,9]=b; T[2,13,12]=a; T[2,13,13]=b; T[2,14,16]=1; T[2,15,11]=a; T[2,15,14]=b; T[2,15,15]=b; T[2,16,16]=1;
T[3,0,0]=b; T[3,0,1]=a; T[3,0,4]=b; T[3,1,1]=b; T[3,1,2]=a; T[3,1,5]=b; T[3,2,2]=b; T[3,2,3]=a; T[3,2,6]=b; T[3,3,3]=a+b; T[3,3,7]=b;
T[3,4,0]=b; T[3,4,5]=a; T[3,4,8]=b; T[3,5,1]=b; T[3,5,6]=a; T[3,5,9]=b; T[3,6,2]=b; T[3,6,7]=a; T[3,6,10]=b; T[3,7,3]=b; T[3,7,7]=a; T[3,7,11]=b;
T[3,8,4]=b; T[3,8,9]=a; T[3,8,12]=b; T[3,9,5]=b; T[3,9,10]=a; T[3,9,13]=b; T[3,10,6]=b; T[3,10,11]=a; T[3,10,14]=b; T[3,11,7]=b; T[3,11,11]=a; T[3,11,15]=b;
T[3,12,8]=b; T[3,12,13]=a; T[3,12,12]=b; T[3,13,9]=b; T[3,13,14]=a; T[3,13,13]=b; T[3,14,16]=1; T[3,15,11]=b; T[3,15,15]=a+b; T[3,16,16]=1;

# Reward function
R = -1 * np.ones([4,17]);
R[:,14] = 100; R[:,9] = -70; R[:,16] = 0;

# Discount factor
discount = 0.95
        
# MDP object
mdp = MDP.MDP(T,R,discount)

# RL problem, rewards sampled from a normal distribution with std=1
# (np.random.normal takes mean and std, we provide a lambda function)
rlProblem = RL.RL(mdp, lambda x: np.random.normal(x, 1))


# --- 實驗參數設定 ---
n_episodes = 200
n_steps = 100
n_trials = 100  # 題目要求：基於 100 次試驗的平均
epsilons_to_test = [0.05, 0.1, 0.3, 0.5]

# 用於儲存每個 epsilon 的平均獎勵曲線
average_rewards_all_epsilons = {}

# --- 開始執行 ---
for eps in epsilons_to_test:
    print(f"開始訓練，epsilon = {eps} ")
    
    # 用於儲存當前 epsilon 下，所有試驗的獎勵歷史
    rewards_all_trials = []
    
    # 使用 tqdm 顯示進度條
    for trial in tqdm(range(n_trials), desc=f"Epsilon={eps}"):
        # 執行 Q-learning
        [Q, policy, rewards_per_episode] = rlProblem.qLearning(
            s0=0,
            initialQ=np.zeros([mdp.nActions, mdp.nStates]),
            nEpisodes=n_episodes,
            nSteps=n_steps,
            epsilon=eps
        )
        rewards_all_trials.append(rewards_per_episode)
    
    # 計算 100 次試驗的平均獎勵
    # np.mean(rewards_all_trials, axis=0) 會計算每個 episode 的平均獎勵
    average_rewards = np.mean(rewards_all_trials, axis=0)
    average_rewards_all_epsilons[eps] = average_rewards
    
    print(f"epsilon = {eps} 訓練完成\n")





plt.figure(figsize=(12, 8))

for eps, avg_rewards in average_rewards_all_epsilons.items():
    plt.plot(avg_rewards, label=f'epsilon = {eps}')

plt.title('Average Cumulative Discounted Reward per Episode (100 Trials)')
plt.xlabel('Episode #')
plt.ylabel('Average Cumulative Discounted Reward')
plt.legend()
plt.grid(True)
plt.show()

