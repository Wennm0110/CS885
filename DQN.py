import torch
import gymnasium as gym
import numpy as np
from copy import deepcopy
# 確保你的 utils 檔案與此檔案在同一個資料夾或 Python 路徑中
import utils.envs, utils.seed, utils.buffers, utils.torch, utils.common

import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# --- 常數設定 (Constants) ---
SEEDS = [1, 2, 3, 4, 5]
t = utils.torch.TorchHelper()
DEVICE = t.device
OBS_N = 4
ACT_N = 2
GAMMA = 0.99
LEARNING_RATE = 5e-4
TRAIN_AFTER_EPISODES = 10
TRAIN_EPOCHS = 5
BUFSIZE = 10000
EPISODES = 300
TEST_EPISODES = 1
HIDDEN = 512
STARTING_EPSILON = 1.0
STEPS_MAX = 10000
EPSILON_END = 0.01

# --- 輔助函式 (Helper Functions) ---

def create_models():
    """建立 Q 網路和 Target Q 網路"""
    Q = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N)
    ).to(DEVICE)
    Qt = deepcopy(Q).to(DEVICE)
    return Q, Qt

def update_target_network(target, source):
    """從來源網路複製權重到目標網路"""
    target.load_state_dict(source.state_dict())

def get_action(Q_net, obs, epsilon):
    """使用 Epsilon-Greedy 策略選擇一個動作"""
    if np.random.rand() < epsilon:
        return np.random.randint(ACT_N)
    else:
        obs_tensor = t.f(obs).view(-1, OBS_N)
        with torch.no_grad():
            q_values = Q_net(obs_tensor)
        return torch.argmax(q_values).item()

def update_networks(buf, Q, Qt, OPT, minibatch_size):
    """執行一步 Q 網路的訓練"""
    S, A, R, S2, D = buf.sample(minibatch_size, t)
    
    q_values = Q(S).gather(1, A.view(-1, 1)).squeeze()

    with torch.no_grad():
        q2_values = torch.max(Qt(S2), dim=1).values
    
    targets = R + GAMMA * q2_values * (1 - D)
    
    loss = torch.nn.MSELoss()(targets, q_values)

    OPT.zero_grad()
    loss.backward()
    OPT.step()
    
    return loss.item()

def plot_results(results, title, xlabel, ylabel, legends):
    """繪製結果的平均值與標準差曲線"""
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'red', 'purple']
    for i, (param_val, curves) in enumerate(results.items()):
        mean = np.mean(curves, axis=0)
        std = np.std(curves, axis=0)
        # CartPole-v1 的滿分是 500
        plt.plot(range(len(mean)), mean, color=colors[i], label=f"{legends}={param_val}")
        plt.fill_between(range(len(mean)), np.maximum(mean - std, 0), np.minimum(mean + std, 500), color=colors[i], alpha=0.2)
    
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc='best')
    plt.grid(True)

# --- 主要訓練函式 (Main Training Function) ---

def train(seed, target_update_freq, minibatch_size):
    """單次執行的主要訓練迴圈"""
    print(f"Seed={seed}, Target Freq={target_update_freq}, Minibatch Size={minibatch_size}")
    
    utils.seed.seed(seed)
    # 使用 CartPole-v1，它的最大步數是 500，獎勵更高，更容易看出學習效果
    env = gym.make("CartPole-v1")
    test_env = gym.make("CartPole-v1")
    
    buf = utils.buffers.ReplayBuffer(BUFSIZE)
    Q, Qt = create_models()
    update_target_network(Qt, Q)
    OPT = torch.optim.Adam(Q.parameters(), lr=LEARNING_RATE)
    
    epsilon = STARTING_EPSILON
    test_rewards_log = []
    avg_rewards_log = []
    
    pbar = tqdm.trange(EPISODES, leave=False)
    for epi in pbar:
        # 1. 遊戲與經驗收集
        obs, _ = env.reset()
        done = False
        while not done:
            action = get_action(Q, obs, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buf.add(obs, action, reward, next_obs, done)
            obs = next_obs
            epsilon = max(EPSILON_END, epsilon - (1.0 / STEPS_MAX))
        
        # 2. 網路訓練
        if epi >= TRAIN_AFTER_EPISODES:
            for _ in range(TRAIN_EPOCHS):
                update_networks(buf, Q, Qt, OPT, minibatch_size)
        
        if (epi + 1) % target_update_freq == 0:
            update_target_network(Qt, Q)
            
        # 3. 性能評估 (Evaluation)
        total_reward = 0
        for _ in range(TEST_EPISODES):
            obs, _ = test_env.reset()
            done = False
            while not done:
                action = get_action(Q, obs, 0.0) # 測試時使用貪婪策略 (epsilon=0)
                
                # 【關鍵修正】評估時，必須使用 test_env 來執行動作
                obs, reward, terminated, truncated, _ = test_env.step(action)
                
                done = terminated or truncated
                total_reward += reward
        test_rewards_log.append(total_reward / TEST_EPISODES)
        
        # 計算移動平均
        avg_reward = np.mean(test_rewards_log[-25:])
        avg_rewards_log.append(avg_reward)
        pbar.set_description(f"R25({avg_reward:.2f})")
        
    env.close()
    test_env.close()
    return avg_rewards_log

# --- 實驗執行區 (Experiment Execution) ---

if __name__ == "__main__":
    
    # --- 實驗 1: Target Network 更新頻率 ---
    target_update_freqs = [1, 10, 50, 100]
    results_freq = {freq: [] for freq in target_update_freqs}
    
    print("--- 執行實驗 1: Target Network 更新頻率 ---")
    for freq in target_update_freqs:
        for seed in SEEDS:
            curve = train(seed=seed, target_update_freq=freq, minibatch_size=10) # 固定 minibatch_size
            results_freq[freq].append(curve)

    # --- 實驗 2: Minibatch 大小 ---
    minibatch_sizes = [1, 10, 50, 100]
    results_minibatch = {size: [] for size in minibatch_sizes}
    
    print("\n--- 執行實驗 2: Minibatch 大小 ---")
    for size in minibatch_sizes:
        for seed in SEEDS:
            curve = train(seed=seed, target_update_freq=10, minibatch_size=size) # 固定 target_update_freq
            results_minibatch[size].append(curve)
            
    # --- 繪圖 ---
    plot_results(results_freq, 
                 'Impact of Target Network Update Frequency',
                 '# of Episodes', 
                 'Average Cumulative Reward (last 25 episodes)',
                 'Update Freq')

    plot_results(results_minibatch, 
                 'Impact of Minibatch Size',
                 '# of Episodes', 
                 'Average Cumulative Reward (last 25 episodes)',
                 'Minibatch Size')

    plt.show()