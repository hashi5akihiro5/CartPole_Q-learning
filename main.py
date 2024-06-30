import gym
import numpy as np

# 環境の生成
env = gym.make('CartPole-v1', render_mode="human")


goal_average_steps = 195
max_number_of_steps = 200
num_consecutive_iterations = 100
num_episodes = 100
last_time_steps = np.zeros(num_consecutive_iterations)


for episode in range(num_episodes):
    # 環境の初期化
    observation, info = env.reset()

    episode_reward = 0

    # ゲームのステップを200回プレイ
    for t in range(max_number_of_steps):
    # 現在の状況を表示させる
        env.render()

        # 環境からランダムな行動を取得
        action = env.action_space.sample()
        # サンプルの行動をさせる。返り値は、[台車及び棒の状態、得られる報酬、ゲーム終了フラグ, info:none]
        # observation: ['カートの位置', 'カートの速度', 'ポールの角度', 'ポールの角速度']
        observation, reward, terminated, truncated, info = env.step(action)
        print("=" * 10)
        print("step = ", t)
        print("action = ", action)
        print("observation = ", observation)
        # カートの位置
        print("Cart Position = ", observation[0])
        # カートの速度
        print("Cart Velocity = ", observation[1])
        # ポールの角度[rad]
        print("Pole Angle[rad] = ", observation[2])
        # ポールの角度[°]-24°〜24°
        print("Pole Angle[°] = ", f"{round(observation[2]*57.29577951307855)}°")
        # ポールの角速度
        print("Pole Velocity At Tip = ", observation[3])
        # ポールが倒れない限り1タイムステップ毎に1得られる。
        print("reward = ", reward)
        print("terminated = ", terminated)
        print("truncated = ", truncated)
        print("info = ", info)
        if terminated:
            print("Episode {} finished after {} timesteps".format(episode, t+1))
            break

# 環境を閉じる
env.close()