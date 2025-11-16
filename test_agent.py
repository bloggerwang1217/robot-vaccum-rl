"""
測試訓練好的 DQN agent

可視化訓練後的 agent 表現
"""

import argparse
import time
import pygame

try:
    from stable_baselines3 import DQN
    SB3_AVAILABLE = True
except ImportError:
    print("Warning: stable-baselines3 not installed")
    SB3_AVAILABLE = False

from gym_wrapper import make_env
from energy_survival_config import get_config


def test_agent(
    model_path: str,
    obs_type: str = 'vector',
    config_name: str = 'base',
    robot_id: int = 0,
    n_episodes: int = 5,
    render: bool = True,
    delay: float = 0.1
):
    """
    測試訓練好的 agent

    Args:
        model_path: 模型路徑
        obs_type: 觀察類型
        config_name: 環境配置
        robot_id: 機器人 ID
        n_episodes: 測試回合數
        render: 是否渲染
        delay: 每步之間的延遲（秒）
    """
    print("=" * 70)
    print("測試 DQN Agent")
    print("=" * 70)

    # 載入模型
    print(f"\n載入模型: {model_path}")
    model = DQN.load(model_path)

    # 創建環境
    config = get_config(config_name)
    env = make_env(
        config=config,
        robot_id=robot_id,
        obs_type=obs_type,
        reward_shaping='dense'
    )

    print(f"\n環境配置: {config_name}")
    print(f"  - 地圖大小: {config['n']}×{config['n']}")
    print(f"  - 觀察模式: {obs_type}")
    print(f"  - 訓練機器人: {robot_id}")
    print(f"\n開始測試 {n_episodes} 個回合...")
    print("=" * 70)

    # 統計信息
    episode_rewards = []
    episode_lengths = []
    survival_count = 0

    try:
        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            print(f"\n回合 {episode + 1}/{n_episodes}")
            print("-" * 70)

            while not done:
                # 預測動作（deterministic=True 表示不探索）
                action, _states = model.predict(obs, deterministic=True)

                # 執行動作
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

                # 渲染
                if render:
                    env.render()
                    time.sleep(delay)

                    # 處理 Pygame 事件
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("\n用戶中斷測試")
                            env.close()
                            return
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                print("\n用戶中斷測試")
                                env.close()
                                return

                # 每 50 步顯示一次狀態
                if episode_length % 50 == 0:
                    print(f"  步數: {episode_length:3d} | "
                          f"能量: {info['energy']:3.0f}/{config['initial_energy']} | "
                          f"充電: {info['charge_count']:2d}次 | "
                          f"獎勵: {episode_reward:6.2f}")

            # 回合結束統計
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if info['is_active']:
                survival_count += 1

            print(f"\n回合 {episode + 1} 結束:")
            print(f"  - 總獎勵: {episode_reward:.2f}")
            print(f"  - 步數: {episode_length}")
            print(f"  - 最終能量: {info['energy']}/{config['initial_energy']}")
            print(f"  - 充電次數: {info['charge_count']}")
            print(f"  - 狀態: {'✓ 存活' if info['is_active'] else '✗ 停機'}")

    except KeyboardInterrupt:
        print("\n\n測試被中斷")

    finally:
        env.close()

    # 顯示統計
    print("\n" + "=" * 70)
    print("測試統計")
    print("=" * 70)

    if episode_rewards:
        import numpy as np
        print(f"平均獎勵: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"平均步數: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"存活率: {survival_count}/{len(episode_rewards)} ({survival_count/len(episode_rewards)*100:.1f}%)")
        print(f"最佳獎勵: {max(episode_rewards):.2f}")
        print(f"最差獎勵: {min(episode_rewards):.2f}")


def compare_with_baseline(
    model_path: str,
    obs_type: str = 'vector',
    config_name: str = 'base',
    robot_id: int = 0,
    n_episodes: int = 10
):
    """
    將訓練的 agent 與 baseline (epsilon-greedy) 比較

    Args:
        model_path: 模型路徑
        obs_type: 觀察類型
        config_name: 環境配置
        robot_id: 機器人 ID
        n_episodes: 測試回合數
    """
    print("=" * 70)
    print("DQN vs Baseline (Epsilon-Greedy) 比較")
    print("=" * 70)

    # 載入模型
    model = DQN.load(model_path)

    # 創建環境
    config = get_config(config_name)
    env = make_env(
        config=config,
        robot_id=robot_id,
        obs_type=obs_type,
        reward_shaping='dense'
    )

    print(f"\n測試 {n_episodes} 個回合...")

    # 測試 DQN agent
    print("\n1. 測試 DQN Agent")
    print("-" * 70)
    dqn_rewards = []
    dqn_survival = 0

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        dqn_rewards.append(episode_reward)
        if info['is_active']:
            dqn_survival += 1

        if (episode + 1) % 5 == 0:
            print(f"  回合 {episode + 1}/{n_episodes} 完成")

    # 測試 baseline (純隨機)
    print("\n2. 測試 Baseline (純隨機)")
    print("-" * 70)
    baseline_rewards = []
    baseline_survival = 0

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = env.action_space.sample()  # 隨機動作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        baseline_rewards.append(episode_reward)
        if info['is_active']:
            baseline_survival += 1

        if (episode + 1) % 5 == 0:
            print(f"  回合 {episode + 1}/{n_episodes} 完成")

    env.close()

    # 比較結果
    import numpy as np

    print("\n" + "=" * 70)
    print("比較結果")
    print("=" * 70)

    print(f"\nDQN Agent:")
    print(f"  平均獎勵: {np.mean(dqn_rewards):.2f} ± {np.std(dqn_rewards):.2f}")
    print(f"  存活率: {dqn_survival}/{n_episodes} ({dqn_survival/n_episodes*100:.1f}%)")

    print(f"\nBaseline (隨機):")
    print(f"  平均獎勵: {np.mean(baseline_rewards):.2f} ± {np.std(baseline_rewards):.2f}")
    print(f"  存活率: {baseline_survival}/{n_episodes} ({baseline_survival/n_episodes*100:.1f}%)")

    improvement = ((np.mean(dqn_rewards) - np.mean(baseline_rewards)) /
                   abs(np.mean(baseline_rewards)) * 100)

    print(f"\n改進幅度:")
    print(f"  獎勵提升: {improvement:+.1f}%")
    print(f"  存活率提升: {(dqn_survival - baseline_survival)/n_episodes*100:+.1f}%")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='測試訓練好的 DQN agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  # 測試 vector 模式的模型
  python test_agent.py --model ./models/dqn_vector/best_model/best_model

  # 測試並比較
  python test_agent.py --model ./models/dqn_vector/final_model --compare

  # 不渲染（快速測試）
  python test_agent.py --model ./models/dqn_vector/final_model --no-render --episodes 20
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='模型路徑（不含 .zip 後綴）'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='vector',
        choices=['vector', 'grid'],
        help='觀察模式（預設：vector）'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='base',
        help='環境配置（預設：base）'
    )

    parser.add_argument(
        '--robot-id',
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help='測試的機器人 ID（預設：0）'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=5,
        help='測試回合數（預設：5）'
    )

    parser.add_argument(
        '--no-render',
        action='store_true',
        help='不渲染（用於快速測試）'
    )

    parser.add_argument(
        '--delay',
        type=float,
        default=0.05,
        help='每步之間的延遲（秒，預設：0.05）'
    )

    parser.add_argument(
        '--compare',
        action='store_true',
        help='與 baseline 比較'
    )

    args = parser.parse_args()

    # 檢查 stable-baselines3
    if not SB3_AVAILABLE:
        print("\nError: stable-baselines3 is required")
        print("Install with: pip install stable-baselines3")
        return

    # 測試
    if args.compare:
        compare_with_baseline(
            model_path=args.model,
            obs_type=args.mode,
            config_name=args.config,
            robot_id=args.robot_id,
            n_episodes=args.episodes
        )
    else:
        test_agent(
            model_path=args.model,
            obs_type=args.mode,
            config_name=args.config,
            robot_id=args.robot_id,
            n_episodes=args.episodes,
            render=not args.no_render,
            delay=args.delay
        )

    print("\n✅ 測試完成！")


if __name__ == '__main__':
    main()
