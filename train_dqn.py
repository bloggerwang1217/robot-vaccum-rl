"""
使用 DQN 訓練單個機器人

示範如何使用 Gym wrapper 訓練強化學習 agent

支援兩種模式：
- vector: 扁平向量 (15維) + MLP
- grid: 網格 (4×n×n) + CNN
"""

import argparse
from pathlib import Path

# Stable-Baselines3 imports
try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
        CallbackList
    )
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    print("Warning: stable-baselines3 not installed")
    print("Install with: pip install stable-baselines3")
    SB3_AVAILABLE = False

from gym_wrapper import make_env
from energy_survival_config import get_config


def train_dqn_vector(
    total_timesteps: int = 100000,
    learning_rate: float = 1e-3,
    config_name: str = 'base',
    robot_id: int = 0,
    save_path: str = './models/dqn_vector'
):
    """
    訓練 DQN (Vector 模式)

    使用扁平向量觀察 + MLP 網絡

    Args:
        total_timesteps: 總訓練步數
        learning_rate: 學習率
        config_name: 環境配置名稱
        robot_id: 訓練的機器人 ID
        save_path: 模型保存路徑
    """
    print("=" * 70)
    print("DQN 訓練 - Vector 模式（MLP）")
    print("=" * 70)

    # 創建環境配置
    config = get_config(config_name)
    print(f"\n環境配置: {config_name}")
    print(f"  - 地圖大小: {config['n']}×{config['n']}")
    print(f"  - 初始能量: {config['initial_energy']}")
    print(f"  - 總回合數: {config['n_steps']}")
    print(f"  - 其他機器人探索率: {config.get('epsilon', 0.1)}")

    # 創建環境
    env = make_env(
        config=config,
        robot_id=robot_id,
        obs_type='vector',
        reward_shaping='dense'
    )
    env = Monitor(env)  # 包裝用於監控

    # 創建評估環境
    eval_env = make_env(
        config=config,
        robot_id=robot_id,
        obs_type='vector',
        reward_shaping='dense'
    )
    eval_env = Monitor(eval_env)

    # 創建 DQN 模型
    print(f"\n創建 DQN 模型...")
    print(f"  - Observation shape: {env.observation_space.shape}")
    print(f"  - Action space: {env.action_space.n}")
    print(f"  - Learning rate: {learning_rate}")

    model = DQN(
        policy="MlpPolicy",  # 使用 MLP
        env=env,
        learning_rate=learning_rate,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        tau=1.0,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=f"{save_path}/tensorboard/"
    )

    # 設置 callbacks
    Path(save_path).mkdir(parents=True, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/best_model",
        log_path=f"{save_path}/eval",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{save_path}/checkpoints",
        name_prefix="dqn_vector"
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # 開始訓練
    print(f"\n開始訓練 {total_timesteps} 步...")
    print("=" * 70)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=100,
        progress_bar=True
    )

    # 保存最終模型
    final_path = f"{save_path}/final_model"
    model.save(final_path)
    print(f"\n最終模型已保存到: {final_path}")

    # 清理
    env.close()
    eval_env.close()

    return model


def train_dqn_grid(
    total_timesteps: int = 100000,
    learning_rate: float = 1e-4,
    config_name: str = 'large',
    robot_id: int = 0,
    save_path: str = './models/dqn_grid'
):
    """
    訓練 DQN (Grid 模式)

    使用網格觀察 + CNN 網絡

    Args:
        total_timesteps: 總訓練步數
        learning_rate: 學習率
        config_name: 環境配置名稱
        robot_id: 訓練的機器人 ID
        save_path: 模型保存路徑
    """
    print("=" * 70)
    print("DQN 訓練 - Grid 模式（CNN）")
    print("=" * 70)

    # 創建環境配置
    config = get_config(config_name)
    print(f"\n環境配置: {config_name}")
    print(f"  - 地圖大小: {config['n']}×{config['n']}")
    print(f"  - 初始能量: {config['initial_energy']}")

    # 創建環境
    env = make_env(
        config=config,
        robot_id=robot_id,
        obs_type='grid',
        reward_shaping='dense'
    )
    env = Monitor(env)

    # 創建評估環境
    eval_env = make_env(
        config=config,
        robot_id=robot_id,
        obs_type='grid',
        reward_shaping='dense'
    )
    eval_env = Monitor(eval_env)

    # 創建 DQN 模型
    print(f"\n創建 DQN 模型...")
    print(f"  - Observation shape: {env.observation_space.shape}")
    print(f"  - Action space: {env.action_space.n}")
    print(f"  - Learning rate: {learning_rate}")

    model = DQN(
        policy="CnnPolicy",  # 使用 CNN
        env=env,
        learning_rate=learning_rate,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=f"{save_path}/tensorboard/"
    )

    # 設置 callbacks
    Path(save_path).mkdir(parents=True, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/best_model",
        log_path=f"{save_path}/eval",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{save_path}/checkpoints",
        name_prefix="dqn_grid"
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # 開始訓練
    print(f"\n開始訓練 {total_timesteps} 步...")
    print("=" * 70)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=100,
        progress_bar=True
    )

    # 保存最終模型
    final_path = f"{save_path}/final_model"
    model.save(final_path)
    print(f"\n最終模型已保存到: {final_path}")

    # 清理
    env.close()
    eval_env.close()

    return model


def main():
    """主函數：解析命令列參數並開始訓練"""
    parser = argparse.ArgumentParser(
        description='訓練 DQN agent 進行能量求生',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  # Vector 模式（預設，適合 3×3）
  python train_dqn.py --mode vector --steps 100000

  # Grid 模式（適合大地圖）
  python train_dqn.py --mode grid --config large --steps 200000

  # 調整學習率
  python train_dqn.py --mode vector --lr 0.0005 --steps 50000
        """
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
        help='環境配置名稱（預設：base）'
    )

    parser.add_argument(
        '--steps',
        type=int,
        default=100000,
        help='訓練步數（預設：100000）'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='學習率（預設：vector=1e-3, grid=1e-4）'
    )

    parser.add_argument(
        '--robot-id',
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help='訓練的機器人 ID（預設：0）'
    )

    parser.add_argument(
        '--save-path',
        type=str,
        default=None,
        help='模型保存路徑（預設：./models/dqn_{mode}）'
    )

    args = parser.parse_args()

    # 檢查 stable-baselines3
    if not SB3_AVAILABLE:
        print("\nError: stable-baselines3 is required for training")
        print("Install with: pip install stable-baselines3")
        return

    # 設定預設值
    if args.lr is None:
        args.lr = 1e-3 if args.mode == 'vector' else 1e-4

    if args.save_path is None:
        args.save_path = f'./models/dqn_{args.mode}'

    # 開始訓練
    if args.mode == 'vector':
        train_dqn_vector(
            total_timesteps=args.steps,
            learning_rate=args.lr,
            config_name=args.config,
            robot_id=args.robot_id,
            save_path=args.save_path
        )
    elif args.mode == 'grid':
        train_dqn_grid(
            total_timesteps=args.steps,
            learning_rate=args.lr,
            config_name=args.config,
            robot_id=args.robot_id,
            save_path=args.save_path
        )

    print("\n✅ 訓練完成！")
    print(f"\n查看訓練日誌：")
    print(f"  tensorboard --logdir {args.save_path}/tensorboard")


if __name__ == '__main__':
    main()
