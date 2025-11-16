"""
Gym-compatible Wrapper for Robot Energy Survival Environment

支援多種 observation 模式：
- 'vector': 扁平向量表示（適合 MLP）
- 'grid': 多通道網格表示（適合 CNN）

設計原則：
- 從簡單開始（vector mode）
- 保持可擴展性（支援 grid mode）
- 統一的 API 接口
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, Literal
import gymnasium as gym
from gymnasium import spaces

from robot_vacuum_env import RobotVacuumEnv, get_rational_action


class GymRobotEnv(gym.Env):
    """
    Gym-compatible wrapper for multi-robot energy survival environment

    支援單個 agent 的訓練（其他 agent 使用 epsilon-greedy）

    Observation Modes:
        - 'vector': 扁平向量 (15維) - 適合 3×3 小地圖 + MLP
        - 'grid': 多通道網格 (4×n×n) - 適合大地圖 + CNN
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(
        self,
        config: Dict[str, Any],
        robot_id: int = 0,
        obs_type: Literal['vector', 'grid'] = 'vector',
        reward_shaping: Literal['simple', 'dense', 'sparse'] = 'dense',
        render_mode: Optional[str] = 'human'
    ):
        """
        初始化 Gym 環境

        Args:
            config: 環境配置字典
            robot_id: 要訓練的機器人 ID (0-3)
            obs_type: 觀察類型 ('vector' 或 'grid')
            reward_shaping: 獎勵塑形方式
                - 'simple': 只有存活/死亡
                - 'dense': 密集獎勵（能量、充電等）
                - 'sparse': 稀疏獎勵（只在結束時）
            render_mode: 渲染模式
        """
        super().__init__()

        self.env = RobotVacuumEnv(config)
        self.robot_id = robot_id
        self.obs_type = obs_type
        self.reward_shaping = reward_shaping
        self.render_mode = render_mode

        # 儲存配置
        self.config = config
        self.n = config.get('n', 3)
        self.initial_energy = config['initial_energy']

        # 用於計算獎勵的上一個狀態
        self.prev_robots = None

        # 定義 Action Space
        self.action_space = spaces.Discrete(5)  # 0-4: 上下左右停

        # 定義 Observation Space
        if obs_type == 'vector':
            # 扁平向量模式：15 維
            # [自己4維] + [到家2維] + [其他機器人3×3=9維]
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(15,),
                dtype=np.float32
            )
        elif obs_type == 'grid':
            # 網格模式：4 通道 × n × n
            # Channel 0: 充電座層
            # Channel 1: 自己的位置
            # Channel 2: 其他機器人
            # Channel 3: 能量信息
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(4, self.n, self.n),
                dtype=np.float32
            )
        else:
            raise ValueError(f"Unknown obs_type: {obs_type}")

        # 機器人的家（充電座）位置
        self.homes = {
            0: (0, 0),
            1: (0, self.n - 1),
            2: (self.n - 1, 0),
            3: (self.n - 1, self.n - 1)
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置環境

        Returns:
            observation: 初始觀察
            info: 額外信息
        """
        super().reset(seed=seed)

        # 重置底層環境
        self.env.reset()

        # 儲存初始機器人狀態
        self.prev_robots = [r.copy() for r in self.env.robots]

        # 獲取初始觀察
        obs = self._get_observation()

        # 額外信息
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        執行一步

        Args:
            action: 要訓練的機器人的動作 (0-4)

        Returns:
            observation: 下一個觀察
            reward: 獎勵值
            terminated: 是否終止（機器人死亡或達到目標）
            truncated: 是否截斷（超時）
            info: 額外信息
        """
        # 生成所有機器人的動作
        actions = self._get_all_actions(action)

        # 執行動作
        _, done = self.env.step(actions)

        # 獲取當前機器人狀態
        robot = self.env.robots[self.robot_id]
        prev_robot = self.prev_robots[self.robot_id]

        # 計算獎勵
        reward = self._compute_reward(robot, prev_robot)

        # 獲取觀察
        obs = self._get_observation()

        # 判斷終止條件
        terminated = not robot['is_active']  # 機器人停機
        truncated = done  # 達到最大回合數

        # 額外信息
        info = self._get_info()

        # 更新 prev_robots
        self.prev_robots = [r.copy() for r in self.env.robots]

        return obs, reward, terminated, truncated, info

    def render(self):
        """渲染環境"""
        if self.render_mode == 'human':
            self.env.render()
        elif self.render_mode == 'rgb_array':
            # TODO: 返回 RGB 陣列
            pass

    def close(self):
        """關閉環境"""
        self.env.close()

    # ==================== 私有方法 ====================

    def _get_observation(self) -> np.ndarray:
        """
        獲取當前觀察

        根據 obs_type 返回不同格式的觀察
        """
        if self.obs_type == 'vector':
            return self._get_vector_observation()
        elif self.obs_type == 'grid':
            return self._get_grid_observation()
        else:
            raise ValueError(f"Unknown obs_type: {self.obs_type}")

    def _get_vector_observation(self) -> np.ndarray:
        """
        獲取扁平向量觀察（15維）

        結構：
        [0-3]:   自己的信息 (x, y, energy, on_charger)
        [4-5]:   到家的距離 (dx, dy)
        [6-14]:  其他3個機器人的相對信息 (dx, dy, active) × 3
        """
        robot = self.env.robots[self.robot_id]
        n = self.n

        obs = []

        # 1. 自己的信息 (4維)
        obs.extend([
            robot['x'] / (n - 1) if n > 1 else 0.5,  # 歸一化 x [0, 1]
            robot['y'] / (n - 1) if n > 1 else 0.5,  # 歸一化 y [0, 1]
            robot['energy'] / self.initial_energy,    # 歸一化能量 [0, 1]
            float(self.env.static_grid[robot['y'], robot['x']] == self.env.CHARGER)  # 在充電座上
        ])

        # 2. 到家的距離 (2維)
        home_y, home_x = self.homes[self.robot_id]
        obs.extend([
            (home_x - robot['x']) / (n - 1) if n > 1 else 0,  # dx [-1, 1]
            (home_y - robot['y']) / (n - 1) if n > 1 else 0   # dy [-1, 1]
        ])

        # 3. 其他機器人的相對信息 (9維)
        for i, other in enumerate(self.env.robots):
            if i != self.robot_id:
                obs.extend([
                    (other['x'] - robot['x']) / (n - 1) if n > 1 else 0,  # 相對 x
                    (other['y'] - robot['y']) / (n - 1) if n > 1 else 0,  # 相對 y
                    float(other['is_active'])  # 是否活躍
                ])

        return np.array(obs, dtype=np.float32)

    def _get_grid_observation(self) -> np.ndarray:
        """
        獲取網格觀察（4×n×n）

        4個通道：
        [0]: 充電座層（所有充電座的位置）
        [1]: 自己的位置
        [2]: 其他機器人的位置
        [3]: 自己的能量信息（在自己位置填入歸一化能量）
        """
        robot = self.env.robots[self.robot_id]
        n = self.n

        # 初始化 4 個通道
        obs = np.zeros((4, n, n), dtype=np.float32)

        # Channel 0: 充電座層
        obs[0] = (self.env.static_grid == self.env.CHARGER).astype(np.float32)

        # Channel 1: 自己的位置
        obs[1, robot['y'], robot['x']] = 1.0

        # Channel 2: 其他機器人
        for i, other in enumerate(self.env.robots):
            if i != self.robot_id and other['is_active']:
                obs[2, other['y'], other['x']] = 1.0

        # Channel 3: 自己的能量信息
        energy_normalized = robot['energy'] / self.initial_energy
        obs[3, robot['y'], robot['x']] = energy_normalized

        return obs

    def _get_all_actions(self, trained_action: int) -> list:
        """
        生成所有機器人的動作列表

        Args:
            trained_action: 訓練中機器人的動作

        Returns:
            4個機器人的動作列表
        """
        actions = []

        for i in range(4):
            if i == self.robot_id:
                # 使用訓練的動作
                actions.append(trained_action)
            else:
                # 其他機器人使用 epsilon-greedy
                robot = self.env.robots[i]

                if not robot['is_active']:
                    actions.append(4)  # 停機，只能停留
                elif np.random.random() < self.env.epsilon:
                    actions.append(np.random.randint(0, 5))  # 探索
                else:
                    # 理性求生
                    actions.append(get_rational_action(robot, self.homes[i], self.n))

        return actions

    def _compute_reward(
        self,
        robot: Dict[str, Any],
        prev_robot: Dict[str, Any]
    ) -> float:
        """
        計算獎勵

        根據 reward_shaping 類型返回不同的獎勵
        """
        if self.reward_shaping == 'simple':
            return self._compute_simple_reward(robot, prev_robot)
        elif self.reward_shaping == 'dense':
            return self._compute_dense_reward(robot, prev_robot)
        elif self.reward_shaping == 'sparse':
            return self._compute_sparse_reward(robot, prev_robot)
        else:
            raise ValueError(f"Unknown reward_shaping: {self.reward_shaping}")

    def _compute_simple_reward(
        self,
        robot: Dict[str, Any],
        prev_robot: Dict[str, Any]
    ) -> float:
        """
        簡單獎勵：只有存活/死亡
        """
        if robot['is_active']:
            return 0.1  # 存活獎勵
        else:
            return -10.0  # 死亡懲罰

    def _compute_dense_reward(
        self,
        robot: Dict[str, Any],
        prev_robot: Dict[str, Any]
    ) -> float:
        """
        密集獎勵：考慮能量變化、充電、碰撞等

        這是**推薦的獎勵設計**，提供更多學習信號
        """
        reward = 0.0

        # 1. 基礎存活獎勵
        if robot['is_active']:
            reward += 0.01
        else:
            # 死亡重罰
            return -5.0

        # 2. 能量變化獎勵
        energy_delta = robot['energy'] - prev_robot['energy']
        reward += energy_delta * 0.01

        # 3. 充電獎勵
        if robot['charge_count'] > prev_robot['charge_count']:
            reward += 0.5

        # 4. 碰撞懲罰（通過能量變化檢測）
        if energy_delta == -self.env.e_collision:
            reward -= 0.3

        # 5. 能量過低警告
        energy_ratio = robot['energy'] / self.initial_energy
        if energy_ratio < 0.2:
            reward -= 0.1  # 鼓勵及時回家充電

        return reward

    def _compute_sparse_reward(
        self,
        robot: Dict[str, Any],
        prev_robot: Dict[str, Any]
    ) -> float:
        """
        稀疏獎勵：只在 episode 結束時給獎勵

        適合測試 agent 的探索能力
        """
        # 只在結束時給獎勵
        if self.env.current_step >= self.env.n_steps:
            return 1.0 if robot['is_active'] else 0.0
        else:
            return 0.0

    def _get_info(self) -> Dict[str, Any]:
        """
        獲取額外信息
        """
        robot = self.env.robots[self.robot_id]

        return {
            'robot_id': self.robot_id,
            'step': self.env.current_step,
            'energy': robot['energy'],
            'energy_ratio': robot['energy'] / self.initial_energy,
            'charge_count': robot['charge_count'],
            'is_active': robot['is_active'],
            'x': robot['x'],
            'y': robot['y'],
            # 全局信息
            'active_robots': sum(1 for r in self.env.robots if r['is_active']),
            'total_charges': sum(r['charge_count'] for r in self.env.robots)
        }


def make_env(
    config: Optional[Dict[str, Any]] = None,
    robot_id: int = 0,
    obs_type: Literal['vector', 'grid'] = 'vector',
    reward_shaping: Literal['simple', 'dense', 'sparse'] = 'dense'
) -> GymRobotEnv:
    """
    便捷函數：創建環境

    Args:
        config: 環境配置（None 則使用預設）
        robot_id: 要訓練的機器人 ID
        obs_type: 觀察類型
        reward_shaping: 獎勵塑形方式

    Returns:
        GymRobotEnv 實例

    Example:
        >>> # 簡單模式（3×3 + vector + MLP）
        >>> env = make_env()
        >>>
        >>> # 大地圖模式（5×5 + grid + CNN）
        >>> config = {'n': 5, 'initial_energy': 150, ...}
        >>> env = make_env(config, obs_type='grid')
    """
    if config is None:
        # 預設配置：3×3 小地圖
        config = {
            'n': 3,
            'initial_energy': 100,
            'e_move': 1,
            'e_charge': 5,
            'e_collision': 3,
            'n_steps': 500,
            'epsilon': 0.1  # 其他機器人的探索率
        }

    return GymRobotEnv(
        config=config,
        robot_id=robot_id,
        obs_type=obs_type,
        reward_shaping=reward_shaping
    )


if __name__ == '__main__':
    # 測試環境
    print("=" * 60)
    print("測試 Gym Wrapper")
    print("=" * 60)

    # 測試 vector 模式
    print("\n1. Vector 模式（15維）")
    print("-" * 60)
    env_vec = make_env(obs_type='vector')
    obs, info = env_vec.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation: {obs}")
    print(f"Info: {info}")

    # 執行幾步
    for i in range(3):
        action = env_vec.action_space.sample()
        obs, reward, terminated, truncated, info = env_vec.step(action)
        print(f"\nStep {i+1}: action={action}, reward={reward:.3f}, energy={info['energy']}")

    env_vec.close()

    # 測試 grid 模式
    print("\n\n2. Grid 模式（4×3×3）")
    print("-" * 60)
    env_grid = make_env(obs_type='grid')
    obs, info = env_grid.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Channel 0 (Chargers):\n{obs[0]}")
    print(f"Channel 1 (Self):\n{obs[1]}")
    print(f"Channel 2 (Others):\n{obs[2]}")
    print(f"Channel 3 (Energy):\n{obs[3]}")

    env_grid.close()

    print("\n" + "=" * 60)
    print("測試完成！")
    print("=" * 60)
