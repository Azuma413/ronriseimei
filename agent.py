"""Q学習エージェント: 状態離散化に基づく表形式Q学習"""
from functools import partial
from math import prod
from typing import NamedTuple
import jax
import jax.numpy as jnp
import env

ROA_ANGLES = (10, 20, 30, 40, 50, 60, 70)  # 吸引領域を調べる初期角度 [deg]

class AgentConfig(NamedTuple):
    n_bins: tuple = (8, 8, 8, 8)  # 各状態次元の分割数
    # 離散化の範囲 [x, x_dot, theta, theta_dot]
    state_low: tuple = (-2.4, -3.0, -0.21, -3.0)
    state_high: tuple = (2.4, 3.0, 0.21, 3.0)
    n_actions: int = 2
    alpha: float = 0.1         # 学習率
    gamma: float = 0.99        # 割引率
    eps_start: float = 1.0     # epsilonの初期値
    eps_end: float = 0.05      # epsilonの最小値
    eps_decay: float = 50000.0 # epsilonの減衰時定数 [step]
    q_init: float = 0.0        # Qテーブルの初期値

class TrainingConfig(NamedTuple):
    total_steps: int
    max_episode_steps: int = 500
    planning_steps: int = 0
    buffer_size: int = 20_000

def discretize(config: AgentConfig, state: jnp.ndarray) -> jnp.ndarray:
    """連続状態を離散状態インデックスへ変換"""
    low, high = jnp.array(config.state_low), jnp.array(config.state_high)
    bins = ((state - low) / (high - low) * jnp.array(config.n_bins)).astype(jnp.int32)
    return jnp.ravel_multi_index(tuple(bins), config.n_bins, mode="clip")  # ビン番号を1つの整数に合成

def select_action(config: AgentConfig, q_table: jnp.ndarray, state: jnp.ndarray, step_count: jnp.ndarray, key: jax.Array):
    """epsilon-greedy方策. epsilonは環境ステップ数に対して指数的に減衰する."""
    key_eps, key_act = jax.random.split(key)
    eps = config.eps_end + (config.eps_start - config.eps_end) * jnp.exp(-step_count / config.eps_decay)
    explore = jax.random.uniform(key_eps) < eps
    return jnp.where(explore, jax.random.randint(key_act, (), 0, config.n_actions), jnp.argmax(q_table[discretize(config, state)]))

def batched_update(config: AgentConfig, q_table: jnp.ndarray, states, actions, rewards, next_states, dones) -> jnp.ndarray:
    """一括でQ学習更新"""
    s_idx = jax.vmap(discretize, in_axes=(None, 0))(config, states)
    ns_idx = jax.vmap(discretize, in_axes=(None, 0))(config, next_states)
    target = rewards + (1.0 - dones) * config.gamma * jnp.max(q_table[ns_idx], axis=-1)
    td_error = target - q_table[s_idx, actions]
    flat = s_idx * config.n_actions + actions  # (状態, 行動) を1次元へ
    total = jnp.zeros(q_table.size).at[flat].add(config.alpha * td_error)
    count = jnp.zeros(q_table.size).at[flat].add(1.0)  # 同一セルへの重複更新は平均を適用する
    return q_table + (total / jnp.maximum(count, 1.0)).reshape(q_table.shape)

@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def train_vec(params: env.CartPoleParams, task: env.TaskConfig, config: AgentConfig, training: TrainingConfig, num_envs: int, key: jax.Array):
    """num_envs個の環境を並列で, 1つのQテーブルを共有して学習"""
    key, key_reset = jax.random.split(key)
    init_carry = (
        jnp.full((prod(config.n_bins), config.n_actions), config.q_init),
        jax.vmap(env.reset, in_axes=(None, 0))(task, jax.random.split(key_reset, num_envs)),
        jnp.zeros((training.buffer_size, 4)),
        jnp.zeros(num_envs),
        (jnp.arange(num_envs) * training.max_episode_steps) // num_envs,  # エピソードの初期位相をずらす
        key
    )
    def scan_step(carry, iteration):
        q_table, states, buffer, ep_return, ep_steps, key = carry
        key, key_act, key_plan, key_reset = jax.random.split(key, 4)
        step_count = iteration * num_envs
        actions = jax.vmap(select_action, in_axes=(None, None, 0, None, 0))(config, q_table, states, step_count, jax.random.split(key_act, num_envs))
        next_states, rewards, dones = jax.vmap(env.step, in_axes=(None, None, 0, 0))(params, task, states, actions)
        q_table = batched_update(config, q_table, states, actions, rewards, next_states, dones.astype(jnp.float32))
        if training.planning_steps > 0:  # 内部モデル (環境と同一の物理モデル) による模擬経験
            slots = (step_count + jnp.arange(num_envs)) % training.buffer_size
            buffer = buffer.at[slots].set(states)
            n_plan = training.planning_steps * num_envs
            k1, k2 = jax.random.split(key_plan)
            plan_s = buffer[jax.random.randint(k1, (n_plan,), 0, jnp.minimum(step_count + num_envs, training.buffer_size))]
            plan_a = jax.random.randint(k2, (n_plan,), 0, config.n_actions)
            plan_ns, plan_r, plan_d = jax.vmap(env.step, in_axes=(None, None, 0, 0))(params, task, plan_s, plan_a)
            q_table = batched_update(config, q_table, plan_s, plan_a, plan_r, plan_ns, plan_d.astype(jnp.float32))
        ep_return = ep_return + rewards
        finished = dones | (ep_steps + 1 >= training.max_episode_steps)
        fresh = jax.vmap(env.reset, in_axes=(None, 0))(task, jax.random.split(key_reset, num_envs))
        states = jnp.where(finished[:, None], fresh, next_states)
        out = (finished, ep_return, ep_steps + 1)
        carry = (q_table, states, buffer, jnp.where(finished, 0.0, ep_return), jnp.where(finished, 0, ep_steps + 1), key)
        return carry, out
    carry, (finished, returns, lengths) = jax.lax.scan(scan_step, init_carry, jnp.arange(training.total_steps // num_envs))
    return carry[0], finished, returns, lengths

@partial(jax.jit, static_argnums=(0, 1, 2, 5))
def rollout_greedy(params: env.CartPoleParams, task: env.TaskConfig, config: AgentConfig, q_table, x0, steps=500):
    """学習後のgreedy方策による軌道"""
    def body(state, _):
        next_state, reward, done = env.step(params, task, state, jnp.argmax(q_table[discretize(config, state)]))
        return next_state, (state, reward, done)
    return jax.lax.scan(body, x0, None, length=steps)[1]

def linearize(params: env.CartPoleParams):
    """直立平衡点まわりで線形化した離散時間の状態方程式"""
    g, l = params.gravity, params.pole_length
    mt = params.cart_mass + params.pole_mass
    ml = params.pole_mass * l
    d = l * (4.0 / 3.0 - params.pole_mass / mt)
    A_c = jnp.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -ml * g / (mt * d), 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, g / d, 0.0]
    ])
    B_c = jnp.array([[0.0], [1.0 / mt + ml / (mt**2 * d)], [0.0], [-1.0 / (mt * d)]])
    return jnp.eye(4) + params.dt * A_c, params.dt * B_c

def solve_dare(A, B, Q, R, iterations=500):
    """Riccati方程式を不動点反復で解き, フィードバックゲインを返す."""
    def body(P, _):
        K = jnp.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        return Q + A.T @ P @ (A - B @ K), None
    P, _ = jax.lax.scan(body, Q, None, length=iterations)
    return jnp.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

@partial(jax.jit, static_argnums=(0, 1, 4, 5))
def simulate_lqr(params: env.CartPoleParams, task: env.TaskConfig, K, x0, steps=500, clip_force=False):
    """LQRフィードバックによる軌道"""
    def body(state, _):
        force = (-K @ state)[0]
        if clip_force:
            force = jnp.clip(force, -params.force_mag, params.force_mag)
        next_state, reward, done = env.transition(params, task, state, force)
        return next_state, (state, reward, done)
    return jax.lax.scan(body, x0, None, length=steps)[1]

@partial(jax.jit, static_argnums=(0, 1, 4, 5))
def roa_sweep(params: env.CartPoleParams, task: env.TaskConfig, K, angles_deg, clip_force=False, steps=1500):
    """初期角度ごとに, 直立へ収束したかとそこまでの最大|x|を返す."""
    def trial(theta0):
        states, _, _ = simulate_lqr(params, task, K, jnp.array([0.0, 0.0, theta0, 0.0]), steps, clip_force)
        fallen = jnp.abs(states[:, 2]) > jnp.pi / 2
        upright = jnp.cumsum(fallen) - fallen == 0  # 転倒した時点までを評価対象とする
        converged = (jnp.abs(states[-1, 2]) < jnp.deg2rad(1.0)) & (jnp.abs(states[-1, 0]) < 0.1)
        return upright[-1] & converged, jnp.max(jnp.where(upright, jnp.abs(states[:, 0]), 0.0))
    return jax.vmap(trial)(jnp.deg2rad(angles_deg))

def region_of_attraction(params: env.CartPoleParams, task: env.TaskConfig, K, angles=ROA_ANGLES):
    """初期角度を振ってLQRの吸引領域を調べ, 結果を表示する."""
    angles_deg = jnp.array(angles, dtype=jnp.float32)
    (free_ok, free_x), (clip_ok, clip_x) = [roa_sweep(params, task, K, angles_deg, clip) for clip in (False, True)]
    print("初期角度 | 入力制限なし (max|x|) | ±10N制限 (max|x|)")
    for i, th in enumerate(angles):
        print(
            f"{th:3d} deg | {'成功' if free_ok[i] else '失敗'} ({free_x[i]:4.1f} m)      | "
            f"{'成功' if clip_ok[i] else '失敗'} ({clip_x[i]:4.1f} m)"
        )
