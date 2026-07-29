"""Q学習エージェント: 状態離散化に基づく表形式Q学習"""
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
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

def init_q_table(config: AgentConfig) -> jnp.ndarray:
    """Qテーブル初期化"""
    n_states = 1
    for n in config.n_bins:
        n_states *= n
    return jnp.full((n_states, config.n_actions), config.q_init)

def discretize(config: AgentConfig, state: jnp.ndarray) -> jnp.ndarray:
    """連続状態を離散状態インデックスへ変換"""
    low, high = jnp.array(config.state_low), jnp.array(config.state_high)
    n_bins = jnp.array(config.n_bins)
    ratio = (state - low) / (high - low)
    bins = jnp.clip((ratio * n_bins).astype(jnp.int32), 0, n_bins - 1)
    weights, w = [], 1
    for n in reversed(config.n_bins):
        weights.append(w)
        w *= n
    return jnp.dot(bins, jnp.array(weights[::-1]))

def epsilon(config: AgentConfig, step_count: jnp.ndarray) -> jnp.ndarray:
    """指数的に減衰するepsilon"""
    return config.eps_end + (config.eps_start - config.eps_end) * jnp.exp(-step_count / config.eps_decay)

def select_action(config: AgentConfig, q_table: jnp.ndarray, state: jnp.ndarray, step_count: jnp.ndarray, key: jax.Array):
    """epsilon-greedy方策"""
    key_eps, key_act = jax.random.split(key)
    s_idx = discretize(config, state)
    greedy = jnp.argmax(q_table[s_idx])
    random_a = jax.random.randint(key_act, (), 0, config.n_actions)
    explore = jax.random.uniform(key_eps) < epsilon(config, step_count)
    return jnp.where(explore, random_a, greedy)

def batched_update(config: AgentConfig, q_table: jnp.ndarray, states, actions, rewards, next_states, dones) -> jnp.ndarray:
    """一括でQ学習更新"""
    s_idx = jax.vmap(discretize, in_axes=(None, 0))(config, states)
    ns_idx = jax.vmap(discretize, in_axes=(None, 0))(config, next_states)
    target = rewards + (1.0 - dones) * config.gamma * jnp.max(q_table[ns_idx], axis=-1)
    td_error = target - q_table[s_idx, actions]
    flat = s_idx * config.n_actions + actions  # (状態, 行動) を1次元へ
    total = jnp.zeros(q_table.size).at[flat].add(config.alpha * td_error)
    count = jnp.zeros(q_table.size).at[flat].add(1.0)
    return q_table + (total / jnp.maximum(count, 1.0)).reshape(q_table.shape)

@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def train_vec(params: env.CartPoleParams, task: env.TaskConfig, config: AgentConfig, training: TrainingConfig, num_envs: int, key: jax.Array):
    """num_envs個の環境を並列で, 1つのQテーブルを共有して学習"""
    iterations = training.total_steps // num_envs
    key, key_reset = jax.random.split(key)
    phase = (jnp.arange(num_envs) * training.max_episode_steps) // num_envs
    init_carry = (
        init_q_table(config),
        jax.vmap(env.reset, in_axes=(None, 0))(task, jax.random.split(key_reset, num_envs)),
        jnp.zeros((training.buffer_size, 4)),
        jnp.zeros(num_envs), phase.astype(jnp.int32), key
    )
    def scan_step(carry, iteration):
        q_table, states, buffer, ep_return, ep_steps, key = carry
        key, key_act, key_plan, key_reset = jax.random.split(key, 4)
        step_count = iteration * num_envs
        actions = jax.vmap(select_action, in_axes=(None, None, 0, None, 0))(config, q_table, states, step_count, jax.random.split(key_act, num_envs))
        next_states, rewards, dones = jax.vmap(env.step, in_axes=(None, None, 0, 0))(params, task, states, actions)
        timeout = ep_steps + 1 >= training.max_episode_steps
        q_table = batched_update(config, q_table, states, actions, rewards, next_states, dones.astype(jnp.float32))
        if training.planning_steps > 0:
            slots = (step_count + jnp.arange(num_envs)) % training.buffer_size
            buffer = buffer.at[slots].set(states)
            valid = jnp.minimum(step_count + num_envs, training.buffer_size)
            n_plan = training.planning_steps * num_envs
            k1, k2 = jax.random.split(key_plan)
            plan_s = buffer[jax.random.randint(k1, (n_plan,), 0, valid)]
            plan_a = jax.random.randint(k2, (n_plan,), 0, config.n_actions)
            plan_ns, plan_r, plan_d = jax.vmap(env.step, in_axes=(None, None, 0, 0))(params, task, plan_s, plan_a)
            q_table = batched_update(config, q_table, plan_s, plan_a, plan_r, plan_ns, plan_d.astype(jnp.float32))
        ep_return = ep_return + rewards
        finished = dones | timeout
        fresh = jax.vmap(env.reset, in_axes=(None, 0))(task, jax.random.split(key_reset, num_envs))
        states = jnp.where(finished[:, None], fresh, next_states)
        out = (finished, ep_return, ep_steps + 1)
        ep_return = jnp.where(finished, 0.0, ep_return)
        ep_steps = jnp.where(finished, 0, ep_steps + 1)
        return (q_table, states, buffer, ep_return, ep_steps, key), out
    carry, (finished, returns, lengths) = jax.lax.scan(scan_step, init_carry, jnp.arange(iterations))
    return carry[0], finished, returns, lengths

@partial(jax.jit, static_argnums=(0, 1, 2, 5))
def rollout_greedy(params: env.CartPoleParams, task: env.TaskConfig, config: AgentConfig, q_table, x0, steps=500):
    """学習後のgreedy方策による軌道"""
    def body(state, _):
        action = jnp.argmax(q_table[discretize(config, state)])
        next_state, reward, done = env.step(params, task, state, action)
        return next_state, (state, reward, done)
    _, (states, rewards, dones) = jax.lax.scan(body, x0, None, length=steps)
    return states, rewards, dones

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
    B_c = jnp.array([
        [0.0],
        [1.0 / mt + ml / (mt**2 * d)],
        [0.0],
        [-1.0 / (mt * d)]
    ])
    A = jnp.eye(4) + params.dt * A_c
    B = params.dt * B_c
    return A, B

def solve_dare(A, B, Q, R, iterations=500):
    """Riccati方程式を不動点反復で解き, フィードバックゲインを返す."""
    def body(P, _):
        K = jnp.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P_next = Q + A.T @ P @ (A - B @ K)
        return P_next, None
    P, _ = jax.lax.scan(body, Q, None, length=iterations)
    return jnp.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

@partial(jax.jit, static_argnums=(0, 1, 4, 5))
def simulate_lqr(params: env.CartPoleParams, task: env.TaskConfig, K, x0, steps=500, clip_force=False):
    """LQRフィードバックを適用"""
    def body(state, _):
        force = (-K @ state)[0]
        if clip_force:
            force = jnp.clip(force, -params.force_mag, params.force_mag)
        next_state, reward, done = env.transition(params, task, state, force)
        return next_state, (state, force, reward, done)
    _, result = jax.lax.scan(body, x0, None, length=steps)
    return result

@partial(jax.jit, static_argnums=(0, 1, 4, 5))
def roa_sweep(params: env.CartPoleParams, task: env.TaskConfig, K, angles_deg, clip=False, steps=1500):
    def trial(theta0):
        def body(carry, _):
            state, max_x, fallen = carry
            u = (-K @ state)[0]
            if clip:
                u = jnp.clip(u, -params.force_mag, params.force_mag)
            next_state, _, _ = env.transition(params, task, state, u)
            state = jnp.where(fallen, state, next_state)
            max_x = jnp.where(fallen, max_x, jnp.maximum(max_x, jnp.abs(next_state[0])))
            fallen = fallen | (jnp.abs(next_state[2]) > jnp.pi / 2)
            return (state, max_x, fallen), None
        init = (jnp.array([0.0, 0.0, theta0, 0.0]), jnp.float32(0.0), jnp.bool_(False))
        (final, max_x, fallen), _ = jax.lax.scan(body, init, None, length=steps)
        converged = (jnp.abs(final[2]) < jnp.deg2rad(1.0)) & (jnp.abs(final[0]) < 0.1)
        return ~fallen & converged, max_x
    return jax.vmap(trial)(jnp.deg2rad(angles_deg))

def region_of_attraction(params: env.CartPoleParams, task: env.TaskConfig, K, angles=ROA_ANGLES):
    """初期角度を振ってLQRの吸引領域を調べ, 結果を表示して返す."""
    angles_deg = jnp.array(angles, dtype=jnp.float32)
    free_ok, free_x = roa_sweep(params, task, K, angles_deg, clip=False)
    clip_ok, clip_x = roa_sweep(params, task, K, angles_deg, clip=True)
    print("初期角度 | 入力制限なし (max|x|) | ±10N制限 (max|x|)")
    for i, th in enumerate(angles):
        print(
            f"{th:3d} deg | {'成功' if free_ok[i] else '失敗'} ({free_x[i]:4.1f} m)      | "
            f"{'成功' if clip_ok[i] else '失敗'} ({clip_x[i]:4.1f} m)"
        )
    return {
        "angles": np.array(angles),
        "free": (np.array(free_ok), np.array(free_x)),
        "clipped": (np.array(clip_ok), np.array(clip_x))
    }