"""Q学習エージェント: 状態離散化に基づく表形式Q学習"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

import env


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
    """Qテーブルを定数で初期化する. 形状は (状態数, n_actions).
    q_initを真の価値より大きく取ると未訪問の状態への探索が促される
    (オプティミスティック初期化)"""
    n_states = 1
    for n in config.n_bins:
        n_states *= n
    return jnp.full((n_states, config.n_actions), config.q_init)


def discretize(config: AgentConfig, state: jnp.ndarray) -> jnp.ndarray:
    """連続状態を離散状態インデックスへ変換する純粋関数"""
    low, high = jnp.array(config.state_low), jnp.array(config.state_high)
    n_bins = jnp.array(config.n_bins)
    ratio = (state - low) / (high - low)
    bins = jnp.clip((ratio * n_bins).astype(jnp.int32), 0, n_bins - 1)
    weights, w = [], 1
    for n in reversed(config.n_bins):  # 各次元のビン番号を1つの整数に合成
        weights.append(w)
        w *= n
    return jnp.dot(bins, jnp.array(weights[::-1]))


def epsilon(config: AgentConfig, step_count: jnp.ndarray) -> jnp.ndarray:
    """指数的に減衰するepsilonを返す"""
    return config.eps_end + (config.eps_start - config.eps_end) * jnp.exp(
        -step_count / config.eps_decay)


def select_action(config: AgentConfig, q_table: jnp.ndarray,
                  state: jnp.ndarray, step_count: jnp.ndarray, key: jax.Array):
    """epsilon-greedy方策に基づき行動を選択する"""
    key_eps, key_act = jax.random.split(key)
    s_idx = discretize(config, state)
    greedy = jnp.argmax(q_table[s_idx])
    random_a = jax.random.randint(key_act, (), 0, config.n_actions)
    explore = jax.random.uniform(key_eps) < epsilon(config, step_count)
    return jnp.where(explore, random_a, greedy)


def update(config: AgentConfig, q_table: jnp.ndarray, state: jnp.ndarray,
           action: jnp.ndarray, reward: jnp.ndarray, next_state: jnp.ndarray,
           done: jnp.ndarray) -> jnp.ndarray:
    """Q学習の更新則により新しいQテーブルを返す純粋関数"""
    s_idx = discretize(config, state)
    ns_idx = discretize(config, next_state)
    target = reward + (1.0 - done) * config.gamma * jnp.max(q_table[ns_idx])
    td_error = target - q_table[s_idx, action]
    return q_table.at[s_idx, action].add(config.alpha * td_error)


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def train(params: env.CartPoleParams, task: env.TaskConfig,
          config: AgentConfig, training: TrainingConfig, key: jax.Array):
    """Q学習を実行する. planning_steps > 0の場合はDyna-Qとなる."""
    key, key_reset = jax.random.split(key)
    init_carry = (init_q_table(config),
                  env.reset(task, key_reset),
                  jnp.zeros((training.buffer_size, 4)),
                  jnp.float32(0.0),
                  jnp.int32(0),
                  key)

    def scan_step(carry, step_count):
        q_table, state, buffer, ep_return, ep_steps, key = carry
        key, key_act, key_plan, key_reset = jax.random.split(key, 4)

        action = select_action(config, q_table, state, step_count, key_act)
        next_state, reward, done = env.step(params, task, state, action)
        timeout = ep_steps + 1 >= training.max_episode_steps
        q_table = update(config, q_table, state, action, reward,
                         next_state, done.astype(jnp.float32))

        if training.planning_steps > 0:
            buffer = buffer.at[step_count % training.buffer_size].set(state)
            valid = jnp.minimum(step_count + 1, training.buffer_size)

            def plan_step(q, key_p):
                k1, k2 = jax.random.split(key_p)
                s = buffer[jax.random.randint(k1, (), 0, valid)]
                a = jax.random.randint(k2, (), 0, config.n_actions)
                s_next, r, d = env.step(params, task, s, a)
                return update(config, q, s, a, r, s_next,
                              d.astype(jnp.float32)), None

            q_table, _ = jax.lax.scan(
                plan_step, q_table,
                jax.random.split(key_plan, training.planning_steps))

        ep_return = ep_return + reward
        finished = done | timeout
        state = jnp.where(
            finished, env.reset(task, key_reset), next_state)
        out = (finished, ep_return)
        ep_return = jnp.where(finished, 0.0, ep_return)
        ep_steps = jnp.where(finished, 0, ep_steps + 1)
        return (q_table, state, buffer, ep_return, ep_steps, key), out

    carry, (finished, returns) = jax.lax.scan(
        scan_step, init_carry, jnp.arange(training.total_steps))
    return carry[0], finished, returns


@partial(jax.jit, static_argnums=(0, 1, 2, 5))
def rollout_greedy(params: env.CartPoleParams, task: env.TaskConfig,
                   config: AgentConfig, q_table, x0, steps=500):
    """学習後の貪欲方策による軌道を返す."""
    def body(state, _):
        action = jnp.argmax(q_table[discretize(config, state)])
        next_state, reward, done = env.step(params, task, state, action)
        return next_state, (state, reward, done)

    _, (states, rewards, dones) = jax.lax.scan(
        body, x0, None, length=steps)
    return states, rewards, dones


def linearize(params: env.CartPoleParams):
    """直立平衡点まわりで線形化した離散時間の状態方程式を返す."""
    g, l = params.gravity, params.pole_length
    mt = params.cart_mass + params.pole_mass
    ml = params.pole_mass * l
    d = l * (4.0 / 3.0 - params.pole_mass / mt)

    A_c = jnp.array([[0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, -ml * g / (mt * d), 0.0],
                     [0.0, 0.0, 0.0, 1.0],
                     [0.0, 0.0, g / d, 0.0]])
    B_c = jnp.array([[0.0],
                     [1.0 / mt + ml / (mt**2 * d)],
                     [0.0],
                     [-1.0 / (mt * d)]])
    A = jnp.eye(4) + params.dt * A_c
    B = params.dt * B_c
    return A, B


def solve_dare(A, B, Q, R, iterations=500):
    """離散時間Riccati方程式を不動点反復で解き, フィードバックゲインKを返す."""
    def body(P, _):
        K = jnp.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P_next = Q + A.T @ P @ (A - B @ K)
        return P_next, None

    P, _ = jax.lax.scan(body, Q, None, length=iterations)
    return jnp.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


@partial(jax.jit, static_argnums=(0, 1, 4, 5))
def simulate_lqr(params: env.CartPoleParams, task: env.TaskConfig,
                 K, x0, steps=500, clip_force=False):
    """非線形ダイナミクスに対しLQRフィードバック u = -Kx を適用する."""
    def body(state, _):
        force = (-K @ state)[0]
        if clip_force:
            force = jnp.clip(force, -params.force_mag, params.force_mag)
        next_state, reward, done = env.transition(params, task, state, force)
        return next_state, (state, force, reward, done)

    _, result = jax.lax.scan(body, x0, None, length=steps)
    return result


def region_of_attraction(params: env.CartPoleParams,
                         task: env.TaskConfig, K):
    """初期角度を振ってLQRの吸引領域を調べる."""
    def run(theta0_deg, clip):
        state = jnp.array([0.0, 0.0, jnp.deg2rad(theta0_deg), 0.0])
        max_x = 0.0
        for _ in range(1500):
            u = (-K @ state)[0]
            if clip:
                u = jnp.clip(u, -params.force_mag, params.force_mag)
            state, _, _ = env.transition(params, task, state, u)
            max_x = max(max_x, abs(float(state[0])))
            if abs(float(state[2])) > jnp.pi / 2:
                return False, max_x
        ok = abs(float(state[2])) < jnp.deg2rad(1) and abs(float(state[0])) < 0.1
        return ok, max_x

    print("初期角度 | 入力制限なし (max|x|) | ±10N制限 (max|x|)")
    for th in [10, 20, 30, 40, 50, 60, 70]:
        ok1, mx1 = run(th, clip=False)
        ok2, mx2 = run(th, clip=True)
        print(f"{th:3d} deg | {'成功' if ok1 else '失敗'} ({mx1:4.1f} m)      | "
              f"{'成功' if ok2 else '失敗'} ({mx2:4.1f} m)")
