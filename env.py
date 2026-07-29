"""カートポール環境: 純粋関数による実装"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class CartPoleParams(NamedTuple):
    """カートポールの物理パラメータ"""
    gravity: float = 9.8       # 重力加速度 [m/s^2]
    cart_mass: float = 1.0     # カートの質量 [kg]
    pole_mass: float = 0.1     # ポールの質量 [kg]
    pole_length: float = 0.5   # ポールの半長 [m]
    force_mag: float = 10.0    # 行動により加える力の大きさ [N]
    dt: float = 0.02           # 積分の時間刻み [s]
    x_limit: float = 2.4       # カート位置の許容範囲 [m]
    theta_limit: float = 12.0 * jnp.pi / 180.0  # ポール角度の許容範囲 [rad]


class TaskConfig(NamedTuple):
    """初期状態・報酬・終端条件をまとめた課題設定."""
    initial_state: tuple = (0.0, 0.0, 0.0, 0.0)
    wrap_angle: bool = False
    terminate_on_limits: bool = True
    uprightness_reward: bool = False


STABILIZATION_TASK = TaskConfig()
SWINGUP_TASK = TaskConfig(
    initial_state=(0.0, 0.0, jnp.pi, 0.0),
    wrap_angle=True,
    terminate_on_limits=False,
    uprightness_reward=True)


def dynamics(params: CartPoleParams, state: jnp.ndarray, force: float) -> jnp.ndarray:
    """運動方程式に基づき1ステップ後の状態を返す純粋関数

    state = [x, x_dot, theta, theta_dot]
    """
    x, x_dot, theta, theta_dot = state
    total_mass = params.cart_mass + params.pole_mass
    ml = params.pole_mass * params.pole_length

    cos_t, sin_t = jnp.cos(theta), jnp.sin(theta)
    temp = (force + ml * theta_dot**2 * sin_t) / total_mass
    theta_acc = (params.gravity * sin_t - cos_t * temp) / (
        params.pole_length * (4.0 / 3.0 - params.pole_mass * cos_t**2 / total_mass))
    x_acc = temp - ml * theta_acc * cos_t / total_mass

    # 半陰的Euler法による積分
    x_dot = x_dot + params.dt * x_acc
    x = x + params.dt * x_dot
    theta_dot = theta_dot + params.dt * theta_acc
    theta = theta + params.dt * theta_dot
    return jnp.array([x, x_dot, theta, theta_dot])


def reset(task: TaskConfig, key: jax.Array) -> jnp.ndarray:
    """課題ごとの初期状態に[-0.05, 0.05]の一様ノイズを加える."""
    noise = jax.random.uniform(key, (4,), minval=-0.05, maxval=0.05)
    return jnp.array(task.initial_state) + noise


def transition(params: CartPoleParams, task: TaskConfig,
               state: jnp.ndarray, force: float):
    """力を受け取り (次状態, 報酬, 終端フラグ) を返す."""
    next_state = dynamics(params, state, force)

    if task.wrap_angle:
        theta = jnp.mod(next_state[2] + jnp.pi, 2.0 * jnp.pi) - jnp.pi
        next_state = next_state.at[2].set(theta)

    if task.terminate_on_limits:
        done = (jnp.abs(next_state[0]) > params.x_limit) | (
            jnp.abs(next_state[2]) > params.theta_limit)
    else:
        done = jnp.bool_(False)

    if task.uprightness_reward:
        reward = (1.0 + jnp.cos(next_state[2])) / 2.0
    else:
        reward = 1.0
    return next_state, reward, done


def step(params: CartPoleParams, task: TaskConfig,
         state: jnp.ndarray, action: int):
    """離散行動を力へ変換して課題を1ステップ進める."""
    force = jnp.where(action == 1, params.force_mag, -params.force_mag)
    return transition(params, task, state, force)
