"""2つの初期化条件と3つの制御手法を総当たりで比較する."""
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import agent
import env


FIG_DIR = Path(__file__).resolve().parent / "fig"

class ExperimentConfig(NamedTuple):
    name: str
    task: env.TaskConfig
    agent_config: agent.AgentConfig
    training: agent.TrainingConfig
    num_seeds: int
    evaluation_state: tuple
    curve_filename: str
    comparison_filename: str
    trajectory_filename: str


class AlgorithmConfig(NamedTuple):
    name: str
    planning_steps: int | None


class TrialResult(NamedTuple):
    q_table: jax.Array
    steps: np.ndarray
    returns: np.ndarray


class ExperimentResult(NamedTuple):
    trials: tuple[TrialResult, ...]
    states: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    forces: np.ndarray | None


EXPERIMENTS = (
    ExperimentConfig(
        name="stabilization",
        task=env.STABILIZATION_TASK,
        agent_config=agent.AgentConfig(),
        training=agent.TrainingConfig(total_steps=600_000),
        num_seeds=5,
        evaluation_state=(0.5, 0.0, 10.0 * jnp.pi / 180.0, 0.0),
        curve_filename="learning_curve.png",
        comparison_filename="dyna_curve.png",
        trajectory_filename="stabilization_traj.png"),
    ExperimentConfig(
        name="swingup",
        task=env.SWINGUP_TASK,
        agent_config=agent.AgentConfig(
            n_bins=(1, 1, 64, 32),
            state_low=(-2.4, -4.0, -jnp.pi, -12.0),
            state_high=(2.4, 4.0, jnp.pi, 12.0),
            eps_decay=150000.0,
            q_init=50.0),
        training=agent.TrainingConfig(total_steps=3_000_000),
        num_seeds=3,
        evaluation_state=(0.0, 0.0, jnp.pi - 0.05, 0.0),
        curve_filename="swingup_curve.png",
        comparison_filename="swingup_dyna_curve.png",
        trajectory_filename="swingup_traj.png"),
)

ALGORITHMS = (
    AlgorithmConfig(name="q-learning", planning_steps=0),
    AlgorithmConfig(name="dyna-q", planning_steps=20),
    AlgorithmConfig(name="lqr", planning_steps=None),
)


def save_figure(filename):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=150)
    plt.close()


def moving_average(values, window=50):
    return np.convolve(values, np.ones(window) / window, mode="valid")


def lqr_gain(params):
    A, B = agent.linearize(params)
    Q = jnp.diag(jnp.array([1.0, 1.0, 10.0, 1.0]))
    R = jnp.array([[0.1]])
    return agent.solve_dare(A, B, Q, R)


def train_trials(params, experiment, training):
    trials = []
    for seed in range(experiment.num_seeds):
        q_table, finished, returns = agent.train(
            params, experiment.task, experiment.agent_config, training,
            jax.random.PRNGKey(seed))
        finished, returns = np.array(finished), np.array(returns)
        trials.append(TrialResult(
            q_table=q_table,
            steps=np.arange(len(finished))[finished],
            returns=returns[finished]))
    return tuple(trials)


def execute(params, experiment, algorithm):
    """1つの課題設定と1つの手法の組を学習・評価する."""
    x0 = jnp.array(experiment.evaluation_state)

    if algorithm.planning_steps is None:
        states, forces, rewards, dones = agent.simulate_lqr(
            params, experiment.task, lqr_gain(params), x0,
            steps=experiment.training.max_episode_steps,
            clip_force=not experiment.task.terminate_on_limits)
        trials = ()
    else:
        training = experiment.training._replace(
            planning_steps=algorithm.planning_steps)
        trials = train_trials(params, experiment, training)
        states, rewards, dones = agent.rollout_greedy(
            params, experiment.task, experiment.agent_config,
            trials[0].q_table, x0,
            steps=training.max_episode_steps)
        forces = None

    result = ExperimentResult(
        trials=trials,
        states=np.array(states),
        rewards=np.array(rewards),
        dones=np.array(dones),
        forces=None if forces is None else np.array(forces))

    label = f"{experiment.name}/{algorithm.name}"
    if experiment.task.terminate_on_limits:
        failures = np.flatnonzero(result.dones)
        survived = failures[0] + 1 if len(failures) else len(result.dones)
        print(f"[{label}] survived: {survived} steps")
    else:
        print(f"[{label}] return: {result.rewards.sum():.1f}")

    if trials:
        final_averages = [moving_average(t.returns)[-1] for t in trials]
        print(f"[{label}] final return (MA): "
              f"{np.mean(final_averages):.1f}")
    return result


def plot_seed_learning(result, filename):
    plt.figure(figsize=(8, 5))
    colors = plt.cm.tab10.colors
    for seed, trial in enumerate(result.trials):
        color = colors[seed]
        average = moving_average(trial.returns)
        average_steps = trial.steps[len(trial.steps) - len(average):]
        plt.plot(trial.steps, trial.returns, color=color, alpha=0.15)
        plt.plot(average_steps, average, color=color, label=f"seed {seed}")
    plt.xlabel("environment steps")
    plt.ylabel("return (moving average)")
    plt.legend()
    save_figure(filename)


def plot_learning_comparison(experiment, results):
    plt.figure(figsize=(8, 5))
    styles = {
        "q-learning": ("tab:blue", "Q-learning"),
        "dyna-q": ("tab:orange", "Dyna-Q"),
    }
    for name, (color, label) in styles.items():
        series = []
        for trial in results[name].trials:
            average = moving_average(trial.returns)
            steps = trial.steps[len(trial.steps) - len(average):]
            series.append((steps, average))
            plt.plot(steps, average, color=color, alpha=0.2)

        start = max(steps[0] for steps, _ in series)
        stop = min(steps[-1] for steps, _ in series)
        grid = np.arange(start, stop + 1, 500)
        mean_curve = np.mean([
            np.interp(grid, steps, values) for steps, values in series
        ], axis=0)
        plt.plot(grid, mean_curve, color=color, lw=2, label=label)

    plt.xlabel("environment steps")
    plt.ylabel("return (moving average)")
    plt.legend()
    save_figure(experiment.comparison_filename)


def plot_trajectories(params, experiment, results):
    steps = experiment.training.max_episode_steps
    t = np.arange(steps) * params.dt

    plt.figure(figsize=(8, 4.5))
    for algorithm in ALGORITHMS:
        theta = results[algorithm.name].states[:, 2]
        if experiment.task.wrap_angle:
            theta = np.unwrap(theta)
        plt.plot(t, np.rad2deg(theta), label=algorithm.name)

    plt.axhline(0.0, color="gray", lw=0.8, ls="--")
    if experiment.task.wrap_angle:
        plt.axhline(180.0, color="gray", lw=0.5)
        plt.axhline(-180.0, color="gray", lw=0.5)
    plt.xlabel("time [s]")
    plt.ylabel("pole angle [deg]  (0 = upright)")
    plt.legend()
    save_figure(experiment.trajectory_filename)


def plot_lqr_diagnostics(params, result):
    t = np.arange(len(result.states)) * params.dt
    _, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(t, result.states[:, 0])
    axes[0].set_ylabel("cart position x [m]")
    axes[1].plot(t, np.rad2deg(result.states[:, 2]))
    axes[1].set_ylabel("pole angle [deg]")
    axes[2].plot(t, result.forces)
    axes[2].set_ylabel("control force u [N]")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.axhline(0.0, color="gray", lw=0.5)
    save_figure("lqr_result.png")

if __name__ == "__main__":
    params = env.CartPoleParams()
    results = {}

    for experiment in EXPERIMENTS:
        task_results = {}
        for algorithm in ALGORITHMS:
            task_results[algorithm.name] = execute(
                params, experiment, algorithm)
        results[experiment.name] = task_results

        plot_seed_learning(
            task_results["q-learning"], experiment.curve_filename)
        plot_learning_comparison(experiment, task_results)
        plot_trajectories(params, experiment, task_results)

    stabilization = results["stabilization"]
    plot_lqr_diagnostics(params, stabilization["lqr"])
    agent.region_of_attraction(
        params, EXPERIMENTS[0].task, lqr_gain(params))
