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
    num_envs: int  # 1つのQテーブルを共有して並列に進める環境数
    evaluation_state: tuple
    comparison_filename: str
    trajectory_filename: str | None  # Noneなら軌道図を出力しない

class AlgorithmConfig(NamedTuple):
    name: str
    planning_steps: int | None

class TrialResult(NamedTuple):
    q_table: jax.Array
    steps: np.ndarray     # 各エピソード終了時点までに消費した環境ステップ数
    returns: np.ndarray   # そのエピソードの収益
    lengths: np.ndarray   # そのエピソードの長さ [step]

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
        num_seeds=64,
        num_envs=64,
        evaluation_state=(0.5, 0.0, 10.0 * jnp.pi / 180.0, 0.0),
        comparison_filename="dyna_curve.png",
        trajectory_filename=None
    ),
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
        num_seeds=64,
        num_envs=64,
        evaluation_state=(0.0, 0.0, jnp.pi - 0.05, 0.0),
        comparison_filename="swingup_dyna_curve.png",
        trajectory_filename="swingup_traj.png"
    ),
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
    """全seedをvmapで同時に学習"""
    keys = jax.random.split(jax.random.PRNGKey(0), experiment.num_seeds)
    q_tables, finished, returns, lengths = jax.jit(jax.vmap(lambda key: agent.train_vec(params, experiment.task, experiment.agent_config, training, experiment.num_envs, key)))(keys)
    finished, returns, lengths = np.array(finished), np.array(returns), np.array(lengths)
    iterations = finished.shape[1]
    index = np.broadcast_to((np.arange(iterations) * experiment.num_envs)[:, None], finished.shape[1:])
    trials = []
    for seed in range(experiment.num_seeds):
        mask = finished[seed].ravel()
        steps = index.ravel()[mask]
        order = np.argsort(steps, kind="stable")
        trials.append(TrialResult(
            q_table=q_tables[seed], steps=steps[order],
            returns=returns[seed].ravel()[mask][order],
            lengths=lengths[seed].ravel()[mask][order]
        ))
    return tuple(trials)

def trial_curves(trials, field="returns"):
    curves = []
    for trial in trials:
        average = moving_average(getattr(trial, field))
        if len(average):
            curves.append((trial.steps[len(trial.steps) - len(average):], average))
    return curves

def aggregate_curves(curves, points=500):
    start = max(steps[0] for steps, _ in curves)
    stop = min(steps[-1] for steps, _ in curves)
    grid = np.linspace(start, stop, points)
    stacked = np.stack([np.interp(grid, steps, values) for steps, values in curves])
    low10, lower, upper, high90 = np.percentile(stacked, [10, 25, 75, 90], axis=0)
    return grid, stacked.mean(axis=0), lower, upper, low10, high90

def execute(params, experiment, algorithm):
    """1つの課題設定と1つの手法の組を学習・評価"""
    x0 = jnp.array(experiment.evaluation_state)
    if algorithm.planning_steps is None:
        states, forces, rewards, dones = agent.simulate_lqr(
            params, experiment.task, lqr_gain(params), x0,
            steps=experiment.training.max_episode_steps,
            clip_force=not experiment.task.terminate_on_limits
        )
        trials = ()
    else:
        training = experiment.training._replace(planning_steps=algorithm.planning_steps)
        trials = train_trials(params, experiment, training)
        states, rewards, dones = agent.rollout_greedy(params, experiment.task, experiment.agent_config, trials[0].q_table, x0, steps=training.max_episode_steps)
        forces = None
    result = ExperimentResult(
        trials=trials,
        states=np.array(states),
        rewards=np.array(rewards),
        dones=np.array(dones),
        forces=None if forces is None else np.array(forces)
    )
    label = f"{experiment.name}/{algorithm.name}"
    if experiment.task.terminate_on_limits:
        failures = np.flatnonzero(result.dones)
        survived = failures[0] + 1 if len(failures) else len(result.dones)
        print(f"[{label}] survived: {survived} steps")
    else:
        print(f"[{label}] return: {result.rewards.sum():.1f}")
    return result

def plot_lqr_reference(results):
    """LQRの収益を比較用の水平線として引く. LQRは学習を伴わないため定数となる."""
    if "lqr" not in results:
        return
    value = results["lqr"].rewards.sum()
    plt.axhline(value, color="tab:red", ls="--", lw=1.5, label=f"LQR (return {value:.0f})")

def plot_learning_comparison(experiment, results):
    plt.figure(figsize=(8, 5))
    styles = {
        "q-learning": ("tab:blue", "Q-learning"),
        "dyna-q": ("tab:orange", "Dyna-Q"),
    }
    for name, (color, label) in styles.items():
        curves = trial_curves(results[name].trials)
        grid, mean_curve, lower, upper, _, _ = aggregate_curves(curves)
        plt.fill_between(grid, lower, upper, color=color, alpha=0.25, lw=0)
        plt.plot(grid, mean_curve, color=color, lw=2, label=f"{label} (mean of {len(curves)} seeds, IQR)")
    plot_lqr_reference(results)
    plt.xlabel("environment steps")
    plt.ylabel("return (moving average)")
    plt.legend()
    plt.grid(alpha=0.25)
    save_figure(experiment.comparison_filename)

def plot_design_ablation(params, experiment, baseline_trials):
    _, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    cap = experiment.training.max_episode_steps
    naive_config = experiment.agent_config._replace(n_bins=(8, 1, 64, 32), q_init=0.0)
    naive = experiment._replace(task=env.SWINGUP_NAIVE_TASK, agent_config=naive_config)
    naive_trials = train_trials(params, naive, naive.training)
    grid, mean_curve, low, high, _, _ = aggregate_curves(trial_curves(naive_trials, "lengths"))
    axes[0].fill_between(grid, low, high, color="tab:red", alpha=0.25, lw=0)
    axes[0].plot(grid, mean_curve, color="tab:red", lw=2, label=f"mean of {len(naive_trials)} seeds (IQR)")
    axes[0].axhline(cap, color="gray", ls=":", lw=1.2, label=f"timeout ({cap} steps)")
    axes[0].set_ylim(0, cap * 1.08)
    axes[0].set_title(r"(a) naive reward $r=\cos\theta$: episode length")
    axes[0].set_ylabel("episode length [steps]")
    axes[0].annotate(f"{mean_curve[0]:.0f} -> {mean_curve[-1]:.0f} steps", xy=(grid[-1], mean_curve[-1]), xytext=(0.35, 0.45), textcoords="axes fraction", color="tab:red", arrowprops=dict(arrowstyle="->", color="tab:red"))
    x0 = jnp.array(experiment.evaluation_state)
    rollout = jax.jit(jax.vmap(lambda q: agent.rollout_greedy(params, naive.task, naive_config, q, x0, cap)))
    states, _, dones = rollout(jnp.stack([t.q_table for t in naive_trials]))
    states, dones = np.array(states), np.array(dones)
    for seed in range(len(naive_trials)):
        end = np.argmax(dones[seed]) + 1 if dones[seed].any() else cap
        t = np.arange(end) * params.dt
        axes[1].plot(t, np.abs(states[seed, :end, 0]), color="tab:red", alpha=0.35, lw=1)
    axes[1].axhline(params.x_limit, color="gray", ls="--", lw=1.2, label=f"rail limit ({params.x_limit} m)")
    axes[1].set_title("(b) naive reward: greedy policy drives to the rail")
    axes[1].set_ylabel("cart position |x| [m]")
    axes[1].set_xlabel("time [s]")
    zero_init = experiment._replace(agent_config=experiment.agent_config._replace(q_init=0.0))
    zero_trials = train_trials(params, zero_init, zero_init.training)
    for trials, color, label in [(zero_trials, "tab:red", r"$Q_0 = 0$"), (baseline_trials, "tab:blue", r"optimistic $Q_0 = 50$")]:
        grid, mean_curve, low, high, _, _ = aggregate_curves(trial_curves(trials))
        axes[2].fill_between(grid, low, high, color=color, alpha=0.25, lw=0)
        axes[2].plot(grid, mean_curve, color=color, lw=2, label=f"{label}: {mean_curve[-1]:.0f}")
    axes[2].set_ylim(0, cap * 1.08)
    axes[2].set_title(r"(c) fixed reward $r=(1+\cos\theta)/2$: return")
    axes[2].set_ylabel("return (moving average)")
    for ax in (axes[0], axes[2]):
        ax.set_xlabel("environment steps")
    for ax in axes:
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.25)
    save_figure("swingup_ablation.png")
    return naive_trials, zero_trials

def plot_trajectories(params, experiment, results):
    plt.figure(figsize=(8, 4.5))
    drawn = []
    for algorithm in ALGORITHMS:
        result = results[algorithm.name]
        theta = result.states[:, 2]
        if experiment.task.wrap_angle:
            theta = np.unwrap(theta)
        failures = np.flatnonzero(result.dones)
        end = failures[0] + 1 if len(failures) else len(theta)
        t = np.arange(end) * params.dt
        degrees = np.rad2deg(theta[:end])
        drawn.append(degrees)
        label = algorithm.name
        if end < len(theta):
            label += f" (terminated at {t[-1]:.2f} s)"
        line, = plt.plot(t, degrees, label=label)
        if end < len(theta):
            plt.plot(t[-1], degrees[-1], "x", color=line.get_color(), ms=9, mew=2)
    low = min(d.min() for d in drawn)
    high = max(d.max() for d in drawn)
    margin = 0.08 * max(high - low, 1.0)
    for level in range(-720, 721, 180):
        if low - margin <= level <= high + margin:
            style = dict(lw=0.8, ls="--") if level % 360 == 0 else dict(lw=0.5)
            plt.axhline(level, color="gray", **style)
    plt.ylim(low - margin, high + margin)
    plt.xlabel("time [s]")
    plt.ylabel("pole angle [deg]  (0 = upright)")
    plt.legend()
    save_figure(experiment.trajectory_filename)

if __name__ == "__main__":
    params = env.CartPoleParams()
    results = {}
    for experiment in EXPERIMENTS:
        task_results = {}
        for algorithm in ALGORITHMS:
            print(f"exp {experiment.name}, algo {algorithm.name}")
            task_results[algorithm.name] = execute(params, experiment, algorithm)
        results[experiment.name] = task_results
        plot_learning_comparison(experiment, task_results)
        if experiment.trajectory_filename:
            plot_trajectories(params, experiment, task_results)
    agent.region_of_attraction(params, EXPERIMENTS[0].task, lqr_gain(params))
    # swing-up課題の報酬設計・探索設計に関する予備実験
    swingup = EXPERIMENTS[1]
    naive_trials, zero_trials = plot_design_ablation(params, swingup, results["swingup"]["q-learning"].trials)
    naive_len = [values[-1] for _, values in trial_curves(naive_trials, "lengths")]
    naive_ret = [values[-1] for _, values in trial_curves(naive_trials)]
    zero_ret = [values[-1] for _, values in trial_curves(zero_trials)]
    print(f"[swingup/naive-reward] episode length (MA): {np.mean(naive_len):.1f} +/- {np.std(naive_len):.1f} steps, return {np.mean(naive_ret):.1f}")
    print(f"[swingup/zero-init]     final return (MA): {np.mean(zero_ret):.1f} +/- {np.std(zero_ret):.1f}")