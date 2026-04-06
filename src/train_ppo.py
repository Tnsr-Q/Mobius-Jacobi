"""
SB3 PPO Training Integration Script for the CJPT / Aletheia Framework.

Usage
-----
    python train_ppo.py

Runs a PPO agent inside AletheiaEnv for 500 000 timesteps, logging phase
diagnostics every 100 steps via CJPTMonitorCallback.  The final policy is
saved to ``aletheia_ppo_final.zip``.

TensorBoard logs are written to ``./aletheia_logs/``; inspect them with::

    python -m tensorboard --logdir ./aletheia_logs/

Deployment notes
----------------
* Replace ``scipy.integrate`` with ``jax.experimental.odeint`` if GPU
  acceleration is required for large batch rollouts.
* Swap the placeholder ``Omega`` and ``delta_kk`` in ``AletheiaEnv.step``
  with ``CausalEnforcer.compute_causal_deviation`` and the Magnus symplectic
  integrator output.
* After ~100 k steps ``g_trap`` should stabilise near 1.0, ``sigma_env``
  should converge to σ_crit, and ``phase`` should lock to
  ``BOUND_RECONSTRUCTION``.
"""

import sys
import os

# Allow imports from the src directory when invoked from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from cjpt_system import CJPTSystem
from jacobi_ode_solver import JacobiODESolver
from aletheia_env import AletheiaEnv


class CJPTMonitorCallback(BaseCallback):
    """Logs CJPT diagnostics every 100 environment steps."""

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            infos = self.locals.get("infos", [{}])
            info = infos[-1] if infos else {}
            if info:
                print(
                    f"Step {self.num_timesteps} | "
                    f"Phase: {info.get('phase', 'N/A')} | "
                    f"G_trap: {info.get('g_trap', 0):.3f} | "
                    f"σ: {info.get('sigma_env', 0):.2e}"
                )
        return True


def build_env(f2: float = 1e-8, xi_H: float = 5e8, H0: float = 1e9,
              max_steps: int = 1000) -> AletheiaEnv:
    """Instantiate and return a fully wired AletheiaEnv."""
    cjpt = CJPTSystem({"f2": f2, "xi_H": xi_H, "M_Pl": 2.435e18})
    ode = JacobiODESolver(f2=f2, xi_H=xi_H, H0=H0)
    return AletheiaEnv(cjpt_system=cjpt, ode_solver=ode, max_steps=max_steps)


def train(total_timesteps: int = 500_000,
          log_dir: str = "./aletheia_logs/",
          save_path: str = "aletheia_ppo_final") -> PPO:
    """
    Train a PPO agent on AletheiaEnv.

    Parameters
    ----------
    total_timesteps : int
        Total environment steps to train for.
    log_dir : str
        TensorBoard log directory.
    save_path : str
        Path (without extension) to save the final model.

    Returns
    -------
    PPO
        Trained model.
    """
    env = build_env()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=log_dir,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=CJPTMonitorCallback(),
    )
    model.save(save_path)
    print(f"\nModel saved to {save_path}.zip")
    return model


if __name__ == "__main__":
    train()
