"""
Visualization tools for CJPT system.

Provides comprehensive plotting for:
- Phase transition diagrams
- Time evolution of order parameters
- PPO state trajectories
- Diagnostic dashboards
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class CJPTVisualizer:
    """Comprehensive visualization for CJPT system."""
    
    def __init__(self, output_dir="/app/outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_phase_diagram(self, cjpt_system, figsize=(12, 8)):
        """
        Plot phase transition diagram from CJPT history.
        """
        if not cjpt_system.phase_history:
            logger.warning("No phase history available.")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        steps = np.arange(len(cjpt_system.phase_history))
        
        # Phase evolution
        phase_map = {
            'MINIMAL_PHASE': 0,
            'TRANSITION': 1,
            'BOUND_RECONSTRUCTION': 2,
            'DUAL_EMERGENCE': 3
        }
        phase_numeric = [phase_map.get(p, 1) for p in cjpt_system.phase_history]
        
        axes[0].plot(steps, phase_numeric, 'o-', markersize=3)
        axes[0].set_ylabel('Phase', fontsize=12)
        axes[0].set_yticks([0, 1, 2, 3])
        axes[0].set_yticklabels(['Minimal', 'Transition', 'Bound Recon', 'Dual Emerg'])
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('CJPT Phase Evolution', fontsize=14)
        
        # Causal deviation
        if cjpt_system.delta_kk_history:
            axes[1].semilogy(steps, cjpt_system.delta_kk_history, 'b-', label='Δ_KK')
            axes[1].set_ylabel('Δ_KK', fontsize=12)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # Trap door and envelope
        if cjpt_system.g_trap_history and cjpt_system.sigma_env_history:
            ax2 = axes[2]
            ax2.plot(steps, cjpt_system.g_trap_history, 'r-', label='G_trap')
            ax2.axhline(1.0, color='orange', linestyle='--', alpha=0.5, label='Phase boundary')
            ax2.axhline(1.5, color='red', linestyle='--', alpha=0.5)
            ax2.set_ylabel('G_trap', fontsize=12)
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            ax2_twin = ax2.twinx()
            ax2_twin.semilogy(steps, cjpt_system.sigma_env_history, 'g-', label='σ_env', alpha=0.7)
            ax2_twin.set_ylabel('σ_env', fontsize=12)
            ax2_twin.legend(loc='upper right')
        
        axes[2].set_xlabel('Step', fontsize=12)
        
        plt.tight_layout()
        output_path = self.output_dir / 'phase_diagram.png'
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved phase diagram to {output_path}")
        plt.close()
    
    def plot_diagnostic_dashboard(self, cjpt_system, tensor_cell, 
                                 causal_enforcer, figsize=(16, 12)):
        """
        Create comprehensive diagnostic dashboard.
        
        Shows all key metrics in a single view:
        - Phase trajectory
        - Causal deviation evolution
        - Trap door status
        - Geometric projection status
        - Fitness evolution
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Top row: Phase and deviation
        ax1 = fig.add_subplot(gs[0, :])
        if cjpt_system.phase_history:
            phase_map = {'MINIMAL_PHASE': 0, 'TRANSITION': 1, 
                        'BOUND_RECONSTRUCTION': 2, 'DUAL_EMERGENCE': 3}
            phase_numeric = [phase_map.get(p, 1) for p in cjpt_system.phase_history]
            steps = np.arange(len(phase_numeric))
            ax1.fill_between(steps, 0, phase_numeric, alpha=0.3, step='mid')
            ax1.plot(steps, phase_numeric, 'k-', linewidth=2)
            ax1.set_ylabel('Phase', fontsize=11)
            ax1.set_yticks([0, 1, 2, 3])
            ax1.set_yticklabels(['Min', 'Trans', 'Bound', 'Dual'], fontsize=9)
            ax1.set_title('Phase Trajectory', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        # Middle left: Causal deviation
        ax2 = fig.add_subplot(gs[1, 0])
        if cjpt_system.delta_kk_history:
            ax2.semilogy(cjpt_system.delta_kk_history, 'b-', linewidth=1.5)
            ax2.set_ylabel('Δ_KK', fontsize=11)
            ax2.set_xlabel('Step', fontsize=11)
            ax2.set_title('Causal Deviation', fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        # Middle center: Trap door
        ax3 = fig.add_subplot(gs[1, 1])
        if cjpt_system.g_trap_history:
            ax3.plot(cjpt_system.g_trap_history, 'r-', linewidth=1.5)
            ax3.axhline(1.0, color='orange', linestyle='--', alpha=0.5, linewidth=1)
            ax3.axhline(1.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax3.set_ylabel('G_trap', fontsize=11)
            ax3.set_xlabel('Step', fontsize=11)
            ax3.set_title('Trap Door Score', fontsize=12)
            ax3.grid(True, alpha=0.3)
        
        # Middle right: Envelope width
        ax4 = fig.add_subplot(gs[1, 2])
        if cjpt_system.sigma_env_history:
            ax4.semilogy(cjpt_system.sigma_env_history, 'g-', linewidth=1.5)
            ax4.set_ylabel('σ_env', fontsize=11)
            ax4.set_xlabel('Step', fontsize=11)
            ax4.set_title('Gaussian Envelope Width', fontsize=12)
            ax4.grid(True, alpha=0.3)
        
        # Bottom left: TensorCell fitness
        ax5 = fig.add_subplot(gs[2, 0])
        if hasattr(tensor_cell, 'step') and tensor_cell.step > 0:
            # Placeholder - would need fitness history
            ax5.text(0.5, 0.5, f'TensorCell\nFitness: {tensor_cell.fitness:.3f}\nStep: {tensor_cell.step}',
                    ha='center', va='center', fontsize=11, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax5.set_title('TensorCell Status', fontsize=12)
            ax5.axis('off')
        
        # Bottom center: Causal projection
        ax6 = fig.add_subplot(gs[2, 1])
        status_text = f"Causal Projection:\n{'ACTIVE' if tensor_cell.causal_projection_active else 'INACTIVE'}"
        if tensor_cell.causal_projection_matrix is not None:
            status_text += f"\nMatrix shape: {tensor_cell.causal_projection_matrix.shape}"
        ax6.text(0.5, 0.5, status_text, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax6.set_title('Projection Status', fontsize=12)
        ax6.axis('off')
        
        # Bottom right: Causal enforcer
        ax7 = fig.add_subplot(gs[2, 2])
        enforcer_text = f"CausalEnforcer\nTolerance: {causal_enforcer.kk_tolerance:.2f}\nHistory: {len(causal_enforcer.causal_deviation_history)} pts"
        ax7.text(0.5, 0.5, enforcer_text, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax7.set_title('Enforcer Status', fontsize=12)
        ax7.axis('off')
        
        fig.suptitle('CJPT Diagnostic Dashboard', fontsize=16, fontweight='bold')
        
        output_path = self.output_dir / 'diagnostic_dashboard.png'
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved diagnostic dashboard to {output_path}")
        plt.close()
    
    def plot_ppo_state_trajectory(self, state_history: List[np.ndarray], 
                                 labels: Optional[List[str]] = None,
                                 figsize=(14, 10)):
        """
        Plot PPO state vector trajectory.
        
        Shows evolution of all 15 state dimensions:
        - Kinematics (6)
        - Transport (3)
        - Symplectic (3)
        - Causal (3)
        """
        if not state_history:
            logger.warning("No state history provided.")
            return
        
        state_array = np.array(state_history)  # Shape: (n_steps, 15)
        n_steps, n_dims = state_array.shape
        
        if labels is None:
            labels = [
                'φ_H', 'φ_S', 'φ̇_H', 'φ̇_S', 'H', 'ε',
                'λ_min', 'λ_max', 'Tr(M)',
                'log|det Ω|', '||Ω-Ω^{-T}||', 'κ(Ω)',
                'Δ_KK', 'σ_env', 'J_bound'
            ]
        
        fig, axes = plt.subplots(5, 3, figsize=figsize, sharex=True)
        axes = axes.flatten()
        
        for i in range(min(n_dims, 15)):
            axes[i].plot(state_array[:, i], linewidth=1.5)
            axes[i].set_ylabel(labels[i], fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(labelsize=9)
        
        for i in range(n_dims, 15):
            axes[i].axis('off')
        
        axes[-2].set_xlabel('Step', fontsize=11)
        axes[-1].set_xlabel('Step', fontsize=11)
        
        fig.suptitle('PPO State Vector Trajectory (Normalized)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'ppo_state_trajectory.png'
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved PPO state trajectory to {output_path}")
        plt.close()
