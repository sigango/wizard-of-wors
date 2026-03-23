import faulthandler

import jaxatari
from jaxatari.wrappers import ObjectCentricWrapper
faulthandler.enable()

import jax
import jax.numpy as jnp
import numpy as np
from ocatari.core import OCAtari

try:
    from train_ocatari_agent import (
        train_ppo_with_ocatari, 
        normalize_observation_ocatari 
    )

    from ppo_agent import (
        create_ppo_train_state,
        ActorCritic,
        TrainState
    )

    from train_jaxatari_agent import (
        train_ppo_with_jaxatari,
        normalize_observation_jaxatari
    )

    from jaxatari.wrappers import AtariWrapper, FlattenObservationWrapper, PixelObsWrapper

except ImportError as e:
    print(f"Error importing modules: {e}")
    exit()

import matplotlib.pyplot as plt
import os
from datetime import datetime
import flax.serialization
import optax # For dummy optimizer if needed
import flax.core 
import pandas as pd
from typing import Dict, Any, Optional, Union, Tuple
import pygame
from tqdm import tqdm

# Constants for visualization
SCALING_FACTOR = 3
WIDTH = 160  # Standard Atari game width
HEIGHT = 210  # Standard Atari game height

# Check if GPU backend is available and set it as the default device.
# jax.devices("gpu") raises on CPU-only setups, so guard it safely.
try:
    gpu_devices = jax.devices("gpu")
except RuntimeError:
    gpu_devices = []

if gpu_devices:
    jax.config.update("jax_platform_name", "gpu")
    print(f"Using GPU for training ({len(gpu_devices)} device(s))")
else:
    print("No GPU backend found, using CPU")

# --- PPO Configuration Presets (Wizard of Wor report protocol) ---
base_wizard_config = {
    "ENV_NAME_OCATARI": "Pong",  # kept for OCAtari compatibility paths
    "ENV_NAME_JAXATARI": "wizardofwor",
    "ENV_TYPE": "jax",  # Can be "ocatari", "jax", or "jaxatari"
    "TOTAL_TIMESTEPS": 20_000_000,
    "TOTAL_TIMESTEPS_PER_EPOCH": 10_000,
    "LR": 3e-4,
    "NUM_ENVS": 64,
    "NUM_STEPS": 256,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "NUM_MINIBATCHES": 8,
    "UPDATE_EPOCHS": 3,
    "CLIP_EPS": 0.1,
    "CLIP_VF_EPS": 0.1,
    "ENT_COEF": 0.005,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "relu",
    "ANNEAL_LR": True,
    "SEED": 42,
    "BUFFER_WINDOW": 4,
    "FRAMESKIP": 4,
    "REPEAT_ACTION_PROBABILITY": 0.25,
    "JAX_OBS_MODE": "object",  # "object" or "pixel"
    "PIXEL_GRAYSCALE": True,
    "PIXEL_RESIZE": True,
    "PIXEL_RESIZE_SHAPE": (84, 84),
    "LOG_INTERVAL_UPDATES": 100,
    "VISUALIZE_AFTER_TRAINING": False,
    "VIZ_STEPS": 1000,
    "VIZ_FPS": 30,
    "SAVE_VIZ_VIDEO": True,
    "USE_WANDB": True,
    "WANDB_PROJECT": "jaxatari-wizardofwor-ppo",
    "WANDB_ENTITY": None,
    "WANDB_MODE": "online",
    "WANDB_RUN_NAME": None,
    "WANDB_GROUP": "wizardofwor",
    "WANDB_TAGS": ["ppo", "jaxatari", "wizardofwor"],
}

config_wizard_object_final = {
    **base_wizard_config,
    "TOTAL_TIMESTEPS": 200_000_000,
    "UPDATE_EPOCHS": 3,
    "JAX_OBS_MODE": "object",
    "LOG_INTERVAL_UPDATES": 100,
    "WANDB_GROUP": "wizardofwor_object_final",
    "WANDB_TAGS": ["ppo", "jaxatari", "wizardofwor", "object", "final"],
}

config_wizard_pixel_final = {
    **base_wizard_config,
    "TOTAL_TIMESTEPS": 100_000_000,
    "UPDATE_EPOCHS": 1,
    "JAX_OBS_MODE": "pixel",
    "NUM_ENVS": 128,
    "NUM_STEPS": 128,
    "NUM_MINIBATCHES": 32,
    "TEST_NUM_ENVS": 128,  # For compatibility with external notes; not used in this PPO pipeline.
    "LOG_INTERVAL_UPDATES": 100,
    "PIXEL_GRAYSCALE": True,
    "PIXEL_RESIZE": True,
    "PIXEL_RESIZE_SHAPE": (84, 84),
    "WANDB_GROUP": "wizardofwor_pixel_final",
    "WANDB_TAGS": ["ppo", "jaxatari", "wizardofwor", "pixel", "final"],
}

# Mac-safe final presets: keep report-required timesteps/epochs, reduce parallelism
# to lower sustained RAM/CPU pressure for long overnight runs.
config_wizard_object_final_macsafe = {
    **config_wizard_object_final,
    "NUM_ENVS": 32,
    "NUM_STEPS": 128,
    "NUM_MINIBATCHES": 8,
    "LOG_INTERVAL_UPDATES": 200,
    "WANDB_GROUP": "wizardofwor_object_final_macsafe",
    "WANDB_TAGS": ["ppo", "jaxatari", "wizardofwor", "object", "final", "macsafe"],
}

config_wizard_pixel_final_macsafe = {
    **config_wizard_pixel_final,
    "NUM_ENVS": 16,
    "NUM_STEPS": 128,
    "NUM_MINIBATCHES": 4,
    "TEST_NUM_ENVS": 16,  # compatibility field; not used in this PPO pipeline.
    "LOG_INTERVAL_UPDATES": 200,
    "WANDB_GROUP": "wizardofwor_pixel_final_macsafe",
    "WANDB_TAGS": ["ppo", "jaxatari", "wizardofwor", "pixel", "final", "macsafe"],
}

config_wizard_object_test = {
    **config_wizard_object_final,
    "TOTAL_TIMESTEPS": 20_000_000,
    "UPDATE_EPOCHS": 3,
    "LOG_INTERVAL_UPDATES": 10,  # 10x smaller than final to keep similar point density
    "WANDB_GROUP": "wizardofwor_object_test",
    "WANDB_TAGS": ["ppo", "jaxatari", "wizardofwor", "object", "test10pct"],
}

config_wizard_pixel_test = {
    **config_wizard_pixel_final,
    "TOTAL_TIMESTEPS": 10_000_000,
    "UPDATE_EPOCHS": 1,
    "TEST_NUM_ENVS": 128,  # For compatibility with external notes; not used in this PPO pipeline.
    "LOG_INTERVAL_UPDATES": 10,  # 10x smaller than final to keep similar point density
    "WANDB_GROUP": "wizardofwor_pixel_test",
    "WANDB_TAGS": ["ppo", "jaxatari", "wizardofwor", "pixel", "test10pct"],
}

CONFIG_PRESETS = {
    "config_wizard_object_final": config_wizard_object_final,
    "config_wizard_pixel_final": config_wizard_pixel_final,
    "config_wizard_object_final_macsafe": config_wizard_object_final_macsafe,
    "config_wizard_pixel_final_macsafe": config_wizard_pixel_final_macsafe,
    "config_wizard_object_test": config_wizard_object_test,
    "config_wizard_pixel_test": config_wizard_pixel_test,
    # Optional short aliases
    "wizard_object_final": config_wizard_object_final,
    "wizard_pixel_final": config_wizard_pixel_final,
    "wizard_object_final_macsafe": config_wizard_object_final_macsafe,
    "wizard_pixel_final_macsafe": config_wizard_pixel_final_macsafe,
    "wizard_object_test": config_wizard_object_test,
    "wizard_pixel_test": config_wizard_pixel_test,
}


def _is_jax_env_type(env_type: str) -> bool:
    return env_type in {"jax", "jaxatari"}


def _resolve_env_name(config_dict: Dict[str, Any], env_type: str) -> str:
    if _is_jax_env_type(env_type):
        return config_dict["ENV_NAME_JAXATARI"]
    return config_dict["ENV_NAME_OCATARI"]


def _build_wrapped_jax_env(config_dict: Dict[str, Any], env_name: str):
    env_base = jaxatari.make(env_name.lower())
    env = AtariWrapper(
        env_base,
        sticky_actions=True,
        frame_stack_size=config_dict["BUFFER_WINDOW"],
        frame_skip=config_dict["FRAMESKIP"],
    )
    obs_mode = config_dict.get("JAX_OBS_MODE", "object").lower()
    if obs_mode == "object":
        env = ObjectCentricWrapper(env)
        env = FlattenObservationWrapper(env)
    elif obs_mode == "pixel":
        env = PixelObsWrapper(
            env,
            do_pixel_resize=config_dict.get("PIXEL_RESIZE", True),
            pixel_resize_shape=tuple(config_dict.get("PIXEL_RESIZE_SHAPE", (84, 84))),
            grayscale=config_dict.get("PIXEL_GRAYSCALE", True),
        )
        env = FlattenObservationWrapper(env)
    else:
        raise ValueError(
            f"Unsupported JAX_OBS_MODE='{obs_mode}'. Use 'object' or 'pixel'."
        )
    return env_base, env

def train_ppo_agent_ocatari(config_dict: Dict[str, Any]) -> Tuple[TrainState, str, Dict[str, Any]]:
    env_type = "ocatari"
    env_name = config_dict["ENV_NAME_OCATARI"]
    
    print(f"Training PPO agent (Distrax base) with {env_type.upper()} environment (Game: {env_name})...")
    
    trained_ppo_state, training_metrics = train_ppo_with_ocatari(config_dict)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/ppo_distrax_{env_type}_{env_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Save model parameters
    model_params_path = os.path.join(results_dir, "ppo_distrax_model_params.npz")
    params_dict_to_save = flax.serialization.to_state_dict(trained_ppo_state.params)
    np.savez(model_params_path, **params_dict_to_save)
    print(f"PPO (Distrax) model parameters saved to {model_params_path}")
    
    # Save metrics as both npz and csv
    metrics_path_npz = os.path.join(results_dir, "training_metrics_ppo_distrax.npz")
    np.savez(metrics_path_npz, **training_metrics)
    print(f"Training metrics saved to {metrics_path_npz}")
    
    # Save metrics as CSV for easier comparison
    metrics_df = pd.DataFrame({
        'timesteps': training_metrics['timesteps'],
        'mean_rewards': training_metrics['mean_rewards'],
        'pg_losses': training_metrics['pg_losses'],
        'vf_losses': training_metrics['vf_losses'],
        'ent_losses': training_metrics['ent_losses']
    })
    metrics_path_csv = os.path.join(results_dir, "training_metrics_ppo_distrax.csv")
    metrics_df.to_csv(metrics_path_csv, index=False)
    print(f"Training metrics saved to {metrics_path_csv}")

    return trained_ppo_state, results_dir, training_metrics


def train_ppo_agent_jaxatari(config_dict: Dict[str, Any]) -> Tuple[TrainState, str, Dict[str, Any]]:
    env_type = config_dict.get("ENV_TYPE", "jax")
    if not _is_jax_env_type(env_type):
        env_type = "jax"
    env_name = config_dict["ENV_NAME_JAXATARI"]

    print(f"Training PPO agent (Distrax base) with {env_type.upper()} environment (Game: {env_name})...")
    
    trained_ppo_state, training_metrics = train_ppo_with_jaxatari(config_dict)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/ppo_distrax_{env_type}_{env_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Save model parameters
    model_params_path = os.path.join(results_dir, "ppo_distrax_model_params.npz")
    params_dict_to_save = flax.serialization.to_state_dict(trained_ppo_state.params)
    np.savez(model_params_path, **params_dict_to_save)
    print(f"PPO (Distrax) model parameters saved to {model_params_path}")
    
    # Save metrics as both npz and csv
    metrics_path_npz = os.path.join(results_dir, "training_metrics_ppo_distrax.npz")
    np.savez(metrics_path_npz, **training_metrics)
    print(f"Training metrics saved to {metrics_path_npz}")
    
    # Save metrics as CSV for easier comparison
    metrics_df = pd.DataFrame({
        'timesteps': training_metrics['timesteps'],
        'mean_rewards': training_metrics['mean_rewards'],
        'pg_losses': training_metrics['pg_losses'],
        'vf_losses': training_metrics['vf_losses'],
        'ent_losses': training_metrics['ent_losses']
    })
    metrics_path_csv = os.path.join(results_dir, "training_metrics_ppo_distrax.csv")
    metrics_df.to_csv(metrics_path_csv, index=False)
    print(f"Training metrics saved to {metrics_path_csv}")
    
    return trained_ppo_state, results_dir, training_metrics


def load_ppo_params_from_npz(model_path: str) -> flax.core.FrozenDict:
    loaded_np = np.load(model_path, allow_pickle=True)
    # Convert numpy arrays to JAX arrays while preserving the nested structure
    def convert_to_jax(x):
        if isinstance(x, dict):
            return {k: convert_to_jax(v) for k, v in x.items()}
        elif isinstance(x, np.ndarray):
            return jnp.array(x)
        return x
    
    # Load and convert the parameters
    loaded_params_dict = {key: convert_to_jax(loaded_np[key].item()) for key in loaded_np.files}
    return flax.core.freeze(loaded_params_dict)

def evaluate_ppo_agent(
    agent_representation: Union[str, TrainState], 
    config_dict: Dict[str, Any], 
    num_episodes: int = 10, 
    eval_seed: int = 123,
    eval_env_type: Optional[str] = None
) -> Tuple[float, float]:
    """
    Evaluate a PPO agent on either OCAtari or JAX environment.
    
    Args:
        agent_representation: Either path to saved params or TrainState object
        config_dict: Configuration dictionary
        num_episodes: Number of episodes to evaluate
        eval_seed: Random seed for evaluation
        eval_env_type: Override environment type for evaluation ("ocatari" or "jax")
    """
    env_type = eval_env_type if eval_env_type is not None else config_dict.get("ENV_TYPE", "jax")
    env_name = _resolve_env_name(config_dict, env_type)
    
    if config_dict.get("BUFFER_WINDOW", None) is None: # fix for old configs
        config_dict["BUFFER_WINDOW"] = config_dict["OCATARI_BUFFER_WINDOW"] 

    if not _is_jax_env_type(env_type):
        eval_env = OCAtari(
            env_name=env_name,
            mode="ram", 
            hud=False, 
            render_mode="rgb_array",
            obs_mode="obj", 
            buffer_window_size=config_dict["BUFFER_WINDOW"],
            frameskip=config_dict["FRAMESKIP"],
            repeat_action_probability=config_dict["REPEAT_ACTION_PROBABILITY"]
        )
    else:  # JAX environment
        _, eval_env = _build_wrapped_jax_env(config_dict, env_name)

    episode_rewards = []
    
    # Determine obs_shape_flat and action_dim for initializing network if loading params
    if not _is_jax_env_type(env_type):
        _obs_temp, _ = eval_env.reset(seed=eval_seed)
        obs_shape_flat_eval = (np.prod(_obs_temp.shape),)
        action_dim_eval = eval_env.action_space.n
    else:
        _obs_temp, _ = eval_env.reset(key=jax.random.PRNGKey(eval_seed))
        obs_shape_flat_eval = (np.prod(_obs_temp.shape),)
        action_dim_eval = eval_env.action_space().n

    if isinstance(agent_representation, str):
        print(f"Loading PPO (Distrax) model params from: {agent_representation}")
        loaded_params = load_ppo_params_from_npz(agent_representation)
        
        dummy_rng = jax.random.PRNGKey(0)
        eval_train_state = create_ppo_train_state(dummy_rng, config_dict, obs_shape_flat_eval, action_dim_eval)
        current_agent_state = eval_train_state.replace(params=loaded_params)
        print("PPO (Distrax) model params loaded for evaluation.")
    elif isinstance(agent_representation, TrainState):
        current_agent_state = agent_representation
        print("Using provided PPO TrainState for evaluation.")
    else:
        raise ValueError("agent_representation must be a path (str) or TrainState object.")

    eval_rng_key = jax.random.PRNGKey(eval_seed)

    for episode in tqdm(range(num_episodes), desc="Evaluating Episodes", unit="ep"):
        if not _is_jax_env_type(env_type):
            obs_stacked, _ = eval_env.reset(seed=eval_seed + episode)
            obs_norm_flat = normalize_observation_ocatari(obs_stacked).reshape(1, -1)
        else:
            obs, state = eval_env.reset(key=jax.random.PRNGKey(eval_seed + episode))
            obs_norm_flat = normalize_observation_jaxatari(obs, eval_env.observation_space()).reshape(1, -1)
        
        episode_reward = 0
        done = False
        step_count = 0

        while not done:
            pi_eval, _ = current_agent_state.apply_fn({'params': current_agent_state.params}, jnp.array(obs_norm_flat))
            action_agent_jnp = pi_eval.mode()
            action_agent = int(action_agent_jnp[0])

            if not _is_jax_env_type(env_type):
                next_obs_stacked, reward, terminated, truncated, _ = eval_env.step(action_agent)
                next_obs_norm_flat = normalize_observation_ocatari(next_obs_stacked).reshape(1, -1)
                done = terminated or truncated
            else:
                next_obs, state, reward, done, _ = eval_env.step(state, action_agent)
                next_obs_norm_flat = normalize_observation_jaxatari(next_obs, eval_env.observation_space()).reshape(1, -1)

            episode_reward += reward
            obs_norm_flat = next_obs_norm_flat
            step_count += 1
            if step_count > 20000:
                print(f"Warning: Eval episode {episode+1} exceeded max steps.")
                break
        
        episode_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")
    
    if not _is_jax_env_type(env_type):
        eval_env.close()
        
    mean_reward = np.mean(episode_rewards) if episode_rewards else 0
    std_reward = np.std(episode_rewards) if episode_rewards else 0
    return mean_reward, std_reward

def plot_training_metrics(metrics: Dict[str, Any], save_path: str, env_name: str) -> None:
    if 'timesteps' not in metrics or 'mean_rewards' not in metrics:
        print("Metrics for plotting mean rewards not found. Skipping reward plot.")
        return

    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training and Evaluation Rewards
    plt.subplot(2, 2, 1)
    plt.plot(metrics['timesteps'], metrics['mean_rewards'], label='Training Reward', color='green')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.title(f'PPO Training vs Evaluation Rewards - {env_name}')
    plt.legend()
    plt.grid(True)

    # Plot 2: Policy Loss
    if 'pg_losses' in metrics:
        plt.subplot(2, 2, 2)
        plt.plot(metrics['timesteps'], metrics['pg_losses'], label='Policy Loss', color='blue')
        plt.xlabel('Timesteps')
        plt.ylabel('Policy Loss')
        plt.title(f'PPO Policy Loss - {env_name}')
        plt.legend()
        plt.grid(True)

    # Plot 3: Value Loss
    if 'vf_losses' in metrics:
        plt.subplot(2, 2, 3)
        plt.plot(metrics['timesteps'], metrics['vf_losses'], label='Value Loss', color='red')
        plt.xlabel('Timesteps')
        plt.ylabel('Value Loss')
        plt.title(f'PPO Value Loss - {env_name}')
        plt.legend()
        plt.grid(True)

    # Plot 4: Entropy
    if 'ent_losses' in metrics:
        plt.subplot(2, 2, 4)
        plt.plot(metrics['timesteps'], metrics['ent_losses'], label='Entropy', color='purple')
        plt.xlabel('Timesteps')
        plt.ylabel('Entropy')
        plt.title(f'PPO Policy Entropy - {env_name}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training plots saved to {save_path}")
    plt.close()

def compare_agents(agent_paths: Dict[str, str], config_dict: Dict[str, Any], num_episodes: int = 100) -> None:
    """
    Compare multiple trained agents by evaluating them on both environments.
    
    Args:
        agent_paths: Dictionary mapping agent names to their parameter file paths
        config_dict: Configuration dictionary
        num_episodes: Number of episodes for evaluation
    """
    results = {}
    
    for agent_name, agent_path in agent_paths.items():
        print(f"\nEvaluating {agent_name}...")
        
        # Evaluate on OCAtari
        ocatari_mean, ocatari_std = evaluate_ppo_agent(
            agent_path, config_dict, num_episodes=num_episodes, eval_env_type="ocatari"
        )
        
        # Evaluate on JAX
        jax_mean, jax_std = evaluate_ppo_agent(
            agent_path, config_dict, num_episodes=num_episodes, eval_env_type="jax"
        )
        
        results[agent_name] = {
            "ocatari_mean": ocatari_mean,
            "ocatari_std": ocatari_std,
            "jax_mean": jax_mean,
            "jax_std": jax_std
        }
    
    # Print comparison table
    print("\nAgent Comparison Results:")
    print("-" * 80)
    print(f"{'Agent Name':<20} {'OCAtari Mean':<15} {'OCAtari Std':<15} {'JAX Mean':<15} {'JAX Std':<15}")
    print("-" * 80)
    
    for agent_name, metrics in results.items():
        print(f"{agent_name:<20} {metrics['ocatari_mean']:<15.2f} {metrics['ocatari_std']:<15.2f} "
              f"{metrics['jax_mean']:<15.2f} {metrics['jax_std']:<15.2f}")


def visualize_agent(agent_path: str, config_dict: Dict[str, Any], num_episodes: int = 10) -> None:
    """
    Visualize a trained agent playing a game.
    
    Args:
        agent_path: Path to the agent's parameter file
        config_dict: Configuration dictionary
        num_episodes: Number of episodes to visualize
    """
    env_type = config_dict.get("ENV_TYPE", "jax")
    env_name = _resolve_env_name(config_dict, env_type)
    
    if config_dict.get("BUFFER_WINDOW", None) is None: # fix for old configs
        config_dict["BUFFER_WINDOW"] = config_dict["OCATARI_BUFFER_WINDOW"] 
    
    # Initialize pygame for visualization
    pygame.init()
    pygame_screen = pygame.display.set_mode((WIDTH * SCALING_FACTOR, HEIGHT * SCALING_FACTOR))
    pygame.display.set_caption(f"Agent Visualization - {env_name}")
    clock = pygame.time.Clock()
    
    # Initialize video recording
    video_writer = None
    cv2 = None
    if config_dict.get("SAVE_VIZ_VIDEO", True):
        import cv2 as _cv2
        cv2 = _cv2
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"agent_visualization_{env_name}_{timestamp}.mp4"
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                video_filename,
                fourcc,
                config_dict.get("VIZ_FPS", 30),
                (WIDTH * SCALING_FACTOR, HEIGHT * SCALING_FACTOR),
            )
        except Exception as e:
            print(f"Could not initialize video writer: {e}. Video will not be saved.")
            video_writer = None
    
    # Load the agent
    loaded_params = load_ppo_params_from_npz(agent_path)
    
    # Initialize environment
    if not _is_jax_env_type(env_type):
        vis_env = OCAtari(
            env_name=env_name,
            mode="ram", 
            hud=False, 
            render_mode="rgb_array",
            obs_mode="obj", 
            buffer_window_size=config_dict["BUFFER_WINDOW"],
            frameskip=config_dict["FRAMESKIP"],
            repeat_action_probability=config_dict["REPEAT_ACTION_PROBABILITY"]
        )
        obs_shape_flat = (np.prod(vis_env.observation_space.shape),)
        action_dim = vis_env.action_space.n
    else:  # JAX environment
        vis_env_base, vis_env = _build_wrapped_jax_env(config_dict, env_name)
        obs_shape_flat = vis_env.reset(key=jax.random.PRNGKey(0))[0].shape
        action_dim = vis_env.action_space().n
    
    # Initialize agent
    dummy_rng = jax.random.PRNGKey(0)
    eval_train_state = create_ppo_train_state(dummy_rng, config_dict, obs_shape_flat, action_dim)
    current_agent_state = eval_train_state.replace(params=loaded_params)
    
    # Initialize visualization
    agent_key = jax.random.PRNGKey(config_dict["SEED"] + 777)
    
    if not _is_jax_env_type(env_type):
        obs_viz, _ = vis_env.reset(seed=config_dict["SEED"])
        print(obs_viz)
        obs_viz_norm_flat = normalize_observation_ocatari(obs_viz).reshape(1, -1)
        current_frame = vis_env.render()
    else:
        print("Resetting environment...")
        vis_reset_key, agent_key = jax.random.split(agent_key)
        obs_viz_raw, state_viz = vis_env.reset(key=vis_reset_key)
        obs_viz_norm_flat = normalize_observation_jaxatari(obs_viz_raw, vis_env.observation_space()).reshape(1, -1)
        print("Environment reset complete")
    
    total_reward_viz = 0
    running = True
    episode_count = 0
        
    while running and episode_count < num_episodes:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
        
        if not running:
            break
            
        # Get action from agent
        pi_viz, _ = current_agent_state.apply_fn({'params': current_agent_state.params}, obs_viz_norm_flat)
        action_viz = pi_viz.mode()  # Use mode for deterministic visualization
        
        # Step environment
        if not _is_jax_env_type(env_type):
            next_obs_viz, reward_viz, terminated, truncated, _ = vis_env.step(int(action_viz[0]))
            next_obs_viz_norm_flat = normalize_observation_ocatari(next_obs_viz).reshape(1, -1)
            done_viz = terminated or truncated
            current_frame = vis_env.render()
            # Convert frame to pygame surface and display
            # Transpose frame from (H, W, C) to (W, H, C) for pygame
            frame_transposed = np.transpose(current_frame, (1, 0, 2))
            frame_surface = pygame.Surface(frame_transposed.shape[:2])
            pygame.pixelcopy.array_to_surface(frame_surface, frame_transposed)
            frame_surface_scaled = pygame.transform.scale(frame_surface, (WIDTH * SCALING_FACTOR, HEIGHT * SCALING_FACTOR))
            pygame_screen.blit(frame_surface_scaled, (0, 0))
            pygame.display.flip()
            
            # Save frame to video
            if video_writer:
                view = pygame.surfarray.array3d(frame_surface_scaled)
                view = view.transpose([1, 0, 2])
                frame_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
        else:
            viz_step_key, agent_key = jax.random.split(agent_key)
            next_obs_viz_raw, state_viz, reward_viz, done_viz, _ = vis_env.step(
                state_viz, action_viz[0] if action_viz.ndim > 0 else action_viz
            )
            next_obs_viz_norm_flat = normalize_observation_jaxatari(next_obs_viz_raw, vis_env.observation_space()).reshape(1, -1)
            # Render the active JAXAtari game state
            raster = vis_env_base.render(state_viz.env_state)
            raster_np = np.asarray(raster)
            if raster_np.dtype != np.uint8:
                raster_np = np.clip(raster_np, 0, 255).astype(np.uint8)
            if raster_np.ndim == 3 and raster_np.shape[-1] == 4:
                raster_np = raster_np[..., :3]

            frame_transposed = np.transpose(raster_np, (1, 0, 2))
            frame_surface = pygame.Surface(frame_transposed.shape[:2])
            pygame.pixelcopy.array_to_surface(frame_surface, frame_transposed)
            frame_surface_scaled = pygame.transform.scale(
                frame_surface, (WIDTH * SCALING_FACTOR, HEIGHT * SCALING_FACTOR)
            )
            pygame_screen.blit(frame_surface_scaled, (0, 0))
            pygame.display.flip()
            
            # Save frame to video
            if video_writer:
                view = pygame.surfarray.array3d(frame_surface_scaled)
                view = view.transpose([1, 0, 2])
                frame_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
        
        total_reward_viz += reward_viz.item() if isinstance(reward_viz, jnp.ndarray) else reward_viz
        obs_viz_norm_flat = next_obs_viz_norm_flat
        
        if done_viz:
            print(f"Episode {episode_count + 1}/{num_episodes} finished with total reward {total_reward_viz:.2f}")
            episode_count += 1
            total_reward_viz = 0
            
            if not _is_jax_env_type(env_type):
                obs_viz, _ = vis_env.reset(seed=config_dict["SEED"] + episode_count)
                obs_viz_norm_flat = normalize_observation_ocatari(obs_viz).reshape(1, -1)
                current_frame = vis_env.render()
            else:
                vis_reset_key, agent_key = jax.random.split(agent_key)
                obs_viz_raw, state_viz = vis_env.reset(key=vis_reset_key)
                obs_viz_norm_flat = normalize_observation_jaxatari(obs_viz_raw, vis_env.observation_space()).reshape(1, -1)
    
    # Release video writer if it exists
    if video_writer:
        video_writer.release()
        print(f"\nVideo saved as: {os.path.abspath(video_filename)}")
    
    if not _is_jax_env_type(env_type):
        vis_env.close()
    pygame.quit()
    print("Visualization finished.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Script to compare PPO agents trained on either OCAtari or JAXAtari')
    parser.add_argument('--mode', type=str, choices=['train-ocatari', 'train-jaxatari', 'eval', 'visualize', 'compare'], required=True,
                      help='Mode: train-ocatari, train-jaxatari, eval, or compare agents')
    parser.add_argument('--preset', type=str, choices=sorted(CONFIG_PRESETS.keys()), default='config_wizard_object_test',
                      help='Configuration preset to load')
    parser.add_argument('--env_type', type=str, choices=['ocatari', 'jax', 'jaxatari'], default=None,
                      help='Optional override for environment type')
    parser.add_argument('--model_path', type=str, help='Path to model parameters for evaluation')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes for evaluation')
    parser.add_argument('--compare_paths', type=str, nargs='+', help='Paths to models for comparison')
    
    args = parser.parse_args()
    
    current_config = CONFIG_PRESETS[args.preset].copy()
    if args.env_type is not None:
        current_config["ENV_TYPE"] = args.env_type
    
    if args.mode == "train-ocatari":
        current_config["ENV_TYPE"] = "ocatari"
        print(f"--- Starting PPO (Distrax base) Agent Training ({current_config['ENV_TYPE']}) ---")
        print(f"Configuration: {current_config}")
        
        trained_ppo_state_obj, ppo_results_dir, training_metrics = train_ppo_agent_ocatari(current_config)
        
        plots_path = os.path.join(ppo_results_dir, "training_plots_ppo_distrax.png")
        plot_training_metrics(training_metrics, plots_path, current_config['ENV_NAME_OCATARI'])
        
        print("\n--- Evaluating trained agent ---")
        mean_reward, std_reward = evaluate_ppo_agent(
            trained_ppo_state_obj, current_config, num_episodes=args.num_episodes
        )
        print(f"Trained Agent - Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
    elif args.mode == "train-jaxatari":
        if not _is_jax_env_type(current_config.get("ENV_TYPE", "jax")):
            current_config["ENV_TYPE"] = "jax"
        print(f"--- Starting PPO (Distrax base) Agent Training ({current_config['ENV_TYPE']}) ---")
        print(f"Configuration: {current_config}")
        
        trained_ppo_state_obj, ppo_results_dir, training_metrics = train_ppo_agent_jaxatari(current_config)

        plots_path = os.path.join(ppo_results_dir, "training_plots_ppo_distrax.png")
        plot_training_metrics(training_metrics, plots_path, current_config['ENV_NAME_JAXATARI'])
        
        print("\n--- Evaluating trained agent ---")
        mean_reward, std_reward = evaluate_ppo_agent(
            trained_ppo_state_obj, current_config, num_episodes=args.num_episodes
        )
        print(f"Trained Agent - Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
    elif args.mode == "eval":
        if not args.model_path:
            raise ValueError("Model path must be provided for evaluation mode")
            
        print(f"--- Evaluating agent from {args.model_path} ---")
        mean_reward, std_reward = evaluate_ppo_agent(
            args.model_path, current_config, num_episodes=args.num_episodes
        )
        print(f"Agent Evaluation - Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
    elif args.mode == "compare":
        if not args.compare_paths:
            if args.model_path: 
                args.compare_paths = [args.model_path]
            else:
                raise ValueError("Model paths must be provided for comparison mode")
            
        agent_paths = {f"agent_{i}": path for i, path in enumerate(args.compare_paths)}
        compare_agents(agent_paths, current_config, num_episodes=args.num_episodes)

    elif args.mode == "visualize":
        if not args.model_path:
            raise ValueError("Model path must be provided for visualization mode")
            
        visualize_agent(args.model_path, current_config, num_episodes=args.num_episodes)

if __name__ == "__main__":
    main()
