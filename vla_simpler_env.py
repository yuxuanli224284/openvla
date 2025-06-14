import os
import sys
import argparse
import time
from typing import List, Dict, Any, Optional, Tuple

import torch
import numpy as np
from PIL import Image
import mediapy
import sapien.core as sapien

# Add SimplerEnv to the Python path
sys.path.append('/files1/Yuxuan_Li/SimplerEnv')
# Import OpenVLA components
from transformers import AutoModelForVision2Seq, AutoProcessor

# Import SimplerEnv components
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict




def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OpenVLA interaction with SimplerEnv")
    
    # Environment options
    parser.add_argument("--task", type=str, default="google_robot_place_in_closed_drawer",
                        help="Task to run in SimplerEnv")
    parser.add_argument("--max-steps", type=int, default=10,
                        help="Maximum number of steps to run")
    
    # Model options
    parser.add_argument("--model", type=str, default="openvla/openvla-7b",
                        help="OpenVLA model to use")
    parser.add_argument("--device", type=str, default="cuda:0", 
                        help="Device to run the model on")
    parser.add_argument("--use-flash-attn", action="store_true", default=True,
                        help="Use flash attention if available")
    
    # Output options
    parser.add_argument("--save-video", action="store_true", default=True,
                        help="Save video of the interaction")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--video-name", type=str, default="vla_interaction.mp4",
                        help="Name of the video file")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print verbose output")
    
    return parser.parse_args()


def list_available_tasks():
    """List available tasks in SimplerEnv."""
    tasks = [
        "google_robot_pick_coke_can",
        "google_robot_move_near",
        "google_robot_open_drawer",
        "google_robot_close_drawer",
        "widowx_spoon_on_towel",
        "widowx_carrot_on_plate", 
        "widowx_stack_cube",
        "widowx_put_eggplant_in_basket"
    ]
    return tasks


def load_vla_model(model_name: str, device: str, use_flash_attn: bool = True):
    """Load the OpenVLA model and processor."""
    print(f"Loading OpenVLA model: {model_name} on {device}")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model with appropriate settings
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True
    }
    
    # Add flash attention if requested and available
    if use_flash_attn:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            model = AutoModelForVision2Seq.from_pretrained(model_name, **model_kwargs).to(device)
            print("Model loaded with flash attention")
        except Exception as e:
            print(f"Flash attention not available, using default attention: {e}")
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForVision2Seq.from_pretrained(model_name, **model_kwargs).to(device)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_name, **model_kwargs).to(device)
    
    return model, processor


def initialize_env(task_name: str):
    """Initialize the SimplerEnv environment."""
    print(f"Initializing environment: {task_name}")
    
    # Create and initialize the environment
    env = simpler_env.make(task_name)
    
    # Disable denoiser for compatibility
    sapien.render_config.rt_use_denoiser = False
    
    # Reset the environment
    obs, reset_info = env.reset()
    
    # Get the language instruction
    instruction = env.get_language_instruction()
    
    print(f"Instruction: {instruction}")
    
    return env, obs, instruction


def get_action_from_vla(
    vla_model, 
    processor, 
    image: np.ndarray, 
    instruction: str, 
    device: str
) -> np.ndarray:
    """Get action prediction from OpenVLA model."""
    # Convert numpy array to PIL image for OpenVLA
    pil_image = Image.fromarray(image)
    
    # Format prompt with instruction
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    
    # Get action prediction from OpenVLA
    inputs = processor(prompt, pil_image).to(device, dtype=torch.bfloat16)
    
    with torch.no_grad():
        action = vla_model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    
    # Convert to numpy if needed
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()
    
    # Ensure action matches the expected format (7-DoF)
    if len(action) > 7:
        action = action[:7]  # Truncate to expected size if needed
    
    return action


def run_vla_interaction(args):
    """Run the VLA interaction with SimplerEnv."""
    # Ensure output directory exists
    if args.save_video:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load VLA model and processor
    vla_model, processor = load_vla_model(
        args.model, 
        args.device, 
        args.use_flash_attn
    )
    
    # Initialize environment
    env, obs, instruction = initialize_env(args.task)
    
    # Main interaction loop
    frames = []
    done, truncated = False, False
    step_count = 0
    
    try:
        print("Starting interaction loop...")
        while not (done or truncated) and step_count < args.max_steps:
            step_count += 1
            if args.verbose:
                print(f"Step {step_count}/{args.max_steps}")
            
            # Get image from observation
            image_array = get_image_from_maniskill2_obs_dict(env, obs)
            frames.append(image_array)
            
            # Get action prediction from OpenVLA
            action = get_action_from_vla(
                vla_model, 
                processor, 
                image_array, 
                instruction, 
                args.device
            )
            
            if args.verbose:
                print(f"Action: {action}")
            
            # Step the environment with the predicted action
            obs, reward, done, truncated, info = env.step(action)
            
            # Check for new instruction (for multi-step tasks)
            new_instruction = env.get_language_instruction()
            if new_instruction != instruction:
                instruction = new_instruction
                print(f"New Instruction: {instruction}")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    except Exception as e:
        print(f"Error during interaction: {e}")
    
    finally:
        # Get episode statistics
        episode_stats = info.get('episode_stats', {})
        print(f"Episode stats: {episode_stats}")
        
        # Save the video of the interaction
        if args.save_video and frames:
            video_path = os.path.join(args.output_dir, args.video_name)
            print(f"Saving video to {video_path}...")
            mediapy.write_video(video_path, frames, fps=10)
            print(f"Video saved: {os.path.abspath(video_path)}")
        
        # Close the environment
        env.close()
        print("Environment closed")
    
    return episode_stats


def main():
    """Main function to run the VLA interaction."""
    args = parse_arguments()
    
    print("Available tasks:", ", ".join(list_available_tasks()))
    print(f"Selected task: {args.task}")
    
    # Run the interaction
    run_vla_interaction(args)


if __name__ == "__main__":
    main() 