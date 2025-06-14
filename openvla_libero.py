import os
import sys
import argparse
import time
from typing import List, Dict, Any, Optional, Tuple

import torch
import numpy as np
from PIL import Image
import imageio

# Add LIBERO to the Python path
sys.path.append('/files1/Yuxuan_Li/LIBERO')

# Import OpenVLA components
from transformers import AutoModelForVision2Seq, AutoProcessor

# Import LIBERO components
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils import get_libero_path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OpenVLA interaction with LIBERO")
    
    # Environment options
    parser.add_argument("--task-suite", type=str, default="libero_10",
                        help="Task suite to use (libero_10, libero_spatial, libero_object, libero_goal)")
    parser.add_argument("--task-id", type=int, default=0,
                        help="Task ID within the suite")
    parser.add_argument("--max-steps", type=int, default=50,
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
    parser.add_argument("--video-name", type=str, default="vla_libero_interaction.mp4",
                        help="Name of the video file")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print verbose output")
    
    return parser.parse_args()


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


def initialize_libero_env(task_suite_name: str, task_id: int):
    """Initialize the LIBERO environment."""
    print(f"Initializing LIBERO environment: {task_suite_name}, Task ID: {task_id}")
    
    # Get benchmark dictionary and task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    
    # Get specific task
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_description = task.language
    
    # Get BDDL file path
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    
    # Initialize environment with camera settings
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 512,
        "camera_widths": 512
    }
    env = OffScreenRenderEnv(**env_args)
    
    # Initialize environment state
    env.seed(0)
    obs = env.reset()
    
    # Get initial states
    init_states = task_suite.get_task_init_states(task_id)
    env.set_init_state(init_states[0])
    
    print(f"Task Description: {task_description}")
    
    return env, obs, task_description


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


def run_vla_libero_interaction(args):
    """Run the VLA interaction with LIBERO environment."""
    # Ensure output directory exists
    if args.save_video:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load VLA model and processor
    vla_model, processor = load_vla_model(
        args.model, 
        args.device, 
        args.use_flash_attn
    )
    
    # Initialize LIBERO environment
    env, obs, instruction = initialize_libero_env(args.task_suite, args.task_id)
    
    # Main interaction loop
    frames = []
    done = False
    step_count = 0
    
    try:
        print("Starting interaction loop...")
        while not done and step_count < args.max_steps:
            step_count += 1
            if args.verbose:
                print(f"Step {step_count}/{args.max_steps}")
            
            # Get image from observation
            if "agentview_rgb" in obs:
                image_array = obs["agentview_rgb"]
            elif "agentview_image" in obs:
                image_array = obs["agentview_image"]
            else:
                # Fallback to render if observation doesn't contain the image
                image_array = env.render(mode='rgb_array')
            
            # Convert image to uint8 if it's float
            if image_array.dtype == np.float32 or image_array.dtype == np.float64:
                image_array = (image_array * 255).astype(np.uint8)
            
            # Store frame
            if args.save_video:
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
            obs, reward, done, info = env.step(action)
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    except Exception as e:
        print(f"Error during interaction: {e}")
        print(f"Available observation keys: {obs.keys() if obs is not None else 'None'}")
    
    finally:
        # Save the video
        if args.save_video and frames:
            video_path = os.path.join(args.output_dir, args.video_name)
            print(f"Saving video to {video_path}...")
            try:
                # Save as GIF first (more reliable)
                gif_path = video_path.replace('.mp4', '.gif')
                imageio.mimsave(gif_path, frames, fps=10)
                print(f"GIF saved to: {os.path.abspath(gif_path)}")
                
                # Try to save as MP4 as well
                try:
                    imageio.mimsave(video_path, frames, fps=10)
                    print(f"MP4 saved to: {os.path.abspath(video_path)}")
                except Exception as e:
                    print(f"Could not save MP4, but GIF was saved successfully. Error: {e}")
            except Exception as e:
                print(f"Error saving video: {e}")
        
        # Close the environment
        env.close()
        print("Environment closed")


def main():
    """Main function to run the VLA interaction with LIBERO."""
    args = parse_arguments()
    
    print(f"Selected task suite: {args.task_suite}")
    print(f"Selected task ID: {args.task_id}")
    
    # Run the interaction
    run_vla_libero_interaction(args)


if __name__ == "__main__":
    main() 