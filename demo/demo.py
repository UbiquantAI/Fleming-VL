"""
Fleming-VL-8B Multi-Modal Inference Script

This script demonstrates three inference modes:
1. Single image inference
2. Video inference (frame-by-frame)
3. 3D medical image (CT/MRI) inference from .npy files

Model: UbiquantAI/Fleming-VL-8B
Based on: InternVL_chat-1.2 template
"""

from transformers import AutoTokenizer, AutoModel
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
from PIL import Image
import torchvision.transforms as T
import numpy as np
import torch
import os


# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = "UbiquantAI/Fleming-VL-8B"

# Prompt template for reasoning-based responses
REASONING_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user a concise "
    "final answer in a short word or phrase. The reasoning process and "
    "answer are enclosed within <think> </think> and <answer> </answer> "
    "tags, respectively, i.e., <think> reasoning process here </think>"
    "<answer> answer here </answer>"
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================================
# Image Preprocessing Functions
# ============================================================================

def build_transform(input_size):
    """Build image transformation pipeline."""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target ratios."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    Dynamically preprocess image by splitting into tiles based on aspect ratio.
    
    Args:
        image: PIL Image
        min_num: Minimum number of tiles
        max_num: Maximum number of tiles
        image_size: Size of each tile
        use_thumbnail: Whether to add a thumbnail image
    
    Returns:
        List of preprocessed PIL Images
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate possible tile configurations
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) 
        for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize and split the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    assert len(processed_images) == blocks
    
    # Add thumbnail if requested
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images


# ============================================================================
# Utility Functions
# ============================================================================

def load_model(model_path, use_flash_attn=True):
    """
    Load the vision-language model and tokenizer.
    
    Args:
        model_path: Path to the pretrained model
        use_flash_attn: Whether to use flash attention (default: True)
    
    Returns:
        tuple: (model, tokenizer)
    """
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=use_flash_attn,
        trust_remote_code=True
    ).eval().cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    return model, tokenizer


# ============================================================================
# Image Inference
# ============================================================================

def inference_single_image(model, tokenizer, image_path, question, 
                          prompt=REASONING_PROMPT, input_size=448, max_num=12):
    """
    Perform inference on a single image.
    
    Args:
        model: Loaded vision-language model
        tokenizer: Loaded tokenizer
        image_path: Path to the input image
        question: Question to ask about the image
        prompt: System prompt template
        input_size: Input image size (default: 448)
        max_num: Maximum number of tiles (default: 12)
    
    Returns:
        str: Model response
    """
    # Load and preprocess image using InternVL's dynamic preprocessing
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()
    
    # Prepare question with prompt and image token
    full_question = f"{prompt}\n<image>\n{question}"
    # print("###",full_question)
    
    # Generate response
    generation_config = dict(max_new_tokens=2048, do_sample=False)
    response = model.chat(tokenizer, pixel_values, full_question, generation_config)
    
    return response


# ============================================================================
# Video Inference
# ============================================================================

def get_frame_indices(bound, fps, max_frame, first_idx=0, num_segments=32):
    """
    Calculate evenly distributed frame indices for video sampling.
    
    Args:
        bound: Tuple of (start_time, end_time) in seconds, or None for full video
        fps: Frames per second of the video
        max_frame: Maximum frame index
        first_idx: First frame index to consider
        num_segments: Number of frames to sample
    
    Returns:
        np.array: Array of frame indices
    """
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """
    Load and preprocess video frames.
    
    Args:
        video_path: Path to the video file
        bound: Time boundary tuple (start, end) in seconds
        input_size: Input image size (default: 448)
        max_num: Maximum number of tiles per frame (default: 1)
        num_segments: Number of frames to extract
    
    Returns:
        tuple: (pixel_values tensor, list of num_patches per frame)
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    
    pixel_values_list = []
    num_patches_list = []
    transform = build_transform(input_size=input_size)
    
    frame_indices = get_frame_indices(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    
    for frame_index in frame_indices:
        # Extract and preprocess frame
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def inference_video(model, tokenizer, video_path, video_duration, question, 
                   prompt=REASONING_PROMPT, input_size=448, max_num=1):
    """
    Perform inference on a video by sampling frames.
    
    Args:
        model: Loaded vision-language model
        tokenizer: Loaded tokenizer
        video_path: Path to the video file
        video_duration: Duration of video in seconds
        question: Question to ask about the video
        prompt: System prompt template
        input_size: Input image size (default: 448)
        max_num: Maximum number of tiles per frame (default: 1)
    
    Returns:
        str: Model response
    """
    # Sample frames from video (1 frame per second)
    num_segments = int(video_duration)
    pixel_values, num_patches_list = load_video(
        video_path, bound=None, input_size=input_size, 
        max_num=max_num, num_segments=num_segments
    )
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    
    # Create image token prefix for all frames
    video_prefix = ''.join([f'<image>\n' for _ in range(len(num_patches_list))])
    
    # Prepare question with prompt and image tokens
    full_question = f"{prompt}\n{video_prefix}{question}"
    
    # Generate response
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    response, history = model.chat(
        tokenizer,
        pixel_values,
        full_question,
        generation_config,
        num_patches_list=num_patches_list,
        history=None,
        return_history=True
    )
    
    return response


# ============================================================================
# 3D Medical Image (NPY) Inference
# ============================================================================

def normalize_image(image):
    """
    Normalize image array to 0-255 range.
    
    Args:
        image: NumPy array of image data
    
    Returns:
        np.array: Normalized image as uint8
    """
    img_min = np.min(image)
    img_max = np.max(image)
    
    if img_max - img_min == 0:
        return np.zeros_like(image, dtype=np.uint8)
    
    return ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)


def convert_npy_to_images(npy_path, input_size=448, max_num=1, num_slices=11):
    """
    Convert 3D medical image (.npy) to multiple 2D RGB images.
    
    Expected input shape: (32, 256, 256) or (1, 32, 256, 256)
    Extracts evenly distributed slices and converts to RGB format.
    
    Args:
        npy_path: Path to the .npy file
        input_size: Input image size (default: 448)
        max_num: Maximum number of tiles per slice (default: 1)
        num_slices: Number of slices to extract (default: 11)
    
    Returns:
        tuple: (pixel_values tensor, list of num_patches per slice) or False if error
    """
    try:
        # Load .npy file
        data = np.load(npy_path)
        
        # Handle shape (1, 32, 256, 256) -> (32, 256, 256)
        if data.ndim == 4 and data.shape[0] == 1:
            data = data[0]
        
        # Validate shape
        if data.shape != (32, 256, 256):
            print(f"Warning: {npy_path} has shape {data.shape}, expected (32, 256, 256), skipping")
            return False
        
        # Select evenly distributed slices from 32 slices
        indices = np.linspace(0, 31, num_slices, dtype=int)
        
        transform = build_transform(input_size=input_size)
        pixel_values_list = []
        num_patches_list = []
        
        # Process each selected slice
        for idx in indices:
            # Get slice
            slice_img = data[idx]
            
            # Normalize to 0-255
            normalized = normalize_image(slice_img)
            
            # Convert grayscale to RGB by stacking
            rgb_img = np.stack([normalized, normalized, normalized], axis=-1)
            
            # Convert to PIL Image
            img = Image.fromarray(rgb_img)
            
            # Preprocess with InternVL's dynamic preprocessing
            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list
    
    except Exception as e:
        print(f"Error processing {npy_path}: {str(e)}")
        return False


def inference_3d_medical_image(model, tokenizer, npy_path, question, 
                              prompt=REASONING_PROMPT, input_size=448, max_num=1):
    """
    Perform inference on 3D medical images stored as .npy files.
    
    Args:
        model: Loaded vision-language model
        tokenizer: Loaded tokenizer
        npy_path: Path to the .npy file (shape: 32x256x256)
        question: Question to ask about the image
        prompt: System prompt template
        input_size: Input image size (default: 448)
        max_num: Maximum number of tiles per slice (default: 1)
    
    Returns:
        str: Model response or None if error
    """
    # Convert 3D volume to multiple 2D slices
    result = convert_npy_to_images(npy_path, input_size=input_size, max_num=max_num)
    
    if result is False:
        return None
    
    pixel_values, num_patches_list = result
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    
    # Create image token prefix for all slices
    image_prefix = ''.join([f'<image>\n' for _ in range(len(num_patches_list))])
    
    # Prepare question with prompt and image tokens
    full_question = f"{prompt}\n{image_prefix}{question}"
    
    # Generate response
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    response, history = model.chat(
        tokenizer,
        pixel_values,
        full_question,
        generation_config,
        num_patches_list=num_patches_list,
        history=None,
        return_history=True
    )
    
    return response


# ============================================================================
# Main Execution Examples
# ============================================================================

def main():
    """
    Main function demonstrating all three inference modes.
    """
    
    # ========================================================================
    # Example 1: Single Image Inference
    # ========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Image Inference")
    print("="*80)
    
    image_path = "./resource/test.jpg"
    question = ' What type of abnormality is present in this image?'
    
    model, tokenizer = load_model(MODEL_PATH, use_flash_attn=True)
    response = inference_single_image(model, tokenizer, image_path, question)
    
    print(f"\nUser: {question}")
    print(f"Assistant: {response}")
    
    # Clean up GPU memory
    del model, tokenizer
    torch.cuda.empty_cache()
    
    # ========================================================================
    # Example 2: Video Inference
    # ========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 2: Video Inference")
    print("="*80)
    
    video_path = "./resource/video.mp4"
    video_duration = 6  # seconds
    question = "Please describe the video."
    
    model, tokenizer = load_model(MODEL_PATH, use_flash_attn=False)
    response = inference_video(model, tokenizer, video_path, video_duration, question)
    
    print(f"\nUser: {question}")
    print(f"Assistant: {response}")
    
    # Clean up GPU memory
    del model, tokenizer
    torch.cuda.empty_cache()
    
    # ========================================================================
    # Example 3: 3D Medical Image Inference
    # ========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 3: 3D Medical Image Inference")
    print("="*80)
    
    npy_path = "./resource/test.npy"
    question = "What device is observed on the chest wall?"
    
    # Example cases:
    # Case 1: /path/to/test_1016_d_2.npy
    #   Question: "Where is the largest lymph node observed?"
    #   Answer: "Right hilar region."
    #
    # Case 2: /path/to/test_1031_a_2.npy
    #   Question: "What device is observed on the chest wall?"
    #   Answer: "Pacemaker."
    
    model, tokenizer = load_model(MODEL_PATH, use_flash_attn=False)
    response = inference_3d_medical_image(model, tokenizer, npy_path, question)
    
    if response:
        print(f"\nUser: {question}")
        print(f"Assistant: {response}")
    else:
        print("\nError: Failed to process 3D medical image")
    
    # Clean up GPU memory
    del model, tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()