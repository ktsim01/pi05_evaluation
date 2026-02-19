#!/usr/bin/env python3
"""
Script to convert a folder of images into a GIF.
The GIF will be saved in the same folder as the input images.
"""

import argparse
from pathlib import Path
from PIL import Image
import re


def natural_sort_key(path):
    """Sort paths naturally by numbers in the filename."""
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    
    filename = path.name
    return [convert(c) for c in re.split(r'(\d+)', filename)]


def images_to_gif(folder_path: Path, output_path: Path = None, output_name: str = "output.gif", fps: int = 10, loop: int = 0):
    """
    Convert all images in a folder to a GIF.
    
    Args:
        folder_path: Path to folder containing images
        output_path: Full path to output GIF file (if None, uses folder_path / output_name)
        output_name: Name of output GIF file (default: "output.gif", used if output_path is None)
        fps: Frames per second for the GIF (default: 10)
        loop: Number of loops (0 = infinite, default: 0)
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif'}
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(folder_path.glob(f"*{ext}"))
        image_files.extend(folder_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return None
    
    # Sort images naturally (by number in filename)
    image_files.sort(key=natural_sort_key)
    
    print(f"Found {len(image_files)} images. Creating GIF...")
    
    # Read all images
    images = []
    for img_path in image_files:
        try:
            img = Image.open(img_path)
            # Convert to RGB if necessary (GIFs don't support RGBA)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create a white background for transparency
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not read {img_path}: {e}")
            continue
    
    if not images:
        print("No valid images could be read.")
        return None
    
    # Create output path
    if output_path is None:
        output_path = folder_path / output_name
    else:
        output_path = Path(output_path)
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as GIF
    print(f"Saving GIF to {output_path}...")
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=1000 // fps,  # Duration in milliseconds
        loop=loop
    )
    
    print(f"Successfully created GIF with {len(images)} frames at {fps} fps: {output_path}")
    return output_path


def process_trials_batch(base_logs_path: Path, task_types: list, fps: int = 10, loop: int = 0):
    """
    Process all trials in specified task directories and create GIFs.
    
    Args:
        base_logs_path: Base path to logs directory (e.g., logs/2026_01_24)
        task_types: List of task types to process (e.g., ['closing', 'grasping'])
        fps: Frames per second for the GIFs (default: 10)
        loop: Number of loops (0 = infinite, default: 0)
    """
    base_logs_path = Path(base_logs_path)
    gifs_output_dir = base_logs_path.parent / "gifs"
    gifs_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing trials from {base_logs_path}")
    print(f"Output directory: {gifs_output_dir}")
    
    total_processed = 0
    total_failed = 0
    
    for task_type in task_types:
        task_dir = base_logs_path / task_type
        if not task_dir.exists():
            print(f"Warning: Task directory does not exist: {task_dir}")
            continue
        
        print(f"\n=== Processing {task_type} ===")
        
        # Find all trial folders
        trial_folders = [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith('trial')]
        trial_folders.sort(key=lambda x: natural_sort_key(x))
        
        for trial_folder in trial_folders:
            external_camera_dir = trial_folder / "external_camera"
            
            if not external_camera_dir.exists():
                print(f"  Skipping {trial_folder.name}: no external_camera folder")
                continue
            
            # Create output GIF name: task_type_trial_name.gif
            gif_name = f"{task_type}_{trial_folder.name}.gif"
            output_path = gifs_output_dir / gif_name
            
            print(f"  Processing {trial_folder.name}...")
            try:
                result = images_to_gif(
                    folder_path=external_camera_dir,
                    output_path=output_path,
                    fps=fps,
                    loop=loop
                )
                if result:
                    total_processed += 1
                else:
                    total_failed += 1
            except Exception as e:
                print(f"  Error processing {trial_folder.name}: {e}")
                total_failed += 1
    
    print(f"\n=== Summary ===")
    print(f"Successfully processed: {total_processed} trials")
    print(f"Failed: {total_failed} trials")
    print(f"All GIFs saved to: {gifs_output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a folder of images into a GIF, or batch process trials",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single folder:
  python images_to_gif.py /path/to/images
  python images_to_gif.py /path/to/images --output animation.gif --fps 15
  
  # Batch process trials:
  python images_to_gif.py --batch --date 2026_01_24
        """
    )
    parser.add_argument(
        "folder",
        type=str,
        nargs='?',
        default=None,
        help="Path to folder containing images (not used with --batch)"
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Batch process all trials in logs/YYYY_MM_DD/closing and logs/YYYY_MM_DD/grasping"
    )
    parser.add_argument(
        "--date", "-d",
        type=str,
        default="2026_01_24",
        help="Date string for batch processing (format: YYYY_MM_DD, default: 2026_01_24)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.gif",
        help="Output GIF filename (default: output.gif, not used with --batch)"
    )
    parser.add_argument(
        "--fps", "-f",
        type=int,
        default=10,
        help="Frames per second for the GIF (default: 10)"
    )
    parser.add_argument(
        "--loop", "-l",
        type=int,
        default=0,
        help="Number of loops (0 = infinite, default: 0)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.batch:
            # Batch processing mode
            logs_base = Path(__file__).parent.parent / "logs" / args.date
            if not logs_base.exists():
                print(f"Error: Logs directory does not exist: {logs_base}")
                return 1
            
            process_trials_batch(
                base_logs_path=logs_base,
                task_types=['closing2'],
                fps=args.fps,
                loop=args.loop
            )
        else:
            # Single folder mode
            if args.folder is None:
                print("Error: folder argument is required when not using --batch mode")
                parser.print_help()
                return 1
            
            images_to_gif(
                folder_path=Path(args.folder),
                output_name=args.output,
                fps=args.fps,
                loop=args.loop
            )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

