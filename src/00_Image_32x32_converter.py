#!/usr/bin/env python3
"""
Image to 32x32 PNG Converter
Converts all images in a folder to 32x32 PNG files with sequential naming.
"""

import os
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Pillow is not installed. Installing now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image


def get_image_files(folder_path: str) -> list:
    """Get all supported image files from the specified folder."""
    supported_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    folder = Path(folder_path)
    
    image_files = []
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in supported_extensions:
            image_files.append(file)
    
    # Sort files for consistent ordering
    image_files.sort(key=lambda x: x.name.lower())
    return image_files


def convert_image_to_32x32(input_path: Path, output_path: Path) -> bool:
    """Convert a single image to 32x32 PNG."""
    try:
        with Image.open(input_path) as img:
            # Convert to RGBA to handle transparency and various formats
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                img = img.convert('RGBA')
            else:
                img = img.convert('RGB')
            
            # Resize to 32x32 using high-quality resampling
            img_resized = img.resize((32, 32), Image.Resampling.LANCZOS)
            
            # Save as PNG
            img_resized.save(output_path, 'PNG')
            return True
            
    except Exception as e:
        print(f"  Error processing {input_path.name}: {e}")
        return False


def convert_folder(input_folder: str, output_folder: str, base_name: str) -> dict:
    """
    Convert all images in a folder to 32x32 PNGs.
    
    Args:
        input_folder: Path to folder containing source images
        output_folder: Path to folder where converted images will be saved
        base_name: Base name for output files (e.g., "icon" -> icon1.png, icon2.png)
    
    Returns:
        Dictionary with conversion statistics
    """
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'files': []
    }
    
    # Validate input folder
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist!")
        return stats
    
    if not input_path.is_dir():
        print(f"Error: '{input_folder}' is not a directory!")
        return stats
    
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = get_image_files(input_folder)
    stats['total'] = len(image_files)
    
    if not image_files:
        print(f"No supported image files found in '{input_folder}'")
        print("Supported formats: JPG, JPEG, PNG, GIF, BMP, TIFF, WEBP")
        return stats
    
    print(f"\nFound {len(image_files)} image(s) to convert")
    print(f"Output folder: {output_path.absolute()}")
    print(f"Naming pattern: {base_name}1.png, {base_name}2.png, ...\n")
    print("-" * 50)
    
    # Process each image
    for index, image_file in enumerate(image_files, start=1):
        output_filename = f"{base_name}{index}.png"
        output_file_path = output_path / output_filename
        
        print(f"Converting: {image_file.name} -> {output_filename}")
        
        if convert_image_to_32x32(image_file, output_file_path):
            stats['success'] += 1
            stats['files'].append({
                'original': image_file.name,
                'converted': output_filename
            })
        else:
            stats['failed'] += 1
    
    return stats


def print_summary(stats: dict):
    """Print conversion summary."""
    print("\n" + "=" * 50)
    print("CONVERSION SUMMARY")
    print("=" * 50)
    print(f"Total files found:    {stats['total']}")
    print(f"Successfully converted: {stats['success']}")
    print(f"Failed:                 {stats['failed']}")
    
    if stats['files']:
        print("\nConverted files:")
        for file_info in stats['files']:
            print(f"  {file_info['original']} -> {file_info['converted']}")


def interactive_mode():
    """Run the converter in interactive mode."""
    print("=" * 50)
    print("  IMAGE TO 32x32 PNG CONVERTER")
    print("=" * 50)
    
    # Get input folder
    while True:
        input_folder = input("\nEnter the path to the folder containing images:\n> ").strip()
        if input_folder:
            # Handle quotes around path
            input_folder = input_folder.strip('"\'')
            if Path(input_folder).exists():
                break
            print(f"Folder '{input_folder}' does not exist. Please try again.")
        else:
            print("Please enter a valid path.")
    
    # Get output folder
    output_folder = input("\nEnter the output folder path (press Enter to use 'output' subfolder):\n> ").strip()
    if not output_folder:
        output_folder = str(Path(input_folder) / "output_32x32")
    else:
        output_folder = output_folder.strip('"\'')
    
    # Get base name
    base_name = input("\nEnter the base name for output files (e.g., 'icon' for icon1.png, icon2.png):\n> ").strip()
    if not base_name:
        base_name = "image"
    
    # Clean the base name (remove invalid characters)
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        base_name = base_name.replace(char, '')
    
    # Perform conversion
    stats = convert_folder(input_folder, output_folder, base_name)
    print_summary(stats)
    
    print(f"\nOutput saved to: {Path(output_folder).absolute()}")


def main():
    """Main function with command-line argument support."""
    if len(sys.argv) == 1:
        # No arguments - run interactive mode
        interactive_mode()
    elif len(sys.argv) == 4:
        # Command line mode: script.py input_folder output_folder base_name
        input_folder = sys.argv[1]
        output_folder = sys.argv[2]
        base_name = sys.argv[3]
        
        stats = convert_folder(input_folder, output_folder, base_name)
        print_summary(stats)
    elif len(sys.argv) == 3:
        # Command line mode: script.py input_folder base_name (output to subfolder)
        input_folder = sys.argv[1]
        base_name = sys.argv[2]
        output_folder = str(Path(input_folder) / "output_32x32")
        
        stats = convert_folder(input_folder, output_folder, base_name)
        print_summary(stats)
    else:
        print("Usage:")
        print("  Interactive mode: python converter.py")
        print("  Command line:     python converter.py <input_folder> <output_folder> <base_name>")
        print("  Command line:     python converter.py <input_folder> <base_name>")
        print("\nExamples:")
        print('  python converter.py "./my_images" "./output" "icon"')
        print('  python converter.py "./my_images" "sprite"')
        sys.exit(1)


if __name__ == "__main__":
    main()