#!/usr/bin/env python3
"""
CIFAR-10 Dataset to Images Converter

This script converts the CIFAR-10 dataset from pickled format to individual image files.
The dataset contains 50,000 training images and 10,000 test images in 10 classes.
Each image is 32x32 pixels with 3 color channels (RGB).
"""

import pickle
import os
import numpy as np
from PIL import Image


def unpickle(file_path):
    """
    Unpickle a CIFAR-10 data file.
    
    Args:
        file_path (str): Path to the pickled data file
        
    Returns:
        dict: Dictionary containing the unpickled data
    """
    with open(file_path, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def load_class_names():
    """
    Load class names from the batches.meta file.
    
    Returns:
        list: List of class names
    """
    meta_data = unpickle('cifar-10-batches-py/batches.meta')
    # Convert bytes to string for class names
    class_names = [name.decode('utf-8') for name in meta_data[b'label_names']]
    return class_names


def reshape_image_data(image_data):
    """
    Reshape flattened image data to 32x32x3 RGB format.
    
    Args:
        image_data (numpy.ndarray): Flattened image data (3072 values)
        
    Returns:
        numpy.ndarray: Reshaped image data (32, 32, 3)
    """
    # CIFAR-10 images are stored as [red_channel, green_channel, blue_channel]
    # Each channel is 32x32 = 1024 values
    red = image_data[0:1024].reshape(32, 32)
    green = image_data[1024:2048].reshape(32, 32)
    blue = image_data[2048:3072].reshape(32, 32)
    
    # Stack channels to create RGB image
    rgb_image = np.stack([red, green, blue], axis=2)
    return rgb_image


def save_images_from_batch(batch_file, batch_name, class_names, output_dir):
    """
    Extract and save images from a single batch file.
    
    Args:
        batch_file (str): Path to the batch file
        batch_name (str): Name of the batch (for organizing files)
        class_names (list): List of class names
        output_dir (str): Output directory path
        
    Returns:
        int: Number of images processed
    """
    print(f"Processing {batch_file}...")
    
    # Load batch data
    batch_data = unpickle(batch_file)
    
    # Extract data and labels
    images = batch_data[b'data']
    labels = batch_data[b'labels']
    
    if b'filenames' in batch_data:
        filenames = [name.decode('utf-8') for name in batch_data[b'filenames']]
    else:
        filenames = [f"image_{i:05d}.png" for i in range(len(labels))]
    
    # Process each image
    images_processed = 0
    for i, (image_data, label, filename) in enumerate(zip(images, labels, filenames)):
        # Get class name
        class_name = class_names[label]
        
        # Create class directory if it doesn't exist
        class_dir = os.path.join(output_dir, batch_name, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Reshape image data
        rgb_image = reshape_image_data(image_data)
        
        # Create PIL Image
        pil_image = Image.fromarray(rgb_image, 'RGB')
        
        # Save image
        image_filename = f"{batch_name}_{i:05d}_{filename}"
        if not image_filename.endswith('.png'):
            image_filename = image_filename.replace('.png', '') + '.png'
        
        image_path = os.path.join(class_dir, image_filename)
        pil_image.save(image_path)
        
        images_processed += 1
        
        # Print progress every 1000 images
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1} images from {batch_name}")
    
    print(f"  Completed {batch_name}: {images_processed} images")
    return images_processed


def convert_cifar10_to_images(output_dir='cifar10_images'):
    """
    Convert the entire CIFAR-10 dataset to individual image files.
    
    Args:
        output_dir (str): Output directory for the images
    """
    print("Starting CIFAR-10 to Images conversion...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load class names
    try:
        class_names = load_class_names()
        print(f"Loaded {len(class_names)} classes: {class_names}")
    except Exception as e:
        print(f"Error loading class names: {e}")
        # Fallback to default CIFAR-10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
        print(f"Using default class names: {class_names}")
    
    total_images = 0
    
    # Process training batches (data_batch_1 to data_batch_5)
    print("\nProcessing training batches...")
    for i in range(1, 6):
        batch_file = f'cifar-10-batches-py/data_batch_{i}'
        batch_name = f'train_batch_{i}'
        
        if os.path.exists(batch_file):
            images_count = save_images_from_batch(batch_file, batch_name, class_names, output_dir)
            total_images += images_count
        else:
            print(f"Warning: {batch_file} not found")
    
    # Process test batch
    print("\nProcessing test batch...")
    test_batch_file = 'cifar-10-batches-py/test_batch'
    if os.path.exists(test_batch_file):
        images_count = save_images_from_batch(test_batch_file, 'test_batch', class_names, output_dir)
        total_images += images_count
    else:
        print(f"Warning: {test_batch_file} not found")
    
    print(f"\nConversion completed!")
    print(f"Total images processed: {total_images}")
    print(f"Images saved to: {output_dir}/")
    print(f"Directory structure:")
    print(f"  {output_dir}/")
    print(f"    ├── train_batch_1/")
    print(f"    ├── train_batch_2/")
    print(f"    ├── train_batch_3/")
    print(f"    ├── train_batch_4/")
    print(f"    ├── train_batch_5/")
    print(f"    └── test_batch/")
    print(f"        ├── airplane/")
    print(f"        ├── automobile/")
    print(f"        ├── bird/")
    print(f"        ├── cat/")
    print(f"        ├── deer/")
    print(f"        ├── dog/")
    print(f"        ├── frog/")
    print(f"        ├── horse/")
    print(f"        ├── ship/")
    print(f"        └── truck/")


def main():
    """Main function to run the conversion."""
    print("CIFAR-10 Dataset to Images Converter")
    print("=" * 40)
    
    # Check if CIFAR-10 data exists
    if not os.path.exists('cifar-10-batches-py'):
        print("Error: 'cifar-10-batches-py' directory not found!")
        print("Please make sure the CIFAR-10 dataset is extracted in the current directory.")
        return
    
    # Run conversion
    try:
        convert_cifar10_to_images()
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()