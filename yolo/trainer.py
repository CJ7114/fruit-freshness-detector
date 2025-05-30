import os
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import yaml
import cv2
from PIL import Image
import shutil
from pathlib import Path
import subprocess
import argparse

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train YOLOv5 for fruit freshness detection')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--weights', type=str, default=None, help='Path to weights file to start from')
    
    return parser.parse_args()

# Set up constants
BATCH_SIZE = 8
EPOCHS = 50
IMAGE_SIZE = 320  # YOLOv5 default size
BASE_DIR = "dataset"  # Change path if needed
# Update this line in the code
CLASSES = ["apple_fresh", "apple_rotten", "banana_fresh", "banana_rotten", 
           "mango_fresh", "mango_rotten", "orange_fresh", "orange_rotten",
           "strawberry_fresh", "strawberry_rotten", "bellpepper_fresh", "bellpepper_rotten",
           "bittergourd_fresh", "bittergourd_rotten", "capsicum_fresh", "capsicum_rotten",
           "carrot_fresh", "carrot_rotten", "cucumber_fresh", "cucumber_rotten",
           "okra_fresh", "okra_rotten", "potato_fresh", "potato_rotten",
           "tomato_fresh", "tomato_rotten"]

def prepare_yolo_dataset():
    """
    Prepares the dataset in YOLOv5 format
    """
    # Create directories
    yolo_dir = Path("yolo_dataset")
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    
    for split in ['train', 'val', 'test']:
        (yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Create data.yaml configuration
    data_yaml = {
        'train': str(os.path.abspath(yolo_dir / 'train' / 'images')),
        'val': str(os.path.abspath(yolo_dir / 'val' / 'images')),
        'test': str(os.path.abspath(yolo_dir / 'test' / 'images')),
        'nc': len(CLASSES),
        'names': CLASSES
    }
    
    with open(yolo_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print("Created YOLOv5 directory structure")
    
    # Process original dataset to YOLOv5 format
    process_original_dataset(BASE_DIR, yolo_dir)
    
    return str(yolo_dir / 'data.yaml')

def process_original_dataset(original_dir, yolo_dir):
    """
    Process original dataset into YOLOv5 format
    """
    for split in ['train', 'validation', 'test']:
        yolo_split = 'val' if split == 'validation' else split
        
        for condition in ['fresh', 'rotten']:
            condition_cap = condition.capitalize()
            
            for fruit_dir in os.listdir(os.path.join(original_dir, split, condition)):
                fruit_path = os.path.join(original_dir, split, condition, fruit_dir)
                
                if not os.path.isdir(fruit_path):
                    continue
                
                # Get class ID
                class_name = f"{fruit_dir}_{condition}"
                class_id = CLASSES.index(class_name)
                
                for img_file in os.listdir(fruit_path):
                    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    
                    # Copy image
                    img_src = os.path.join(fruit_path, img_file)
                    img_dst = str(yolo_dir / yolo_split / 'images' / f"{split}_{condition}_{fruit_dir}_{img_file}")
                    shutil.copy(img_src, img_dst)
                    
                    # Create label file (assuming the fruit is centered and takes up ~80% of the image)
                    img = Image.open(img_src)
                    width, height = img.size
                    
                    # YOLO format: <class> <x_center> <y_center> <width> <height>
                    # All values normalized between 0 and 1
                    label_txt = f"{class_id} 0.5 0.5 0.8 0.8\n"
                    
                    # Save label file with same name as image but .txt extension
                    label_path = str(yolo_dir / yolo_split / 'labels' / f"{split}_{condition}_{fruit_dir}_{os.path.splitext(img_file)[0]}.txt")
                    with open(label_path, 'w') as f:
                        f.write(label_txt)
    
    print(f"Processed dataset to YOLOv5 format")

def install_yolov5():
    """
    Install YOLOv5 if not already installed
    """
    if not os.path.exists('yolov5'):
        print("Installing YOLOv5...")
        subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5'], check=True)
        subprocess.run(['pip', 'install', '-r', 'yolov5/requirements.txt'], check=True)
    else:
        print("YOLOv5 already installed")

def train_yolov5(data_yaml, weights='yolov5s.pt', resume=False):
    """
    Train YOLOv5 model on the prepared dataset
    """
    # Convert data_yaml to absolute path with forward slashes
    abs_data_yaml = os.path.abspath(data_yaml).replace('\\', '/')
    print(f"Using data.yaml at: {abs_data_yaml}")
    
    # Save current directory
    original_dir = os.getcwd()
    
    # Change to yolov5 directory
    os.chdir('yolov5')
    
    # Build command with subprocess for better error handling
    cmd = [
        "python", "train.py",
        "--img", str(IMAGE_SIZE),
        "--batch", str(BATCH_SIZE),
        "--epochs", str(EPOCHS),
        "--data", abs_data_yaml,
        "--weights", weights,
        "--workers", "2",
        "--cache",
        "--project", "../fruit_freshness_results",
        "--device", "0"
    ]
    
    # Add resume flag if continuing from checkpoint
    if resume:
        cmd.append("--resume")
        print("Resuming training from last checkpoint")
    
    print(f"Training command: {' '.join(cmd)}")
    
    try:
        # Use subprocess.Popen for live output instead of subprocess.run
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Print output in real-time
        print("\nTraining progress:")
        for line in iter(process.stdout.readline, ''):
            print(line, end='')  # Print each line as it comes
            if not line:  # Break if end of output
                break
                
        process.wait()  # Wait for process to complete
        if process.returncode != 0:
            print("Error during training")
        else:
            print("Training completed successfully")
    except Exception as e:
        print(f"Exception during training: {str(e)}")
    
    # Return to original directory
    os.chdir(original_dir)
    
    # The weights should now be in the project root
    weights_path = os.path.join('fruit_freshness_results', 'exp', 'weights', 'best.pt')
    
    # Verify weights file exists
    if not os.path.exists(weights_path):
        print(f"WARNING: Weights not found at expected path: {weights_path}")
        # Try to locate weights
        weights_path = find_weights_file('fruit_freshness_results')
        if weights_path:
            print(f"Found weights at: {weights_path}")
        else:
            print("No weights file found. Will use default YOLOv5s weights for evaluation.")
            weights_path = 'yolov5s.pt'
    
    return weights_path

def find_weights_file(base_dir):
    """Find the best.pt weights file by searching recursively."""
    if not os.path.exists(base_dir):
        return None
        
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'best.pt':
                return os.path.join(root, file)
    return None

def evaluate_yolov5(weights_path, data_yaml):
    """
    Evaluate YOLOv5 model on test set
    """
    # Convert to absolute paths with forward slashes
    abs_weights_path = os.path.abspath(weights_path).replace('\\', '/')
    abs_data_yaml = os.path.abspath(data_yaml).replace('\\', '/')
    
    print(f"Using weights at: {abs_weights_path}")
    print(f"Using data.yaml at: {abs_data_yaml}")
    
    # Save current directory
    original_dir = os.getcwd()
    
    # Change to yolov5 directory
    os.chdir('yolov5')
    
    # Build command with subprocess for better error handling
    cmd = [
        "python", "val.py",
        "--img", str(IMAGE_SIZE),
        "--batch", str(BATCH_SIZE),
        "--data", abs_data_yaml,
        "--weights", abs_weights_path,
        "--task", "test",
        "--save-txt",
        "--save-conf",
        "--project", "../fruit_freshness_eval"
    ]
    
    print(f"Evaluation command: {' '.join(cmd)}")
    
    try:
        # Use subprocess.Popen for live output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Print output in real-time
        print("\nEvaluation progress:")
        for line in iter(process.stdout.readline, ''):
            print(line, end='')  # Print each line as it comes
            if not line:  # Break if end of output
                break
                
        process.wait()  # Wait for process to complete
        if process.returncode != 0:
            print("Error during evaluation")
        else:
            print("Evaluation completed successfully")
    except Exception as e:
        print(f"Exception during evaluation: {str(e)}")
    
    # Return to original directory
    os.chdir(original_dir)
    
    # Results should be in the project root
    results_dir = os.path.join('fruit_freshness_eval', 'exp')
    
    return results_dir

def predict_image(weights_path, image_path):
    """
    Predict freshness and type of fruits/vegetables in a single image
    """
    # Convert to absolute path
    abs_weights_path = os.path.abspath(weights_path)
    abs_image_path = os.path.abspath(image_path)
    
    print(f"Loading model from: {abs_weights_path}")
    print(f"Predicting on image: {abs_image_path}")
    
    try:
        # Check if weights file exists
        if not os.path.exists(abs_weights_path):
            print(f"Warning: Weights file not found at {abs_weights_path}")
            print("Falling back to default YOLOv5s model")
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        else:
            # Load model with force_reload to avoid cache issues
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=abs_weights_path, force_reload=True)
        
        # Make prediction
        results = model(abs_image_path)
        
        # Display results
        results.print()  # Print results to console
        results.show()   # Display results
        
        # Save results to file
        results.save(save_dir='prediction_results')
        
        # Get detailed results
        predictions = results.pandas().xyxy[0]
        
        if len(predictions) == 0:
            print("No objects detected in the image.")
        else:
            for i, row in predictions.iterrows():
                if 'name' in row:
                    class_name = row['name']
                    confidence = row['confidence']
                    if '_' in class_name:
                        fruit_type, freshness = class_name.split('_')
                        print(f"Detected {fruit_type} ({freshness}) with {confidence:.2f} confidence")
                    else:
                        print(f"Detected {class_name} with {confidence:.2f} confidence")
        
        return predictions
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print("Troubleshooting steps:")
        print(f"1. Check if the weights file exists: {os.path.exists(abs_weights_path)}")
        print(f"2. Check if the image file exists: {os.path.exists(abs_image_path)}")
        print("3. Try running with a different weights file, like 'yolov5s.pt'")
        
        # Try with default model as fallback
        try:
            print("Attempting prediction with default YOLOv5s model...")
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            results = model(abs_image_path)
            results.print()
            results.show()
            return results.pandas().xyxy[0]
        except Exception as e2:
            print(f"Fallback prediction also failed: {str(e2)}")
            return None

def create_custom_confusion_matrix(results_dir):
    """
    Create a custom confusion matrix from YOLOv5 results
    """
    # Parse prediction results
    all_preds = []
    all_targets = []
    
    labels_dir = os.path.join(results_dir, 'labels')
    if not os.path.exists(labels_dir):
        print("No labels directory found. Cannot create confusion matrix.")
        return
    
    for file in os.listdir(labels_dir):
        if not file.endswith('.txt'):
            continue
        
        # Get prediction file
        pred_file = os.path.join(labels_dir, file)
        
        # Get corresponding target file (from test set)
        img_name = file.replace('_labels.txt', '.jpg')
        target_file = os.path.join('yolo_dataset', 'test', 'labels', img_name.replace('.jpg', '.txt'))
        
        if not os.path.exists(target_file):
            continue
        
        # Read prediction
        with open(pred_file, 'r') as f:
            lines = f.readlines()
            if lines:
                parts = lines[0].strip().split()
                pred_class = int(float(parts[0]))
                all_preds.append(pred_class)
        
        # Read target
        with open(target_file, 'r') as f:
            lines = f.readlines()
            if lines:
                parts = lines[0].strip().split()
                target_class = int(float(parts[0]))
                all_targets.append(target_class)
    
    # Create confusion matrix
    if all_preds and all_targets:
        cm = confusion_matrix(all_targets, all_preds, labels=range(len(CLASSES)))
        
        plt.figure(figsize=(20, 20))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=CLASSES,
            yticklabels=CLASSES
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        print("\nClassification Report:")
        print(classification_report(all_targets, all_preds, target_names=CLASSES))
    else:
        print("Not enough data to create confusion matrix")

def batch_prediction(weights_path, folder_path):
    """
    Process a folder of images and generate predictions
    """
    # Check if weights file exists
    if not os.path.exists(weights_path):
        print(f"Warning: Weights file not found at {weights_path}")
        print("Falling back to default YOLOv5s model")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    else:
        # Load YOLOv5 model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
    
    # Create output directory
    output_dir = 'batch_predictions'
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    for img_file in os.listdir(folder_path):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        img_path = os.path.join(folder_path, img_file)
        
        # Make prediction
        pred = model(img_path)
        
        # Store results
        results[img_file] = pred.pandas().xyxy[0]
        
        # For visualization (optional)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        
        for i, row in results[img_file].iterrows():
            box = row[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(int)
            
            # Handle both custom and default model predictions
            if 'name' in row:
                class_name = row['name']
                if '_' in class_name:
                    fruit_type, freshness = class_name.split('_')
                    label = f"{fruit_type} ({freshness}): {row['confidence']:.2f}"
                else:
                    label = f"{class_name}: {row['confidence']:.2f}"
            else:
                label = f"Class {row.get('class', '?')}: {row['confidence']:.2f}"
            
            # Draw bounding box
            plt.gca().add_patch(plt.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                fill=False, edgecolor='red', linewidth=2
            ))
            
            # Add label
            plt.text(
                box[0], box[1]-10, label,
                color='white', fontsize=10, backgroundcolor='red'
            )
        
        plt.title(f"Predictions for {img_file}")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"pred_{img_file}"))
        plt.close()  # Close the figure to save memory
    
    print(f"Batch prediction completed. Results saved to {output_dir}")
    return results

def test_with_default_model(test_image=None):
    """Test a predefined YOLOv5 model on an image"""
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    
    # Find a test image if none provided
    if not test_image:
        test_image_dir = os.path.join('yolo_dataset', 'test', 'images')
        if os.path.exists(test_image_dir) and len(os.listdir(test_image_dir)) > 0:
            test_image = os.path.join(test_image_dir, os.listdir(test_image_dir)[0])
    
    if test_image and os.path.exists(test_image):
        print(f"Testing on image: {test_image}")
        
        results = model(test_image)
        results.show()
        results.print()
        
        # Save results
        results.save(save_dir='test_results')
        return True
    else:
        print("No test images found or specified image doesn't exist.")
        return False

def main():
    """
    Main function to run the entire pipeline
    """
    # Parse arguments
    args = parse_arguments()
    
    # Update global constants
    global EPOCHS, BATCH_SIZE
    if args.epochs:
        EPOCHS = args.epochs
    if args.batch:
        BATCH_SIZE = args.batch
    
    print(f"Using batch size: {BATCH_SIZE}, epochs: {EPOCHS}")
    
    print("1. Installing YOLOv5...")
    install_yolov5()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("2. Preparing dataset in YOLOv5 format...")
    data_yaml = prepare_yolo_dataset()
    
    # Check for existing weights to resume training
    resume_training = args.resume
    
    if resume_training:
        if args.weights:
            weights_path = args.weights
            print(f"Resuming training from specified weights: {weights_path}")
        else:
            weights_path = find_weights_file('fruit_freshness_results')
            if weights_path:
                print(f"Found existing weights at: {weights_path}")
                print("Resuming training from last checkpoint")
            else:
                print("No existing weights found for resuming. Using default weights.")
                weights_path = 'yolov5s.pt'
    else:
        weights_path = args.weights if args.weights else 'yolov5s.pt'
    
    print("3. Training YOLOv5 model...")
    weights_path = train_yolov5(data_yaml, weights=weights_path, resume=resume_training)
    
    print("4. Evaluating YOLOv5 model...")
    results_dir = evaluate_yolov5(weights_path, data_yaml)
    
    print("5. Creating confusion matrix...")
    create_custom_confusion_matrix(results_dir)
    
    print("6. YOLOv5 training and evaluation complete!")
    print(f"Model saved at: {weights_path}")
    
    # Example of using the model for prediction
    print("\nExample prediction:")
    test_image_dir = os.path.join('yolo_dataset', 'test', 'images')
    if os.path.exists(test_image_dir) and len(os.listdir(test_image_dir)) > 0:
        test_image = os.path.join(test_image_dir, os.listdir(test_image_dir)[0])
        predict_image(weights_path, test_image)
    else:
        print("No test images found. Cannot run prediction example.")
        print("Trying to run with default YOLOv5s model...")
        test_with_default_model()

if __name__ == "__main__":
    main()