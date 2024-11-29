import torch
from torchvision import transforms
from PIL import Image
import os
import pickle
from basemodels import cusResNet18  # Replace with the actual file where cusResNet18 is defined

# Step 1: Load the Custom Model
model = cusResNet18(n_classes=2, pretrained=True)  # Replace n_classes with your task's number of classes
model.eval()  # Set to evaluation mode

# Step 2: Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.Lambda(lambda img: img.convert('RGB')),  # Convert B/W to RGB by duplicating the channel
    transforms.ToTensor(),          # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

# Step 3: Function to Extract Features
def extract_features(image_path, model):
    """
    Extract features for a single image using cusResNet18.
    Returns both FC layer output and AvgPool layer output.
    """
    img = Image.open(image_path).convert('L')  # Open the image as B/W (grayscale)
    img = transform(img).unsqueeze(0)  # Apply transformations and add batch dimension
    with torch.no_grad():
        fc_out, avgpool_out = model.inference(img)  # Extract features
    return {'fc': fc_out.numpy(), 'avgpool': avgpool_out.numpy()}

# Step 4: Process All Images in a Directory
def process_images(image_dir, output_path, model):
    """
    Process all images in a directory, extract features, and save to a pickle file.
    """
    features = {}
    for image_name in os.listdir(image_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ensure valid image files
            image_path = os.path.join(image_dir, image_name)
            features[image_name] = extract_features(image_path, model)
            print(f"Processed: {image_name}")

    # Save features to a pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(features, f)
    print(f"Features saved to {output_path}")

# Step 5: Define Paths and Run Feature Extraction
image_dir = r"C:\Users\saium\Desktop\thyroid\augmented_images"  # Replace with the directory containing your images
output_path = r"C:\Users\saium\Desktop\thyroid\pkls\features.pkl"  # Path to save the pickle file

# Extract features and save them
process_images(image_dir, output_path, model)