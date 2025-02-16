import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Preprocessing steps for image transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Pre-trained mean and std
])

# Custom Dataset for Loading Images
class LungCancerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def prepare_data(image_folder, label_file):
    # Load images and labels (you should adjust based on your actual dataset format)
    image_paths = []
    labels = []

    # Dummy example: Assuming 'image_folder' has images and 'label_file' has corresponding labels
    # image_paths = [os.path.join(image_folder, img_name) for img_name in os.listdir(image_folder)]
    # labels = read_labels_from_file(label_file)  # Implement your own logic to load labels
    
    # Here, let's mock image paths and labels for testing:
    image_paths = ["path_to_image1.jpg", "path_to_image2.jpg"]  # Replace with actual paths
    labels = [0, 1]  # Example label (0: Adenocarcinoma, 1: Squamous Cell Carcinoma, etc.)

    # Create Dataset and DataLoader
    dataset = LungCancerDataset(image_paths, labels, transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    return dataloader
