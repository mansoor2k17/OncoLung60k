import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.fusion_model import FusionTransformerClassifier
from utils.dataset import prepare_data
from utils.train import train_model
from utils.evaluate import evaluate_model

from efficientnet_b7 import EfficientNetB7
from resnet152 import ResNet152
from densenet121 import DenseNet121
from transformer import TransformerFeatureExtractor

def main():
    # Set the paths to the Kemianet weights files for each model
    efficientnet_b7_weight_path = 'path_to_kemianet_efficientnet_b7_weights.pth'
    resnet152_weight_path = 'path_to_kemianet_resnet152_weights.pth'
    densenet121_weight_path = 'path_to_kemianet_densenet121_weights.pth'
    
    # Initialize the models with Kemianet weights
    efficientnet_b7_model = EfficientNetB7(pretrained=True, weight_path=efficientnet_b7_weight_path)
    resnet152_model = ResNet152(pretrained=True, weight_path=resnet152_weight_path)
    densenet121_model = DenseNet121(pretrained=True, weight_path=densenet121_weight_path)
    transformer_model = TransformerFeatureExtractor()
    
    # Continue with your training pipeline
    # Your code for data loading, training, and evaluation will follow
    
if __name__ == "__main__":
    main()
