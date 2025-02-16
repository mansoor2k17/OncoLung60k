import torch
import torch.nn as nn
import timm
from torchvision import models

class FusionTransformerClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(FusionTransformerClassifier, self).__init__()

        # Pre-trained models for feature extraction
        self.efficientnet_b7 = timm.create_model('efficientnet_b7', pretrained=True, num_classes=0)  # Remove final FC layer
        self.resnet152 = models.resnet152(pretrained=True)
        self.densenet121 = models.densenet121(pretrained=True)

        # Transformer for feature extraction from input images
        self.transformer = nn.Transformer(
            d_model=512,  # Adjust input dimension as per the feature size
            nhead=8, 
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048
        )
        
        # FC layers for final classification
        self.fc1 = nn.Linear(512 * 4 + 512, 512)  # 512 * 4 from the pre-trained models and 512 from the transformer
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Extract features from the pre-trained models
        eff_b7_features = self.efficientnet_b7(x)
        resnet152_features = self.resnet152(x)
        densenet121_features = self.densenet121(x)

        # Prepare input for transformer (convert to appropriate shape for transformer)
        x_transformer = x.view(x.size(0), -1, 512)  # Assume 512x512 input image, adjust according to your needs
        transformer_out = self.transformer(x_transformer, x_transformer)  # Self-attention

        # Gather output from transformer (using the last token)
        transformer_features = transformer_out[-1, :, :]

        # Concatenate all features
        combined_features = torch.cat((eff_b7_features, resnet152_features, densenet121_features, transformer_features), dim=1)

        # Classification layers
        x = self.fc1(combined_features)
        x = torch.relu(x)
        x = self.fc2(x)
        
        return x
