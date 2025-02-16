import timm
import torch.nn as nn

class EfficientNetB7(nn.Module):
    def __init__(self, pretrained=False, num_classes=4, weight_path=None):
        super(EfficientNetB7, self).__init__()
        self.model = timm.create_model('efficientnet_b7', pretrained=False, num_classes=num_classes)
        
        if pretrained and weight_path is not None:
            # Load Kemianet weights
            self.model.load_state_dict(torch.load(weight_path, map_location='cpu'))
            print(f"Loaded Kemianet weights from {weight_path}")
        
    def forward(self, x):
        return self.model(x)
