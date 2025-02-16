import torch.nn as nn
import torchvision.models as models

class DenseNet121(nn.Module):
    def __init__(self, pretrained=False, num_classes=4, weight_path=None):
        super(DenseNet121, self).__init__()
        self.model = models.densenet121(pretrained=False)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

        if pretrained and weight_path is not None:
            # Load Kemianet weights
            self.model.load_state_dict(torch.load(weight_path, map_location='cpu'))
            print(f"Loaded Kemianet weights from {weight_path}")

    def forward(self, x):
        return self.model(x)
