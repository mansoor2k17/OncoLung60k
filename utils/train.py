import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from models.fusion_model import FusionTransformerClassifier

def train_model(model, dataloader, num_epochs=10, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        all_labels = []
        all_preds = []

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate average loss and accuracy
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
