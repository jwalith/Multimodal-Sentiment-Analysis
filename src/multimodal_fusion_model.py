import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttentionFusion(nn.Module):
    def __init__(self, image_dim: int, text_dim: int, hidden_dim: int = 256):
        super(AttentionFusion, self).__init__()
        
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        image_proj = self.image_projection(image_embeddings)
        text_proj = self.text_projection(text_embeddings)
        
        image_proj = image_proj.unsqueeze(1)
        text_proj = text_proj.unsqueeze(1)
        
        sequence = torch.cat([image_proj, text_proj], dim=1)
        attended, _ = self.attention(sequence, sequence, sequence)
        
        fused = attended.mean(dim=1)
        return self.layer_norm(fused)


class MultimodalSentimentClassifier(nn.Module):
    def __init__(self, image_dim: int, text_dim: int, num_classes: int = 3, 
                 hidden_dim: int = 256, dropout: float = 0.3, fusion_method: str = "attention"):
        super(MultimodalSentimentClassifier, self).__init__()
        
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.fusion_method = fusion_method
        
        if fusion_method == "attention":
            self.fusion = AttentionFusion(image_dim, text_dim, hidden_dim)
            fusion_output_dim = hidden_dim
        elif fusion_method == "concat":
            fusion_output_dim = image_dim + text_dim
        elif fusion_method == "add":
            assert image_dim == text_dim, "For additive fusion, image and text dimensions must be equal"
            fusion_output_dim = image_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        if self.fusion_method == "attention":
            fused_embeddings = self.fusion(image_embeddings, text_embeddings)
        elif self.fusion_method == "concat":
            fused_embeddings = torch.cat([image_embeddings, text_embeddings], dim=-1)
        elif self.fusion_method == "add":
            fused_embeddings = image_embeddings + text_embeddings
        
        return self.classifier(fused_embeddings)
    
    def predict(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(image_embeddings, text_embeddings)
            return F.softmax(logits, dim=-1)
    
    def predict_classes(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict sentiment classes.
        
        Args:
            image_embeddings: CLIP image embeddings
            text_embeddings: BERT text embeddings
            
        Returns:
            torch.Tensor: Predicted class indices (batch_size,)
        """
        with torch.no_grad():
            logits = self.forward(image_embeddings, text_embeddings)
            predictions = torch.argmax(logits, dim=-1)
        return predictions


class MultimodalSentimentTrainer:
    """
    Trainer class for the multimodal sentiment classifier.
    """
    
    def __init__(self, model: MultimodalSentimentClassifier, device: str = "auto"):
        """
        Initialize the trainer.
        
        Args:
            model: Multimodal sentiment classifier
            device: Device to use for training
        """
        self.model = model
        self.device = self._get_device(device)
        self.model.to(self.device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the appropriate device to use."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def train_epoch(self, train_loader, optimizer, criterion) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Tuple[float, float]: Average loss and accuracy for the epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (image_embeddings, text_embeddings, labels) in enumerate(train_loader):
            # Move to device
            image_embeddings = image_embeddings.to(self.device)
            text_embeddings = text_embeddings.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(image_embeddings, text_embeddings)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple[float, float]: Average loss and accuracy
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for image_embeddings, text_embeddings, labels in val_loader:
                # Move to device
                image_embeddings = image_embeddings.to(self.device)
                text_embeddings = text_embeddings.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(image_embeddings, text_embeddings)
                loss = criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs: int = 50, 
              learning_rate: float = 0.001, weight_decay: float = 1e-4) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            
        Returns:
            Dict[str, list]: Training history
        """
        # Setup optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        logger.info("Training completed!")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def evaluate(self, test_loader) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for image_embeddings, text_embeddings, labels in test_loader:
                # Move to device
                image_embeddings = image_embeddings.to(self.device)
                text_embeddings = text_embeddings.to(self.device)
                labels = labels.to(self.device)
                
                # Predict
                predictions = self.model.predict_classes(image_embeddings, text_embeddings)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_multimodal_model(image_dim: int, text_dim: int, num_classes: int = 3, 
                          fusion_method: str = "attention", hidden_dim: int = 256, 
                          dropout: float = 0.3) -> MultimodalSentimentClassifier:
    """
    Factory function to create a multimodal sentiment classifier.
    
    Args:
        image_dim: Dimension of CLIP image embeddings
        text_dim: Dimension of BERT text embeddings
        num_classes: Number of sentiment classes
        fusion_method: Method for fusing embeddings
        hidden_dim: Hidden dimension for the model
        dropout: Dropout rate
        
    Returns:
        MultimodalSentimentClassifier: Configured model
    """
    return MultimodalSentimentClassifier(
        image_dim=image_dim,
        text_dim=text_dim,
        num_classes=num_classes,
        fusion_method=fusion_method,
        hidden_dim=hidden_dim,
        dropout=dropout
    )


if __name__ == "__main__":
    # Example usage
    image_dim = 512  # CLIP embedding dimension
    text_dim = 768  # BERT embedding dimension
    num_classes = 3  # negative, neutral, positive
    
    # Create model
    model = create_multimodal_model(image_dim, text_dim, num_classes, fusion_method="attention")
    
    # Print model architecture
    print("Multimodal Sentiment Classifier:")
    print(model)
    
    # Test forward pass
    batch_size = 4
    image_embeddings = torch.randn(batch_size, image_dim)
    text_embeddings = torch.randn(batch_size, text_dim)
    
    with torch.no_grad():
        logits = model(image_embeddings, text_embeddings)
        probabilities = model.predict(image_embeddings, text_embeddings)
        predictions = model.predict_classes(image_embeddings, text_embeddings)
    
    print(f"\nInput shapes: Image {image_embeddings.shape}, Text {text_embeddings.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output probabilities shape: {probabilities.shape}")
    print(f"Predictions: {predictions}")
