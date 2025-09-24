import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from clip_embedder import create_clip_embedder
from bert_sentiment_embedder import create_bert_sentiment_embedder
from multimodal_fusion_model import create_multimodal_model, MultimodalSentimentTrainer
from data_processor import create_data_processor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MultimodalTrainingPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("Initializing embedders...")
        self.clip_embedder = create_clip_embedder(
            model_name=config.get('clip_model', 'openai/clip-vit-base-patch32'),
            device=self.device
        )
        
        self.bert_embedder = create_bert_sentiment_embedder(
            model_name=config.get('bert_model', 'cardiffnlp/twitter-roberta-base-sentiment-latest'),
            device=self.device,
            max_length=config.get('max_text_length', 512)
        )
        
        self.image_dim = self.clip_embedder.get_embedding_dimension()
        self.text_dim = self.bert_embedder.get_embedding_dimension()
        
        logger.info(f"Image embedding dimension: {self.image_dim}")
        logger.info(f"Text embedding dimension: {self.text_dim}")
        
        self.model = create_multimodal_model(
            image_dim=self.image_dim,
            text_dim=self.text_dim,
            num_classes=config.get('num_classes', 3),
            fusion_method=config.get('fusion_method', 'attention'),
            hidden_dim=config.get('hidden_dim', 256),
            dropout=config.get('dropout', 0.3)
        )
        
        self.trainer = MultimodalSentimentTrainer(self.model, device=self.device)
        logger.info("Training pipeline initialized")
    
    def extract_embeddings(self, data_loader: DataLoader) -> tuple:
        logger.info("Extracting embeddings...")
        
        all_image_embeddings = []
        all_text_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, texts, labels) in enumerate(tqdm(data_loader, desc="Extracting embeddings")):
                # Extract CLIP embeddings
                batch_image_embeddings = []
                for image in images:
                    if isinstance(image, torch.Tensor):
                        # Convert tensor to PIL Image
                        image_np = image.permute(1, 2, 0).numpy()
                        image_np = (image_np * 255).astype(np.uint8)
                        from PIL import Image
                        pil_image = Image.fromarray(image_np)
                    else:
                        pil_image = image
                    
                    embedding = self.clip_embedder.extract_embedding(pil_image)
                    batch_image_embeddings.append(embedding)
                
                # Extract BERT embeddings
                batch_text_embeddings = self.bert_embedder.extract_batch_embeddings(texts)
                
                all_image_embeddings.extend(batch_image_embeddings)
                all_text_embeddings.extend(batch_text_embeddings)
                all_labels.extend(labels.numpy())
        
        image_embeddings = torch.tensor(np.array(all_image_embeddings), dtype=torch.float32)
        text_embeddings = torch.tensor(np.array(all_text_embeddings), dtype=torch.float32)
        labels = torch.tensor(all_labels, dtype=torch.long)
        
        logger.info(f"Extracted {len(image_embeddings)} embeddings")
        return image_embeddings, text_embeddings, labels
    
    def create_embedding_data_loader(self, image_embeddings: torch.Tensor, 
                                   text_embeddings: torch.Tensor, 
                                   labels: torch.Tensor, 
                                   batch_size: int = 32, 
                                   shuffle: bool = True) -> DataLoader:
        dataset = torch.utils.data.TensorDataset(image_embeddings, text_embeddings, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        logger.info("Starting training...")
        
        logger.info("Extracting training embeddings...")
        train_image_embeddings, train_text_embeddings, train_labels = self.extract_embeddings(train_loader)
        
        logger.info("Extracting validation embeddings...")
        val_image_embeddings, val_text_embeddings, val_labels = self.extract_embeddings(val_loader)
        
        train_embedding_loader = self.create_embedding_data_loader(
            train_image_embeddings, train_text_embeddings, train_labels,
            batch_size=self.config.get('batch_size', 32), shuffle=True
        )
        
        val_embedding_loader = self.create_embedding_data_loader(
            val_image_embeddings, val_text_embeddings, val_labels,
            batch_size=self.config.get('batch_size', 32), shuffle=False
        )
        
        history = self.trainer.train(
            train_loader=train_embedding_loader,
            val_loader=val_embedding_loader,
            num_epochs=self.config.get('num_epochs', 50),
            learning_rate=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        return history
    
    def evaluate(self, test_loader: DataLoader) -> dict:
        logger.info("Evaluating model...")
        
        test_image_embeddings, test_text_embeddings, test_labels = self.extract_embeddings(test_loader)
        
        test_embedding_loader = self.create_embedding_data_loader(
            test_image_embeddings, test_text_embeddings, test_labels,
            batch_size=self.config.get('batch_size', 32), shuffle=False
        )
        
        metrics = self.trainer.evaluate(test_embedding_loader)
        
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test Precision: {metrics['precision']:.4f}")
        logger.info(f"Test Recall: {metrics['recall']:.4f}")
        logger.info(f"Test F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def save_model(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, 'multimodal_sentiment_model.pth')
        torch.save(self.model.state_dict(), model_path)
        
        config_path = os.path.join(save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save training history if available
        if hasattr(self.trainer, 'train_losses'):
            history = {
                'train_losses': self.trainer.train_losses,
                'val_losses': self.trainer.val_losses,
                'train_accuracies': self.trainer.train_accuracies,
                'val_accuracies': self.trainer.val_accuracies
            }
            history_path = os.path.join(save_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
        
        logger.info(f"Model saved to {save_dir}")


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def create_default_config() -> dict:
    return {
        'clip_model': 'openai/clip-vit-base-patch32',
        'bert_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'max_text_length': 512,
        'num_classes': 3,
        'fusion_method': 'attention',
        'hidden_dim': 256,
        'dropout': 0.3,
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'data_dir': 'data',
        'output_dir': 'models'
    }


def main():
    parser = argparse.ArgumentParser(description='Train Multimodal Sentiment Analysis Model')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='models', help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--create_sample_data', action='store_true', help='Create sample data for testing')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    config['data_dir'] = args.data_dir
    config['output_dir'] = args.output_dir
    config['num_epochs'] = args.num_epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    
    logger.info("Starting multimodal sentiment analysis training")
    logger.info(f"Configuration: {config}")
    
    try:
        data_processor = create_data_processor(config['data_dir'])
        
        # Load or create data
        if args.create_sample_data or not os.path.exists(os.path.join(config['data_dir'], 'train.csv')):
            logger.info("Creating sample data...")
            df = data_processor.create_sample_data(num_samples=1000)
            train_df, val_df, test_df = data_processor.split_data(df)
            data_processor.save_processed_data(train_df, val_df, test_df, config['data_dir'])
        else:
            logger.info("Loading existing data...")
            train_df = pd.read_csv(os.path.join(config['data_dir'], 'train.csv'))
            val_df = pd.read_csv(os.path.join(config['data_dir'], 'val.csv'))
            test_df = pd.read_csv(os.path.join(config['data_dir'], 'test.csv'))
        
        train_loader, val_loader, test_loader = data_processor.create_data_loaders(
            train_df, val_df, test_df, 
            batch_size=config['batch_size'],
            num_workers=0  # Disable multiprocessing to avoid PIL Image issues
        )
        
        pipeline = MultimodalTrainingPipeline(config)
        history = pipeline.train(train_loader, val_loader)
        metrics = pipeline.evaluate(test_loader)
        
        # Save the model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(config['output_dir'], f'model_{timestamp}')
        pipeline.save_model(save_dir)
        
        # Save evaluation metrics
        metrics_path = os.path.join(save_dir, 'evaluation_metrics.json')
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            else:
                serializable_metrics[key] = value
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        # Plot training history
        pipeline.trainer.plot_training_history(
            save_path=os.path.join(save_dir, 'training_history.png')
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {save_dir}")
        logger.info(f"Final test accuracy: {metrics['accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
