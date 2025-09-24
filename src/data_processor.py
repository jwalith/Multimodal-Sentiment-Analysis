import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def custom_collate_fn(batch):
    images, texts, labels = zip(*batch)
    
    # Convert string labels to integers
    if isinstance(labels[0], str):
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        labels = [label_map[label] for label in labels]
    
    return list(images), list(texts), torch.tensor(labels, dtype=torch.long)


class MultimodalDataset(Dataset):
    def __init__(self, image_paths: List[str], texts: List[str], labels: List[int], 
                 image_transform=None, text_transform=None):
        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels
        self.image_transform = image_transform
        self.text_transform = text_transform
        
        assert len(image_paths) == len(texts) == len(labels), "All lists must have the same length"
        logger.info(f"Created dataset with {len(self.image_paths)} samples")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.image_transform:
                image = self.image_transform(image)
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='white')
            if self.image_transform:
                image = self.image_transform(image)
        
        text = self.texts[idx]
        if self.text_transform:
            text = self.text_transform(text)
        
        return image, text, self.labels[idx]


class MultimodalDataProcessor:
    """
    Data processor for multimodal sentiment analysis.
    
    This class handles data loading, preprocessing, and preparation for training.
    """
    
    def __init__(self, data_dir: str, image_size: Tuple[int, int] = (224, 224)):
        self.data_dir = data_dir
        self.image_size = image_size
        self.label_encoder = LabelEncoder()
        
        self.sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.label_to_id = {'negative': 0, 'neutral': 1, 'positive': 2}
        
        logger.info(f"Initialized data processor for directory: {data_dir}")
    
    def load_data_from_csv(self, csv_path: str, image_column: str = 'image_path', 
                          text_column: str = 'text', label_column: str = 'sentiment') -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            image_column: Name of the column containing image paths
            text_column: Name of the column containing text
            label_column: Name of the column containing sentiment labels
            
        Returns:
            pd.DataFrame: Loaded data
        """
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_columns = [image_column, text_column, label_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert sentiment labels to numeric if needed
        if df[label_column].dtype == 'object':
            df[label_column] = df[label_column].map(self.label_to_id)
        
        logger.info(f"Loaded {len(df)} samples from {csv_path}")
        return df
    
    def load_data_from_json(self, json_path: str) -> pd.DataFrame:
        """
        Load data from a JSON file.
        
        Expected JSON format:
        [
            {
                "image_path": "path/to/image.jpg",
                "text": "Caption or description",
                "sentiment": "positive"  # or "negative", "neutral"
            },
            ...
        ]
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # Convert sentiment labels to numeric
        df['sentiment'] = df['sentiment'].map(self.label_to_id)
        
        logger.info(f"Loaded {len(df)} samples from {json_path}")
        return df
    
    def create_sample_data(self, num_samples: int = 100) -> pd.DataFrame:
        """
        Create sample data for testing purposes.
        
        Args:
            num_samples: Number of sample data points to create
            
        Returns:
            pd.DataFrame: Sample data
        """
        # Sample texts with different sentiments
        sample_texts = {
            'positive': [
                "This is amazing! I love it!",
                "Best thing ever! So happy!",
                "Incredible! This made my day!",
                "Fantastic! Highly recommended!",
                "Wonderful! Can't get enough!"
            ],
            'negative': [
                "This is terrible! I hate it!",
                "Worst experience ever!",
                "Completely disappointed!",
                "Awful! Don't waste your time!",
                "Horrible! Regret buying this!"
            ],
            'neutral': [
                "It's okay, nothing special.",
                "Average quality, as expected.",
                "Not bad, but not great either.",
                "It works, but could be better.",
                "Standard product, meets expectations."
            ]
        }
        
        # Create sample data
        data = []
        for i in range(num_samples):
            sentiment = np.random.choice(['positive', 'negative', 'neutral'])
            text = np.random.choice(sample_texts[sentiment])
            
            # Create a dummy image path (you would replace this with actual image paths)
            image_path = f"sample_image_{i:04d}.jpg"
            
            data.append({
                'image_path': image_path,
                'text': text,
                'sentiment': sentiment
            })
        
        df = pd.DataFrame(data)
        df['sentiment'] = df['sentiment'].map(self.label_to_id)
        
        logger.info(f"Created {len(df)} sample data points")
        return df
    
    def preprocess_images(self, image_paths: List[str]) -> List[np.ndarray]:
        """
        Preprocess images for training.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List[np.ndarray]: Preprocessed images
        """
        processed_images = []
        
        for image_path in image_paths:
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Could not load image: {image_path}")
                    # Create a blank image as fallback
                    image = np.zeros((*self.image_size[::-1], 3), dtype=np.uint8)
                else:
                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Resize image
                    image = cv2.resize(image, self.image_size)
                
                processed_images.append(image)
                
            except Exception as e:
                logger.warning(f"Error processing image {image_path}: {e}")
                # Create a blank image as fallback
                image = np.zeros((*self.image_size[::-1], 3), dtype=np.uint8)
                processed_images.append(image)
        
        return processed_images
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess texts for training.
        
        Args:
            texts: List of texts
            
        Returns:
            List[str]: Preprocessed texts
        """
        processed_texts = []
        
        for text in texts:
            # Basic text preprocessing
            text = str(text).strip()
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Handle empty texts
            if not text:
                text = "No text provided"
            
            processed_texts.append(text)
        
        return processed_texts
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input dataframe
            test_size: Proportion of data for test set
            val_size: Proportion of data for validation set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test dataframes
        """
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, 
            stratify=df['sentiment']
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size_adjusted, random_state=random_state,
            stratify=train_val_df['sentiment']
        )
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def create_data_loaders(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           test_df: pd.DataFrame, batch_size: int = 32, 
                           num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch data loaders for train, validation, and test sets.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes for data loading
            
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders
        """
        # Create datasets
        train_dataset = MultimodalDataset(
            train_df['image_path'].tolist(),
            train_df['text'].tolist(),
            train_df['sentiment'].tolist()
        )
        
        val_dataset = MultimodalDataset(
            val_df['image_path'].tolist(),
            val_df['text'].tolist(),
            val_df['sentiment'].tolist()
        )
        
        test_dataset = MultimodalDataset(
            test_df['image_path'].tolist(),
            test_df['text'].tolist(),
            test_df['sentiment'].tolist()
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            collate_fn=custom_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            collate_fn=custom_collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            collate_fn=custom_collate_fn
        )
        
        logger.info(f"Created data loaders with batch size {batch_size}")
        
        return train_loader, val_loader, test_loader
    
    def get_class_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Get the distribution of classes in the dataset.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dict[str, int]: Class distribution
        """
        distribution = df['sentiment'].value_counts().to_dict()
        
        # Convert numeric labels to string labels
        label_distribution = {}
        for label_id, count in distribution.items():
            label_name = self.sentiment_labels[label_id]
            label_distribution[label_name] = count
        
        return label_distribution
    
    def save_processed_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           test_df: pd.DataFrame, output_dir: str):
        """
        Save processed data to files.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        
        logger.info(f"Saved processed data to {output_dir}")


def create_data_processor(data_dir: str, image_size: Tuple[int, int] = (224, 224)) -> MultimodalDataProcessor:
    """
    Factory function to create a data processor.
    
    Args:
        data_dir: Directory containing the dataset
        image_size: Target size for images
        
    Returns:
        MultimodalDataProcessor: Configured data processor
    """
    return MultimodalDataProcessor(data_dir=data_dir, image_size=image_size)


if __name__ == "__main__":
    # Example usage
    processor = create_data_processor("data")
    
    # Create sample data for testing
    sample_df = processor.create_sample_data(num_samples=50)
    
    # Check class distribution
    distribution = processor.get_class_distribution(sample_df)
    print("Class distribution:", distribution)
    
    # Split data
    train_df, val_df, test_df = processor.split_data(sample_df)
    
    # Create data loaders
    train_loader, val_loader, test_loader = processor.create_data_loaders(
        train_df, val_df, test_df, batch_size=8
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test data loading
    for batch_idx, (images, texts, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Images shape: {images.shape}, Labels: {labels}")
        if batch_idx >= 2:  # Only show first few batches
            break
