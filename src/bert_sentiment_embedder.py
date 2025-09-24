import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import numpy as np
from typing import Union, List, Optional, Dict
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTSentimentEmbedder:
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest", 
                 device: str = "auto", max_length: int = 512):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.max_length = max_length
        
        logger.info(f"Loading BERT model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        try:
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.sentiment_model.to(self.device)
            self.has_sentiment_head = True
        except:
            logger.warning("Could not load sentiment classification head, using base model only")
            self.sentiment_model = None
            self.has_sentiment_head = False
        
        self.model.to(self.device)
        self.model.eval()
        
        if self.has_sentiment_head:
            self.sentiment_model.eval()
        
        logger.info(f"BERT model loaded successfully on {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the appropriate device to use."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better sentiment analysis.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle common social media patterns
        text = re.sub(r'@\w+', '[USER]', text)  # Replace mentions
        text = re.sub(r'http\S+', '[URL]', text)  # Replace URLs
        text = re.sub(r'#\w+', '[HASHTAG]', text)  # Replace hashtags
        
        return text
    
    def extract_embedding(self, text: str, pooling_strategy: str = "cls") -> np.ndarray:
        """
        Extract BERT embedding from a single text.
        
        Args:
            text: Input text
            pooling_strategy: Strategy for pooling token embeddings ('cls', 'mean', 'max')
            
        Returns:
            np.ndarray: BERT text embedding
        """
        # Preprocess text
        text = self._preprocess_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            
            # Apply pooling strategy
            if pooling_strategy == "cls":
                embedding = last_hidden_states[:, 0, :]  # CLS token
            elif pooling_strategy == "mean":
                attention_mask = inputs["attention_mask"]
                embedding = (last_hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            elif pooling_strategy == "max":
                embedding = last_hidden_states.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        
        return embedding.cpu().numpy().flatten()
    
    def extract_batch_embeddings(self, texts: List[str], batch_size: int = 32, 
                                pooling_strategy: str = "cls") -> np.ndarray:
        """
        Extract BERT embeddings from a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            pooling_strategy: Strategy for pooling token embeddings
            
        Returns:
            np.ndarray: Array of BERT embeddings (n_texts, embedding_dim)
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Preprocess texts
            batch_texts = [self._preprocess_text(text) for text in batch_texts]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=self.max_length,
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                
                # Apply pooling strategy
                if pooling_strategy == "cls":
                    batch_embeddings = last_hidden_states[:, 0, :]
                elif pooling_strategy == "mean":
                    attention_mask = inputs["attention_mask"]
                    batch_embeddings = (last_hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
                elif pooling_strategy == "max":
                    batch_embeddings = last_hidden_states.max(dim=1)[0]
                else:
                    raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
            
            embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def get_sentiment_scores(self, text: str) -> Dict[str, float]:
        """
        Get sentiment scores for the text using the sentiment classification head.
        
        Args:
            text: Input text
            
        Returns:
            Dict[str, float]: Dictionary with sentiment scores
        """
        if not self.has_sentiment_head:
            logger.warning("Sentiment classification head not available")
            return {}
        
        # Preprocess text
        text = self._preprocess_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get sentiment predictions
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Map to sentiment labels (this depends on the specific model)
        # For twitter-roberta-base-sentiment-latest: LABEL_0=negative, LABEL_1=neutral, LABEL_2=positive
        sentiment_labels = ["negative", "neutral", "positive"]
        sentiment_scores = {}
        
        for i, label in enumerate(sentiment_labels):
            sentiment_scores[label] = float(probabilities[0][i])
        
        return sentiment_scores
    
    def extract_sentiment_aware_embedding(self, text: str, pooling_strategy: str = "cls") -> np.ndarray:
        """
        Extract embedding that combines BERT features with sentiment information.
        
        Args:
            text: Input text
            pooling_strategy: Strategy for pooling token embeddings
            
        Returns:
            np.ndarray: Combined embedding with sentiment information
        """
        # Get base BERT embedding
        base_embedding = self.extract_embedding(text, pooling_strategy)
        
        # Get sentiment scores
        sentiment_scores = self.get_sentiment_scores(text)
        
        if sentiment_scores:
            # Convert sentiment scores to array
            sentiment_array = np.array([
                sentiment_scores.get("negative", 0.0),
                sentiment_scores.get("neutral", 0.0),
                sentiment_scores.get("positive", 0.0)
            ])
            
            # Concatenate base embedding with sentiment scores
            combined_embedding = np.concatenate([base_embedding, sentiment_array])
        else:
            # If no sentiment scores available, just return base embedding
            combined_embedding = base_embedding
        
        return combined_embedding
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the BERT embeddings."""
        # Create a dummy text to get the embedding dimension
        dummy_text = "This is a test sentence."
        embedding = self.extract_embedding(dummy_text)
        return len(embedding)
    
    def get_sentiment_aware_embedding_dimension(self) -> int:
        """Get the dimension of the sentiment-aware embeddings."""
        dummy_text = "This is a test sentence."
        embedding = self.extract_sentiment_aware_embedding(dummy_text)
        return len(embedding)


def create_bert_sentiment_embedder(model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest", 
                                  device: str = "auto", max_length: int = 512) -> BERTSentimentEmbedder:
    """
    Factory function to create a BERT sentiment embedder instance.
    
    Args:
        model_name: Name of the BERT model
        device: Device to use
        max_length: Maximum sequence length
        
    Returns:
        BERTSentimentEmbedder: Configured BERT sentiment embedder instance
    """
    return BERTSentimentEmbedder(model_name=model_name, device=device, max_length=max_length)


if __name__ == "__main__":
    # Example usage
    embedder = create_bert_sentiment_embedder()
    
    # Test with sample texts
    sample_texts = [
        "I love this movie! It's amazing!",
        "This is terrible, I hate it.",
        "It's okay, nothing special.",
        "OMG this is the best thing ever!!! 😍"
    ]
    
    print(f"BERT embedding dimension: {embedder.get_embedding_dimension()}")
    print(f"Sentiment-aware embedding dimension: {embedder.get_sentiment_aware_embedding_dimension()}")
    
    for text in sample_texts:
        print(f"\nText: {text}")
        sentiment_scores = embedder.get_sentiment_scores(text)
        print(f"Sentiment scores: {sentiment_scores}")
        
        embedding = embedder.extract_embedding(text)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding (first 5 values): {embedding[:5]}")
