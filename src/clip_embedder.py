import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import Union, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIPImageEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "auto"):
        self.model_name = model_name
        self.device = self._get_device(device)
        
        logger.info(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"CLIP model loaded successfully on {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def extract_embedding(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().flatten()
    
    def extract_batch_embeddings(self, images: List[Union[str, Image.Image, np.ndarray]], 
                                batch_size: int = 32) -> np.ndarray:
        """
        Extract CLIP embeddings from a batch of images.
        
        Args:
            images: List of images (paths, PIL Images, or numpy arrays)
            batch_size: Batch size for processing
            
        Returns:
            np.ndarray: Array of CLIP embeddings (n_images, embedding_dim)
        """
        embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Convert batch to PIL Images
            pil_images = []
            for img in batch_images:
                if isinstance(img, str):
                    pil_images.append(Image.open(img).convert('RGB'))
                elif isinstance(img, np.ndarray):
                    pil_images.append(Image.fromarray(img).convert('RGB'))
                else:
                    pil_images.append(img.convert('RGB'))
            
            # Process batch
            inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            embeddings.append(image_features.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the CLIP embeddings."""
        # Create a dummy image to get the embedding dimension
        dummy_image = Image.new('RGB', (224, 224), color='white')
        embedding = self.extract_embedding(dummy_image)
        return len(embedding)
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text using CLIP's text encoder.
        This can be useful for creating text-image similarity scores.
        
        Args:
            text: Input text to encode
            
        Returns:
            np.ndarray: CLIP text embedding
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy().flatten()
    
    def compute_image_text_similarity(self, image: Union[str, Image.Image, np.ndarray], 
                                    text: str) -> float:
        """
        Compute similarity between an image and text using CLIP.
        
        Args:
            image: Input image
            text: Input text
            
        Returns:
            float: Cosine similarity score between image and text embeddings
        """
        image_embedding = self.extract_embedding(image)
        text_embedding = self.encode_text(text)
        
        # Compute cosine similarity
        similarity = np.dot(image_embedding, text_embedding) / (
            np.linalg.norm(image_embedding) * np.linalg.norm(text_embedding)
        )
        
        return float(similarity)


def create_clip_embedder(model_name: str = "openai/clip-vit-base-patch32", 
                        device: str = "auto") -> CLIPImageEmbedder:
    """
    Factory function to create a CLIP embedder instance.
    
    Args:
        model_name: Name of the CLIP model
        device: Device to use
        
    Returns:
        CLIPImageEmbedder: Configured CLIP embedder instance
    """
    return CLIPImageEmbedder(model_name=model_name, device=device)


if __name__ == "__main__":
    # Example usage
    embedder = create_clip_embedder()
    
    # Test with a sample image (you would replace this with actual image paths)
    print(f"CLIP embedding dimension: {embedder.get_embedding_dimension()}")
    
    # Example of creating a dummy image for testing
    dummy_image = Image.new('RGB', (224, 224), color='red')
    embedding = embedder.extract_embedding(dummy_image)
    print(f"Sample embedding shape: {embedding.shape}")
    print(f"Sample embedding (first 5 values): {embedding[:5]}")
