"""
Example Usage of Multimodal Sentiment Analysis

This script demonstrates how to use the multimodal sentiment analysis system
for detecting sentiment in memes, ads, and social media posts.
"""

import sys
import os
import json
from PIL import Image
import requests
import io

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from clip_embedder import create_clip_embedder
from bert_sentiment_embedder import create_bert_sentiment_embedder
from multimodal_fusion_model import create_multimodal_model


def main():
    """Main example function."""
    print("🚀 Multimodal Sentiment Analysis Example")
    print("=" * 50)
    
    # Initialize embedders
    print("📦 Initializing embedders...")
    clip_embedder = create_clip_embedder()
    bert_embedder = create_bert_sentiment_embedder()
    
    print(f"✅ CLIP embedding dimension: {clip_embedder.get_embedding_dimension()}")
    print(f"✅ BERT embedding dimension: {bert_embedder.get_embedding_dimension()}")
    
    # Create model
    print("\n🤖 Creating multimodal model...")
    model = create_multimodal_model(
        image_dim=clip_embedder.get_embedding_dimension(),
        text_dim=bert_embedder.get_embedding_dimension(),
        num_classes=3,
        fusion_method="attention"
    )
    
    # Example 1: Test with sample data
    print("\n📝 Example 1: Testing with sample data")
    print("-" * 40)
    
    # Create sample image and text
    sample_image = Image.new('RGB', (224, 224), color='red')
    sample_text = "I love this red color! It's so vibrant and beautiful!"
    
    # Extract embeddings
    image_embedding = clip_embedder.extract_embedding(sample_image)
    text_embedding = bert_embedder.extract_embedding(sample_text)
    
    # Convert to tensors
    import torch
    image_tensor = torch.tensor(image_embedding, dtype=torch.float32).unsqueeze(0)
    text_tensor = torch.tensor(text_embedding, dtype=torch.float32).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        logits = model(image_tensor, text_tensor)
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    print(f"Image: Red colored image")
    print(f"Text: {sample_text}")
    print(f"Predicted sentiment: {sentiment_labels[predicted_class]}")
    print(f"Confidence: {probabilities[0][predicted_class].item():.4f}")
    print(f"All probabilities: {probabilities[0].numpy()}")
    
    # Get BERT sentiment scores
    bert_scores = bert_embedder.get_sentiment_scores(sample_text)
    print(f"BERT sentiment scores: {bert_scores}")
    
    # Get image-text similarity
    similarity = clip_embedder.compute_image_text_similarity(sample_image, sample_text)
    print(f"Image-text similarity: {similarity:.4f}")
    
    # Example 2: Test with real image from URL
    print("\n🌐 Example 2: Testing with real image from URL")
    print("-" * 50)
    
    try:
        # Download image from URL
        image_url = "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=400"
        response = requests.get(image_url, timeout=10)
        real_image = Image.open(io.BytesIO(response.content)).convert('RGB')
        
        real_text = "This sunset is absolutely breathtaking! I love it!"
        
        # Extract embeddings
        image_embedding = clip_embedder.extract_embedding(real_image)
        text_embedding = bert_embedder.extract_embedding(real_text)
        
        # Convert to tensors
        image_tensor = torch.tensor(image_embedding, dtype=torch.float32).unsqueeze(0)
        text_tensor = torch.tensor(text_embedding, dtype=torch.float32).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            logits = model(image_tensor, text_tensor)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
        
        print(f"Image URL: {image_url}")
        print(f"Text: {real_text}")
        print(f"Predicted sentiment: {sentiment_labels[predicted_class]}")
        print(f"Confidence: {probabilities[0][predicted_class].item():.4f}")
        
        # Get BERT sentiment scores
        bert_scores = bert_embedder.get_sentiment_scores(real_text)
        print(f"BERT sentiment scores: {bert_scores}")
        
        # Get image-text similarity
        similarity = clip_embedder.compute_image_text_similarity(real_image, real_text)
        print(f"Image-text similarity: {similarity:.4f}")
        
    except Exception as e:
        print(f"❌ Error processing real image: {e}")
    
    # Example 3: Batch processing
    print("\n📊 Example 3: Batch processing")
    print("-" * 30)
    
    # Create multiple samples
    samples = [
        {
            "image": Image.new('RGB', (224, 224), color='green'),
            "text": "This green is amazing! I love it!",
            "description": "Green image with positive text"
        },
        {
            "image": Image.new('RGB', (224, 224), color='blue'),
            "text": "This blue is terrible, I hate it!",
            "description": "Blue image with negative text"
        },
        {
            "image": Image.new('RGB', (224, 224), color='yellow'),
            "text": "It's okay, nothing special.",
            "description": "Yellow image with neutral text"
        }
    ]
    
    print("Processing batch of samples:")
    for i, sample in enumerate(samples, 1):
        print(f"\n{i}. {sample['description']}")
        
        # Extract embeddings
        image_embedding = clip_embedder.extract_embedding(sample['image'])
        text_embedding = bert_embedder.extract_embedding(sample['text'])
        
        # Convert to tensors
        image_tensor = torch.tensor(image_embedding, dtype=torch.float32).unsqueeze(0)
        text_tensor = torch.tensor(text_embedding, dtype=torch.float32).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            logits = model(image_tensor, text_tensor)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
        
        print(f"   Text: {sample['text']}")
        print(f"   Predicted sentiment: {sentiment_labels[predicted_class]}")
        print(f"   Confidence: {probabilities[0][predicted_class].item():.4f}")
    
    print("\n🎉 Example completed successfully!")
    print("\n💡 Next steps:")
    print("1. Train the model with real data: python train.py --create_sample_data")
    print("2. Run the API server: python app.py")
    print("3. Test the API endpoints with your own images and text")
    print("4. Integrate into your applications using the REST API")


if __name__ == "__main__":
    main()
