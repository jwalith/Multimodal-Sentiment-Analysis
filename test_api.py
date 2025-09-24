"""
Test script for the Multimodal Sentiment Analysis API
"""

import requests
import json

def test_prediction():
    """Test the prediction endpoint"""
    url = 'http://127.0.0.1:5000/predict'
    
    # Test data
    data = {
        'image_url': 'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=400',
        'text': 'This sunset is absolutely breathtaking! I love it!'
    }
    
    try:
        print("🚀 Testing prediction endpoint...")
        print(f"📝 Text: {data['text']}")
        print(f"🖼️ Image: {data['image_url']}")
        print("-" * 50)
        
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"🎯 Predicted sentiment: {result['result']['predicted_sentiment']}")
            print(f"📊 Confidence: {result['result']['confidence']:.4f}")
            print(f"📈 All probabilities:")
            for sentiment, prob in result['result']['class_probabilities'].items():
                print(f"   {sentiment}: {prob:.4f}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_model_info():
    """Test the model info endpoint"""
    url = 'http://127.0.0.1:5000/model_info'
    
    try:
        print("\n🔍 Testing model info endpoint...")
        response = requests.get(url)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Model info retrieved successfully!")
            print(f"📋 Model config: {json.dumps(result['model_config'], indent=2)}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_prediction()
    test_model_info()
