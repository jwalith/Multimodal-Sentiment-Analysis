import os
import sys
import json
import base64
import io
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from clip_embedder import create_clip_embedder
from bert_sentiment_embedder import create_bert_sentiment_embedder
from multimodal_fusion_model import create_multimodal_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

clip_embedder = None
bert_embedder = None
multimodal_model = None
model_config = None
sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}


class MultimodalSentimentPredictor:
    def __init__(self, model_path: str, config_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.clip_embedder = create_clip_embedder(
            model_name=self.config.get('clip_model', 'openai/clip-vit-base-patch32'),
            device=self.device
        )
        
        self.bert_embedder = create_bert_sentiment_embedder(
            model_name=self.config.get('bert_model', 'cardiffnlp/twitter-roberta-base-sentiment-latest'),
            device=self.device,
            max_length=self.config.get('max_text_length', 512)
        )
        
        self.model = create_multimodal_model(
            image_dim=self.clip_embedder.get_embedding_dimension(),
            text_dim=self.bert_embedder.get_embedding_dimension(),
            num_classes=self.config.get('num_classes', 3),
            fusion_method=self.config.get('fusion_method', 'attention'),
            hidden_dim=self.config.get('hidden_dim', 256),
            dropout=self.config.get('dropout', 0.3)
        )
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Predictor initialized")
    
    def predict_from_image_and_text(self, image: Image.Image, text: str) -> Dict[str, Any]:
        image_embedding = self.clip_embedder.extract_embedding(image)
        text_embedding = self.bert_embedder.extract_embedding(text)
        
        image_tensor = torch.tensor(image_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        text_tensor = torch.tensor(text_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(image_tensor, text_tensor)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
        
        bert_sentiment_scores = self.bert_embedder.get_sentiment_scores(text)
        
        result = {
            'predicted_sentiment': sentiment_labels[predicted_class],
            'confidence': float(probabilities[0][predicted_class]),
            'class_probabilities': {
                sentiment_labels[i]: float(probabilities[0][i]) 
                for i in range(len(sentiment_labels))
            },
            'bert_sentiment_scores': bert_sentiment_scores,
            'image_text_similarity': float(self.clip_embedder.compute_image_text_similarity(image, text))
        }
        
        return result
    
    def predict_from_urls(self, image_url: str, text: str) -> Dict[str, Any]:
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            return self.predict_from_image_and_text(image, text)
        except Exception as e:
            logger.error(f"Error processing image URL {image_url}: {e}")
            return {'error': f'Failed to process image URL: {str(e)}'}
    
    def predict_from_base64(self, image_base64: str, text: str) -> Dict[str, Any]:
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            return self.predict_from_image_and_text(image, text)
        except Exception as e:
            logger.error(f"Error processing base64 image: {e}")
            return {'error': f'Failed to process base64 image: {str(e)}'}


def initialize_models(model_dir: str = "models"):
    global clip_embedder, bert_embedder, multimodal_model, model_config
    
    try:
        model_dirs = [d for d in os.listdir(model_dir) if d.startswith('model_')]
        if not model_dirs:
            logger.error(f"No trained models found in {model_dir}")
            return False
        
        latest_model_dir = sorted(model_dirs)[-1]
        model_path = os.path.join(model_dir, latest_model_dir, 'multimodal_sentiment_model.pth')
        config_path = os.path.join(model_dir, latest_model_dir, 'config.json')
        
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            logger.error(f"Model files not found in {latest_model_dir}")
            return False
        
        predictor = MultimodalSentimentPredictor(model_path, config_path)
        
        clip_embedder = predictor.clip_embedder
        bert_embedder = predictor.bert_embedder
        multimodal_model = predictor.model
        model_config = predictor.config
        
        logger.info("Models initialized")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        return False


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': multimodal_model is not None
    })


@app.route('/predict', methods=['POST'])
def predict_sentiment():
    # Expected: {"image_url": "...", "text": "..."} or {"image_base64": "...", "text": "..."}
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if multimodal_model is None:
            return jsonify({'error': 'Models not loaded'}), 500
        
        # Load latest model
        model_dirs = sorted([d for d in os.listdir("models") if d.startswith('model_')])
        latest_model = model_dirs[-1]
        model_path = os.path.join("models", latest_model, 'multimodal_sentiment_model.pth')
        config_path = os.path.join("models", latest_model, 'config.json')
        
        predictor = MultimodalSentimentPredictor(model_path, config_path)
        
        # Process based on input type
        if 'image_url' in data:
            result = predictor.predict_from_urls(data['image_url'], text)
        elif 'image_base64' in data:
            result = predictor.predict_from_base64(data['image_base64'], text)
        else:
            return jsonify({'error': 'Either image_url or image_base64 is required'}), 400
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_sentiment_batch():
    # Expected: {"samples": [{"image_url": "...", "text": "..."}, ...]}
    try:
        data = request.get_json()
        
        if not data or 'samples' not in data:
            return jsonify({'error': 'No samples provided'}), 400
        
        samples = data['samples']
        if not isinstance(samples, list):
            return jsonify({'error': 'Samples must be a list'}), 400
        
        if multimodal_model is None:
            return jsonify({'error': 'Models not loaded'}), 500
        
        # Load latest model
        model_dirs = sorted([d for d in os.listdir("models") if d.startswith('model_')])
        latest_model = model_dirs[-1]
        model_path = os.path.join("models", latest_model, 'multimodal_sentiment_model.pth')
        config_path = os.path.join("models", latest_model, 'config.json')
        
        predictor = MultimodalSentimentPredictor(model_path, config_path)
        
        results = []
        for i, sample in enumerate(samples):
            try:
                text = sample.get('text', '')
                if not text:
                    results.append({'error': f'Sample {i}: Text is required'})
                    continue
                
                if 'image_url' in sample:
                    result = predictor.predict_from_urls(sample['image_url'], text)
                elif 'image_base64' in sample:
                    result = predictor.predict_from_base64(sample['image_base64'], text)
                else:
                    results.append({'error': f'Sample {i}: Either image_url or image_base64 is required'})
                    continue
                
                if 'error' in result:
                    results.append({'error': f'Sample {i}: {result["error"]}'})
                else:
                    results.append({'success': True, 'result': result})
                    
            except Exception as e:
                results.append({'error': f'Sample {i}: {str(e)}'})
        
        return jsonify({
            'success': True,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500


@app.route('/model_info', methods=['GET'])
def get_model_info():
    if model_config is None:
        return jsonify({'error': 'No model loaded'}), 500
    
    return jsonify({
        'model_config': model_config,
        'sentiment_labels': sentiment_labels,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/embed_text', methods=['POST'])
def embed_text():
    # Expected: {"text": "..."}
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        if bert_embedder is None:
            return jsonify({'error': 'BERT embedder not loaded'}), 500
        
        text = data['text']
        embedding = bert_embedder.extract_embedding(text)
        sentiment_scores = bert_embedder.get_sentiment_scores(text)
        
        return jsonify({
            'success': True,
            'embedding': embedding.tolist(),
            'sentiment_scores': sentiment_scores,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Text embedding error: {e}")
        return jsonify({'error': f'Text embedding failed: {str(e)}'}), 500


@app.route('/embed_image', methods=['POST'])
def embed_image():
    # Expected: {"image_url": "..."} or {"image_base64": "..."}
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if clip_embedder is None:
            return jsonify({'error': 'CLIP embedder not loaded'}), 500
        
        if 'image_url' in data:
            response = requests.get(data['image_url'], timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
        elif 'image_base64' in data:
            image_data = base64.b64decode(data['image_base64'])
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        else:
            return jsonify({'error': 'Either image_url or image_base64 is required'}), 400
        
        embedding = clip_embedder.extract_embedding(image)
        
        return jsonify({
            'success': True,
            'embedding': embedding.tolist(),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Image embedding error: {e}")
        return jsonify({'error': f'Image embedding failed: {str(e)}'}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


def main():
    if not initialize_models():
        logger.warning("Models not loaded. Please train a model first.")
    
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)


if __name__ == '__main__':
    main()
