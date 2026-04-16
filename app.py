import os
import json
import io
import base64
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from PIL import Image
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import torchvision.transforms as transforms

# Suppress TensorFlow/other warnings
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"Using device: {device}")


# ============================================================================
# MODEL LOADING
# ============================================================================

def get_device():
    """Get the best device available"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_classification_model():
    """Load the EfficientNet classification model"""
    try:
        from efficientnet_pytorch import EfficientNet
        
        model_path = Path('best_model.pth')
        if not model_path.exists():
            print(f"Classification model not found at {model_path}")
            return None
            
        state_dict = torch.load(str(model_path), map_location=device)
        
        # Infer model architecture from state_dict
        if "_fc.weight" in state_dict and state_dict["_fc.weight"].ndim == 2:
            in_features = int(state_dict["_fc.weight"].shape[1])
            by_fc_width = {
                1280: "efficientnet-b0", 1408: "efficientnet-b2", 1536: "efficientnet-b3",
                1792: "efficientnet-b4", 2048: "efficientnet-b5", 2304: "efficientnet-b6",
                2560: "efficientnet-b7",
            }
            model_arch = by_fc_width.get(in_features, "efficientnet-b4")
        else:
            model_arch = "efficientnet-b4"
        
        model = EfficientNet.from_name(model_arch)
        model._fc = torch.nn.Linear(model._fc.in_features, 2)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        print(f"Loaded classification model: {model_arch}")
        return model
    except Exception as e:
        print(f"Error loading classification model: {e}")
        return None


def load_segmentation_model():
    """Load the segmentation model (DeepLabV3+ or U-Net)"""
    try:
        from seg_model import create_segmentation_model
        
        arch = os.getenv("SEG_MODEL_ARCH", "deeplabv3plus").strip().lower()
        encoder = os.getenv("SEG_MODEL_ENCODER", "resnet34")
        
        model_path = f"seg_model_{arch}.pth" if arch != "unet" else "seg_model.pth"
        if not Path(model_path).exists():
            model_path = "seg_model.pth"
        
        if not Path(model_path).exists():
            print(f"Segmentation model not found")
            return None
        
        model = create_segmentation_model(
            arch=arch,
            out_channels=2,
            encoder_name=encoder,
            encoder_weights=None,
        )
        
        ckpt = torch.load(str(model_path), map_location=device)
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            elif "model_state_dict" in ckpt:
                ckpt = ckpt["model_state_dict"]
            ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        
        model.load_state_dict(ckpt)
        model = model.to(device)
        model.eval()
        
        print(f"Loaded segmentation model: {arch} with {encoder} encoder")
        return model
    except Exception as e:
        print(f"Error loading segmentation model: {e}")
        return None


device = get_device()

# Initialize models as None (will be loaded on startup)
classification_model = None
segmentation_model = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def image_to_base64(image_array):
    """Convert numpy image array to base64 string"""
    try:
        if isinstance(image_array, torch.Tensor):
            image_array = image_array.cpu().numpy()
        
        # Normalize to 0-255 range if needed
        if image_array.max() <= 1:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)
        
        # Convert to PIL Image and encode
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            pil_img = Image.fromarray(image_array, 'RGB')
        else:
            pil_img = Image.fromarray(image_array, 'L')
        
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        return img_base64
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None


def overlay_masks_on_image(original_image, disc_mask, cup_mask):
    """Create an overlay visualization of masks on the original image"""
    try:
        # Ensure images are in the right format
        if isinstance(original_image, str):
            original_image = cv2.imread(original_image)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Resize masks to match original image size
        if isinstance(disc_mask, torch.Tensor):
            disc_mask = disc_mask.cpu().numpy()
        if isinstance(cup_mask, torch.Tensor):
            cup_mask = cup_mask.cpu().numpy()
        
        # Normalize masks to 0-1 range
        if disc_mask.max() > 1:
            disc_mask = disc_mask / 255.0
        if cup_mask.max() > 1:
            cup_mask = cup_mask / 255.0
        
        h, w = original_image.shape[:2]
        disc_mask_resized = cv2.resize(disc_mask, (w, h))
        cup_mask_resized = cv2.resize(cup_mask, (w, h))
        
        # Create overlay with transparency
        overlay = original_image.copy().astype(float)
        
        # Red for disc mask (opacity 0.3)
        disc_region = disc_mask_resized > 0.5
        overlay[disc_region] = overlay[disc_region] * 0.7 + np.array([255, 0, 0]) * 0.3
        
        # Blue for cup mask (opacity 0.3)
        cup_region = cup_mask_resized > 0.5
        overlay[cup_region] = overlay[cup_region] * 0.7 + np.array([0, 0, 255]) * 0.3
        
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return overlay
    except Exception as e:
        print(f"Error creating overlay: {e}")
        return original_image


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)


@app.route('/api/classify', methods=['POST'])
def api_classify():
    """
    Classification endpoint
    Upload: POST /api/classify with image file
    Returns: {prediction, confidence, probabilities}
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Use PNG, JPG, JPEG, or BMP'}), 400
        
        if classification_model is None:
            return jsonify({'error': 'Classification model not loaded'}), 500
        
        # Save temporary file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Perform classification
        try:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            image = Image.open(filepath).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = classification_model(tensor)
                probs = torch.softmax(logits, dim=1)
                confidence, pred_idx = torch.max(probs, dim=1)
            
            classes = ["glaucoma", "normal"]
            prediction = classes[pred_idx.item()]
            conf_score = float(confidence.item())
            glaucoma_prob = float(probs[0, 0].item())
            normal_prob = float(probs[0, 1].item())
            
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': conf_score,
            'probabilities': {
                'glaucoma': glaucoma_prob,
                'normal': normal_prob
            },
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        print(f"Error in classify endpoint: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/segment', methods=['POST'])
def api_segment():
    """
    Segmentation endpoint
    Upload: POST /api/segment with image file
    Returns: {disc_mask, cup_mask, original_image, overlay}
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Use PNG, JPG, JPEG, or BMP'}), 400
        
        if segmentation_model is None:
            return jsonify({'error': 'Segmentation model not loaded'}), 500
        
        # Save temporary file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load original image
            original_img = Image.open(filepath).convert('RGB')
            original_img_np = np.array(original_img)
            
            # Perform segmentation
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            
            tensor = transform(original_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = segmentation_model(tensor)
                # Output is [B, 2, H, W] where dim 0 is disc, dim 1 is cup
                output = torch.softmax(output, dim=1)
                disc_mask = output[0, 0].cpu().numpy()
                cup_mask = output[0, 1].cpu().numpy()
            
            # Resize masks back to original size
            h, w = original_img_np.shape[:2]
            disc_mask = cv2.resize(disc_mask, (w, h))
            cup_mask = cv2.resize(cup_mask, (w, h))
            
            # Create overlay
            overlay = overlay_masks_on_image(original_img_np, disc_mask, cup_mask)
            
            # Convert to base64 for JSON response
            disc_mask_b64 = image_to_base64(disc_mask)
            cup_mask_b64 = image_to_base64(cup_mask)
            original_b64 = image_to_base64(original_img_np)
            overlay_b64 = image_to_base64(overlay)
            
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
        
        return jsonify({
            'success': True,
            'disc_mask': disc_mask_b64,
            'cup_mask': cup_mask_b64,
            'original_image': original_b64,
            'overlay': overlay_b64,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        print(f"Error in segment endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/combined', methods=['POST'])
def api_combined():
    """
    Combined diagnosis endpoint (classification + segmentation)
    Upload: POST /api/combined with image file
    Returns: {prediction, confidence, disc_mask, cup_mask, overlay}
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Use PNG, JPG, JPEG, or BMP'}), 400
        
        if classification_model is None or segmentation_model is None:
            return jsonify({'error': 'One or more models not loaded'}), 500
        
        # Save temporary file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load original image
            original_img = Image.open(filepath).convert('RGB')
            original_img_np = np.array(original_img)
            
            # === CLASSIFICATION ===
            transform_clf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            tensor_clf = transform_clf(original_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = classification_model(tensor_clf)
                probs = torch.softmax(logits, dim=1)
                confidence, pred_idx = torch.max(probs, dim=1)
            
            classes = ["glaucoma", "normal"]
            prediction = classes[pred_idx.item()]
            conf_score = float(confidence.item())
            glaucoma_prob = float(probs[0, 0].item())
            normal_prob = float(probs[0, 1].item())
            
            # === SEGMENTATION ===
            transform_seg = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            
            tensor_seg = transform_seg(original_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = segmentation_model(tensor_seg)
                output = torch.softmax(output, dim=1)
                disc_mask = output[0, 0].cpu().numpy()
                cup_mask = output[0, 1].cpu().numpy()
            
            # Resize masks back to original size
            h, w = original_img_np.shape[:2]
            disc_mask = cv2.resize(disc_mask, (w, h))
            cup_mask = cv2.resize(cup_mask, (w, h))
            
            # Create overlay
            overlay = overlay_masks_on_image(original_img_np, disc_mask, cup_mask)
            
            # Convert to base64 for JSON response
            disc_mask_b64 = image_to_base64(disc_mask)
            cup_mask_b64 = image_to_base64(cup_mask)
            original_b64 = image_to_base64(original_img_np)
            overlay_b64 = image_to_base64(overlay)
            
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': conf_score,
            'probabilities': {
                'glaucoma': glaucoma_prob,
                'normal': normal_prob
            },
            'disc_mask': disc_mask_b64,
            'cup_mask': cup_mask_b64,
            'original_image': original_b64,
            'overlay': overlay_b64,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        print(f"Error in combined endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch', methods=['POST'])
def api_batch():
    """
    Batch processing endpoint
    Upload: POST /api/batch with multiple image files
    Returns: array of results for each image
    """
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No image files provided'}), 400
        
        files = request.files.getlist('images')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        if classification_model is None or segmentation_model is None:
            return jsonify({'error': 'One or more models not loaded'}), 500
        
        results = []
        
        for file in files:
            if not allowed_file(file.filename):
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': 'Invalid file format'
                })
                continue
            
            try:
                # Save temporary file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Load original image
                original_img = Image.open(filepath).convert('RGB')
                original_img_np = np.array(original_img)
                
                # === CLASSIFICATION ===
                transform_clf = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                
                tensor_clf = transform_clf(original_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits = classification_model(tensor_clf)
                    probs = torch.softmax(logits, dim=1)
                    confidence, pred_idx = torch.max(probs, dim=1)
                
                classes = ["glaucoma", "normal"]
                prediction = classes[pred_idx.item()]
                conf_score = float(confidence.item())
                glaucoma_prob = float(probs[0, 0].item())
                normal_prob = float(probs[0, 1].item())
                
                # === SEGMENTATION ===
                transform_seg = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor()
                ])
                
                tensor_seg = transform_seg(original_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = segmentation_model(tensor_seg)
                    output = torch.softmax(output, dim=1)
                    disc_mask = output[0, 0].cpu().numpy()
                    cup_mask = output[0, 1].cpu().numpy()
                
                # Resize masks back to original size
                h, w = original_img_np.shape[:2]
                disc_mask = cv2.resize(disc_mask, (w, h))
                cup_mask = cv2.resize(cup_mask, (w, h))
                
                # Create overlay
                overlay = overlay_masks_on_image(original_img_np, disc_mask, cup_mask)
                overlay_b64 = image_to_base64(overlay)
                
                results.append({
                    'filename': file.filename,
                    'success': True,
                    'prediction': prediction,
                    'confidence': conf_score,
                    'probabilities': {
                        'glaucoma': glaucoma_prob,
                        'normal': normal_prob
                    },
                    'overlay': overlay_b64,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Clean up
                if os.path.exists(filepath):
                    os.remove(filepath)
            
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'total': len(files),
            'successful': sum(1 for r in results if r.get('success', False)),
            'results': results
        }), 200
    
    except Exception as e:
        print(f"Error in batch endpoint: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'device': str(device),
        'models_loaded': {
            'classification': classification_model is not None,
            'segmentation': segmentation_model is not None
        }
    }), 200


if __name__ == '__main__':
    port = 5001  # Using 5001 instead of 5000 to avoid macOS AirTunes conflict
    print(f"\n{'='*60}")
    print(f"Loading Models...")
    print(f"{'='*60}")
    
    # Load models on startup
    classification_model = load_classification_model()
    segmentation_model = load_segmentation_model()
    
    print(f"\n{'='*60}")
    print(f"Starting Glaucoma Detection Web App")
    print(f"{'='*60}")
    print(f"Server: http://localhost:{port}")
    print(f"Device: {device}")
    print(f"Classification Model: {'✓ Loaded' if classification_model else '✗ Not Loaded'}")
    print(f"Segmentation Model: {'✓ Loaded' if segmentation_model else '✗ Not Loaded'}")
    print(f"{'='*60}\n")
    
    app.run(debug=False, host='127.0.0.1', port=port, threaded=True)
