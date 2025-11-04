import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class YOLOGradCAM:
    """
    Grad-CAM implementation for YOLO models
    """
    def __init__(self, model, target_layer=None):
        """
        Args:
            model: YOLO model (loaded via ultralytics)
            target_layer: Target layer for Grad-CAM (default: last conv layer)
        """
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Get the model's underlying PyTorch model
        if hasattr(model, 'model'):
            self.pytorch_model = model.model
        else:
            self.pytorch_model = model
            
        # Find target layer (last convolutional layer by default)
        if target_layer is None:
            self.target_layer = self._find_target_layer()
        else:
            self.target_layer = target_layer
            
        # Register hooks
        self._register_hooks()
    
    def _find_target_layer(self):
        """Find the last convolutional layer in the model"""
        # For YOLOv8, typically the last conv layer before detection head
        # This may need adjustment based on your specific YOLO version
        for name, module in self.pytorch_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target = module
        return target
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, image_path, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            image_path: Path to input image
            target_class: Target class index (if None, uses predicted class)
            
        Returns:
            cam: Grad-CAM heatmap (numpy array)
            prediction: Model prediction
        """
        # Load and preprocess image
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model(image_path, verbose=False)
        
        # Get prediction
        if len(results) > 0 and len(results[0].boxes) > 0:
            prediction = int(results[0].boxes.cls[0].item())
        else:
            prediction = None
            return None, None
        
        # Use predicted class if target_class not specified
        if target_class is None:
            target_class = prediction
        
        # Get model output for backprop
        # This requires accessing the raw model output before NMS
        self.pytorch_model.eval()
        
        # Prepare input tensor
        from ultralytics.data.augment import LetterBox
        transform = LetterBox(self.model.overrides.get('imgsz', 640))
        img_transformed = transform(image=img_rgb)
        img_tensor = torch.from_numpy(img_transformed).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(next(self.pytorch_model.parameters()).device)
        
        # Forward pass
        output = self.pytorch_model(img_tensor)
        
        # Get the output corresponding to target class
        # This depends on YOLO architecture - adjust as needed
        if isinstance(output, (list, tuple)):
            output = output[0]
        
        # Calculate gradients
        self.pytorch_model.zero_grad()
        
        # For classification, use the class score
        # You may need to adjust this based on your YOLO output format
        class_score = output[0, target_class] if output.dim() > 1 else output[target_class]
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling on gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize to input image size
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
        
        return cam, prediction
    
    def visualize_cam(self, image_path, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Create visualization of Grad-CAM overlay on original image
        
        Args:
            image_path: Path to original image
            cam: Grad-CAM heatmap
            alpha: Transparency of overlay
            colormap: OpenCV colormap
            
        Returns:
            overlay: Image with Grad-CAM overlay
            heatmap: Colored heatmap
        """
        # Load original image
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply colormap to CAM
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)
        
        return overlay, heatmap


def plot_gradcam_results(original_img, overlay_img, heatmap, singh_grade):
    """
    Create beautiful Grad-CAM visualization for Streamlit
    
    Args:
        original_img: Original X-ray image (numpy array)
        overlay_img: Image with Grad-CAM overlay
        heatmap: Grad-CAM heatmap
        singh_grade: Singh Index grade
        
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='none')
    fig.patch.set_alpha(0)
    
    # Original image
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Original X-Ray', fontsize=14, fontweight='bold', 
                      color='white', pad=15)
    axes[0].axis('off')
    
    # Grad-CAM overlay
    axes[1].imshow(overlay_img)
    axes[1].set_title(f'Grad-CAM Overlay\nSingh Grade: {singh_grade}', 
                     fontsize=14, fontweight='bold', color='white', pad=15)
    axes[1].axis('off')
    
    # Heatmap only
    axes[2].imshow(heatmap)
    axes[2].set_title('Activation Heatmap', fontsize=14, fontweight='bold', 
                     color='white', pad=15)
    axes[2].axis('off')
    
    # Add colorbar for heatmap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    
    # Create custom colormap
    colors = ['#0000ff', '#00ffff', '#00ff00', '#ffff00', '#ff0000']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('gradcam', colors, N=n_bins)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Activation', rotation=270, labelpad=20, color='white', fontsize=11)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    plt.tight_layout()
    return fig


# Helper function to integrate with existing pipeline
def generate_gradcam_for_prediction(model_path, image_path, output_dir=None):
    """
    Generate Grad-CAM visualization for a YOLO prediction
    
    Args:
        model_path: Path to YOLO model weights
        image_path: Path to input image
        output_dir: Directory to save outputs (optional)
        
    Returns:
        cam: Grad-CAM heatmap
        overlay: Overlay image
        heatmap: Colored heatmap
        prediction: Model prediction
    """
    from ultralytics import YOLO
    
    # Load model
    print("Pass")
    model = YOLO(model_path)
    print("Model Loaded")
    # Initialize Grad-CAM
    gradcam = YOLOGradCAM(model)
    print("GradCAM Initialized")
    
    # Generate CAM
    cam, prediction = gradcam.generate_cam(image_path)
    print("Generated parsed")
    
    if cam is None:
        print("No CAM generated")
        return None, None, None, None
    print("CAM Generated")
    
    # Create visualization
    overlay, heatmap = gradcam.visualize_cam(image_path, cam)
    print("Visualization Created")
    
    # Save if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        img_name = Path(image_path).stem
        cv2.imwrite(str(output_dir / f"{img_name}_gradcam_overlay.png"), 
                   cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_dir / f"{img_name}_gradcam_heatmap.png"), 
                   cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        print("Saved outputs")
    return cam, overlay, heatmap, prediction
