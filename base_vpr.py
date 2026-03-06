import os
import logging
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseVPR(ABC):
    """
    Abstract base class for Visual Place Recognition (VPR) models.
    Provides common functionality for model loading, image transformation, and feature extraction.
    """
    
    def __init__(self, backbone='ResNet50', fc_output_dim=2048):
        """
        Initialize the VPR model.
        
        Args:
            backbone (str): The backbone architecture to use (e.g., 'ResNet50')
            fc_output_dim (int): The dimensionality of the output descriptors
        """
        self.backbone = backbone
        self.fc_output_dim = fc_output_dim
        self.model = None
        self.transform = None
        self.model_type = self.__class__.__name__  # Automatically detect model type
        
        # # Respect subclass-provided fc_output_dim; only infer when not provided
        # self._set_correct_fc_output_dim()
    
    # def _set_correct_fc_output_dim(self):
    #     """
    #     If a subclass did not provide fc_output_dim, infer a sensible default
    #     based on model type and backbone/descriptor settings.
    #     """
    #     # Only compute a default when not provided by the concrete model class
    #     if self.fc_output_dim is None:
    #         if self.model_type == 'Cosplace':
    #             # CosPlace uses backbone-specific dimensions
    #             self.fc_output_dim = self._get_cosplace_dimensions(self.backbone)
    #         elif self.model_type == 'EigenPlaces':
    #             # EigenPlaces uses backbone-specific dimensions
    #             self.fc_output_dim = self._get_eigenplaces_dimensions(self.backbone)
    #         elif self.model_type == 'NetVLAD':
    #             # NetVLAD uses num_clusters * encoder_dim (64 * 512 = 32768)
    #             num_clusters = getattr(self, 'num_clusters', 64)
    #             encoder_dim = 512  # VGG16 encoder dimension
    #             self.fc_output_dim = num_clusters * encoder_dim
    #         elif self.model_type == 'PatchNetVLAD':
    #             # PatchNetVLAD uses descriptor_dim parameter
    #             self.fc_output_dim = getattr(self, 'descriptor_dim', 512)
    #         # For other models, keep fc_output_dim as None until resolved later
        
    def _get_cosplace_dimensions(self, backbone):
        """Get CosPlace dimensions based on backbone."""
        cosplace_dims = {
            "ResNet18": 512,
            "ResNet50": 2048,
            "ResNet101": 2048,
            "ResNet152": 2048,
            "VGG16": 512,
            "EfficientNet-B0": 1280,
            "EfficientNet-B7": 2560
        }
        return cosplace_dims.get(backbone, 2048)  # Default to 2048
    
    def _get_eigenplaces_dimensions(self, backbone):
        """Get EigenPlaces dimensions based on backbone."""
        eigenplaces_dims = {
            "ResNet18": 512,
            "ResNet50": 2048,
            "ResNet101": 2048,
            "ResNet152": 2048,
            "VGG16": 512
        }
        return eigenplaces_dims.get(backbone, 2048)  # Default to 2048
        
    @abstractmethod
    def load_model(self):
        """
        Load the VPR model from PyTorch Hub.
        
        Returns:
            model: The loaded VPR model in evaluation mode
        """
        pass

    def setup_image_transform(self):
        """
        Create model-specific image transformation pipeline.
        Automatically applies the correct resize transformation based on model type.
        
        Model-specific resize requirements:
        - CosPlace: Resize(512, antialias=True) - resize shorter side to 512, maintain aspect ratio
        - NetVLAD: Resize((480, 640)) - fixed size resize
        - EigenPlaces: Resize((224, 224)) ??
        - PatchNetVLAD: Resize((480, 640)) - fixed size resize
        - SALAD: Resize((322, 322)) - fixed size resize (optimal performance size from paper)
        """
        # Model-specific resize configurations
        resize_configs = {
            'Cosplace': transforms.Resize(512, antialias=True),
            'EigenPlaces': transforms.Resize(512, antialias=True),
            'NetVLAD': transforms.Resize((480, 640)),
            'PatchNetVLAD': transforms.Resize((480, 640)),
            'MixVPR': transforms.Resize((320, 320)),
            'Salad': transforms.Resize((322, 322), interpolation=transforms.InterpolationMode.BILINEAR),
            'Salad512': transforms.Resize((322, 322), interpolation=transforms.InterpolationMode.BILINEAR),
            'CricaVPR': transforms.Resize((518, 518))
        }
        
        # Get the appropriate resize transform for this model
        try:
            resize_transform = resize_configs.get(self.model_type)
        except Exception as e:
            raise RuntimeError(f"Error getting resize transform for {self.model_type}: {e}")
        
        self.transform = transforms.Compose([
            resize_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return self.transform

    def extract_features(self, image_input, transform=None):
        """
        Enhanced feature extraction supporting both image paths and PIL Image objects.
        
        Args:
            image_input (str | PIL.Image.Image): Path to image file or PIL Image object
            transform: Image transformation pipeline
            
        Returns:
            numpy.ndarray: Extracted features as a flattened array, or None if failed
        """
        if self.model is None:
            self.load_model()
            
        if transform is None:
            transform = self.transform or self.setup_image_transform()
        
        try:
            # Handle both path and PIL Image object
            if isinstance(image_input, str):
                img = Image.open(image_input).convert('RGB')
            else:
                # Assume it's a PIL Image object
                img = image_input.convert('RGB')
            
            img_tensor = transform(img).unsqueeze(0)
            
            # Move tensor to the same device as the model
            if hasattr(self.model, 'device'):
                img_tensor = img_tensor.to(self.model.device)
            elif hasattr(self, 'device'):
                img_tensor = img_tensor.to(self.device)
            
            with torch.no_grad():
                features = self.model(img_tensor)
            
            # Move to CPU and clear GPU memory
            result = features.cpu().numpy().flatten()
            del features, img_tensor
            torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {image_input}: {e}")
            return None

    def extract_features_batch(self, image_inputs, batch_size=None, transform=None):
        """
        Batch feature extraction with optional fixed batch size and adaptive fallback.
        If batch_size is provided, it will be used as the preferred per-forward size;
        on OOM or 32-bit index errors, the chunk size is halved until successful.
        """
        if self.model is None:
            self.load_model()
        if transform is None:
            transform = self.transform or self.setup_image_transform()

        # Preprocess all inputs to tensors on CPU
        tensors = []
        for image_input in image_inputs:
            try:
                img = Image.open(image_input).convert('RGB') if isinstance(image_input, str) else image_input.convert('RGB')
                tensors.append(transform(img))
            except Exception as e:
                logger.error(f"Error loading image {image_input}: {e}")
                return None

        if not tensors:
            return None

        # Adaptive chunked forwarding
        device = getattr(self.model, 'device', getattr(self, 'device', 'cpu'))
        # Prefer user-provided batch_size; otherwise fall back to a safe default
        if batch_size is not None and isinstance(batch_size, int) and batch_size > 0:
            max_chunk = batch_size
        else:
            max_chunk = getattr(self, 'safe_batch_size', 8 if (isinstance(device, torch.device) and device.type == 'cuda') else 32)
        features_list = []
        idx = 0
        while idx < len(tensors):
            chunk_size = min(max_chunk, len(tensors) - idx)
            success = False
            while chunk_size >= 1 and not success:
                try:
                    batch_tensor = torch.stack(tensors[idx:idx+chunk_size], dim=0)
                    if isinstance(device, torch.device):
                        batch_tensor = batch_tensor.to(device)
                    with torch.no_grad():
                        out = self.model(batch_tensor)
                    features_list.append(out.detach().cpu())
                    del out, batch_tensor
                    torch.cuda.empty_cache()
                    success = True
                    idx += chunk_size
                except RuntimeError as re:
                    msg = str(re)
                    # Reduce chunk size on CUDA OOM or 32-bit index errors
                    if ('CUDA' in msg or 'cuda' in msg or 'out of memory' in msg or 'canUse32BitIndexMath' in msg):
                        torch.cuda.empty_cache()
                        chunk_size //= 2
                        if chunk_size < 1:
                            logger.error(f"Batch forwarding failed even at chunk_size=1: {re}")
                            return None
                    else:
                        logger.error(f"Error processing batch chunk: {re}")
                        return None
                except Exception as e:
                    logger.error(f"Error processing batch chunk: {e}")
                    return None

        return torch.cat(features_list, dim=0).numpy()

    def extract_features_batch_backup(self, image_inputs, transform=None):
        """
        Batch feature extraction with adaptive chunking.
        Automatically reduces per-forward batch size on CUDA errors or 32-bit index math errors.
        """
        if self.model is None:
            self.load_model()
        if transform is None:
            transform = self.transform or self.setup_image_transform()

        # Preprocess all inputs to tensors on CPU
        tensors = []
        for image_input in image_inputs:
            try:
                img = Image.open(image_input).convert('RGB') if isinstance(image_input, str) else image_input.convert('RGB')
                tensors.append(transform(img))
            except Exception as e:
                logger.error(f"Error loading image {image_input}: {e}")
                return None

        if not tensors:
            return None

        # Adaptive chunked forwarding
        device = getattr(self.model, 'device', getattr(self, 'device', 'cpu'))
        max_chunk = getattr(self, 'safe_batch_size', 8 if (isinstance(device, torch.device) and device.type == 'cuda') else 32)
        features_list = []
        idx = 0
        while idx < len(tensors):
            chunk_size = min(max_chunk, len(tensors) - idx)
            success = False
            while chunk_size >= 1 and not success:
                try:
                    batch_tensor = torch.stack(tensors[idx:idx+chunk_size], dim=0)
                    if isinstance(device, torch.device):
                        batch_tensor = batch_tensor.to(device)
                    with torch.no_grad():
                        out = self.model(batch_tensor)
                    features_list.append(out.detach().cpu())
                    del out, batch_tensor
                    torch.cuda.empty_cache()
                    success = True
                    idx += chunk_size
                except RuntimeError as re:
                    msg = str(re)
                    # Reduce chunk size on CUDA OOM or 32-bit index errors
                    if ('CUDA' in msg or 'cuda' in msg or 'out of memory' in msg or 'canUse32BitIndexMath' in msg):
                        torch.cuda.empty_cache()
                        chunk_size //= 2
                        if chunk_size < 1:
                            logger.error(f"Batch forwarding failed even at chunk_size=1: {re}")
                            return None
                    else:
                        logger.error(f"Error processing batch chunk: {re}")
                        return None
                except Exception as e:
                    logger.error(f"Error processing batch chunk: {e}")
                    return None

        return torch.cat(features_list, dim=0).numpy()

    def get_model(self):
        """Get the loaded model."""
        if self.model is None:
            self.load_model()
        return self.model
    
    def get_transform(self):
        """Get the image transform pipeline."""
        if self.transform is None:
            self.setup_image_transform()
        return self.transform 
    
    def get_feature_dimensions(self):
        """
        Get the output feature dimensions of the VPR model.
        Uses model-specific logic to determine dimensions accurately.
        
        Returns:
            int: Feature dimension size
        """
        # For most models, we can return the pre-calculated fc_output_dim
        # which was set correctly during initialization
        if hasattr(self, 'fc_output_dim') and self.fc_output_dim is not None:
            return self.fc_output_dim
        
        # Fallback: Try to determine from loaded model
        if self.model is not None:
            # Method 1: Check the model's final layer output features
            if hasattr(self.model, 'aggregation') and hasattr(self.model.aggregation, 'fc'):
                return self.model.aggregation.fc.out_features
            
            # Method 2: Check if there's a direct classifier/fc layer
            if hasattr(self.model, 'fc'):
                return self.model.fc.out_features
            
            # Method 3: Look for the last linear layer in the model
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, torch.nn.Linear):
                    return module.out_features
            
            # Method 4: Create a dummy input to get dimensions (fallback)
            try:
                dummy_input = torch.randn(1, 3, 224, 224)  # Typical input size
                with torch.no_grad():
                    dummy_output = self.model(dummy_input)
                return dummy_output.shape[-1]  # Last dimension is feature size
            except Exception as e:
                logger.warning(f"Could not determine feature dimensions from model: {e}")
        
        # Final fallback: return None if we can't determine dimensions
        logger.warning(f"Could not determine feature dimensions for {self.model_type}")
        return None
