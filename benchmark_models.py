import os
import logging
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Optional
from base_vpr import BaseVPR

logger = logging.getLogger(__name__)


class Cosplace(BaseVPR):
    """
    CosPlace VPR model loader.
    Loads a pre-trained CosPlace model from PyTorch Hub (gmberton/cosplace).
    Exposes the BaseVPR API for feature extraction.
    """

    def __init__(self, backbone='ResNet50', fc_output_dim=512):
        """
        Args:
            backbone: Backbone architecture ('ResNet18', 'ResNet50', 'ResNet101', 'ResNet152', 'VGG16').
            fc_output_dim: Output descriptor dimensionality (e.g. 512, 2048).
        """
        super().__init__(backbone, fc_output_dim)

    def load_model(self):
        """Load the CosPlace model from PyTorch Hub."""
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            self.model = torch.hub.load(
                "gmberton/cosplace",
                "get_trained_model",
                backbone=self.backbone,
                fc_output_dim=self.fc_output_dim
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            setattr(self.model, 'device', self.device)
            logger.info("CosPlace model loaded successfully")
            return self.model
        except Exception as e:
            raise RuntimeError(f"Failed to load CosPlace model: {e}")

    @staticmethod
    def get_cosplace_dimensions_quick(model_name="ResNet50"):
        """Quick lookup for standard CosPlace model output dimensions."""
        standard_dims = {
            "ResNet18": 512,
            "ResNet50": 2048,
            "ResNet101": 2048,
            "ResNet152": 2048,
            "VGG16": 512,
            "EfficientNet-B0": 1280,
            "EfficientNet-B7": 2560,
        }
        return standard_dims.get(model_name, None)
