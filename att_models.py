import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import logging
from project_utils import llm_grid_to_attention_map, llm_coord_to_attention_map

logger = logging.getLogger(__name__)

"""
AttentionCosPlace: LLM-guided spatial attention for CosPlace descriptors.

Standard GeM (Generalized Mean Pooling) for a feature map of shape [C, H, W]:
    y_c = ( (1/(H*W)) * Σ_i Σ_j [x_{c,i,j}^p] )^(1/p)

Weighted GeM with LLM attention weights (w_{i,j} in [0,2]) and att_ratio (α in [0,1]):
    w_final_{i,j} = existing_weights + α × (attention_weights - existing_weights)

    y_c = ( ( Σ_i Σ_j [ w_final_{i,j} * x_{c,i,j}^p ] ) / ( Σ_i Σ_j w_final_{i,j} ) )^(1/p)

    - α=0: reduces to original GeM pattern (unweighted)
    - α=1: full LLM attention weighting
"""


class AttentionCosPlace(nn.Module):
    """
    CosPlace wrapper that injects LLM-guided spatial attention into the GeM pooling stage.

    The LLM attention is provided as a grid dict (e.g. {"A1": 1.2, "B3": 0.5, ...})
    describing per-region importance weights.  The model blends these weights with the
    existing GeM contribution pattern using `att_ratio` before re-pooling.
    """

    def __init__(self, model_handler=None, transform=None, backbone='ResNet50', fc_output_dim=2048):
        """
        Args:
            model_handler: Optional Cosplace instance (from benchmark_matcher). If provided,
                           the pre-loaded model and transform are reused to avoid reloading.
            transform: Optional torchvision transform; inferred from model_handler if None.
            backbone: Backbone name used when loading from Hub directly (no model_handler).
            fc_output_dim: Output descriptor dimension used when loading from Hub directly.
        """
        super().__init__()

        if model_handler is not None:
            self.cosplace = model_handler.get_model()
            self.transform = transform or model_handler.get_transform()
            self.backbone = model_handler.backbone
            self.fc_output_dim = model_handler.fc_output_dim
        else:
            self.cosplace = torch.hub.load(
                'gmberton/cosplace', 'get_trained_model',
                backbone=backbone, fc_output_dim=fc_output_dim
            )
            self.backbone = backbone
            self.fc_output_dim = fc_output_dim
            self.transform = transform or transforms.Compose([
                transforms.Resize((512), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        self.original_aggregation = self.cosplace.aggregation
        self.feature_dim = self._get_feature_dim(self.backbone)

    def _get_feature_dim(self, backbone):
        dims = {
            'ResNet18': 512, 'ResNet50': 2048, 'ResNet101': 2048, 'ResNet152': 2048,
            'VGG16': 512, 'EfficientNet-B0': 1280, 'EfficientNet-B1': 1280,
            'EfficientNet-B2': 1408, 'EfficientNet-B3': 1536, 'EfficientNet-B4': 1792,
            'EfficientNet-B5': 2048, 'EfficientNet-B6': 2304, 'EfficientNet-B7': 2560
        }
        return dims.get(backbone, 2048)

    def weighted_gem_target(self, x, attention_map=None, att_ratio=0.1):
        """
        Weighted GeM pooling blending the existing GeM contribution pattern with LLM attention.

        Mathematical formula:
            existing_weights = normalized GeM per-location contribution
            w_final = existing_weights + α × (attention_map - existing_weights)
            result_c = ( Σ_ij [w_final_ij * x_cij^p] / Σ_ij w_final_ij )^(1/p)

        Args:
            x: L2-normalized feature tensor [B, C, H, W]
            attention_map: Spatial weights [H, W] derived from the LLM grid dict
            att_ratio: Blend factor α ∈ [0, 1]

        Returns:
            Pooled tensor [B, C, 1, 1]
        """
        if attention_map is None:
            return self._standard_gem(x)

        B, C, H, W = x.shape

        if attention_map.dim() == 2:
            attention_map = attention_map.unsqueeze(0).expand(B, -1, -1)
        else:
            raise ValueError("attention_map must be 2D [H, W]")

        if attention_map.shape[1:] != (H, W):
            attention_map = F.interpolate(
                attention_map.unsqueeze(1), size=(H, W),
                mode='bilinear', align_corners=False
            ).squeeze(1)

        gem_module = self.original_aggregation[1]
        p = getattr(gem_module, 'p', torch.tensor(3.0))
        eps = getattr(gem_module, 'eps', 1e-6)

        x_p = x.clamp(min=eps).pow(p)

        num_before = x_p.sum(dim=1)
        den_before = torch.ones_like(num_before) * (H * W) + eps
        contrib_before_root = (num_before / den_before).clamp(min=eps).pow(1.0 / p)
        existing_weights = contrib_before_root / (contrib_before_root.mean(dim=(-2, -1), keepdim=True) + eps)

        w_final = existing_weights + att_ratio * (attention_map - existing_weights)

        w_final_expanded = w_final.unsqueeze(1)
        numerator = (w_final_expanded * x_p).sum(dim=(-2, -1))
        denominator = w_final.sum(dim=(-2, -1)).unsqueeze(-1) + eps
        result = (numerator / denominator).pow(1.0 / p)

        return result.unsqueeze(-1).unsqueeze(-1)

    def _standard_gem(self, x):
        gem_module = self.original_aggregation[1]
        return gem_module(x)

    def forward(self, x, attention_map=None, att_ratio: float = 0.1):
        """
        Forward pass.

        Args:
            x: Input image tensor [B, 3, H, W]
            attention_map: Spatial attention [H_feat, W_feat] from llm_grid_to_attention_map
            att_ratio: Attention blend factor ∈ [0, 1]

        Returns:
            L2-normalized descriptor [B, fc_output_dim]
        """
        features = self.cosplace.backbone(x)

        if attention_map is not None:
            l2norm1 = self.original_aggregation[0]
            features_norm = l2norm1(features)

            pooled = self.weighted_gem_target(features_norm, attention_map, att_ratio)

            flatten = self.original_aggregation[2]
            linear = self.original_aggregation[3]
            l2norm2 = self.original_aggregation[4]

            return l2norm2(linear(flatten(pooled)))
        else:
            return self.original_aggregation(features)

    def extract_features_with_attention(self, image, llm_dict, att_ratio: float = 0.1, interpolate: bool = False):
        """
        Extract descriptor from a single image with optional LLM attention.

        Args:
            image: File path (str) or PIL Image.
            llm_dict: Grid-based attention weights {"A1": w, ...} or coordinate list,
                      or None for standard extraction.
            att_ratio: Blend factor ∈ [0, 1] (0 = original CosPlace, 1 = full attention).
            interpolate: If True, bilinear-interpolate the grid attention map.

        Returns:
            Descriptor tensor [1, fc_output_dim]
        """
        if isinstance(image, str):
            from PIL import Image as _Image
            image = _Image.open(image).convert('RGB')
        if hasattr(image, 'convert'):
            image = self.transform(image).unsqueeze(0)

        device = next(self.cosplace.parameters()).device
        image = image.to(device)

        if att_ratio <= 1e-6 or llm_dict is None:
            return self.forward(image, attention_map=None, att_ratio=att_ratio)

        with torch.no_grad():
            feats = self.cosplace.backbone(image)
            _, _, H_feat, W_feat = feats.shape
            del feats

        if isinstance(llm_dict, dict) and 'result' in llm_dict:
            llm_dict = llm_dict['result']

        if isinstance(llm_dict, dict):
            attention_map = llm_grid_to_attention_map(llm_dict, H_feat, W_feat, device=device, interpolate=interpolate)
        elif isinstance(llm_dict, list):
            attention_map = llm_coord_to_attention_map(llm_dict, H_feat, W_feat, device=device)
        else:
            attention_map = None

        return self.forward(image, attention_map=attention_map, att_ratio=att_ratio)
