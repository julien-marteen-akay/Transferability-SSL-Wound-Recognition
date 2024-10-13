import torch
from torch import nn
import torchvision
from lightning.pytorch import seed_everything


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x
    

class ToFeatureVector(nn.Module):
    """Handles tuple outputs from models with multi-stage feature maps or multiple branches and then flattens them to be compatible with a linear layer"""
    def __init__(self, model, sample_input=torch.rand((2, 3, 224, 224))):
        super().__init__()
        self.handle_tuple = self.tuple_to_tensor if self.is_tuple(model, sample_input=sample_input) else Identity()

    def tuple_to_tensor(self, x, index=-1):
        """They did this in their paper, too."""
        return x[index]
    
    def is_tuple(self, model, sample_input):
        with torch.no_grad():
            out = model(sample_input)
        return isinstance(out, tuple)
    
    def flatten(self, x):
        return x.view(x.size(0), -1)

    def forward(self, x):
        return self.flatten(self.handle_tuple(x))


class Backbone(nn.Module):
    def __init__(self,
                 backbone: str,
                 seed: int = 42,
                 **kwargs):
        super().__init__(**kwargs)
        seed_everything(seed)

        # init_backbone will set these
        self.model = None
        self.target_layer = None

        self.backbone = backbone
        self.init_backbone(backbone)

        # backbone is extended to provide flattened outputs, dealing with edge cases e.g. tuple outputs, etc.
        self.to_feature_vector = ToFeatureVector(model=self.model)

        # keep track of other useful information
        self.model_size = self.get_model_size(only_trainable=True)

    def raise_backbone_error(self, backbone=None):
        raise ValueError(
            "Provide for the argument <backbone_architecture> in vicregl one of the following:\n"
            "- 'resnet_imagenet'\n"
            "- 'resnet'\n"
            "- 'vicreg'\n"
            "- 'vicregl'\n"
            f"You provided '{backbone}'"
            )

    def get_model_size(self, only_trainable=True):
        if only_trainable:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.model.parameters())

    def init_backbone(self, backbone=None):
        backbone = backbone or self.backbone
        if backbone == "resnet_imagenet":
            self.model = self.get_resnet(pretrained=True)
        elif backbone == "resnet":
            self.model = self.get_resnet(pretrained=False)
        elif backbone == "vicreg":
            self.model = self.get_vicreg_backbone()
        elif "vicregl" in backbone:
            self.model = self.get_vicregl_backbone(backbone_architecture=backbone)
        else:
            self.raise_backbone_error(backbone=backbone)

    def get_resnet(self, pretrained=False):
        """Pure ResNets come with classification layer (multi-class fully connected). We the layer here.
        Note that SSL methods using ResNets do not have this problem, thus we only do it in this function."""
        model = torchvision.models.resnet50(weights=("DEFAULT" if pretrained else None))
        model = torch.nn.Sequential(*list(model.children())[:-1])
        return model
    
    def get_vicreg_backbone(self):
        return torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
    
    def get_vicregl_backbone(self, backbone_architecture):
        # resnet vs. convnext
        if "resnet" in backbone_architecture:
            return torch.hub.load('facebookresearch/vicregl:main', 'resnet50_alpha0p9')
        elif "convnext" in backbone_architecture:
            # answers: which size of covnext?
            if "small" in backbone_architecture:
                return torch.hub.load('facebookresearch/vicregl:main', 'convnext_small_alpha0p9')
            elif "base" in backbone_architecture:
                return torch.hub.load('facebookresearch/vicregl:main', 'convnext_base_alpha0p9')
            elif "xlarge" in backbone_architecture:
                return torch.hub.load('facebookresearch/vicregl:main', 'convnext_xlarge_alpha0p75')
        
        raise ValueError(
            "Provide for the argument <backbone_architecture> in vicregl one of the following:\n"
            "- 'resnet'\n"
            "- 'convnext_small'\n"
            "- 'convnext_base'\n"
            "- 'convnext_xlarge'\n"
            f"You provided '{backbone_architecture}'"
            )
    
    def set_target_layer(self, backbone=None):
        backbone = backbone or self.backbone
        if backbone == "resnet_imagenet":
            self.target_layer = self.model[-2][-1].conv3
        elif backbone == "resnet":
            self.target_layer = self.model[-2][-1].conv3
        elif backbone == "vicreg":
            self.target_layer = self.model.layer4[-1].conv3
        elif "vicregl" in backbone:
            if "resnet" in backbone:
                self.target_layer = self.model.layer4[-1].conv3
            elif "convnext" in backbone:
                # answers: which size of covnext?
                if "small" in backbone:
                    self.target_layer = self.model.stages[-1][-1].dwconv
                elif "base" in backbone:
                    self.target_layer = self.model.stages[-1][-1].dwconv
                elif "xlarge" in backbone:
                    self.target_layer = self.model.stages[-1][-1].dwconv
        else:
            self.raise_backbone_error(backbone=backbone)

    def forward(self, x):
        return self.to_feature_vector(self.model(x))
