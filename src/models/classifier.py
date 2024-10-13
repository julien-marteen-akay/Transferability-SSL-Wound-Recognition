import torch
from torch import nn
from torch.nn import functional as F
import lightning as L
from lightning.pytorch import seed_everything
from torchmetrics.functional import classification as metrics
from torchmetrics import Accuracy, F1Score
from torchmetrics.classification import MatthewsCorrCoef, AUROC

# internals
from ..models.backbone import Backbone
from ..utils import misc


class Classifier(L.LightningModule):
    def __init__(self,
                 backbone: str,
                 num_mlp_layers: int,
                 num_classes: int,
                 num_mlp_neurons: int = 512,
                 freeze_backbone: bool = True,
                 seed: int = 42,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        # save hparams in modelcheckpoint
        self.save_hyperparameters()
        seed_everything(seed)

        # get backbone
        self.backbone = Backbone(backbone=backbone, seed=seed)
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        # classification layer
        self.num_mlp_layers = max(num_mlp_layers, 1)
        self.num_classes = num_classes
        self.num_mlp_neurons = num_mlp_neurons
        self.classifier = self.get_classification_layer()

        # metrics
        self.set_metrics(stage="val")
        self.set_metrics(stage="test")

    @property
    def target_layer(self):
        if not self.backbone.target_layer:
            print("Running gradcam_setup() to automatically fetch the target_layer.")
            self.gradcam_setup()
        return self.backbone.target_layer

    def gradcam_setup(self):
        self.backbone.set_target_layer()

    def unfreeze_backbone(self):
        self.backbone.requires_grad_(True)
        self.backbone.train()

    def get_classification_layer(self):
        """Gets classification head. MLP, with adjustable number of layers and number of neurons per layer. Backbone-agnostic"""
        # retrieve output shape of backbone
        with torch.no_grad():
            out = self.backbone(torch.rand(1, 3, 224, 224))  # e.g. out.size(-1)=2048
        
        # accumulate all shapes to be reached from start to end of mlp (both inclusive), e.g. [2048, 512, 10] means we need Linear layers (2048, 512) and (512, 10)
        in_to_out = [self.num_mlp_neurons for _ in range(self.num_mlp_layers - 1)]  # if layers=1 --> []; if layers=2 --> [512] e.g. for self.num_mlp_layers=512
        in_to_out.insert(0, out.size(-1))
        in_to_out.append(self.num_classes)
        in_to_out = [(in_to_out[i-1], in_to_out[i]) for i in range(1, len(in_to_out))]  # [2048, 512, 10] --> [(2048, 512), (512, 10)]
        in_to_out = [nn.Linear(*in_to_out.pop(0)) if misc.is_even(i) else nn.ReLU() for i in range(2 * len(in_to_out) - 1)]  # [(2048, 512), (512, 10)] -> [Linear, ReLU, Linear]
        return nn.Sequential(*in_to_out)
    
    def set_metrics(self, stage, task="multiclass"):
        setattr(self, f"{stage}_acc_macro", Accuracy(num_classes=self.num_classes, task=task, average="macro"))
        setattr(self, f"{stage}_acc_micro", Accuracy(num_classes=self.num_classes, task=task, average="micro"))
        setattr(self, f"{stage}_rocauc", AUROC(num_classes=self.num_classes, task=task, average="macro", thresholds=None))
        setattr(self, f"{stage}_f1_macro", F1Score(num_classes=self.num_classes, task=task, average="macro"))
        setattr(self, f"{stage}_f1_micro", F1Score(num_classes=self.num_classes, task=task, average="micro"))
        setattr(self, f"{stage}_mcc", MatthewsCorrCoef(num_classes=self.num_classes, task=task))

    def get_metrics(self, stage):
        return {
            f"{stage}_acc_macro": getattr(self, f"{stage}_acc_macro"),
            f"{stage}_acc_micro": getattr(self, f"{stage}_acc_micro"),
            f"{stage}_rocauc": getattr(self, f"{stage}_rocauc"),
            f"{stage}_f1_macro": getattr(self, f"{stage}_f1_macro"),
            f"{stage}_f1_micro": getattr(self, f"{stage}_f1_micro"),
            f"{stage}_mcc": getattr(self, f"{stage}_mcc")
        }

    def compute_metrics(self, *, y, y_tilde, stage):
        """y=target or label, y_tilde=y predicted from x"""
        y_tilde_prob = F.softmax(y_tilde, dim=-1)
        for metric in self.get_metrics(stage).values():
            metric.update(y_tilde_prob, y)
    
    def dict_val_to_item(self, dictionary):
        """key is metric's name and value is the tensor of dim=0, I extract the item of the tensor to get a simple python float"""
        return {key: val.item() for key, val in dictionary.items()}
    
    # ----------------
    # overridden hooks
    # ----------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def forward(self, x):
        return self.classifier(self.backbone(x))

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        # feed forward
        y_tilde = self(x)

        # calculate loss
        loss = F.cross_entropy(input=y_tilde, target=y)

        # logging
        values = self.dict_val_to_item({"loss": loss})
        self.log_dict(values, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        # feed forward
        y_tilde = self(x)

        # calculate loss
        loss = F.cross_entropy(input=y_tilde, target=y)

        # calculate metrics
        self.compute_metrics(y=y, y_tilde=y_tilde, stage="val")
        
        # logging
        values = {**self.dict_val_to_item({"val_loss": loss}), **self.get_metrics("val")}
        self.log_dict(values, on_epoch=True, prog_bar=True)

    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch

        # feed forward
        y_tilde = self(x)

        # calculate loss
        loss = F.cross_entropy(input=y_tilde, target=y)

        # calculate metrics
        self.compute_metrics(y=y, y_tilde=y_tilde, stage="test")

        # logging
        values = {**self.dict_val_to_item({"test_loss": loss}), **self.get_metrics("test")}
        self.log_dict(values, on_epoch=True, prog_bar=True)
        