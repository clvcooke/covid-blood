import torch
import torch.nn as nn
import torch.nn.functional as F
from models.imagenet import get_model
from torch.utils.checkpoint import checkpoint


# TODO: merge these classes, near 80% code overlap
# From: https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
class GatedAttentionModel(nn.Module):
    def __init__(self, backbone_name, instance_hidden_size=32, hidden_size=64, num_classes=2, pretrained_backbone=True):
        super(GatedAttentionModel, self).__init__()
        self.instance_hidden_size = instance_hidden_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.backbone = get_model(backbone_name, self.instance_hidden_size, pretrained_backbone)
        # # unfreeze backbone weights
        for param in self.backbone.features[-2:].parameters():
            param.requires_grad = True

        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

        # binary classification
        self.sil_classifier = nn.Sequential(
            nn.Linear(self.instance_hidden_size, 2)
        )

        self.attention_v = nn.Sequential(
            nn.Linear(self.instance_hidden_size, self.hidden_size),
            nn.Tanh()
        )

        self.attention_u = nn.Sequential(
            nn.Linear(self.instance_hidden_size, self.hidden_size),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.hidden_size, 1)

        self.classifier = nn.Sequential(
            nn.Linear(self.instance_hidden_size, self.num_classes)
        )

    def forward(self, x):
        x = x.squeeze(0)
        features = self.backbone(x)
        sil_classification = self.sil_classifier(features)
        attention_v = self.attention_v(features)
        attention_u = self.attention_u(features)
        attention = self.attention_weights(attention_v * attention_u)
        attention = torch.transpose(attention, 1, 0)
        attention = F.softmax(attention, dim=1)
        gated_features = torch.mm(attention, features)
        classification = self.classifier(gated_features)
        return classification, sil_classification


class TransformerMIL(nn.Module):
    def __init__(self, backbone_name, instance_hidden_size=32, hidden_size=64, num_classes=2, pretrained_backbone=True):
        super(TransformerMIL, self).__init__()
        self.instance_hidden_size = instance_hidden_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.backbone = get_model(backbone_name, 2, pretrained_backbone)
        N_HEADS = 1
        D_MODEL = 16
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
        self.reduction_layer = nn.Sequential(nn.Linear(1024, D_MODEL), nn.ReLU())
        self.mil_classifier = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=1, dim_feedforward=D_MODEL), 2)
        self.classifier_layer = nn.Linear(D_MODEL, 2)

    def forward(self, x):
        x = x.squeeze(0)
        features = self.backbone.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        sil_out = torch.flatten(out, 1).view(1, -1, 1024)
        reduced = self.reduction_layer(sil_out)
        out = self.mil_classifier(reduced)
        out_avg = torch.mean(out, dim=1)
        mil_classification = self.classifier_layer(out_avg)
        return mil_classification


class SimpleMIL(nn.Module):
    def __init__(self, backbone_name, instance_hidden_size=32, hidden_size=64, num_classes=2, pretrained_backbone=True):
        super(SimpleMIL, self).__init__()
        self.instance_hidden_size = instance_hidden_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.backbone = get_model(backbone_name, 2, pretrained_backbone)

        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

        # binary classification
        self.sil_classifier = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(self.instance_hidden_size, 2)
        )

        self.mil_classifier = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(self.instance_hidden_size, 2),
        )

    def forward(self, x):
        x = x.squeeze(0)
        features = self.backbone(x)
        sil_classification = features
        # sil_classification = self.sil_classifier(features)
        # mean_feat = torch.mean(features, dim=0, keepdim=True)
        mil_classification = torch.mean(sil_classification, dim=0, keepdim=True)
        # mil_classification = self.mil_classifier(mean_feat)
        return mil_classification, sil_classification


class AttentionModel(nn.Module):
    def __init__(self, backbone_name, instance_hidden_size=32, hidden_size=64, num_classes=2, pretrained_backbone=True):
        super(AttentionModel, self).__init__()
        self.instance_hidden_size = instance_hidden_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.backbone = get_model(backbone_name, self.instance_hidden_size, pretrained_backbone)
        self.attention = nn.Sequential(
            nn.Linear(self.instance_hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.instance_hidden_size, self.num_classes)
        )

    def forward(self, x):
        x = x.squeeze(0)
        features = self.backbone(x)
        attention = self.attention(features)
        attention = torch.transpose(attention, 1, 0)
        attention_weights = F.softmax(attention, dim=1)

        aggregated_features = torch.mm(attention_weights, features)
        classification = self.classifier(aggregated_features)
        return classification
