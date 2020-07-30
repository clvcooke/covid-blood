import torch
import torch.nn as nn
import torch.nn.functional as F
from models.imagenet import get_model


# TODO: merge these classes, near 80% code overlap
# From: https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
class MultiHeadedAttentionModel(nn.Module):
    def __init__(self, backbone_name, num_heads=2, instance_hidden_size=32, hidden_size=64, num_classes=2, pretrained_backbone=True):
        super(MultiHeadedAttentionModel, self).__init__()

        self.instance_hidden_size = instance_hidden_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.backbone = get_model(backbone_name, self.instance_hidden_size, pretrained_backbone)
        # for param in self.backbone.features[-2:].parameters():
        #     param.requires_grad = True

        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

        self.multi_headed_attention = nn.Sequential(
            nn.Linear(self.instance_hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_heads)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.instance_hidden_size*self.num_heads, self.num_classes)
        )

    def forward(self, x):
        # Batch size == 1 for now
        x = x.squeeze(0)
        instance_count = x.shape[0]
        # B x features
        features = self.backbone(x)
        # B x Num Heads
        attention = self.multi_headed_attention(features)
        # softmax across B
        attention = F.softmax(attention, dim=0).view(instance_count, 1, self.num_heads)
        # reshape features: B x features x 1
        features_view = features.view(instance_count, -1, 1)
        # broadcasting: B x features x num_heads --> features x num_heads
        aggregated_features = torch.sum(attention*features_view, dim=0)
        # output of: 1 x C (C = num classes)
        classification = self.classifier(aggregated_features.view(1, -1))
        return classification



class GatedAttentionModel(nn.Module):
    def __init__(self, backbone_name, instance_hidden_size=32, hidden_size=64, num_classes=2, num_heads=1, pretrained_backbone=True):
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
        attention_v = self.attention_v(features)
        attention_u = self.attention_u(features)
        attention = self.attention_weights(attention_v * attention_u)
        attention = torch.transpose(attention, 1, 0)
        attention = F.softmax(attention, dim=1)
        gated_features = torch.mm(attention, features)
        classification = self.classifier(gated_features)

        return classification


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
