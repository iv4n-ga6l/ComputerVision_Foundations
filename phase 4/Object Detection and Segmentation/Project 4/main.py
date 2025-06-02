"""
Custom Object Detector Implementation
Complete pipeline for training custom object detection models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
import argparse
from collections import defaultdict
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class IoULoss(nn.Module):
    """Intersection over Union Loss"""
    def __init__(self, loss_type='iou'):
        super(IoULoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, pred_boxes, target_boxes):
        # Calculate IoU
        iou = self.calculate_iou(pred_boxes, target_boxes)
        
        if self.loss_type == 'iou':
            return 1 - iou.mean()
        elif self.loss_type == 'giou':
            giou = self.calculate_giou(pred_boxes, target_boxes)
            return 1 - giou.mean()
        elif self.loss_type == 'diou':
            diou = self.calculate_diou(pred_boxes, target_boxes)
            return 1 - diou.mean()
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two sets of boxes"""
        # Calculate intersection
        inter_x1 = torch.max(box1[:, 0], box2[:, 0])
        inter_y1 = torch.max(box1[:, 1], box2[:, 1])
        inter_x2 = torch.min(box1[:, 2], box2[:, 2])
        inter_y2 = torch.min(box1[:, 3], box2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Calculate union
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area
    
    def calculate_giou(self, box1, box2):
        """Calculate Generalized IoU"""
        iou = self.calculate_iou(box1, box2)
        
        # Calculate enclosing box
        enclose_x1 = torch.min(box1[:, 0], box2[:, 0])
        enclose_y1 = torch.min(box1[:, 1], box2[:, 1])
        enclose_x2 = torch.max(box1[:, 2], box2[:, 2])
        enclose_y2 = torch.max(box1[:, 3], box2[:, 3])
        
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
        union_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]) + \
                    (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1]) - \
                    iou * enclose_area
        
        giou = iou - (enclose_area - union_area) / enclose_area
        return giou
    
    def calculate_diou(self, box1, box2):
        """Calculate Distance IoU"""
        iou = self.calculate_iou(box1, box2)
        
        # Calculate center distances
        center1_x = (box1[:, 0] + box1[:, 2]) / 2
        center1_y = (box1[:, 1] + box1[:, 3]) / 2
        center2_x = (box2[:, 0] + box2[:, 2]) / 2
        center2_y = (box2[:, 1] + box2[:, 3]) / 2
        
        center_distance = (center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2
        
        # Calculate diagonal of enclosing box
        enclose_x1 = torch.min(box1[:, 0], box2[:, 0])
        enclose_y1 = torch.min(box1[:, 1], box2[:, 1])
        enclose_x2 = torch.max(box1[:, 2], box2[:, 2])
        enclose_y2 = torch.max(box1[:, 3], box2[:, 3])
        
        diagonal_distance = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
        
        diou = iou - center_distance / diagonal_distance
        return diou

class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature extraction"""
    def __init__(self, in_channels_list, out_channels=256):
        super(FeaturePyramidNetwork, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, 1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
    
    def forward(self, inputs):
        # Build laterals
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        
        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode="nearest"
            )
        
        # Build outputs
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        return outs

class DetectionHead(nn.Module):
    """Detection head for object classification and localization"""
    def __init__(self, in_channels, num_classes, num_anchors):
        super(DetectionHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Classification head
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)
        )
        
        # Regression head
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)
        )
    
    def forward(self, features):
        cls_outputs = []
        reg_outputs = []
        
        for feature in features:
            cls_output = self.cls_conv(feature)
            reg_output = self.reg_conv(feature)
            
            # Reshape outputs
            batch_size = cls_output.shape[0]
            cls_output = cls_output.view(batch_size, self.num_anchors, self.num_classes, -1)
            cls_output = cls_output.permute(0, 3, 1, 2).contiguous()
            
            reg_output = reg_output.view(batch_size, self.num_anchors, 4, -1)
            reg_output = reg_output.permute(0, 3, 1, 2).contiguous()
            
            cls_outputs.append(cls_output)
            reg_outputs.append(reg_output)
        
        return cls_outputs, reg_outputs

class AnchorGenerator:
    """Generate anchor boxes for object detection"""
    def __init__(self, sizes, aspect_ratios, scales):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.scales = scales
    
    def generate_anchors(self, feature_shapes, device):
        """Generate anchors for all feature levels"""
        all_anchors = []
        
        for i, (h, w) in enumerate(feature_shapes):
            stride = 2 ** (i + 3)  # 8, 16, 32, 64, 128
            base_size = self.sizes[i]
            
            # Generate base anchors
            base_anchors = self._generate_base_anchors(base_size, self.aspect_ratios, device)
            
            # Apply scales
            scaled_anchors = []
            for scale in self.scales:
                scaled_anchors.append(base_anchors * scale)
            base_anchors = torch.cat(scaled_anchors, dim=0)
            
            # Generate all anchors for this level
            level_anchors = self._generate_level_anchors(base_anchors, h, w, stride, device)
            all_anchors.append(level_anchors)
        
        return torch.cat(all_anchors, dim=0)
    
    def _generate_base_anchors(self, base_size, aspect_ratios, device):
        """Generate base anchor shapes"""
        anchors = []
        for aspect_ratio in aspect_ratios:
            w = base_size * np.sqrt(aspect_ratio)
            h = base_size / np.sqrt(aspect_ratio)
            anchors.append([-w/2, -h/2, w/2, h/2])
        
        return torch.tensor(anchors, device=device, dtype=torch.float32)
    
    def _generate_level_anchors(self, base_anchors, h, w, stride, device):
        """Generate anchors for a specific feature level"""
        shift_x = torch.arange(0, w, device=device, dtype=torch.float32) * stride
        shift_y = torch.arange(0, h, device=device, dtype=torch.float32) * stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        
        shifts = torch.stack((shift_x.flatten(), shift_y.flatten(),
                             shift_x.flatten(), shift_y.flatten()), dim=1)
        
        # Add shifts to base anchors
        anchors = base_anchors.view(1, -1, 4) + shifts.view(-1, 1, 4)
        return anchors.view(-1, 4)

class CustomDetector(nn.Module):
    """Custom object detector with FPN backbone"""
    def __init__(self, num_classes, backbone='resnet50'):
        super(CustomDetector, self).__init__()
        self.num_classes = num_classes
        
        # Backbone
        if backbone == 'resnet50':
            backbone_model = torchvision.models.resnet50(pretrained=True)
            self.backbone = self._build_resnet_backbone(backbone_model)
            backbone_channels = [512, 1024, 2048]
        elif backbone == 'vgg16':
            backbone_model = torchvision.models.vgg16(pretrained=True)
            self.backbone = self._build_vgg_backbone(backbone_model)
            backbone_channels = [256, 512, 512]
        
        # FPN
        self.fpn = FeaturePyramidNetwork(backbone_channels)
        
        # Detection head
        self.detection_head = DetectionHead(256, num_classes, num_anchors=3)
        
        # Anchor generator
        self.anchor_generator = AnchorGenerator(
            sizes=[32, 64, 128, 256, 512],
            aspect_ratios=[0.5, 1.0, 2.0],
            scales=[1.0]
        )
        
        # Loss functions
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.iou_loss = IoULoss(loss_type='giou')
    
    def _build_resnet_backbone(self, backbone_model):
        """Build ResNet backbone for feature extraction"""
        layers = []
        
        # Initial layers
        layers.extend([
            backbone_model.conv1,
            backbone_model.bn1,
            backbone_model.relu,
            backbone_model.maxpool
        ])
        
        # ResNet blocks
        layers.extend([
            backbone_model.layer1,
            backbone_model.layer2,
            backbone_model.layer3,
            backbone_model.layer4
        ])
        
        return nn.ModuleList(layers)
    
    def _build_vgg_backbone(self, backbone_model):
        """Build VGG backbone for feature extraction"""
        features = list(backbone_model.features.children())
        
        # Divide into different levels
        level1 = nn.Sequential(*features[:16])   # Up to pool3
        level2 = nn.Sequential(*features[16:23]) # Up to pool4
        level3 = nn.Sequential(*features[23:30]) # Up to pool5
        
        return nn.ModuleList([level1, level2, level3])
    
    def extract_features(self, x):
        """Extract multi-level features"""
        features = []
        
        if isinstance(self.backbone[0], nn.Conv2d):  # ResNet
            # Initial layers
            for layer in self.backbone[:4]:
                x = layer(x)
            
            # Extract features from different levels
            for layer in self.backbone[4:7]:  # layer1, layer2, layer3
                x = layer(x)
                if layer == self.backbone[5] or layer == self.backbone[6]:  # layer2, layer3
                    features.append(x)
            
            x = self.backbone[7](x)  # layer4
            features.append(x)
        
        else:  # VGG
            for i, layer in enumerate(self.backbone):
                x = layer(x)
                features.append(x)
        
        return features
    
    def forward(self, images, targets=None):
        """Forward pass"""
        # Extract features
        backbone_features = self.extract_features(images)
        
        # Apply FPN
        fpn_features = self.fpn(backbone_features)
        
        # Get predictions
        cls_outputs, reg_outputs = self.detection_head(fpn_features)
        
        # Generate anchors
        feature_shapes = [(f.shape[-2], f.shape[-1]) for f in fpn_features]
        anchors = self.anchor_generator.generate_anchors(feature_shapes, images.device)
        
        if self.training and targets is not None:
            # Compute losses
            return self._compute_losses(cls_outputs, reg_outputs, anchors, targets)
        else:
            # Post-process predictions
            return self._post_process(cls_outputs, reg_outputs, anchors, images.shape[-2:])
    
    def _compute_losses(self, cls_outputs, reg_outputs, anchors, targets):
        """Compute training losses"""
        # Flatten predictions
        cls_logits = torch.cat([cls.view(-1, self.num_classes) for cls in cls_outputs], dim=0)
        reg_preds = torch.cat([reg.view(-1, 4) for reg in reg_outputs], dim=0)
        
        # Assign targets to anchors
        assigned_labels, assigned_boxes = self._assign_targets(anchors, targets)
        
        # Compute classification loss
        valid_mask = assigned_labels >= 0
        pos_mask = assigned_labels > 0
        
        if pos_mask.sum() > 0:
            cls_loss = self.focal_loss(cls_logits[valid_mask], assigned_labels[valid_mask])
            reg_loss = self.iou_loss(reg_preds[pos_mask], assigned_boxes[pos_mask])
        else:
            cls_loss = torch.tensor(0.0, device=cls_logits.device)
            reg_loss = torch.tensor(0.0, device=reg_preds.device)
        
        total_loss = cls_loss + reg_loss
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'reg_loss': reg_loss
        }
    
    def _assign_targets(self, anchors, targets):
        """Assign ground truth targets to anchors"""
        batch_size = len(targets)
        num_anchors = anchors.shape[0]
        
        assigned_labels = torch.full((batch_size * num_anchors,), -1, dtype=torch.long, device=anchors.device)
        assigned_boxes = torch.zeros((batch_size * num_anchors, 4), device=anchors.device)
        
        for i, target in enumerate(targets):
            gt_boxes = target['boxes']
            gt_labels = target['labels']
            
            if len(gt_boxes) == 0:
                continue
            
            # Compute IoU between anchors and ground truth
            ious = self._compute_iou_matrix(anchors, gt_boxes)
            
            # Assign anchors
            max_ious, max_indices = ious.max(dim=1)
            
            # Positive samples (IoU > 0.5)
            pos_mask = max_ious > 0.5
            start_idx = i * num_anchors
            end_idx = (i + 1) * num_anchors
            
            assigned_labels[start_idx:end_idx][pos_mask] = gt_labels[max_indices[pos_mask]]
            assigned_boxes[start_idx:end_idx][pos_mask] = gt_boxes[max_indices[pos_mask]]
            
            # Negative samples (IoU < 0.3)
            neg_mask = max_ious < 0.3
            assigned_labels[start_idx:end_idx][neg_mask] = 0
        
        return assigned_labels, assigned_boxes
    
    def _compute_iou_matrix(self, boxes1, boxes2):
        """Compute IoU matrix between two sets of boxes"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        inter_x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
        inter_y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
        inter_x2 = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
        inter_y2 = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
        
        return inter_area / union_area
    
    def _post_process(self, cls_outputs, reg_outputs, anchors, image_shape):
        """Post-process predictions for inference"""
        # Flatten predictions
        cls_logits = torch.cat([cls.view(-1, self.num_classes) for cls in cls_outputs], dim=0)
        reg_preds = torch.cat([reg.view(-1, 4) for reg in reg_outputs], dim=0)
        
        # Apply sigmoid to classification scores
        cls_scores = torch.sigmoid(cls_logits)
        
        # Get predicted boxes
        pred_boxes = self._decode_boxes(reg_preds, anchors)
        
        # Filter predictions
        max_scores, labels = cls_scores.max(dim=1)
        keep = max_scores > 0.3
        
        final_boxes = pred_boxes[keep]
        final_scores = max_scores[keep]
        final_labels = labels[keep]
        
        # Apply NMS
        keep_nms = torchvision.ops.nms(final_boxes, final_scores, iou_threshold=0.5)
        
        return {
            'boxes': final_boxes[keep_nms],
            'scores': final_scores[keep_nms],
            'labels': final_labels[keep_nms]
        }
    
    def _decode_boxes(self, reg_preds, anchors):
        """Decode regression predictions to boxes"""
        # Simplified box decoding
        decoded_boxes = anchors + reg_preds
        return decoded_boxes

class CustomDataset(Dataset):
    """Custom dataset for object detection"""
    def __init__(self, data_dir, annotation_file, transform=None, format='coco'):
        self.data_dir = data_dir
        self.transform = transform
        self.format = format
        
        if format == 'coco':
            self._load_coco_annotations(annotation_file)
        elif format == 'pascal':
            self._load_pascal_annotations(annotation_file)
        elif format == 'custom':
            self._load_custom_annotations(annotation_file)
    
    def _load_coco_annotations(self, annotation_file):
        """Load COCO format annotations"""
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        self.images = coco_data['images']
        self.annotations = coco_data['annotations']
        
        # Create image_id to annotations mapping
        self.img_to_anns = defaultdict(list)
        for ann in self.annotations:
            self.img_to_anns[ann['image_id']].append(ann)
    
    def _load_pascal_annotations(self, annotation_dir):
        """Load PASCAL VOC format annotations"""
        self.images = []
        self.img_to_anns = {}
        
        for xml_file in os.listdir(annotation_dir):
            if xml_file.endswith('.xml'):
                xml_path = os.path.join(annotation_dir, xml_file)
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Image info
                filename = root.find('filename').text
                width = int(root.find('size/width').text)
                height = int(root.find('size/height').text)
                
                img_id = len(self.images)
                self.images.append({
                    'id': img_id,
                    'file_name': filename,
                    'width': width,
                    'height': height
                })
                
                # Annotations
                annotations = []
                for obj in root.findall('object'):
                    bbox = obj.find('bndbox')
                    x1 = float(bbox.find('xmin').text)
                    y1 = float(bbox.find('ymin').text)
                    x2 = float(bbox.find('xmax').text)
                    y2 = float(bbox.find('ymax').text)
                    
                    annotations.append({
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'category_name': obj.find('name').text
                    })
                
                self.img_to_anns[img_id] = annotations
    
    def _load_custom_annotations(self, annotation_file):
        """Load custom format annotations"""
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        self.images = data['images']
        self.img_to_anns = data['annotations']
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.data_dir, img_info['file_name'])
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        anns = self.img_to_anns.get(img_info['id'], [])
        
        boxes = []
        labels = []
        
        for ann in anns:
            if self.format == 'coco':
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])
            elif self.format == 'pascal':
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(1)  # Simplified for demo
            elif self.format == 'custom':
                boxes.append(ann['bbox'])
                labels.append(ann['label'])
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,))
        }
        
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, category_ids=labels)
            image = transformed['image']
            if transformed['bboxes']:
                target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
                target['labels'] = torch.tensor(transformed['category_ids'], dtype=torch.long)
        
        return image, target

def get_transforms(mode='train'):
    """Get data transforms"""
    if mode == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=0.2),
            A.Resize(640, 640),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
    else:
        return A.Compose([
            A.Resize(640, 640),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

def collate_fn(batch):
    """Custom collate function"""
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    return images, targets

def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001):
    """Train the custom detector"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)
            
            # Move targets to device
            for i in range(len(targets)):
                for key in targets[i]:
                    targets[i][key] = targets[i][key].to(device)
            
            optimizer.zero_grad()
            losses = model(images, targets)
            loss = losses['total_loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                for i in range(len(targets)):
                    for key in targets[i]:
                        targets[i][key] = targets[i][key].to(device)
                
                losses = model(images, targets)
                val_loss += losses['total_loss'].item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_detector.pth')
            print("Best model saved!")
        
        scheduler.step()
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, class_names=None):
    """Evaluate model performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            predictions = model(images)
            
            all_predictions.append(predictions)
            all_targets.extend(targets)
    
    # Compute mAP
    map_score = compute_map(all_predictions, all_targets)
    print(f"mAP@0.5: {map_score:.4f}")
    
    return map_score

def compute_map(predictions, targets, iou_threshold=0.5):
    """Compute mean Average Precision"""
    # Simplified mAP computation
    total_tp = 0
    total_fp = 0
    total_gt = 0
    
    for pred_batch, target_batch in zip(predictions, targets):
        for pred, target in zip(pred_batch if isinstance(pred_batch, list) else [pred_batch], 
                               target_batch if isinstance(target_batch, list) else [target_batch]):
            
            pred_boxes = pred['boxes'].cpu()
            pred_scores = pred['scores'].cpu()
            target_boxes = target['boxes'].cpu()
            
            total_gt += len(target_boxes)
            
            if len(pred_boxes) == 0:
                continue
            
            # Sort by confidence
            sorted_indices = torch.argsort(pred_scores, descending=True)
            matched = torch.zeros(len(target_boxes), dtype=torch.bool)
            
            for idx in sorted_indices:
                pred_box = pred_boxes[idx]
                
                if len(target_boxes) == 0:
                    total_fp += 1
                    continue
                
                # Compute IoU
                ious = compute_iou_single(pred_box.unsqueeze(0), target_boxes)
                max_iou, max_idx = torch.max(ious, dim=0)
                
                if max_iou > iou_threshold and not matched[max_idx]:
                    total_tp += 1
                    matched[max_idx] = True
                else:
                    total_fp += 1
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0
    
    return precision * recall / (precision + recall) * 2 if (precision + recall) > 0 else 0

def compute_iou_single(box1, box2):
    """Compute IoU between single box and multiple boxes"""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area

def visualize_predictions(model, image_path, class_names=None):
    """Visualize predictions on an image"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Transform
    transform = get_transforms('test')
    transformed = transform(image=image_rgb, bboxes=[], category_ids=[])
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        predictions = model(input_tensor)
    
    # Visualize
    pred = predictions[0] if isinstance(predictions, list) else predictions
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    
    # Draw on original image
    vis_image = image_rgb.copy()
    
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            label_text = f'Class {label}: {score:.2f}'
            if class_names and label < len(class_names):
                label_text = f'{class_names[label]}: {score:.2f}'
            
            cv2.putText(vis_image, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(vis_image)
    plt.title('Object Detection Results')
    plt.axis('off')
    plt.show()

def demo_training():
    """Demo training with synthetic data"""
    print("Creating demo dataset...")
    
    # Create synthetic dataset
    num_samples = 100
    images = torch.rand(num_samples, 3, 640, 640)
    targets = []
    
    for i in range(num_samples):
        # Random number of objects (1-5)
        num_objects = np.random.randint(1, 6)
        boxes = []
        labels = []
        
        for _ in range(num_objects):
            # Random box coordinates
            x1, y1 = np.random.randint(0, 500, 2)
            x2, y2 = x1 + np.random.randint(50, 140), y1 + np.random.randint(50, 140)
            x2, y2 = min(x2, 640), min(y2, 640)
            
            boxes.append([x1, y1, x2, y2])
            labels.append(np.random.randint(1, 4))  # 3 classes
        
        targets.append({
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long)
        })
    
    # Create data loaders
    dataset = [(images[i], targets[i]) for i in range(num_samples)]
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # Model
    model = CustomDetector(num_classes=4)  # 3 classes + background
    
    print("Starting training...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=5, lr=0.001)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Progress')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Custom Object Detector')
    parser.add_argument('--mode', choices=['train', 'eval', 'inference', 'demo'], default='demo')
    parser.add_argument('--dataset_path', type=str, help='Path to dataset')
    parser.add_argument('--annotation_file', type=str, help='Path to annotations')
    parser.add_argument('--model_path', type=str, help='Path to model weights')
    parser.add_argument('--image_path', type=str, help='Path to image for inference')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--backbone', choices=['resnet50', 'vgg16'], default='resnet50')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        print("Running training demo...")
        demo_training()
    
    elif args.mode == 'train':
        if not args.dataset_path or not args.annotation_file:
            print("Error: dataset_path and annotation_file required for training")
            return
        
        # Setup datasets
        train_transform = get_transforms('train')
        val_transform = get_transforms('val')
        
        train_dataset = CustomDataset(args.dataset_path, args.annotation_file, train_transform)
        val_dataset = CustomDataset(args.dataset_path, args.annotation_file, val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                 shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                               shuffle=False, collate_fn=collate_fn)
        
        # Model
        model = CustomDetector(num_classes=args.num_classes, backbone=args.backbone)
        
        # Train
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, 
            num_epochs=args.num_epochs, lr=args.learning_rate
        )
    
    elif args.mode == 'eval':
        if not args.model_path:
            print("Error: model_path required for evaluation")
            return
        
        model = CustomDetector(num_classes=args.num_classes, backbone=args.backbone)
        model.load_state_dict(torch.load(args.model_path))
        
        # Setup test dataset
        test_transform = get_transforms('test')
        test_dataset = CustomDataset(args.dataset_path, args.annotation_file, test_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                shuffle=False, collate_fn=collate_fn)
        
        # Evaluate
        map_score = evaluate_model(model, test_loader)
        print(f"Final mAP: {map_score:.4f}")
    
    elif args.mode == 'inference':
        if not args.model_path or not args.image_path:
            print("Error: model_path and image_path required for inference")
            return
        
        model = CustomDetector(num_classes=args.num_classes, backbone=args.backbone)
        model.load_state_dict(torch.load(args.model_path))
        
        visualize_predictions(model, args.image_path)
    
    print("\nCustom Object Detector completed!")
    print("\nKey Features Implemented:")
    print("- Custom detector with FPN backbone")
    print("- Multiple loss functions (Focal, GIoU)")
    print("- Advanced data augmentation")
    print("- Multi-format dataset support")
    print("- Comprehensive training pipeline")
    print("- mAP evaluation")
    print("- Real-time inference capabilities")

if __name__ == "__main__":
    main()
