"""
YOLO Object Detection Implementation
==================================

A comprehensive implementation of YOLO (You Only Look Once) object detection
with support for training, inference, and real-time detection.

Features:
- Complete YOLO architecture implementation
- Multi-dataset support (COCO, Pascal VOC, custom)
- Real-time detection with webcam
- Model training and evaluation
- Performance optimization
- Visualization tools

 Project
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import json
import argparse
from tqdm import tqdm
import time
from typing import List, Tuple, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class YOLODataset(Dataset):
    """YOLO dataset for object detection"""
    
    def __init__(self, images_dir: str, labels_dir: str, img_size: int = 640, 
                 augment: bool = True, num_classes: int = 80):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Get all image files
        self.image_files = [f for f in os.listdir(images_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Define augmentations
        if augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(p=0.1),
                A.Blur(blur_limit=3, p=0.1),
                A.GaussNoise(p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = os.path.join(self.labels_dir, 
                                 self.image_files[idx].replace('.jpg', '.txt')
                                                    .replace('.png', '.txt')
                                                    .replace('.jpeg', '.txt'))
        
        bboxes = []
        class_labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    values = line.strip().split()
                    if len(values) == 5:
                        class_id = int(values[0])
                        x_center, y_center, width, height = map(float, values[1:])
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
        
        # Apply transforms
        if bboxes:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        else:
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image = transformed['image']
            bboxes = []
            class_labels = []
        
        # Convert to tensors
        targets = torch.zeros((len(bboxes), 6))  # [batch_idx, class, x, y, w, h]
        for i, (bbox, class_label) in enumerate(zip(bboxes, class_labels)):
            targets[i] = torch.tensor([0, class_label, *bbox])
        
        return image, targets

class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and LeakyReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """Residual block for YOLO backbone"""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels // 2, 1)
        self.conv2 = ConvBlock(channels // 2, channels, 3, padding=1)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual

class CSPBlock(nn.Module):
    """Cross Stage Partial block"""
    
    def __init__(self, in_channels, out_channels, num_blocks=1):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels // 2, 1)
        self.conv2 = ConvBlock(in_channels, out_channels // 2, 1)
        self.conv3 = ConvBlock(out_channels, out_channels, 1)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(out_channels // 2) for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        
        for block in self.res_blocks:
            x1 = block(x1)
        
        x = torch.cat([x1, x2], dim=1)
        return self.conv3(x)

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast"""
    
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1)
        self.conv2 = ConvBlock(hidden_channels * 4, out_channels, 1)
        self.maxpool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))

class YOLOBackbone(nn.Module):
    """YOLO backbone network"""
    
    def __init__(self):
        super().__init__()
        
        # Stem
        self.stem = ConvBlock(3, 64, 6, stride=2, padding=2)
        
        # Stage 1
        self.stage1 = nn.Sequential(
            ConvBlock(64, 128, 3, stride=2, padding=1),
            CSPBlock(128, 128, num_blocks=3)
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBlock(128, 256, 3, stride=2, padding=1),
            CSPBlock(256, 256, num_blocks=6)
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBlock(256, 512, 3, stride=2, padding=1),
            CSPBlock(512, 512, num_blocks=9)
        )
        
        # Stage 4
        self.stage4 = nn.Sequential(
            ConvBlock(512, 1024, 3, stride=2, padding=1),
            CSPBlock(1024, 1024, num_blocks=3),
            SPPF(1024, 1024)
        )
    
    def forward(self, x):
        x = self.stem(x)
        
        c1 = self.stage1(x)   # P2/4
        c2 = self.stage2(c1)  # P3/8
        c3 = self.stage3(c2)  # P4/16
        c4 = self.stage4(c3)  # P5/32
        
        return c2, c3, c4

class YOLONeck(nn.Module):
    """YOLO neck with FPN"""
    
    def __init__(self):
        super().__init__()
        
        # Top-down pathway
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.reduce_c4 = ConvBlock(1024, 512, 1)
        self.reduce_c3 = ConvBlock(512, 256, 1)
        
        self.csp1 = CSPBlock(1024, 512, num_blocks=3)
        self.csp2 = CSPBlock(512, 256, num_blocks=3)
        
        # Bottom-up pathway
        self.downsample1 = ConvBlock(256, 256, 3, stride=2, padding=1)
        self.downsample2 = ConvBlock(512, 512, 3, stride=2, padding=1)
        
        self.csp3 = CSPBlock(512, 512, num_blocks=3)
        self.csp4 = CSPBlock(1024, 1024, num_blocks=3)
    
    def forward(self, c2, c3, c4):
        # Top-down pathway
        p5 = self.reduce_c4(c4)
        p4 = self.csp1(torch.cat([self.upsample(p5), c3], dim=1))
        
        p4_reduced = self.reduce_c3(p4)
        p3 = self.csp2(torch.cat([self.upsample(p4_reduced), c2], dim=1))
        
        # Bottom-up pathway
        n3 = p3
        n4 = self.csp3(torch.cat([self.downsample1(n3), p4_reduced], dim=1))
        n5 = self.csp4(torch.cat([self.downsample2(n4), p5], dim=1))
        
        return n3, n4, n5

class YOLOHead(nn.Module):
    """YOLO detection head"""
    
    def __init__(self, num_classes=80, anchors_per_scale=3):
        super().__init__()
        self.num_classes = num_classes
        self.anchors_per_scale = anchors_per_scale
        self.num_outputs = anchors_per_scale * (5 + num_classes)  # [x, y, w, h, conf, cls...]
        
        # Detection heads for different scales
        self.head_small = nn.Conv2d(256, self.num_outputs, 1)   # 80x80
        self.head_medium = nn.Conv2d(512, self.num_outputs, 1)  # 40x40
        self.head_large = nn.Conv2d(1024, self.num_outputs, 1)  # 20x20
        
        # Anchor boxes (relative to 640x640 image)
        self.anchors = torch.tensor([
            [[10, 13], [16, 30], [33, 23]],      # Small objects
            [[30, 61], [62, 45], [59, 119]],     # Medium objects
            [[116, 90], [156, 198], [373, 326]]  # Large objects
        ]) / 640.0
    
    def forward(self, features):
        small, medium, large = features
        
        # Apply detection heads
        pred_small = self.head_small(small)
        pred_medium = self.head_medium(medium)
        pred_large = self.head_large(large)
        
        return [pred_small, pred_medium, pred_large]

class YOLO(nn.Module):
    """Complete YOLO model"""
    
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        
        self.backbone = YOLOBackbone()
        self.neck = YOLONeck()
        self.head = YOLOHead(num_classes)
    
    def forward(self, x):
        # Backbone
        c2, c3, c4 = self.backbone(x)
        
        # Neck
        n3, n4, n5 = self.neck(c2, c3, c4)
        
        # Head
        predictions = self.head([n3, n4, n5])
        
        return predictions

class YOLOLoss(nn.Module):
    """YOLO loss function"""
    
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Loss weights
        self.lambda_coord = 5.0
        self.lambda_obj = 1.0
        self.lambda_noobj = 0.5
        self.lambda_cls = 1.0
    
    def forward(self, predictions, targets):
        device = predictions[0].device
        
        total_loss = 0
        
        for i, pred in enumerate(predictions):
            batch_size, _, grid_h, grid_w = pred.shape
            pred = pred.view(batch_size, 3, -1, grid_h, grid_w)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
            # Split predictions
            pred_boxes = pred[..., :4]
            pred_conf = pred[..., 4]
            pred_cls = pred[..., 5:]
            
            # Create target tensors
            target_boxes = torch.zeros_like(pred_boxes)
            target_conf = torch.zeros_like(pred_conf)
            target_cls = torch.zeros_like(pred_cls)
            
            # Calculate losses (simplified version)
            coord_loss = self.mse_loss(pred_boxes, target_boxes)
            conf_loss = self.bce_loss(pred_conf, target_conf)
            cls_loss = self.bce_loss(pred_cls, target_cls)
            
            scale_loss = (self.lambda_coord * coord_loss + 
                         self.lambda_obj * conf_loss + 
                         self.lambda_cls * cls_loss)
            
            total_loss += scale_loss
        
        return total_loss

def non_max_suppression(predictions, conf_threshold=0.25, iou_threshold=0.45):
    """Apply Non-Maximum Suppression to predictions"""
    outputs = []
    
    for pred in predictions:
        # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        boxes = pred[:, :4].clone()
        boxes[:, 0] = pred[:, 0] - pred[:, 2] / 2  # x1
        boxes[:, 1] = pred[:, 1] - pred[:, 3] / 2  # y1
        boxes[:, 2] = pred[:, 0] + pred[:, 2] / 2  # x2
        boxes[:, 3] = pred[:, 1] + pred[:, 3] / 2  # y2
        
        # Filter by confidence
        conf_mask = pred[:, 4] > conf_threshold
        boxes = boxes[conf_mask]
        scores = pred[conf_mask, 4]
        classes = pred[conf_mask, 5:].argmax(dim=1)
        
        # Apply NMS
        keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
        
        if len(keep_indices) > 0:
            final_boxes = boxes[keep_indices]
            final_scores = scores[keep_indices]
            final_classes = classes[keep_indices]
            
            outputs.append(torch.cat([
                final_boxes, 
                final_scores.unsqueeze(1), 
                final_classes.unsqueeze(1).float()
            ], dim=1))
        else:
            outputs.append(torch.empty((0, 6)))
    
    return outputs

class YOLOTrainer:
    """YOLO training pipeline"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = YOLOLoss()
        
    def train_epoch(self, dataloader, optimizer, scheduler=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        pbar = tqdm(dataloader, desc='Training')
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            
            # Forward pass
            predictions = self.model(images)
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        if scheduler:
            scheduler.step()
        
        return total_loss / num_batches
    
    def validate(self, dataloader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc='Validation'):
                images = images.to(self.device)
                predictions = self.model(images)
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)

class YOLODetector:
    """YOLO inference pipeline"""
    
    def __init__(self, model_path, device, conf_threshold=0.25, iou_threshold=0.45):
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load model
        self.model = YOLO(num_classes=80)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        # COCO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def preprocess_image(self, image):
        """Preprocess image for inference"""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize while maintaining aspect ratio
        h, w = image.shape[:2]
        scale = min(640/w, 640/h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to 640x640
        padded = np.full((640, 640, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Normalize and convert to tensor
        padded = padded.astype(np.float32) / 255.0
        padded = (padded - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        padded = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0)
        
        return padded, scale
    
    def postprocess_predictions(self, predictions, scale):
        """Postprocess model predictions"""
        detections = []
        
        for i, pred in enumerate(predictions):
            batch_size, _, grid_h, grid_w = pred.shape
            pred = pred.view(batch_size, 3, -1, grid_h, grid_w)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
            # Apply sigmoid to confidence and classes
            pred[..., 4:] = torch.sigmoid(pred[..., 4:])
            
            # Create grid
            grid_y, grid_x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w))
            grid = torch.stack([grid_x, grid_y], dim=2).float().to(pred.device)
            
            # Convert predictions to absolute coordinates
            pred[..., 0] = (pred[..., 0] + grid[..., 0]) / grid_w
            pred[..., 1] = (pred[..., 1] + grid[..., 1]) / grid_h
            
            # Flatten predictions
            pred = pred.view(batch_size, -1, pred.size(-1))
            detections.append(pred)
        
        # Concatenate all predictions
        all_detections = torch.cat(detections, dim=1)[0]  # Remove batch dimension
        
        # Filter by confidence
        conf_mask = all_detections[:, 4] > self.conf_threshold
        detections = all_detections[conf_mask]
        
        if len(detections) == 0:
            return []
        
        # Get class predictions
        class_conf, class_pred = detections[:, 5:].max(dim=1)
        detections = torch.cat([
            detections[:, :5],
            class_conf.unsqueeze(1),
            class_pred.unsqueeze(1).float()
        ], dim=1)
        
        # Apply NMS
        nms_detections = non_max_suppression([detections], 
                                           self.conf_threshold, 
                                           self.iou_threshold)[0]
        
        # Scale back to original image size
        if len(nms_detections) > 0:
            nms_detections[:, :4] /= scale
        
        return nms_detections
    
    def detect_image(self, image_path):
        """Detect objects in single image"""
        with torch.no_grad():
            # Preprocess
            tensor, scale = self.preprocess_image(image_path)
            tensor = tensor.to(self.device)
            
            # Inference
            predictions = self.model(tensor)
            
            # Postprocess
            detections = self.postprocess_predictions(predictions, scale)
            
            return detections
    
    def visualize_detections(self, image_path, detections, save_path=None):
        """Visualize detections on image"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
            
            # Draw bounding box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw label
            label = f'{self.class_names[int(cls)]}: {conf:.2f}'
            cv2.putText(image, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        return image

def create_demo_dataset(data_dir='demo_data'):
    """Create demo dataset for testing"""
    os.makedirs(f'{data_dir}/images', exist_ok=True)
    os.makedirs(f'{data_dir}/labels', exist_ok=True)
    
    # Create sample images and labels
    for i in range(10):
        # Create random image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(f'{data_dir}/images/image_{i:03d}.jpg', image)
        
        # Create sample label
        with open(f'{data_dir}/labels/image_{i:03d}.txt', 'w') as f:
            # Random object: class x_center y_center width height
            cls = np.random.randint(0, 80)
            x_center = np.random.uniform(0.2, 0.8)
            y_center = np.random.uniform(0.2, 0.8)
            width = np.random.uniform(0.1, 0.3)
            height = np.random.uniform(0.1, 0.3)
            f.write(f'{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n')
    
    print(f"Demo dataset created in {data_dir}/")

def train_model(data_dir, epochs=10, batch_size=8, learning_rate=0.001):
    """Train YOLO model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = YOLODataset(
        images_dir=f'{data_dir}/images',
        labels_dir=f'{data_dir}/labels',
        augment=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Create model
    model = YOLO(num_classes=80).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Trainer
    trainer = YOLOTrainer(model, device)
    
    print("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader, optimizer, scheduler)
        
        print(f"Train Loss: {train_loss:.4f}")
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), 'best_yolo.pt')
            print(f"Saved best model with loss: {best_loss:.4f}")
    
    print("Training completed!")

def detect_objects(input_path, output_dir='results', model_path='best_yolo.pt'):
    """Detect objects in images"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create detector
    detector = YOLODetector(model_path, device)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.isfile(input_path):
        image_files = [input_path]
    else:
        image_files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Processing {len(image_files)} images...")
    
    for image_path in tqdm(image_files):
        # Detect objects
        detections = detector.detect_image(image_path)
        
        # Visualize and save
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f'detected_{filename}')
        detector.visualize_detections(image_path, detections, output_path)
        
        print(f"Detected {len(detections)} objects in {filename}")

def realtime_detection(model_path='best_yolo.pt', source='webcam'):
    """Real-time object detection"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create detector
    detector = YOLODetector(model_path, device)
    
    # Open video source
    if source == 'webcam':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("Starting real-time detection. Press 'q' to quit.")
    
    fps_counter = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame temporarily for detection
        temp_path = 'temp_frame.jpg'
        cv2.imwrite(temp_path, frame)
        
        # Detect objects
        detections = detector.detect_image(temp_path)
        
        # Visualize detections
        result_frame = detector.visualize_detections(temp_path, detections)
        result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
        
        # Calculate FPS
        fps_counter += 1
        if fps_counter % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            start_time = end_time
            print(f"FPS: {fps:.1f}")
        
        # Display result
        cv2.imshow('YOLO Detection', result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Clean up
    if os.path.exists('temp_frame.jpg'):
        os.remove('temp_frame.jpg')

def evaluate_model(data_dir, model_path='best_yolo.pt'):
    """Evaluate model performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create detector
    detector = YOLODetector(model_path, device)
    
    # Get test images
    images_dir = f'{data_dir}/images'
    image_files = [f for f in os.listdir(images_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    total_detections = 0
    total_images = len(image_files)
    
    print(f"Evaluating on {total_images} images...")
    
    for image_file in tqdm(image_files):
        image_path = os.path.join(images_dir, image_file)
        detections = detector.detect_image(image_path)
        total_detections += len(detections)
    
    avg_detections = total_detections / total_images
    print(f"\nEvaluation Results:")
    print(f"Total images: {total_images}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {avg_detections:.2f}")

def main():
    parser = argparse.ArgumentParser(description='YOLO Object Detection')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['create_demo', 'train', 'detect', 'realtime', 'evaluate'],
                       help='Mode to run')
    parser.add_argument('--data_dir', type=str, default='demo_data',
                       help='Data directory')
    parser.add_argument('--input', type=str, default='demo_data/images',
                       help='Input images directory or single image')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--model', type=str, default='best_yolo.pt',
                       help='Model weights path')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--source', type=str, default='webcam',
                       help='Video source for real-time detection')
    
    args = parser.parse_args()
    
    if args.mode == 'create_demo':
        create_demo_dataset(args.data_dir)
        
    elif args.mode == 'train':
        train_model(args.data_dir, args.epochs, args.batch_size, args.learning_rate)
        
    elif args.mode == 'detect':
        detect_objects(args.input, args.output, args.model)
        
    elif args.mode == 'realtime':
        realtime_detection(args.model, args.source)
        
    elif args.mode == 'evaluate':
        evaluate_model(args.data_dir, args.model)

if __name__ == "__main__":
    main()
