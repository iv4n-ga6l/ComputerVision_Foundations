"""
Production Computer Vision Pipeline

A comprehensive production-ready computer vision pipeline with MLOps integration,
monitoring, scaling, and enterprise-grade features for deploying CV systems at scale.

Author: Computer Vision Engineer
Date: 2024
"""

import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import time
import uuid
import json
import logging
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import mlflow
import mlflow.pytorch
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
import yaml
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
import queue
from contextlib import asynccontextmanager
import aioredis
import aiofiles
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import boto3
from minio import Minio
import hashlib
import jwt
from passlib.context import CryptContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('cv_pipeline_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('cv_pipeline_request_duration_seconds', 'Request duration')
INFERENCE_DURATION = Histogram('cv_pipeline_inference_duration_seconds', 'Inference duration', ['model'])
ACTIVE_CONNECTIONS = Gauge('cv_pipeline_active_connections', 'Active connections')
MODEL_ACCURACY = Gauge('cv_pipeline_model_accuracy', 'Model accuracy', ['model', 'version'])
ERROR_RATE = Gauge('cv_pipeline_error_rate', 'Error rate')

# Database models
Base = declarative_base()

class InferenceResult(Base):
    __tablename__ = 'inference_results'
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    input_hash = Column(String, nullable=False)
    predictions = Column(JSON, nullable=False)
    confidence_scores = Column(JSON, nullable=False)
    processing_time_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)

class ModelMetrics(Base):
    __tablename__ = 'model_metrics'
    
    id = Column(String, primary_key=True)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    inference_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    avg_processing_time_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

@dataclass
class PredictionRequest:
    """Request model for inference."""
    image_data: bytes
    model_name: Optional[str] = "default"
    confidence_threshold: float = 0.5
    return_features: bool = False
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PredictionResponse:
    """Response model for inference."""
    id: str
    predictions: List[Dict[str, Any]]
    confidence_scores: List[float]
    processing_time_ms: float
    model_name: str
    model_version: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

class ConfigManager:
    """Configuration management with environment overrides."""
    
    def __init__(self, config_path: str = "config/production.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with environment overrides."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            config = self._get_default_config()
        
        # Override with environment variables
        self._override_with_env(config)
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 4,
                'max_request_size': 10485760  # 10MB
            },
            'models': {
                'default': {
                    'name': 'resnet50',
                    'path': 'models/resnet50.pth',
                    'type': 'classification',
                    'batch_size': 32,
                    'warmup': True
                }
            },
            'database': {
                'url': 'postgresql://user:pass@localhost:5432/cvpipeline',
                'pool_size': 10,
                'max_overflow': 20
            },
            'cache': {
                'url': 'redis://localhost:6379/0',
                'ttl': 3600,
                'max_connections': 10
            },
            'storage': {
                'type': 'local',  # local, s3, minio
                'bucket': 'cv-pipeline-models',
                'path': './models'
            },
            'monitoring': {
                'prometheus': {'enabled': True, 'port': 9090},
                'sentry': {'enabled': False, 'dsn': ''},
                'log_level': 'INFO'
            },
            'security': {
                'jwt_secret': 'your-secret-key',
                'token_expire_hours': 24,
                'rate_limit': {'requests_per_minute': 100}
            }
        }
    
    def _override_with_env(self, config: Dict[str, Any]):
        """Override configuration with environment variables."""
        # Database URL
        if 'DATABASE_URL' in os.environ:
            config['database']['url'] = os.environ['DATABASE_URL']
        
        # Redis URL
        if 'REDIS_URL' in os.environ:
            config['cache']['url'] = os.environ['REDIS_URL']
        
        # JWT Secret
        if 'JWT_SECRET' in os.environ:
            config['security']['jwt_secret'] = os.environ['JWT_SECRET']
        
        # MLflow tracking URI
        if 'MLFLOW_TRACKING_URI' in os.environ:
            config['mlflow'] = {'tracking_uri': os.environ['MLFLOW_TRACKING_URI']}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key path (e.g., 'api.host')."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

class ModelManager:
    """Manages model loading, versioning, and inference."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.models = {}
        self.model_versions = {}
        self.model_metrics = {}
        self.lock = threading.Lock()
        
        # Initialize MLflow
        mlflow_uri = config.get('mlflow.tracking_uri', 'file:./mlruns')
        mlflow.set_tracking_uri(mlflow_uri)
        
        self._load_models()
    
    def _load_models(self):
        """Load all configured models."""
        models_config = self.config.get('models', {})
        
        for model_name, model_config in models_config.items():
            try:
                self._load_model(model_name, model_config)
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
    
    def _load_model(self, model_name: str, model_config: Dict[str, Any]):
        """Load a specific model."""
        model_path = model_config['path']
        model_type = model_config.get('type', 'classification')
        
        # Load model based on type
        if model_type == 'classification':
            model = self._load_classification_model(model_path)
        elif model_type == 'object_detection':
            model = self._load_detection_model(model_path)
        elif model_type == 'segmentation':
            model = self._load_segmentation_model(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Set up preprocessing
        transform = self._get_transform(model_type)
        
        with self.lock:
            self.models[model_name] = {
                'model': model,
                'type': model_type,
                'transform': transform,
                'config': model_config,
                'loaded_at': datetime.utcnow(),
                'inference_count': 0
            }
            
            # Version tracking
            version = model_config.get('version', '1.0.0')
            self.model_versions[model_name] = version
            
            # Initialize metrics
            self.model_metrics[model_name] = {
                'accuracy': 0.0,
                'inference_count': 0,
                'error_count': 0,
                'total_time_ms': 0.0
            }
        
        # Warmup model if configured
        if model_config.get('warmup', False):
            self._warmup_model(model_name)
    
    def _load_classification_model(self, model_path: str) -> torch.nn.Module:
        """Load classification model."""
        if model_path.endswith('.pth') or model_path.endswith('.pt'):
            # PyTorch model
            model = torch.load(model_path, map_location='cpu')
            model.eval()
            return model
        else:
            # Load from MLflow
            return mlflow.pytorch.load_model(model_path)
    
    def _load_detection_model(self, model_path: str) -> torch.nn.Module:
        """Load object detection model."""
        # Simplified YOLO loading
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        model.eval()
        return model
    
    def _load_segmentation_model(self, model_path: str) -> torch.nn.Module:
        """Load segmentation model."""
        # Placeholder for segmentation model loading
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        return model
    
    def _get_transform(self, model_type: str) -> transforms.Compose:
        """Get preprocessing transforms for model type."""
        if model_type == 'classification':
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        elif model_type == 'object_detection':
            return transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor()
            ])
        else:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
    
    def _warmup_model(self, model_name: str):
        """Warmup model with dummy input."""
        try:
            model_info = self.models[model_name]
            model = model_info['model']
            transform = model_info['transform']
            
            # Create dummy input
            dummy_image = Image.new('RGB', (224, 224), color='red')
            dummy_input = transform(dummy_image).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                _ = model(dummy_input)
            
            logger.info(f"Model {model_name} warmed up successfully")
            
        except Exception as e:
            logger.error(f"Failed to warmup model {model_name}: {e}")
    
    def infer(self, model_name: str, image_data: bytes, 
              confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Run inference on image data."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_info = self.models[model_name]
        model = model_info['model']
        transform = model_info['transform']
        model_type = model_info['type']
        
        start_time = time.time()
        
        try:
            # Preprocess image
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                if model_type == 'classification':
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    predictions = self._process_classification_output(probabilities, confidence_threshold)
                
                elif model_type == 'object_detection':
                    outputs = model(input_tensor)
                    predictions = self._process_detection_output(outputs, confidence_threshold)
                
                elif model_type == 'segmentation':
                    outputs = model(input_tensor)
                    predictions = self._process_segmentation_output(outputs, confidence_threshold)
                
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update metrics
            with self.lock:
                self.model_metrics[model_name]['inference_count'] += 1
                self.model_metrics[model_name]['total_time_ms'] += processing_time
            
            # Record Prometheus metrics
            INFERENCE_DURATION.labels(model=model_name).observe(processing_time / 1000)
            
            return {
                'predictions': predictions['predictions'],
                'confidence_scores': predictions['confidence_scores'],
                'processing_time_ms': processing_time,
                'model_version': self.model_versions[model_name]
            }
            
        except Exception as e:
            # Update error metrics
            with self.lock:
                self.model_metrics[model_name]['error_count'] += 1
            
            logger.error(f"Inference error for model {model_name}: {e}")
            raise
    
    def _process_classification_output(self, probabilities: torch.Tensor,
                                     confidence_threshold: float) -> Dict[str, Any]:
        """Process classification model output."""
        probs = probabilities.cpu().numpy()[0]
        
        # Get top predictions above threshold
        indices = np.where(probs >= confidence_threshold)[0]
        
        predictions = []
        confidence_scores = []
        
        for idx in indices:
            predictions.append({
                'class_id': int(idx),
                'class_name': f'class_{idx}',  # Would map to actual class names
                'confidence': float(probs[idx])
            })
            confidence_scores.append(float(probs[idx]))
        
        # Sort by confidence
        sorted_indices = np.argsort(confidence_scores)[::-1]
        predictions = [predictions[i] for i in sorted_indices]
        confidence_scores = [confidence_scores[i] for i in sorted_indices]
        
        return {'predictions': predictions, 'confidence_scores': confidence_scores}
    
    def _process_detection_output(self, outputs: torch.Tensor,
                                confidence_threshold: float) -> Dict[str, Any]:
        """Process object detection model output."""
        # Simplified detection processing
        detections = outputs.cpu().numpy()
        
        predictions = []
        confidence_scores = []
        
        # Process detections above threshold
        for detection in detections:
            if len(detection) > 4 and detection[4] >= confidence_threshold:
                predictions.append({
                    'bbox': [float(x) for x in detection[:4]],
                    'class_id': int(detection[5]) if len(detection) > 5 else 0,
                    'confidence': float(detection[4])
                })
                confidence_scores.append(float(detection[4]))
        
        return {'predictions': predictions, 'confidence_scores': confidence_scores}
    
    def _process_segmentation_output(self, outputs: torch.Tensor,
                                   confidence_threshold: float) -> Dict[str, Any]:
        """Process segmentation model output."""
        # Simplified segmentation processing
        segmentation_map = outputs.cpu().numpy()[0]
        
        # Get unique classes above threshold
        unique_classes = np.unique(segmentation_map)
        
        predictions = []
        confidence_scores = []
        
        for class_id in unique_classes:
            if class_id > 0:  # Ignore background
                mask_area = np.sum(segmentation_map == class_id)
                confidence = min(1.0, mask_area / (segmentation_map.size * 0.1))
                
                if confidence >= confidence_threshold:
                    predictions.append({
                        'class_id': int(class_id),
                        'mask_area': int(mask_area),
                        'confidence': float(confidence)
                    })
                    confidence_scores.append(float(confidence))
        
        return {'predictions': predictions, 'confidence_scores': confidence_scores}
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a loaded model."""
        if model_name not in self.models:
            return None
        
        model_info = self.models[model_name]
        metrics = self.model_metrics[model_name]
        
        avg_time = (metrics['total_time_ms'] / metrics['inference_count'] 
                   if metrics['inference_count'] > 0 else 0.0)
        
        return {
            'name': model_name,
            'type': model_info['type'],
            'version': self.model_versions[model_name],
            'loaded_at': model_info['loaded_at'].isoformat(),
            'inference_count': metrics['inference_count'],
            'error_count': metrics['error_count'],
            'avg_processing_time_ms': avg_time,
            'accuracy': metrics['accuracy']
        }
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all loaded models."""
        return [self.get_model_info(name) for name in self.models.keys()]

class DatabaseManager:
    """Database operations manager."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        database_url = config.get('database.url')
        pool_size = config.get('database.pool_size', 10)
        max_overflow = config.get('database.max_overflow', 20)
        
        self.engine = create_engine(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True
        )
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
    
    def save_inference_result(self, result: PredictionResponse, user_id: str, input_hash: str):
        """Save inference result to database."""
        session = self.get_session()
        try:
            db_result = InferenceResult(
                id=result.id,
                user_id=user_id,
                model_name=result.model_name,
                model_version=result.model_version,
                input_hash=input_hash,
                predictions=result.predictions,
                confidence_scores=result.confidence_scores,
                processing_time_ms=result.processing_time_ms,
                metadata=result.metadata
            )
            session.add(db_result)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save inference result: {e}")
            raise
        finally:
            session.close()
    
    def get_inference_history(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get inference history for user."""
        session = self.get_session()
        try:
            results = session.query(InferenceResult)\
                           .filter_by(user_id=user_id)\
                           .order_by(InferenceResult.created_at.desc())\
                           .limit(limit)\
                           .all()
            
            return [
                {
                    'id': r.id,
                    'model_name': r.model_name,
                    'model_version': r.model_version,
                    'predictions': r.predictions,
                    'confidence_scores': r.confidence_scores,
                    'processing_time_ms': r.processing_time_ms,
                    'created_at': r.created_at.isoformat(),
                    'metadata': r.metadata
                }
                for r in results
            ]
        finally:
            session.close()

class CacheManager:
    """Redis cache manager."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.redis_url = config.get('cache.url')
        self.ttl = config.get('cache.ttl', 3600)
        self.redis_client = None
        self._connect()
    
    def _connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                max_connections=self.config.get('cache.max_connections', 10),
                decode_responses=False
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def get(self, key: str) -> Optional[bytes]:
        """Get value from cache."""
        if not self.redis_client:
            return None
        
        try:
            return self.redis_client.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self.redis_client:
            return False
        
        try:
            ttl = ttl or self.ttl
            return self.redis_client.setex(key, ttl, value)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if not self.redis_client:
            return False
        
        try:
            return self.redis_client.delete(key) > 0
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def generate_key(self, prefix: str, *args) -> str:
        """Generate cache key."""
        key_parts = [prefix] + [str(arg) for arg in args]
        key_string = ':'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

class AuthManager:
    """Authentication and authorization manager."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.jwt_secret = config.get('security.jwt_secret')
        self.token_expire_hours = config.get('security.token_expire_hours', 24)
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Check expiration
            exp = payload.get('exp')
            if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
                return None
            
            return payload
        except jwt.PyJWTError:
            return None
    
    def create_token(self, user_id: str, permissions: List[str] = None) -> str:
        """Create JWT token."""
        expire = datetime.utcnow() + timedelta(hours=self.token_expire_hours)
        payload = {
            'user_id': user_id,
            'permissions': permissions or [],
            'exp': expire,
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

# Global instances
config = ConfigManager()
model_manager = ModelManager(config)
db_manager = DatabaseManager(config)
cache_manager = CacheManager(config)
auth_manager = AuthManager(config)

# Security
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current user from JWT token."""
    token = credentials.credentials
    payload = auth_manager.verify_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return payload

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Production CV Pipeline...")
    
    # Initialize monitoring
    if config.get('monitoring.prometheus.enabled', True):
        logger.info("Prometheus metrics enabled")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Production CV Pipeline...")

# FastAPI application
app = FastAPI(
    title="Production Computer Vision Pipeline",
    description="Enterprise-grade computer vision inference API",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request/Response middleware for metrics
@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Middleware for collecting metrics."""
    start_time = time.time()
    
    ACTIVE_CONNECTIONS.inc()
    
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception as e:
        status = 500
        response = JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )
    
    # Record metrics
    process_time = time.time() - start_time
    REQUEST_DURATION.observe(process_time)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=status
    ).inc()
    
    ACTIVE_CONNECTIONS.dec()
    
    return response

# API Routes
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/v1/infer", response_model=PredictionResponse)
async def infer_single(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    model_name: str = "default",
    confidence_threshold: float = 0.5,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Single image inference endpoint."""
    
    # Validate file type
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image data
    image_data = await image.read()
    
    # Generate input hash for caching
    input_hash = hashlib.md5(image_data + model_name.encode()).hexdigest()
    
    # Check cache
    cache_key = cache_manager.generate_key("inference", input_hash)
    cached_result = cache_manager.get(cache_key)
    
    if cached_result:
        logger.info("Returning cached result")
        return PredictionResponse(**json.loads(cached_result))
    
    try:
        # Run inference
        result = model_manager.infer(model_name, image_data, confidence_threshold)
        
        # Create response
        response = PredictionResponse(
            id=str(uuid.uuid4()),
            predictions=result['predictions'],
            confidence_scores=result['confidence_scores'],
            processing_time_ms=result['processing_time_ms'],
            model_name=model_name,
            model_version=result['model_version'],
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Cache result
        cache_manager.set(cache_key, json.dumps(asdict(response)).encode())
        
        # Save to database (background task)
        background_tasks.add_task(
            db_manager.save_inference_result,
            response,
            current_user['user_id'],
            input_hash
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/v1/batch_infer")
async def infer_batch(
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(...),
    model_name: str = "default",
    confidence_threshold: float = 0.5,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Batch inference endpoint."""
    
    if len(images) > 32:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size too large (max 32)")
    
    results = []
    
    for image in images:
        try:
            # Validate file type
            if not image.content_type.startswith('image/'):
                continue
            
            # Read image data
            image_data = await image.read()
            
            # Run inference
            result = model_manager.infer(model_name, image_data, confidence_threshold)
            
            # Create response
            response = PredictionResponse(
                id=str(uuid.uuid4()),
                predictions=result['predictions'],
                confidence_scores=result['confidence_scores'],
                processing_time_ms=result['processing_time_ms'],
                model_name=model_name,
                model_version=result['model_version'],
                timestamp=datetime.utcnow().isoformat()
            )
            
            results.append(response)
            
        except Exception as e:
            logger.error(f"Batch inference error for image {image.filename}: {e}")
            continue
    
    return {"results": results, "processed_count": len(results)}

@app.get("/v1/models")
async def list_models(current_user: Dict[str, Any] = Depends(get_current_user)):
    """List available models."""
    return {"models": model_manager.list_models()}

@app.get("/v1/models/{model_name}")
async def get_model_info(
    model_name: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get information about a specific model."""
    model_info = model_manager.get_model_info(model_name)
    
    if not model_info:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return model_info

@app.get("/v1/history")
async def get_inference_history(
    limit: int = 100,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get inference history for current user."""
    history = db_manager.get_inference_history(current_user['user_id'], limit)
    return {"history": history}

@app.post("/v1/auth/token")
async def create_auth_token(user_id: str, permissions: List[str] = None):
    """Create authentication token (for testing)."""
    token = auth_manager.create_token(user_id, permissions)
    return {"access_token": token, "token_type": "bearer"}

def create_production_app():
    """Create production-ready FastAPI application."""
    return app

def run_server():
    """Run the server with production configuration."""
    host = config.get('api.host', '0.0.0.0')
    port = config.get('api.port', 8000)
    workers = config.get('api.workers', 4)
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True,
        reload=False
    )

if __name__ == "__main__":
    run_server()
