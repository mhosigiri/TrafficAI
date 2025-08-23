# TrafficAI Docker Deployment Guide

## 🐳 Container Overview

This project includes Docker configurations for easy deployment and scaling of the TrafficAI system.

## 📦 Available Docker Images

### 1. **CPU Version** (`Dockerfile`)
- Lightweight image for CPU-only deployment
- Includes Tesseract OCR
- Suitable for testing or CPU-based inference

### 2. **GPU Version** (`Dockerfile.gpu`)
- NVIDIA CUDA 12.1 support
- Optimized for GPU acceleration
- Includes all ML dependencies with CUDA support

## 🚀 Quick Start

### Build and Run with Docker Compose

```bash
# For development (with GPU support)
docker-compose up -d

# For production deployment
docker-compose -f docker-compose.prod.yml up -d

# For CPU-only version
docker build -t trafficai:cpu .
docker run -it trafficai:cpu
```

## 📋 Docker Services

### Development Services (`docker-compose.yml`)

1. **trafficai** - Main monitoring service with camera access
2. **trafficai-batch** - Batch processing service
3. **trafficai-train** - Model training service

### Production Services (`docker-compose.prod.yml`)

1. **trafficai-prod** - Production monitoring with auto-restart
2. **trafficai-api** - API service (for future web interface)

## 🔧 Configuration

### Environment Variables

```yaml
NVIDIA_VISIBLE_DEVICES: all      # GPU selection
NVIDIA_DRIVER_CAPABILITIES: compute,utility,video
TZ: UTC                          # Timezone
```

### Volume Mounts

- `/app/data` - Training data
- `/app/models` - Trained models
- `/app/violations` - Violation evidence
- `/app/logs` - System logs

## 🖥️ Usage Examples

### 1. Run Health Check
```bash
docker run --rm trafficai:latest python deploy_traffic_monitor.py --mode health
```

### 2. Process Video File
```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  trafficai:latest \
  python deploy_traffic_monitor.py --mode video --input /app/input/traffic.mp4
```

### 3. Batch Process Images
```bash
docker run --rm \
  -v $(pwd)/images:/app/input \
  -v $(pwd)/results:/app/output \
  trafficai:latest \
  python deploy_traffic_monitor.py --mode batch --input /app/input --output /app/output
```

### 4. Live Camera (Linux)
```bash
docker run --rm \
  --device=/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  trafficai:latest \
  python deploy_traffic_monitor.py --mode camera
```

## 🚀 GPU Support

### Prerequisites
- NVIDIA Docker runtime installed
- NVIDIA drivers on host
- CUDA-compatible GPU

### Install NVIDIA Container Toolkit
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Verify GPU Access
```bash
docker run --rm --runtime=nvidia --gpus all trafficai:gpu nvidia-smi
```

## 🏭 Production Deployment

### 1. Build Production Image
```bash
docker build -f Dockerfile.gpu -t trafficai:gpu .
```

### 2. Deploy with Docker Compose
```bash
# Start services
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f trafficai-prod
```

### 3. Scale Services
```bash
# Scale batch processing
docker-compose -f docker-compose.prod.yml up -d --scale trafficai-batch=3
```

## 🔍 Monitoring

### Health Check
```bash
docker exec trafficai-production python deploy_traffic_monitor.py --mode health
```

### Container Logs
```bash
docker logs -f trafficai-production
```

### System Stats
```bash
docker stats trafficai-production
```

## 🛠️ Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi

# Check container GPU access
docker exec trafficai-production python -c "import torch; print(torch.cuda.is_available())"
```

### Permission Issues
```bash
# Fix volume permissions
docker exec trafficai-production chown -R 1000:1000 /app/violations /app/logs
```

### Memory Issues
```bash
# Limit container memory
docker run -m 8g trafficai:gpu
```

## 🔐 Security Considerations

1. **Run as non-root user** (add to Dockerfile):
   ```dockerfile
   RUN useradd -m -u 1000 trafficai
   USER trafficai
   ```

2. **Use secrets for sensitive data**:
   ```yaml
   secrets:
     api_key:
       file: ./secrets/api_key.txt
   ```

3. **Network isolation**:
   ```yaml
   networks:
     frontend:
     backend:
       internal: true
   ```

## 📊 Performance Tuning

### GPU Memory
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0
  - TF_FORCE_GPU_ALLOW_GROWTH=true
```

### CPU Limits
```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
```

## 🚢 Kubernetes Deployment

For Kubernetes deployment, convert docker-compose to K8s manifests:

```bash
kompose convert -f docker-compose.prod.yml
```

Or use the provided Helm chart (future enhancement).

## 📝 Notes

- The container includes all necessary dependencies
- Models should be mounted or copied to `/app/models/`
- Tesseract OCR is pre-installed
- Default command runs health check
- Supports both interactive and batch modes

Happy Containerized Deployment! 🐳🚀
