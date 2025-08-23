.PHONY: help build build-gpu run run-gpu test clean push

# Default target
help:
	@echo "TrafficAI Docker Commands:"
	@echo "  make build       - Build CPU Docker image"
	@echo "  make build-gpu   - Build GPU Docker image"
	@echo "  make run         - Run container (CPU)"
	@echo "  make run-gpu     - Run container (GPU)"
	@echo "  make test        - Run health check"
	@echo "  make clean       - Clean up containers and images"
	@echo "  make compose-up  - Start services with docker-compose"
	@echo "  make compose-down - Stop services"

# Build targets
build:
	docker build -t trafficai:latest -t trafficai:cpu .

build-gpu:
	docker build -f Dockerfile.gpu -t trafficai:gpu .

# Run targets
run:
	docker run --rm -it \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/violations:/app/violations \
		-v $(PWD)/logs:/app/logs \
		trafficai:latest

run-gpu:
	docker run --rm -it \
		--gpus all \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/violations:/app/violations \
		-v $(PWD)/logs:/app/logs \
		trafficai:gpu

# Test target
test:
	docker run --rm trafficai:latest python deploy_traffic_monitor.py --mode health

test-gpu:
	docker run --rm --gpus all trafficai:gpu python deploy_traffic_monitor.py --mode health

# Docker Compose targets
compose-up:
	docker-compose up -d

compose-down:
	docker-compose down

compose-logs:
	docker-compose logs -f

# Production targets
prod-up:
	docker-compose -f docker-compose.prod.yml up -d

prod-down:
	docker-compose -f docker-compose.prod.yml down

prod-logs:
	docker-compose -f docker-compose.prod.yml logs -f

# Clean up
clean:
	docker-compose down -v
	docker system prune -f

# Push to registry (update with your registry)
push:
	docker tag trafficai:gpu your-registry/trafficai:gpu
	docker tag trafficai:latest your-registry/trafficai:latest
	docker push your-registry/trafficai:gpu
	docker push your-registry/trafficai:latest
