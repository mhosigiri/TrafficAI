@echo off
echo Building TrafficAI Docker Images...
echo ==================================

REM Build CPU version
echo Building CPU version...
docker build -t trafficai:latest -t trafficai:cpu .
if %errorlevel% neq 0 (
    echo Failed to build CPU image
    exit /b %errorlevel%
)

REM Build GPU version
echo.
echo Building GPU version...
docker build -f Dockerfile.gpu -t trafficai:gpu .
if %errorlevel% neq 0 (
    echo Failed to build GPU image
    exit /b %errorlevel%
)

echo.
echo Build complete!
echo.
echo To run:
echo   CPU: docker run -it trafficai:cpu
echo   GPU: docker run --gpus all -it trafficai:gpu
echo.
