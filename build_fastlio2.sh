#!/bin/bash

set -e 

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

git submodule update --init --recursive

source /opt/ros/humble/setup.bash
cd ros2_ws

PROJECT_ROOT=$(pwd)
log_info "ws path: $PROJECT_ROOT"

log_info "step 1/6: 安装 cmake..."
sudo apt update && sudo apt install -y cmake

log_info "step 2/6: 编译 Livox-SDK2..."
cd src/Livox-SDK2
mkdir -p build && cd build
cmake .. && make -j$(nproc)
sudo make install

log_info "step 3/6: 编译 livox_ros_driver2..."
cd "$PROJECT_ROOT/src/ws_livox/src/livox_ros_driver2"
./build.sh humble

cd "$PROJECT_ROOT"

log_info "step 4/6: 加载Livox环境..."
source src/ws_livox/install/setup.bash

log_info "step 5/6: 复制FAST_LIO配置文件..."

cp -r ../src/fastlio2_config/* src/FAST_LIO/config/
log_info "✓ 配置文件已复制到 src/FAST_LIO/config/"

log_info "step 6/6: 编译 FAST_LIO..."
colcon build --symlink-install --packages-select fast_lio

log_info "✓ 全部构建完成！"