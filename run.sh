#!/bin/bash

# 激活虚拟环境并运行应用
echo "Starting Gradio + Hugging Face application..."

# 检查参数
if [ "$1" == "advanced" ]; then
    echo "Running advanced multi-task NLP app..."
    uv run python advanced_app.py
else
    echo "Running simple sentiment analysis app..."
    echo "Tip: Use './run.sh advanced' to run the advanced app"
    uv run python app.py
fi