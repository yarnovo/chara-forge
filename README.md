# Chara Forge

🎮 **AI 驱动的游戏角色图片生成服务**

Chara Forge 是 Dream Forge 平台的核心子模块，专为独立游戏开发者提供高质量的 AI 角色图片生成服务。无论您是缺乏美术技能的程序员，还是需要快速迭代角色设计的小型团队，Chara Forge 都能帮助您快速生成专业级的游戏角色图片。

## ✨ 特性

- 🎨 **多样化风格支持**：像素艺术、日式动漫、欧美卡通、写实风格等
- 🚀 **三层接口设计**：Gradio Web UI、RESTful API、MCP 协议支持
- 🎯 **游戏角色专精**：专门针对游戏角色设计优化
- 🔧 **模块化架构**：易于扩展和定制化
- 📦 **容器化部署**：Docker 支持，一键部署
- 💡 **AI 助手集成**：通过 MCP 协议与 AI 助手无缝协作

## 🚀 快速开始

### 前置要求

- Python 3.12+
- NVIDIA GPU (推荐 8GB+ VRAM)
- Docker (可选，用于容器化部署)

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-org/chara-forge.git
cd chara-forge

# 使用 uv 安装依赖
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 启动服务

```bash
# 启动 Gradio Web 界面
python -m chara_forge.web

# 启动 FastAPI 服务器
python -m chara_forge.api

# 启动 MCP 服务器
python -m chara_forge.mcp
```

### Docker 部署

```bash
# 构建镜像
docker build -t chara-forge .

# 启动服务
docker-compose up -d
```

## 💻 使用方式

### 1. Gradio Web 界面

访问 `http://localhost:7860` 打开 Web 界面：

1. 在文本框中输入角色描述
2. 选择艺术风格（动漫、卡通、写实等）
3. 调整生成参数（图片尺寸、数量等）
4. 点击生成按钮获取角色图片

### 2. RESTful API

```bash
# 生成角色图片
curl -X POST "http://localhost:8000/api/v1/generate/character" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "beautiful fantasy warrior princess",
    "style": "anime",
    "width": 768,
    "height": 768,
    "num_images": 1
  }'
```

### 3. MCP 协议集成

```python
# 在 AI 助手中调用
result = await mcp_client.call_tool(
    "generate_character",
    {
        "prompt": "cyberpunk hacker girl with neon hair",
        "style": "anime",
        "size": "768x768"
    }
)
```

## 🛠️ 技术栈

- **后端**：Python 3.12, FastAPI, Gradio
- **AI 模型**：Stable Diffusion XL, LoRA Models
- **包管理**：uv
- **容器化**：Docker
- **协议**：MCP (Model Context Protocol)

## 📁 项目结构

```
chara-forge/
├── chara_forge/           # 核心模块
│   ├── core/             # 核心生成引擎
│   ├── web/              # Gradio Web 界面
│   ├── api/              # FastAPI 服务
│   ├── mcp/              # MCP 服务器
│   └── models/           # 模型管理
├── models/               # 预训练模型
├── configs/              # 配置文件
├── tests/                # 测试用例
├── docs/                 # 文档
├── docker/               # Docker 相关文件
└── scripts/              # 工具脚本
```

## 🎯 支持的角色类型

- **角色立绘**：全身、半身、头像
- **角色表情**：喜怒哀乐等表情变化
- **角色服装**：不同服装和装备
- **角色姿势**：站立、坐姿、动作姿势
- **角色风格**：像素艺术、动漫、卡通、写实

## 🔧 配置

### 环境变量

```bash
# 模型路径
MODEL_PATH=/path/to/models

# GPU 设置
CUDA_VISIBLE_DEVICES=0

# API 配置
API_HOST=0.0.0.0
API_PORT=8000

# MCP 配置
MCP_HOST=localhost
MCP_PORT=8001
```

### 模型配置

在 `configs/models.yaml` 中配置使用的模型：

```yaml
models:
  base_model: "stabilityai/stable-diffusion-xl-base-1.0"
  lora_models:
    - "character-lora-v1"
    - "anime-style-lora"
  controlnet_models:
    - "controlnet-openpose"
```

## 📚 API 文档

### 核心端点

- `POST /api/v1/generate/character` - 生成角色图片
- `GET /api/v1/tasks/{task_id}/status` - 查询任务状态
- `GET /api/v1/tasks/{task_id}/result` - 获取生成结果
- `POST /api/v1/generate/character/batch` - 批量生成

详细的 API 文档请访问：`http://localhost:8000/docs`

## 🤝 MCP 工具

### 可用工具

- `generate_character` - 生成游戏角色图片
- `get_character_styles` - 获取支持的艺术风格
- `get_generation_status` - 查询生成任务状态
- `list_recent_characters` - 列出最近生成的角色

### 集成示例

```json
{
  "name": "generate_character",
  "description": "Generate game character images using AI",
  "input_schema": {
    "type": "object",
    "properties": {
      "prompt": {
        "type": "string",
        "description": "Character description"
      },
      "style": {
        "type": "string",
        "enum": ["anime", "cartoon", "realistic", "pixel"],
        "description": "Art style"
      },
      "size": {
        "type": "string",
        "enum": ["512x512", "768x768", "1024x1024"],
        "description": "Image dimensions"
      }
    },
    "required": ["prompt"]
  }
}
```

## 🔍 性能指标

- 角色图片生成：< 20秒/张（GPU）
- API 响应时间：< 200ms（不含生成时间）
- MCP 调用响应：< 100ms（不含生成时间）
- 并发处理：支持最多 10 个并发任务
- 内存使用：< 8GB VRAM（SDXL 模型）

## 🛡️ 安全特性

- API Key 认证和访问控制
- 请求速率限制
- 内容审核（NSFW 检测）
- 用户数据隔离

## 📊 开发状态

### Phase 1：核心功能（MVP）
- [x] 基础 Gradio 界面框架
- [ ] 角色图片生成功能（SDXL）
- [ ] RESTful API 实现
- [ ] Docker 容器化

### Phase 2：接口扩展
- [ ] MCP 服务器实现
- [ ] WebSocket 实时推送
- [ ] 批量处理支持
- [ ] 基础 LoRA 模型集成

### Phase 3：功能优化
- [ ] 角色一致性模型（IP-Adapter）
- [ ] ControlNet 姿势控制
- [ ] 商业 API 集成（可选）
- [ ] 性能优化和缓存机制

## 🤝 贡献指南

我们欢迎各种形式的贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与项目。

### 开发环境设置

```bash
# 安装开发依赖
uv sync --dev

# 运行测试
make test

# 代码格式化
make format

# 类型检查
make check
```

## 📄 许可证

本项目基于 MIT 许可证开源。详情请参阅 [LICENSE](LICENSE) 文件。

## 🆘 支持

- 📖 [文档](docs/)
- 🐛 [问题反馈](https://github.com/your-org/chara-forge/issues)
- 💬 [讨论区](https://github.com/your-org/chara-forge/discussions)
- 📧 [邮件支持](mailto:support@chara-forge.com)

## 🏆 致谢

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - 核心图片生成模型
- [Gradio](https://gradio.app/) - Web 界面框架
- [FastAPI](https://fastapi.tiangolo.com/) - API 框架
- [MCP](https://github.com/modelcontextprotocol/python-sdk) - 协议支持

---

🎨 **让 AI 为您的游戏角色注入生命力！**