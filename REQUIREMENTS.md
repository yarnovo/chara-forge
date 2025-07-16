# Chara Forge 需求文档

## 项目概述

Chara Forge 是 Dream Forge 平台的角色图片生成子模块，专注于为独立游戏开发者提供 AI 驱动的游戏角色图片生成服务。该模块旨在帮助缺乏美术技能的程序员快速生成高质量的游戏角色图片资源。

## 目标用户

- **独立游戏开发者**：特别是程序员背景，需要快速生成角色原画的开发者
- **小型游戏开发团队**：需要快速迭代角色设计的团队
- **AI 工具开发者**：需要通过 MCP 接口集成角色生成功能的开发者

## 核心功能

### 1. 角色图片生成

专注于游戏角色图片的生成，支持多种角色类型：
- **角色立绘**：全身、半身、头像等不同构图
- **角色表情**：喜怒哀乐等多种表情变化
- **角色服装**：不同服装和装备的变换
- **角色姿势**：站立、坐姿、动作姿势等

### 2. 三层用户接口

#### 2.1 Gradio Web 界面
- 直观的角色生成参数输入表单
- 角色风格预设（像素艺术、日式动漫、欧美卡通、写实等）
- 实时预览和迭代优化
- 批量生成和变体生成
- 历史记录和收藏功能

#### 2.2 RESTful API 接口
- 完整的角色生成 API 端点
- 支持异步任务处理
- WebSocket 实时进度推送
- 批量任务管理
- API Key 认证机制

#### 2.3 MCP (Model Context Protocol) 接口
- 实现标准 MCP 服务器协议
- 提供 `generate_character` 工具函数
- 支持上下文感知的角色生成
- 结构化的参数验证和错误处理
- 与 AI 助手的无缝集成

### 3. 模型支持

#### 3.1 开源模型
- Stable Diffusion XL（主要模型）
- 专门的角色生成 LoRA 模型
- ControlNet 支持（姿势控制）
- 角色一致性模型（IP-Adapter）

#### 3.2 商业 API（可选）
- OpenAI DALL-E 3
- Midjourney（通过第三方 API）
- NovelAI（动漫风格专精）

## 技术架构

### 1. 核心技术栈
- **后端**：Python 3.12+
- **Web 框架**：Gradio（UI）+ FastAPI（API）
- **模型框架**：Diffusers, PyTorch
- **MCP 框架**：mcp (Model Context Protocol SDK)
- **包管理**：uv
- **容器化**：Docker

### 2. 部署方案
- Docker 镜像分发
- docker-compose 编排
- 支持 GPU 加速（NVIDIA Docker）
- 环境变量配置管理

### 3. 系统架构
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Gradio UI     │     │  RESTful API    │     │   MCP Server    │
│  (Web 界面)      │     │   (FastAPI)      │     │  (MCP SDK)      │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                         │
         └───────────────────────┴─────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │  Character Generator    │
                    │     Core Engine         │
                    └────────────┬────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌────────┴────────┐   ┌─────────┴────────┐   ┌─────────┴────────┐
│  Local Models   │   │  LoRA Models     │   │  Commercial APIs │
│ (SDXL Base)     │   │  (Character)     │   │  (Optional)      │
└─────────────────┘   └──────────────────┘   └──────────────────┘
```

## 功能详细说明

### 1. Gradio Web 界面

**主要功能区域**：
- **角色描述输入**：多行文本框，支持详细的角色描述
- **风格选择器**：下拉菜单，预设多种艺术风格
- **参数调节**：滑动条和数值输入，控制生成细节
- **预览区域**：实时显示生成的角色图片
- **操作面板**：生成、保存、历史记录等操作按钮

**输入参数**：
- 角色描述文本（prompt）
- 负面提示词（negative prompt）
- 艺术风格（anime, cartoon, realistic, pixel art 等）
- 图片尺寸（512x512, 768x768, 1024x1024）
- 生成数量（1-8 张）
- 随机种子（seed）
- 引导强度（guidance scale）
- 推理步数（steps）

**输出**：
- PNG 格式图片（支持透明通道）
- 生成参数的 JSON 记录
- 缩略图预览

### 2. RESTful API 接口

**核心端点**：
```
POST /api/v1/generate/character
GET  /api/v1/tasks/{task_id}/status
GET  /api/v1/tasks/{task_id}/result
POST /api/v1/generate/character/batch
```

**请求参数**：
```json
{
  "prompt": "beautiful fantasy warrior princess",
  "negative_prompt": "low quality, blurry",
  "style": "anime",
  "width": 768,
  "height": 768,
  "num_images": 1,
  "seed": 42,
  "guidance_scale": 7.5,
  "steps": 20,
  "callback_url": "https://your-app.com/webhook"
}
```

**响应格式**：
```json
{
  "task_id": "uuid-string",
  "status": "completed",
  "images": [
    {
      "url": "https://example.com/image.png",
      "base64": "data:image/png;base64,..."
    }
  ],
  "metadata": {
    "generation_time": 15.2,
    "model_used": "stable-diffusion-xl-base-1.0"
  }
}
```

### 3. MCP (Model Context Protocol) 接口

**工具定义**：
```python
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

**使用示例**：
```python
# AI 助手调用
result = await mcp_client.call_tool(
    "generate_character",
    {
        "prompt": "cyberpunk hacker girl with neon hair",
        "style": "anime",
        "size": "768x768"
    }
)
```

**返回数据**：
```python
{
  "success": True,
  "images": [
    {
      "data": "base64_encoded_image",
      "format": "png",
      "width": 768,
      "height": 768
    }
  ],
  "metadata": {
    "model": "stable-diffusion-xl-base-1.0",
    "generation_time": 12.5
  }
}
```

## 非功能性需求

### 1. 性能要求
- 角色图片生成：< 20秒/张（GPU）
- API 响应时间：< 200ms（不含生成时间）
- MCP 调用响应：< 100ms（不含生成时间）
- 并发处理：支持最多 10 个并发生成任务
- 内存使用：< 8GB VRAM（SDXL 模型）

### 2. 可扩展性
- 模块化设计，易于添加新的角色生成模型
- 插件系统支持自定义 LoRA 模型
- 配置驱动的模型切换和参数调整
- 支持水平扩展（多 GPU 部署）

### 3. 用户体验
- 直观的 Gradio 界面设计
- 实时生成进度显示
- 历史记录和收藏功能
- 批量生成和导出功能
- 预设角色模板快速生成

### 4. 安全性
- API Key 认证和访问控制
- 请求速率限制和防滥用机制
- 内容审核（NSFW 检测）
- 用户数据隔离和隐私保护

## 开发计划

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

### Phase 4：生产就绪
- [ ] 完整的 API 文档
- [ ] 部署指南和监控
- [ ] 用户管理和认证系统
- [ ] 性能监控和日志记录

## 成功指标

1. **技术指标**
   - 支持至少 5 种角色艺术风格
   - API 可用性 > 99.5%
   - 平均生成时间 < 20秒
   - MCP 集成成功率 > 95%

2. **用户指标**
   - 用户界面易用性评分 > 4.0/5.0
   - 角色生成质量满意度 > 85%
   - API 集成便利性高
   - MCP 工具使用成功率 > 90%

3. **项目指标**
   - Docker 镜像大小 < 15GB
   - 部署时间 < 15分钟
   - 文档完整性 100%

## 约束和限制

1. **技术约束**
   - 需要 NVIDIA GPU 支持（CUDA 11.8+）
   - 最低内存要求：16GB RAM, 8GB VRAM
   - Python 3.12+ 环境
   - 专注于 2D 角色图片生成

2. **法律约束**
   - 遵守 Stable Diffusion 的开源协议
   - 商业 API 使用需要合法授权
   - 生成内容版权说明
   - NSFW 内容过滤

3. **功能约束**
   - 专注于游戏角色生成
   - 不支持 3D 模型生成
   - 不支持动画生成
   - 图片尺寸限制在 1024x1024 以内

## 风险评估

1. **技术风险**
   - SDXL 模型兼容性问题
   - GPU 资源成本和可用性
   - 生成质量的一致性问题
   - MCP 协议的稳定性

2. **业务风险**
   - 竞品功能快速迭代
   - 用户对角色生成质量的期望
   - 法律法规变化（AI 生成内容）

3. **缓解措施**
   - 建立模型版本管理和测试机制
   - 提供多种部署选项（CPU/GPU）
   - 持续收集用户反馈和质量评估
   - 密切关注 MCP 协议发展

## MCP 接口详细设计

### 1. 服务器配置
```json
{
  "name": "chara-forge",
  "version": "1.0.0",
  "description": "Game character image generation service",
  "author": "Chara Forge Team",
  "homepage": "https://github.com/your-org/chara-forge",
  "license": "MIT"
}
```

### 2. 工具列表
- `generate_character`: 生成游戏角色图片
- `get_character_styles`: 获取支持的艺术风格列表
- `get_generation_status`: 查询生成任务状态
- `list_recent_characters`: 列出最近生成的角色

### 3. 资源类型
- `character_image`: 角色图片资源
- `generation_history`: 生成历史记录
- `style_presets`: 风格预设配置

### 4. 错误处理
- 参数验证失败
- 模型加载失败
- GPU 内存不足
- 生成超时处理
- 内容审核失败