# Dream Forge Makefile
# 管理 Gradio 应用的启动、停止和重启

# 默认应用文件
APP ?= app.py
# PID 文件
PID_FILE = app.pid
# 日志文件
LOG_FILE = app.log
# Python 命令
PYTHON = uv run python

# 帮助信息
.PHONY: help
help:
	@echo "Dream Forge 管理命令:"
	@echo "  make start          - 在后台启动应用 (默认: app.py)"
	@echo "  make start APP=advanced_app.py - 启动高级应用"
	@echo "  make stop           - 停止应用"
	@echo "  make restart        - 重启应用"
	@echo "  make status         - 查看应用状态"
	@echo "  make logs           - 查看日志"
	@echo "  make tail           - 实时查看日志"
	@echo "  make clean          - 清理日志和 PID 文件"

# 启动应用
.PHONY: start
start:
	@if [ -f $(PID_FILE) ]; then \
		PID=$$(cat $(PID_FILE)); \
		if ps -p $$PID > /dev/null 2>&1; then \
			echo "应用已在运行 (PID: $$PID)"; \
			exit 1; \
		else \
			echo "发现旧的 PID 文件，清理中..."; \
			rm -f $(PID_FILE); \
		fi \
	fi
	@echo "启动应用: $(APP)"
	@> $(LOG_FILE)  # 清空日志文件
	@nohup $(PYTHON) $(APP) > $(LOG_FILE) 2>&1 & echo $$! > $(PID_FILE)
	@sleep 2  # 等待应用启动
	@if [ -f $(PID_FILE) ]; then \
		PID=$$(cat $(PID_FILE)); \
		if ps -p $$PID > /dev/null 2>&1; then \
			echo "✅ 应用已启动 (PID: $$PID)"; \
			echo "📝 日志文件: $(LOG_FILE)"; \
			echo "🌐 访问地址: http://localhost:7860"; \
		else \
			echo "❌ 应用启动失败"; \
			rm -f $(PID_FILE); \
			tail -n 20 $(LOG_FILE); \
			exit 1; \
		fi \
	fi

# 停止应用
.PHONY: stop
stop:
	@if [ -f $(PID_FILE) ]; then \
		PID=$$(cat $(PID_FILE)); \
		if ps -p $$PID > /dev/null 2>&1; then \
			echo "停止应用 (PID: $$PID)..."; \
			kill $$PID; \
			sleep 2; \
			if ps -p $$PID > /dev/null 2>&1; then \
				echo "强制停止应用..."; \
				kill -9 $$PID; \
			fi; \
			rm -f $(PID_FILE); \
			echo "✅ 应用已停止"; \
		else \
			echo "应用未在运行"; \
			rm -f $(PID_FILE); \
		fi \
	else \
		echo "未找到 PID 文件，应用可能未运行"; \
	fi

# 重启应用
.PHONY: restart
restart:
	@echo "重启应用..."
	@$(MAKE) stop
	@sleep 1
	@$(MAKE) start APP=$(APP)

# 查看状态
.PHONY: status
status:
	@if [ -f $(PID_FILE) ]; then \
		PID=$$(cat $(PID_FILE)); \
		if ps -p $$PID > /dev/null 2>&1; then \
			echo "✅ 应用正在运行 (PID: $$PID)"; \
			echo "📝 日志文件: $(LOG_FILE)"; \
			echo "🌐 访问地址: http://localhost:7860"; \
			echo ""; \
			echo "进程信息:"; \
			ps -f -p $$PID; \
		else \
			echo "❌ 应用未运行 (发现过期的 PID 文件)"; \
			rm -f $(PID_FILE); \
		fi \
	else \
		echo "❌ 应用未运行"; \
	fi

# 查看日志
.PHONY: logs
logs:
	@if [ -f $(LOG_FILE) ]; then \
		cat $(LOG_FILE); \
	else \
		echo "日志文件不存在"; \
	fi

# 实时查看日志
.PHONY: tail
tail:
	@if [ -f $(LOG_FILE) ]; then \
		tail -f $(LOG_FILE); \
	else \
		echo "日志文件不存在"; \
	fi

# 清理文件
.PHONY: clean
clean:
	@echo "清理日志和 PID 文件..."
	@rm -f $(PID_FILE) $(LOG_FILE)
	@echo "✅ 清理完成"

# 检查依赖
.PHONY: check
check:
	@echo "检查项目依赖..."
	@if ! command -v uv > /dev/null 2>&1; then \
		echo "❌ 未安装 uv，请先安装 uv"; \
		exit 1; \
	fi
	@echo "✅ uv 已安装"
	@echo "同步依赖..."
	@uv sync
	@echo "✅ 依赖检查完成"

# 默认目标
.DEFAULT_GOAL := help