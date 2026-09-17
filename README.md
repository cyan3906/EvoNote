# EvoNote

EvoNote 是一个智能笔记知识演化系统，支持 Markdown 笔记保存、LLM 结构化抽取、ES + Milvus 混合检索、claim 级合并建议和人工复核闭环。

## 本地启动教程

### 1. 安装 Docker Desktop

先安装 Docker Desktop，用来启动 Elasticsearch 和 Milvus。

- Windows 下载地址：https://www.docker.com/products/docker-desktop/
- 安装完成后打开 Docker Desktop，等待左下角显示 Docker Engine 已启动。
- 在终端中确认 Docker 可用：

```powershell
docker --version
docker compose version
```

### 2. 准备后端环境变量

进入后端目录：

```powershell
cd E:\huadian\evonote\backend\python
```

如果还没有 `.env` 文件，可以从示例文件复制一份：

```powershell
copy .env.example .env
```

然后编辑 `.env`，至少确认大模型相关配置可用：

```env
AGENT_API_BASE_URL=你的模型服务地址
AGENT_API_KEY=你的 API Key
AGENT_MODEL=你的模型名

EVOLUTION_API_BASE_URL=你的模型服务地址
EVOLUTION_API_KEY=你的 API Key

EMBEDDING_MODEL=text-embedding-3-large
```

### 3. 启动 Elasticsearch 和 Milvus

回到项目根目录：

```powershell
cd E:\huadian\evonote
```

只启动检索依赖，不启动 app 容器：

```powershell
docker compose up -d elasticsearch milvus
```

查看容器状态：

```powershell
docker compose ps
```

正常情况下能看到：

```text
evonote-elasticsearch
evonote-milvus
```

如果需要查看日志：

```powershell
docker compose logs -f elasticsearch
docker compose logs -f milvus
```

### 4. 安装 Python 依赖

进入后端 Python 目录：

```powershell
cd E:\huadian\evonote\backend\python
```

创建并激活虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```powershell
pip install -r requirements.txt
```

### 5. 启动后端 main

在 `backend\python` 目录下启动：

```powershell
python app\main.py
```

也可以使用 uvicorn 热更新启动：

```powershell
uvicorn app.main:app --reload
```

启动成功后访问：

```text
http://127.0.0.1:8000/
```

API 文档：

```text
http://127.0.0.1:8000/docs
```

默认开发密码：

```text
evonote2026
```

### 6. 停止服务

停止后端：在运行后端的终端按 `Ctrl + C`。

停止 Elasticsearch 和 Milvus：

```powershell
cd E:\huadian\evonote
docker compose down
```

## 常用命令

重新启动 ES 和 Milvus：

```powershell
docker compose restart elasticsearch milvus
```

重建 Milvus 和 Elasticsearch 索引：

```text
POST http://127.0.0.1:8000/api/evolution/index/rebuild
```

运行 embedding 并发探测：

```powershell
cd E:\huadian\evonote\backend\python\tests
python .\probe_embedding_concurrency.py
```

运行 embedding AB 测试：

```powershell
cd E:\huadian\evonote\backend\python
python tests\benchmark_embedding_ab.py
```
