# ai-ma

商品目录的 AI 侧车：文字 / 外观 / 图片检索，以及基于通义千问的工具化导购对话。

- 向量库：本机 Docker 里的 Qdrant  
- 对话模型：阿里云百炼（DashScope）在线 Qwen  
- Python / LangChain：跑在宿主机，不放进 Docker（Mac 上便于用本机算力）

---

## 环境要求

- Python 3.12+（当前可用 3.14）
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- 对话需要 [阿里云百炼](https://bailian.console.aliyun.com/) 的 API Key

---

## 一、首次安装

在项目根目录执行：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，至少填好对话密钥：

```bash
DASHSCOPE_API_KEY=你的密钥
```

本地开发可以暂时留空 `API_KEY`。若填写了 `API_KEY`，调用搜索 / 对话 / 导入接口时需带请求头 `X-API-Key`。

---

## 二、启动依赖（Qdrant + Redis）

```bash
docker compose up -d
```

- Qdrant：`http://localhost:6333`
- Redis：`localhost:6379`（密码见 `.env` 里的 `REDIS_PASSWORD`）

可选：

```bash
# 本地 Ollama（Docker 内仅 CPU，Mac 更建议本机装 Ollama）
docker compose --profile llm up -d

# MinIO 对象存储（本地演示用）
docker compose --profile storage up -d
```

查看状态：

```bash
docker compose ps
```

---

## 三、导入 / 同步商品

演示数据在 `data.json`（两支长笛）。

### 一次性全量导入（旧方式，仍可用）

```bash
.venv/bin/python -m app.services.importer data.json
```

### 增量同步（推荐）

按「水位时间 + 内容 hash」只写入真正变化的商品，复用同一套切块与向量写入：

```bash
# 首次或对账：扫全量，hash 相同会跳过
.venv/bin/python -m app.services.sync --full

# 之后按水位增量
.venv/bin/python -m app.services.sync

# 定时循环（例如每小时）
.venv/bin/python -m app.services.sync --schedule 3600

# 模拟 ATG 更新了某 SKU 的时间戳（内容未变则仍跳过重嵌）
.venv/bin/python -m app.services.sync --touch site7sku1497780
.venv/bin/python -m app.services.sync
```

也可用 HTTP 触发（需服务已启动；若配置了 `API_KEY` 要带密钥）：

```bash
curl -X POST 'http://127.0.0.1:8000/products/sync?full=false'
```

同步状态存在 `.cache/sync_state.sqlite`。当前变更源是本地 JSON（`SYNC_SOURCE=file`），以后接 ATG 时实现同一套 `ChangeSource` 即可。

核对切块后检索是否正常：

```bash
.venv/bin/python -m app.evals
```

---

## 四、启动 Web 服务

```bash
.venv/bin/fastapi dev
```

- 页面：http://127.0.0.1:8000  
- 接口文档：http://127.0.0.1:8000/docs  
- 健康检查：http://127.0.0.1:8000/health  

修改 `.env` 后需要重启 `fastapi dev` 才会生效。

左边是检索结果，右边是对话；对话历史由服务端会话保存（优先 Redis，连不上则用内存），浏览器里只保留 `session_id`。

---

## 五、常用接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 商品库是否连通、商品数 |
| `POST` | `/search` | 文字 / 外观检索 |
| `POST` | `/search/image` | 上传图片检索 |
| `POST` | `/chat` | 导购对话（需 `DASHSCOPE_API_KEY`） |
| `POST` | `/products/import` | 按文件全量导入 |
| `POST` | `/products/sync` | 增量同步 |
| `GET` | `/products/{sku}` | 按 SKU 读取 |

配置了 `API_KEY` 时，除 `/`、`/health` 外需带：

```bash
curl -H 'X-API-Key: 你的密钥' ...
```

---

## 六、项目结构（简要）

```
app/
  api/           HTTP 路由、鉴权、Request-Id
  core/          配置与日志
  schemas/       请求 / 响应模型
  services/
    catalog.py       解析 data.json
    documents.py     长文切块
    embeddings.py    BGE / CLIP
    importer.py      下载图并写入
    store/           Qdrant 写 / 读
    sync/            定时增量同步
    chat/            工具化对话 + 会话
web/             前端页面
data.json        演示商品
docker-compose.yml
```

---

## 七、日常开发最短路径

```bash
docker compose up -d
.venv/bin/pip install -r requirements.txt   # 仅依赖有变更时
cp -n .env.example .env                     # 首次
.venv/bin/python -m app.services.sync --full
.venv/bin/fastapi dev
```

浏览器打开 http://127.0.0.1:8000 即可使用。
