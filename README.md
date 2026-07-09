# Chaoke Du - Portfolio Website

个人作品集网站，基于 Next.js + Docker 部署，支持中英双语、Markdown 博客、嵌入式工具和 HTTPS。

线上地址：[https://vinkoai.top](https://vinkoai.top)

## 技术栈

- **框架**：Next.js 16 (App Router, standalone output)
- **语言**：TypeScript + React 19
- **样式**：Tailwind CSS 4
- **国际化**：next-intl（中文 / 英文）
- **博客**：Markdown + gray-matter + remark-html
- **部署**：Docker + Nginx + Let's Encrypt HTTPS
- **服务器**：阿里云 ECS（Ubuntu 26.04, 2C2G）

## 项目结构

```
├── content/blog/          # Markdown 博客文章（24 篇）
├── public/
│   ├── blog-assets/       # 博客图片资源（42 张）
│   └── tools/             # 嵌入式工具静态产物（gitignore）
├── src/
│   ├── app/[locale]/      # 页面路由（首页/项目/博客/留言/联系/关于）
│   ├── app/api/           # API 路由（留言/联系/搜索/登录/管理）
│   ├── components/        # 组件（Navbar/Footer/ProjectCard/BlogList 等）
│   ├── lib/               # 数据层（posts.ts/projects.ts/db.ts/auth.ts）
│   └── i18n/              # 国际化配置
├── messages/              # 中英文翻译文件
├── nginx.conf             # Nginx 配置（HTTPS + 限流 + 安全头）
├── docker-compose.yml     # Docker Compose 编排
├── Dockerfile             # 多阶段构建
└── .env.example           # 环境变量模板
```

## 功能

- **博客**：24 篇 AI/深度学习技术文章，支持中英文切换、标签筛选、搜索
- **项目展示**：PaperReader、AI Image Studio、Music Visualizer、SchemaForge、Solution Architect、CompetitorAI
- **嵌入式工具**：4 个静态工具直接挂载到 `/tools/` 路径
- **留言板**：公开留言，管理后台审核
- **联系表单**：访客可发送消息，管理员在后台查看
- **管理后台**：JWT 登录，管理留言和联系消息
- **HTTPS**：Let's Encrypt 免费证书，自动续期
- **安全**：Nginx 限流、安全响应头、输入校验、数据持久化

## 本地开发

```bash
npm install
npm run dev
```

打开 [http://localhost:3000](http://localhost:3000)

## 部署流程

### 1. 本地构建 Docker 镜像（在 WSL 中执行）

```bash
cd /mnt/c/.../portfolio-website
docker build -t portfolio-web:latest .
```

### 2. 导出并上传镜像到服务器

```bash
docker save portfolio-web:latest | gzip > /tmp/portfolio-web.tar.gz
scp /tmp/portfolio-web.tar.gz root@39.106.45.168:/tmp/
```

### 3. 服务器加载镜像并重启

```bash
ssh root@39.106.45.168
cd /opt/portfolio
git pull                          # 同步配置文件
docker load -i /tmp/portfolio-web.tar.gz
rm /tmp/portfolio-web.tar.gz
docker compose up -d --force-recreate
```

> 注意：不要在服务器上执行 `docker build`，服务器只有 2C2G，构建在本地 WSL 完成。

## 环境变量

复制 `.env.example` 为 `.env` 并填写：

```env
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your-strong-password
ADMIN_JWT_SECRET=your-random-secret
PORTFOLIO_API_KEY=your-api-key
PORTFOLIO_URL=https://vinkoai.top
```

服务器上 `.env` 权限应为 `600`，不要提交到 Git。

## HTTPS 证书

使用 Let's Encrypt 免费证书，有效期 90 天，已配置自动续期：

```bash
# 手动申请
docker run --rm \
  -v /opt/portfolio/letsencrypt:/etc/letsencrypt \
  -v /opt/portfolio/certbot/www:/var/www/certbot \
  certbot/certbot certonly --webroot \
  -w /var/www/certbot \
  -d vinkoai.top -d www.vinkoai.top \
  --agree-tos --register-unsafely-without-email --non-interactive

# 自动续期（已配置 cron）
# /etc/cron.d/portfolio-certbot
```

## 嵌入式工具

以下工具构建为静态产物，挂载到 `/tools/` 路径：

| 工具 | 路径 | 源码位置 |
|------|------|----------|
| AI Image Studio | `/tools/ai-image-studio/` | workspace/ai-image-studio |
| Music Visualizer | `/tools/music-visualizer/` | workspace/music-visualizer |
| SchemaForge | `/tools/schemaforge/` | workspace/db-schema-designer |
| Solution Architect | `/tools/solution-architect/` | workspace/solution-architect |

工具构建后需复制到 `public/tools/` 再重新构建主站镜像。

## 博客文章导入

使用 `import_posts.py` 从多个源目录导入 Markdown 文章：

```bash
python import_posts.py --all --portfolio ./portfolio-website --dry-run
python import_posts.py --all --portfolio ./portfolio-website
```

## License

MIT
