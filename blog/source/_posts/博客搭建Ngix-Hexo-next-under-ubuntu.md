---
title: 博客搭建Ngix+Hexo+next(under ubuntu)
date: 2025-04-22 19:41:34
tags:
    - 博客
categories:
    - 博客
---
本文逐步说明如何利用ngix + hexo + next主题 搭建博客

服务器配置:

```bash
#操作系统: Ubuntu Server 20.04 LTS 64bit
#cpu: 2核
#内存: 2gb
#硬盘: 50gb
```

---

<!--more-->


# 安装
## 安装ngix
```bash
sudo apt install nginx
# 启动ngix
ngix
```

## 安装Hexo
#### 安装git
```bash
sudo apt-get install git-core
```

#### 安装Node.js
```bash
# 下载安装nvm:
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
# 代替重新启动 shell
\. "$HOME/.nvm/nvm.sh"
# 下载安装 Node.js:
nvm install 22
# Verify the Node.js version:
node -v # Should print "v22.14.0".
nvm current # Should print "v22.14.0".
# Verify npm version:
npm -v # Should print "10.9.2".
```

#### 安装Hexo
```bash
npm install -g hexo-cli
### 将 Hexo 所在的目录下的 node_modules 添加到环境变量之中即可直接使用 hexo <command>
echo 'PATH="$PATH:./node_modules/.bin"' >> ~/.profile
```

# 建站
## hexo建立相关文件夹
```bash
hexo init <folder>
cd <folder>
npm install
```

初始化后，您的项目文件夹将如下所示：

```bash
.
├── _config.yml
├── package.json
├── scaffolds
├── source
|   ├── _drafts
|   └── _posts
└── themes
```

## 安装主题next
### 下载next
```bash
cd your_site
git clone https://github.com/iissnan/hexo-theme-next themes/next
```

### 启用主题
打开your_site/_config.yml,  找到theme字段

```bash
(your_site/_config.yml)
theme : next
```

# hexo映射到ngix
更新hexo配置

```bash
hexo clean
hexo g
```

将生成的前端代码(your_site/public/*)转移到ngix存储静态网页的地方(/var/www/html)

```bash
sudo cp -rf your_site/public/* /var/www/html
```

**到此Ngix+Hexo+text主题初步部署完毕**

