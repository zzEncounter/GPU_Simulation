# Collaboration Guide (SSH)

本文档说明如何在当前服务器上协作开发本项目。

## 1. 当前已搭好内容（可直接用）

已创建并推送一个裸仓库（bare repo）：

- Bare repo path: `/home/jqliu/collab-remotes/pennylane-qubit-rotation-demo.git`
- 默认分支：`main`
- 已配置：`receive.denyNonFastforwards=true`
- 已配置跨账号可写权限（当前仓库目录为 `drwsrwsrwx`）
- 已放开目录穿透权限：`/home/jqliu` 为 `711`

当前本地开发仓库的 `origin` 也已指向该 bare repo。

## 2. 协作者如何开始（不同 Linux 账号也可）

协作者使用各自 Linux 账号通过 SSH 登录后，可直接 clone：

```bash
# 在协作者自己的终端（本地机器）执行
# 服务器示例主机：biglittle
# 示例 IP：192.168.50.129

git clone ssh://<your_linux_user>@192.168.50.129/home/jqliu/collab-remotes/pennylane-qubit-rotation-demo.git
cd pennylane-qubit-rotation-demo

python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python setup.py build_ext --inplace
```

首次使用 Git 请配置身份（每位协作者都要设置自己的名字/邮箱）：

```bash
git config user.name "Your Name"
git config user.email "your_email@example.com"
```

## 3. 推荐日常协作流程

```bash
# 先同步 main
git checkout main
git pull --ff-only

# 新建功能分支
git checkout -b feat/your-topic

# 开发 + 提交
git add .
git commit -m "feat: your change"

# 推送分支
git push -u origin feat/your-topic
```

无代码托管平台时，可把分支名和 commit hash 发给维护者，由维护者在服务器上合并。

## 4. 维护者合并流程（无 PR 平台）

```bash
git fetch origin
git checkout main
git pull --ff-only

# 例如合并某个功能分支（可改为 --no-ff）
git merge --ff-only origin/feat/your-topic

git push origin main
```

## 5. 更规范的管理员方案（推荐，需 sudo）

虽然当前方案可用，但更推荐把 bare repo 放到公共路径（例如 `/srv/git`），并用共享组授权。

示例（需要管理员权限）：

```bash
sudo groupadd qrotdev
sudo usermod -aG qrotdev <user1>
sudo usermod -aG qrotdev <user2>

sudo mkdir -p /srv/git
sudo chgrp qrotdev /srv/git
sudo chmod 2775 /srv/git

sudo git init --bare --shared=group /srv/git/pennylane-qubit-rotation-demo.git
sudo git -C /srv/git/pennylane-qubit-rotation-demo.git config receive.denyNonFastforwards true
sudo git -C /srv/git/pennylane-qubit-rotation-demo.git symbolic-ref HEAD refs/heads/main
```

然后维护者把当前仓库推送过去并切换 remote：

```bash
git remote add shared /srv/git/pennylane-qubit-rotation-demo.git
git push -u shared main
git remote set-url origin /srv/git/pennylane-qubit-rotation-demo.git
```

协作者使用各自账号 clone：

```bash
git clone ssh://<user>@<server>/srv/git/pennylane-qubit-rotation-demo.git
```

## 6. 注意事项与安全建议

- 不要多人在同一个工作树目录里同时改代码；每人各自 clone 一份。
- `old/` 已被 `.gitignore` 排除，不会进入版本库。
- 修改 `cpp/*.cu` 或 `cpp/*.cpp` 后，记得重新编译扩展。
- 当前 `/home/jqliu/collab-remotes` 方案为了快速协作放宽了权限；如果有管理员支持，建议尽快迁移到 `/srv/git + 共享组`。
