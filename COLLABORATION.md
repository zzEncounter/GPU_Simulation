# Collaboration Guide (SSH)

本文档说明如何在当前服务器上协作开发本项目。

## 1. 当前已搭好内容

已创建并推送一个裸仓库（bare repo）：

- Bare repo path: `/home/jqliu/collab-remotes/pennylane-qubit-rotation-demo.git`
- 默认分支：`main`
- 已配置：`receive.denyNonFastforwards=true`

当前本地开发仓库的 `origin` 也已指向该 bare repo。

## 2. 协作者如何开始（同账号 SSH 场景）

如果协作者使用同一个账号（`jqliu`）通过 SSH 登录这台机器：

```bash
# 在协作者自己的终端（本地机器）执行
# 服务器主机：biglittle
# 示例 IP：192.168.50.129

git clone ssh://jqliu@192.168.50.129/home/jqliu/collab-remotes/pennylane-qubit-rotation-demo.git
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

## 5. 多账号 SSH 场景（推荐更规范）

如果每位协作者使用不同 Linux 账号登录，建议把 bare repo 放到公共路径（例如 `/srv/git`），并用共享组授权。

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

然后维护者把当前仓库推送过去：

```bash
git remote add shared /srv/git/pennylane-qubit-rotation-demo.git
git push -u shared main
```

协作者使用各自账号 clone：

```bash
git clone ssh://<user>@<server>/srv/git/pennylane-qubit-rotation-demo.git
```

## 6. 注意事项

- 不要多人在同一个工作树目录里同时改代码；每人各自 clone 一份。
- `old/` 已被 `.gitignore` 排除，不会进入版本库。
- 修改 `cpp/*.cu` 或 `cpp/*.cpp` 后，记得重新编译扩展。
