# Collaboration Guide (GitHub SSH)

本文档说明当前项目的协作方式。

## 1. Canonical Remote

从现在开始，统一使用以下远端仓库：

- `git@github.com:zzEncounter/GPU_Simulation.git`

本地仓库的 `origin` 已切换到该地址。

## 2. 协作者如何开始

每位协作者在本地机器执行：

```bash
git clone git@github.com:zzEncounter/GPU_Simulation.git
cd GPU_Simulation

python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python setup.py build_ext --inplace
```

首次使用 Git 请设置身份：

```bash
git config user.name "Your Name"
git config user.email "your_email@example.com"
```

## 3. 日常协作流程

```bash
# 同步主分支
git checkout main
git pull --ff-only

# 开新分支
git checkout -b feat/your-topic

# 提交
git add .
git commit -m "feat: your change"

# 推送分支
git push -u origin feat/your-topic
```

推荐通过 GitHub PR 合并到 `main`。

## 4. 维护者合并建议

- 分支保护建议：开启 `main` 的保护规则（至少要求 PR、禁止 force push）。
- 仓库权限建议：协作者使用 GitHub collaborator 或 team 管理。

## 5. SSH 权限排障

如果 `git push` 报权限错误，先检查当前 SSH 绑定账号：

```bash
ssh -T git@github.com
```

输出应为你预期的 GitHub 用户名。

若用户名不对：

- 给当前 SSH key 绑定正确的 GitHub 账号；或
- 在 `~/.ssh/config` 为该仓库配置专用 key（例如 `Host github-zz` + `IdentityFile`），并把 remote 改为该 Host。

## 6. 注意事项

- 不要多人在同一个工作树目录里同时改代码；每人各自 clone 一份。
- `old/` 已被 `.gitignore` 排除，不会进入版本库。
- 修改 `cpp/*.cu` 或 `cpp/*.cpp` 后，记得重新编译扩展。
