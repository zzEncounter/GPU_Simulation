# Collaboration Guide (GitHub SSH)

从现在开始，统一使用以下远端仓库：

- `git@github.com:zzEncounter/GPU_Simulation.git`

本地仓库的 `origin` 已切换到该地址。

## 协作者如何开始

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