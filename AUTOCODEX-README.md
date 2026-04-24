# AutoCodex

这是公司内定制版 AutoCodex。

## 主要特性
- 优先读取同目录的 `autocodex.toml`
- 默认连接 `https://api.autocodex.net`
- 首次启动优先复用 `CODEX_HOME/auth.json` 中的 KEY
- 如未检测到有效 KEY，只会提示输入一次 KEY
- 启动时会检查 `https://api.autocodex.net/update` 并后台下载新版本
- 超过有效期后会提示前往 `https://autocodex.net` 获取最新版本

## 配置文件
`autocodex.toml` 仅保留模型、工作目录等本地偏好；KEY 存储在 `CODEX_HOME/auth.json`。

## 获取 KEY
请访问 https://autocodex.net
