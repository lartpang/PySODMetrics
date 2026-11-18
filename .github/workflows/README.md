# GitHub Actions 自动构建文档

`Build and Deploy Documentation` 构建 Sphinx 文档，并将 HTML 产物部署到 GitHub Pages。页面显示的版本取自构建时安装的 PySODMetrics 包。

## 自动部署

- 发布 `v` 开头的正式 GitHub Release（例如 `v1.7.0`）时，工作流检出该标签所指的提交，构建并部署该版本的文档。预发布会跳过部署。
- 推送到 `main` 或 `master` 时，修改文档、指标源码、`pyproject.toml` 或文档工作流也会更新同一站点。站点展示最近一次成功部署的内容。

正式发布前，先推送包含文档和新版工作流的提交，再为该提交创建标签并发布 Release。[GitHub Release 触发规则](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#release)

## 手动部署已有标签

在 **Actions → Build and Deploy Documentation → Run workflow** 中选择 `main` 上的工作流，在 `tag` 中填写版本标签，例如 `v1.7.0`。留空时使用所选工作流的分支或标签。如果 Release 已经发布，可用这个入口补部署。

## GitHub Pages 设置

在 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。首次配置后，工作流通过 `actions/deploy-pages` 发布构建产物。如果 `github-pages` 环境配置了分支或标签限制，在 **Settings → Environments → github-pages** 中保留主分支规则，并添加允许 `v*` 的标签规则。[Pages 发布来源](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#publishing-with-a-custom-github-actions-workflow) · [部署环境规则](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments#deployment-branches-and-tags)

在 Actions 的 Deploy 作业中查看部署结果和页面地址。[PySODMetrics 文档站点](https://lartpang.github.io/PySODMetrics/)

## 本地预览

在仓库根目录安装文档依赖并构建 HTML。

```bash
python -m pip install -e ".[docs]"
python -m sphinx -b html deploy deploy/_build
```

打开 `deploy/_build/index.html` 查看生成的文档。`deploy/.gitignore` 已配置忽略构建产物。
