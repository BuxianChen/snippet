# 环境配置

参考 [https://vuejs.org/guide/quick-start.html](https://vuejs.org/guide/quick-start.html), 文档中介绍了两种方式: Creating a Vue Application 和 Using Vue from CDN, 前者是做一个完整的 vue 项目额方式, 后者是在 javascript 直接将 vue.js 进行引用. 这里主要将前一种

**node.js & npm**

首先需要安装 node.js: 以 Linux Ubuntu 为例, 首先去 [https://nodejs.org/en/download/](https://nodejs.org/en/download/) 下载 `node-v20.11.1-linux-x64.tar.xz`, 放在某个目录后解压, 再将其添加至 `PATH` 环境变量即可 (这种安装方式可以方便卸载, 直接删掉这个目录即可)

```bash
mv node-v20.11.1-linux-x64.tar.xz ~/software
tar -xvf node-v20.11.1-linux-x64.tar.xz

# vim ~/.bashrc
# export PATH=~/software/node-v20.11.1-linux-x64/bin:$PATH
```

这里简要说明一下 `node-v20.11.1-linux-x64` 的目录结构

```
bin/
  - corepack
  - node
  - npm  # 自带 npm
  - npx
include/
  - node/**/*.h
lib/
  - node_modules/   # 不确定之后安装的包是否也在此目录下
    - corepack/
    - npm/
share/
  - doc/
  - man/
CHANGELOG.md
LICENSE.md
README.md
```

**Vue3**

```bash
# cd <projects_dir>
npm create vue@latest   # 注意一开始会填写项目名称, 会创建一个与项目名称相同的文件夹, 其余选项搞不懂可以先全部通过敲回车选择 NO
cd <your-project-name>
```

原始目录结构是一个 demo, 目录结构如下:

```
node_modules/   # 依赖项的安装目录, Vue 是每个项目都把需要的包安装一遍, 执行 npm install 之后才有这个目录
public/
  - favicon.ico   # 一个 Vue 的 icon 图标
src/              # 代码
  - App.vue
  - main.js
  - assets/
    - base.css
    - logo.svg
    - main.css
  - components/
    - icons/  # 这些文件是样例项目包含 svg 的文件, 但多包了一层 template 标签, 因此无法直接用浏览器打开
      - IconCommunity.vue
      - IconDocumentation.vue
      - IconEcosystem.vue
      - IconSupport.vue
      - IconTooling.vue
    HelloWorld.vue
    TheWelcome.vue
    WelcomeItem.vue
.gitignore       # 注意 npm create 并没有 git init, 但是自动生成了 .gitignore 文件
index.html       # 首页?
jsconfig.json
package.json     # 依赖包记录
package-lock.json  # 依赖包记录, 执行 npm install 时会自动生成
README.md
vite.config.js   # 不确定是不是与项目相关, demo 项目似乎依赖 vite
```

接下来使用这些命令安装依赖项, 运行, 以及 build

```bash
npm install     # 依赖包
npm run dev     # 运行服务, 这时可以用浏览器访问
npm run build   # 将 vue 代码编译为 javascript, css, html 代码, 放在 dist 目录下
```