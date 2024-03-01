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
  - favicon.ico   # 一个 Vue 的 icon 图标, 网页的标签页图标
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

# npm & npx

```bash
npm install prettier
# npm install -g <package-name>  # 全局安装, 不推荐
```

此命令会在当前文件夹下新建或更新 node_modules/prettier 文件夹, 创建或更新 package.json, package-lock.json 文件, 因此是局部安装, 自动创建的 `package.json` 文件内容如下:

```json
{
  "dependencies": {
    "prettier": "^3.2.5"
  }
}
```

ChatGPT (不确定真伪) : npx 是 npm 的一部分, npx 会临时安装并执行依赖项，然后在执行完成后将其删除

# hello world

执行 `npm create vue@latest` 后, 删除不必要的文件, 目录结构如下

```
src/
  - assets/      # 空文件夹
  - components/  # 空文件夹
  - App.vue
  - main.js
index.html       # 使用默认生成的
jsconfig.json    # 使用默认生成的
package.json     # 使用默认生成的
vite.config.js   # 使用默认生成的
```

`main.js` 内容

```javascript
import { createApp } from 'vue'
import App from './App.vue'

createApp(App).mount("#app")
```

`App.vue` 内容

```vue
<template>
  <h1> title </h1>
  <p>{{ content }}</p>
</template>

<script>
export default {
  data(){
    return {content: "hello world"}
  }
}
</script>
```

以下均为使用 `npm create vue@latest` 创建文件夹时生成的原始内容

`index.html` 内容

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <link rel="icon" href="/favicon.ico">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vite App</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.js"></script>
  </body>
</html>
```

`jsconfig.json` 文件内容

```json
{
  "compilerOptions": {
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "exclude": ["node_modules", "dist"]
}
```

`package.json` 文件内容

```json
{
  "name": "vue_hello",
  "version": "0.0.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "vue": "^3.4.15"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^5.0.3",
    "vite": "^5.0.11"
  }
}
```

`vite.config.js` 文件内容

```javascript
import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    vue(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  }
})
```

此时可以使用如下命令安装依赖及运行

```
npm install     # 依赖包
npm run dev     # 运行服务, 这时可以用浏览器访问
```