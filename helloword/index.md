# 零基础用 GitHub Pages + Hugo 搭建个人网站


<!--more-->

[Hugo](https://gohugo.io/)是用Go编写的静态站点生成器,生成速度（不到1s）比其他生成器快了许多,且配置比Hexo等简单，目前主要存在的缺点是主题不够丰富和成熟、插件较少。

话不多说，开始搭建~

## 创建GitHub Pages

1.  在GitHub新建一个Repository命名为\[你的GitHub账户名\].github.io，其他可以不用填写，直接创建即可。
    
    ![](/_resources/59a23947be34486da5708e51b094e0fc.png)
    
2.  进入刚创建的Repository，点击右边的Settings，下滑找到GitHub Pages，点击`Choose a theme`，随意选择一个theme（因为不会用到），点击`commit changes`就创建完成了。
    
    ![](/_resources/c7799b39ec714fc286c5da2901cbb66d.png)
    

## 安装Hugo和Git

### 安装Hugo

#### macOS使用Homebrew

`brew install hugo`

#### Windows使用Chocolatey

`choco install hugo -confirm`

详见 https://gohugo.io/getting-started/installing

### 安装Git

在[Git官网](https://git-scm.com/downloads)下载安装。

命令行输入 `git version` 显示 Git 的版本号，说明安装成功。

*注意需要将 Hugo 和 Git 安装目录都加入系统Path环境变量。*

## 创建Site 生成网站

### 创建site

在用来存放博客的路径下，命令行执行：

`hugo new site mysite`

该命令创建一个名为 `mysite` 的文件夹来存放你的博客。

`mysite` 的目录结构如下：

```sh
├── archetypes // .md 博客原型 模版
├── content // .md 存放你写的 Markdown 文件
├── data // YAML, JSON, or TOML等配置文件
├── layouts // .html 网站模版
├──  static  // images, CSS, JavaScript 等，决定网站的外观。
├── themes // 存放网站主题
└── config.toml // 网站的配置文件`
```

### 安装主题

你可以在[主题](https://themes.gohugo.io/)选择自己喜欢的主题， 本站使用的是 [LeaveIt](https://themes.gohugo.io/leaveit/)。

执行：

`cd mysite/themes`

进入该文件夹用来存放主题的文件夹, 执行：

`git clone https://github.com/liuzc/LeaveIt.git`

### 设置模版

回到`mysite`:

`cd ..`

打开 `mysite/archetypes` 目录下的 模版 `default.md` ，更改为：

```toml
+++
title =  "{{ replace .Name "-" "  " | title }}"  # 文章标题
date =  {{  .Date  }}  # 自动添加日期
draft =  true  # 是否为草稿
categories =  [""]  # 目录（数组）
tags =  [""]  # 标签（数组）
description =  ""  # 描述
comments =  true  # 是否开启评论
share =  true  # 是否开启分享
+++
```

执行：

```
hugo new about.md
hugo new posts/firstBlog.md
```

在 `content` 文件夹创建 `about.md` 和 `firstBlog.md` ，打开他们随意输入些内容。

### 配置 config.toml

按照以下 `config.toml` 模板更改, 不知道的可以先空着。

在 `mysite/static` 文件夹中新建 `images` 文件夹，将你的头像文件放入其中。

```toml
baseURL =  "https://[你的GirHub用户名].github.io"
title =  "My Site"  # 网站标题
languageCode =  "zh-cn"  # 语言
hasCJKLanguage =  true  # 字数统计时统计汉字
theme =  "LeaveIt"  # 主题
paginate =  # 每页博客数
enableEmoji =  true  # 支持 Emoji
enableRobotsTXT =  true  # 支持 robots.txt
googleAnalytics =  ""  # Google 统计 id
preserveTaxonomyNames =  true
[blackfriday]  # Markdown 渲染引擎
hrefTargetBlank =  true  # Open external links in a new window/tab.
nofollowLinks =  true
noreferrerLinks =  true
[Permalinks]
posts =  "/:year/:month/:title/"
[menu]
[[menu.main]]
name =  "Blog"
url =  "/posts/"
weight =  1
[[menu.main]]
name =  "Categories"
url =  "/categories/"
weight =  2
[[menu.main]]
name =  "Tags"
url =  "/tags/"
weight =  3
[[menu.main]]
name =  "About"
url =  "/about/"
weight =  4
[params]
since =  2019
author =  "你的名字"
avatar =  "/images/avatar.png"  # 头像文件路径` 
`subtitle =  "Hugo is Absurdly Fast!"
home_mode =  ""  # 填post则在首页post博客
enableGitalk =  true  # gitalk 评论系统
google_verification =  ""
description =  ""  # 描述
keywords =  ""  # site keywords
beian =  ""
baiduAnalytics =  ""
license=  '本文采用<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/" target="_blank">知识共享署名-非商业性使用 4.0 国际许可协议</a>进行许可'
[params.social]
GitHub  =  ""
Twitter  =  ""
Email  =  ""
Instagram  =  ""
Wechat  =  "/images/me/wechat.png"  # Wechat QRcode image
Facebook  =  ""
Telegram  =  ""
Dribbble  =  ""
Medium  =  ""`
```
### 生成草稿网站

配置完 `config.toml` 后在 `mysite` 文件夹执行：

`hugo server -D`

打开 http://localhost:1313/ 即可查看生成的静态网站啦。

## 发布到GitHub Pages

生成网站命令：

`hugo`

生成网站存储在 public 文件夹中。

执行：

`cd public`

进入 public 文件夹，然后：
```sh
git init
git remote add origin https://github.com/[Github 用户名]/[Github 用户名].github.io.git
git add .
git commit -m "init commit"
git push -u origin master -f
```

这里 `-u origin master` 指定了默认主机，后面可以不加参数直接push了。

以后再发布时的命令如下：
```sh
git add .
git commit -m "the commit message"
git push`
```

接下来稍等片刻你便可以在 https://\[你的GitHub用户名\].github.io/ 访问你的网站啦！

## 主题优化

下面主要是对 [LeaveIt](https://themes.gohugo.io/leaveit/) 主题的一些优化。

### 解决黑色主题闪屏问题

切换为黑色主题后，每次打开新页面都会网页都会闪一下，亮瞎狗眼有没有！

解决方法只需要更改 `mysite/themes/LeaveIt/layouts/_default/baseof.html` 的一行代码：

`<body  class="dark-theme ">`

将默认加载的主题设为黑色即可。

### 站点流量统计

利用[不蒜子](https://busuanzi.ibruce.info/)，两行代码实现访问量统计，将以下代码

```js
<script  async  src="//busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js"></script>
```

可以加在 `mysite/themes/LeaveIt/layouts/partials/js.html` 中。

再将以下代码加到需要的位置即可：

网站总访问量Page View:(可以放在`partials/footer.html`)

```js
<span  id="busuanzi_container_site_pv">本站总访问量<span  id="busuanzi_value_site_pv"></span>次</span>
```

网站总访客数Unique Visitor:

```js
<span  id="busuanzi_container_site_uv">您是第<span  id="busuanzi_value_site_uv"></span>位访客</span>
```

博客总阅读量:（放在`_default/simple.html`）

```js
<span  id="busuanzi_container_site_pv">本文总阅读量<span  id="busuanzi_value_page_pv"></span>次</span>
```

### 博客目录

### 站内搜索

留点坑，有空再来填。

* * *

最后，致谢GitHub, Hugo, [LeaveIt](https://github.com/liuzc/leaveit), 和博主[Mogeko](https://mogeko.me/)！

添加个人域名、CDN、评论区等更多优化可以参考[Mogeko](https://mogeko.me/)的[博客](https://mogeko.me/2018/025/)。
