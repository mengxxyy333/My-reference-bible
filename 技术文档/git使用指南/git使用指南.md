# 使用git

## 一、下载git

​	https://git-scm.com/

## 二、基本命令

​	git version：查看git版本

​	git config --global user.name "用户名"：设置用户名

​	git config --global user.email "邮箱"：设置邮箱

​	git init：初始化当前文件夹，.git文件中会记录所有文件版本

​	git add 文件名：添加当前文件夹下指定的文件，注意后缀别写错

​	git add .：添加当前文件夹下所有文件

​	**add只是暂时保存，还需要提交上去**

​	git commit：提交，之后会打开vim编辑器写说明，关于vim操作这里不再赘述

​	git commit -m "提交说明"：上一个命令的简化版操作

​	git log：查看日志，提交信息

​	git reset --hard commitid：回退到某个提交版本，这里的commitid需要替换成你想回退的版本

​	git branch 0.2：创建一个0.2分支

​	git branch -a：查看所有分支

​	git checkout 0.2：切换到0.2分支

​	git checkout master：切换到主分支

​	git merge 0.2：将当前分支与0.2分支合并

## 三、代码仓库

​	这里使用github。注册登录

​	点击右上角加号 --> new repository，选择归属，起个仓库名，添加描述

​	选择公开（public）或者不公开（private）

​	点击create repository

​	按照指引走即可：

​	create a new repository：

```
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/mengxxyy333/testgit.git
git push -u origin main
```

​	push an existing repository：

```
git remote add origin https://github.com/mengxxyy333/testgit.git
git branch -M main
git push -u origin main
```

​	当你提交文件后有更新时，想要提交更新的内容，相关命令如下：

```
git status：查看有变动的文件
git add 文件名：添加变动文件
git commoit -m "说明"：添加修改说明
git push：直接推送更新
```

## 四、新手配置SSH链接问题

​	1、首先查看是否有SSH密钥：

```
ls -al ~/.ssh
```

​	如果有以下文件，说明有SSH：

- `id_rsa` 和 `id_rsa.pub`（RSA密钥）
- `id_ed25519` 和 `id_ed25519.pub`（Ed25519密钥）

​	2、如果没有，生成新的SSH密钥：

```
# 推荐使用Ed25519算法（更安全、更快）
ssh-keygen -t ed25519 -C "2230570133@qq.com"

# 或者使用RSA算法（兼容性更好）
# ssh-keygen -t rsa -b 4096 -C "2230570133@qq.com"
```

​	按照提示操作：

​		（1）保存位置：按Enter使用默认位置（`~/.ssh/id_ed25519`）

​		（2）设置密码：可选，按Enter跳过或设置密码（更安全）

​	3、接下来查看并复制公钥：

```
# 查看公钥内容
cat ~/.ssh/id_ed25519.pub
# 或者（如果是RSA）
# cat ~/.ssh/id_rsa.pub
```

​	输出类似这样的内容：

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIM1wvQ8fJgLmN9QzX4jKp7JzT8VqXwY0L9oPqRsT6b 2230570133@qq.com
```

​	全选并复制整个输出内容。

​	4、将SSH密钥添加到github

（1）登录github，点击个人头像 --> Settings

（2）左侧选择SSH and GPG keys

（3）点击New SSH Key

（4）填写：title（给自己的SSH密钥起个名字）、key type（默认Authentication Key）、key（复制第三步输出的一长串）

（5）点击Add SSH key

​	5、修改远程仓库地址

​	查看当前远程地址：

```
git remote -v
```

​	将HTTPS地址改为SSH地址：

```
git remote set-url origin git@github.com:mengxxyy777/My-reference-bible.git
```

​	再次确认：

```
git remote -v
```

​	应该显示这样的内容：

```
origin  git@github.com:mengxxyy777/My-reference-bible.git (fetch)
origin  git@github.com:mengxxyy777/My-reference-bible.git (push)
```

​	6、现在可以测试链接：

```
ssh -T git@github.com
```

​	第一次连接会显示：

```
The authenticity of host 'github.com (IP地址)' can't be established.
ED25519 key fingerprint is SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU.
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

​	输入yes，会看到如下内容：

```
Hi mengxxyy777! You've successfully authenticated, but GitHub does not provide shell access.
```

​	然后可以推送：

```
git push -u origin main
```

​	7、如果遇到每次操作都需要输入密钥密码，可以这样做：

​	MacOS：

```
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
```

## 五、一些其他问题

### 1、遇到不同系统提交相同文件时，换行符转换问题，git通常会提示：

```
warning: in the working copy of '技术文档/Sublime Text配置C++竞赛环境/Sublime Text配置C++.md', LF will be replaced by CRLF the next time Git touches it
```

​	这个问题是mac和windows之间换行符格式不一致的问题，可以禁用自动转换：

```
# 全局禁用换行符自动转换
git config --global core.autocrlf false

# 或者只对当前仓库禁用
git config core.autocrlf false
```

​	但是这里我们其实需要它的自动转换功能，所以这样做：

​	在项目根目录（也就是我们可以看到.git所在那个状态），新建一个文件：.gitattributes.txt。内容如下：

```
# 自动检测文本文件并进行换行符转换
* text=auto

# 明确指定某些文件类型为文本
*.md text
*.cpp text
*.h text
*.txt text
*.json text
*.yml text
*.yaml text

# 二进制文件不转换
*.png binary
*.jpg binary
*.pdf binary
*.zip binary
```

​	这样就可以解决了。如果还有后续一些转换问题，尝试重新规范化：

```
# 保存当前更改
git stash

# 重新规范化所有文件
git add --renormalize .

# 恢复更改
git stash pop

# 重新添加
git add .
```

