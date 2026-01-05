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