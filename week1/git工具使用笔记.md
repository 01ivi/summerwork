## git工具使用笔记

#### 创建git版本库并添加文件

---

利用`git init`命令创建版本库；利用`git add`将文件添加到暂存区；利用`git commit -m"文件名"`将文件提交到仓库

![创建git版本库并添加文件图片](https://i.postimg.cc/rzXcKzYt/git.png)

#### 将本地仓库上传到github并导入gitee

---

使用`git remote add origin github网址`命令连接在github上创建的远程仓库；使用`git push -u oringin master`命令将当前分支内容同步到远程仓库；gitee中可一键导入github仓库

![](https://i.postimg.cc/Xv1Dxq6G/2.png)



#### 创建与合并分支

---

使用`git checkout –b name`命令创建分支；使用`git branch`查看分支

![创建dev分支](https://i.postimg.cc/wv6LtcGD/dev.png)

使用`git checkout name`切换分支；使用`git merge name`合并某分支到当前分支；使用`git branch –d name`删除分支

![合并分支](https://i.postimg.cc/2yxz1hgL/image.png)

