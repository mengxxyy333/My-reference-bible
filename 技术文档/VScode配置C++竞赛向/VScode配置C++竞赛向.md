# VScode配置C++竞赛向

​	1、首先要下载VScode：

​	2、然后要有clang++的环境，g++应该也可以，没试过，Windows还是去mingw64里面下：github.com/mstorsjo/llvm-mingw/releases，然后下载msvcrt-x86_64.zip这项，下载好了win中直接把include放进去，但是Mac或Linux需要c++/v1目录；MacOS自带.

​	3、搜索cps，设置中设置默认语言，向下找Use Short Code Forces Name，这个可以勾选，能缩短文件名，再找Submission Compiler，改成最高的，Command改成clang++，Args写上：-std=c++2a，就可以了，到这就可以解决大部分测试样例情况了，我们发现，点击小绿之后VScode直接就有样例可以跑了，但是不是自动的，可以修改编译快捷键.

​	4、处理交互题目：搜索插件：Code Runner，点settings -> 找到Code-runner:Executor Map，进入下面的Edit in settings.json，找到cpp那一行，将g++改为：clang++ -std=c++2a，这个配置默认不走终端，可以在下面添加代码：

```
"code-runner.runInTerminal": true,
```

​	5、搜索插件clangs，安装，安装这个插件时，还要注意搜索c++那两个插件，需要卸载，这两个是微软自带的，安装好后还是找到settings，找到Clangd:Path，对应填上你的目录，是bin下的clangd可执行文件；再找到Clangd:Arguments，添加两条：

```
--query-driver=[你的clang可执行文件目录]
```

```
--header-insertion=never
```

​	找到Clangd:Fallback Flags，添加：

```
-std=c++2a
```

​	6、接下来美化，插件搜索background，然后自己在settings.json中设置吧，格式大概如下：

```
"background.fullscreen": {
        "images": ["/Users/Zhuanz/Desktop/pic/夜、萤火虫和你.png"],
        "opacity": 0.3,
        "size": "cover",
        "position": "center",
        "interval": 0,
        "random": false
    }
```

​	7、VScode也可以自己配置snippet，类似于模板代码，这个自己搜索教程即可.