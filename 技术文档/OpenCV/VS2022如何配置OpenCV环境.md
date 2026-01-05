# VS2022如何配置OpenCV环境

​	首先下载OpenCV，官网：https://opencv.org/releases/

​	下载好之后放到一个全英文目录中，如：

![image-20250518165210501](保存路径示例.png)

​	打开VS，并新建空项目，右击项目，属性，VC++目录，包含目录中添加上图opencv文件夹中build里的include文件，如：D:\codetools\OpenCV\opencv\build\include。库目录添加build中的x64中的vcxx中的lib，如：D:\codetools\OpenCV\opencv\build\x64\vc16\lib。此时点击链接器，输入，附加依赖项添加：opencv_world4110d.lib（注意此处要让编译器处于Debug模式，并且world后面的版本号要匹配，如我的opencv版本号为4.11.0）。

​	确定，保存项目，去配置环境变量。选择用户变量中的Path，编辑，新建，将x64下的vcxx目录下的bin文件目录粘贴进来，如：D:\codetools\OpenCV\opencv\build\x64\vc16\bin。确定。

​	重启电脑即可。