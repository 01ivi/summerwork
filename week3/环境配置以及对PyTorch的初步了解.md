## 环境配置以及对PyTorch的初步了解

#### 安装anaconda

+ 从官网下载anaconda；使用`bash Anaconda3-2025.06-0-Linux-x86_64.sh`命令安装；使用`conda -V`查看conda版本

  [![anaconda.png](https://i.postimg.cc/1tPHYM27/anaconda.png)](https://postimg.cc/mcXCP3b7)

+ 在conda下载好了之后默认是在bash环境中的，需要创建一个新环境去使用，使用`conda create -n wangzirui python=3.9 `创建一个新的环境

  ![](https://i.postimg.cc/WzQyf0v9/image.png)

  使用`conda activate wangzirui`命令进入新环境

  #### 配置PyTorch环境

  + 这里我配置的是gpu版本的，方便配合电脑自带的显卡使用，官网下载一直超时，所以需要切换国内镜像源`conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
    conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
    conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/`
  
  + 使用`pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 -f https://mirrors.aliyun.com/pytorch-wheels/cu118`命令安装pytorch和CUDA，国内镜像源只有阿里源还有CUDA
  
    [![pytorch.png](https://i.postimg.cc/9FmWX3wX/pytorch.png)](https://postimg.cc/6TPszmhF)
  
  + 这一步因为家里的wifi过于不稳定，我一直重复下载了很多次，下载速度特别慢而且经常timeout，受不了了最后是开流量才能下载好，所以这一步浪费了很多时间。
  
  + 验证是否安装完成
  
    [![image.png](https://i.postimg.cc/GtPMsMWD/image.png)](https://postimg.cc/V069xFLs)
    
    

#### 安装jupyter notebook

+ 使用`conda install -c conda-forge jupyterlab`命令和`conda install jupyter notebook `命令安装

  [![jupyterlab安装.png](https://i.postimg.cc/Gt0TR4Dq/jupyterlab安装.png)](https://postimg.cc/cgmLM4Xn)

+ 安装后输入jupyter notebook命令得到网址

  [![jupyter.png](https://i.postimg.cc/3JsJNsfb/jupyter.png)](https://postimg.cc/PPMkFRjm)

+ 在浏览器输入网址就能看到可视化界面了

  [![jupyter.png](https://i.postimg.cc/JnMrwCdS/jupyter.png)](https://postimg.cc/R3p5fymT)