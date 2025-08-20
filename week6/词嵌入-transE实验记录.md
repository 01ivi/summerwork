## TransE

+ 首先安装`rarfile`库，以及系统级的`unrar`工具，使用`unrar x FB15k.rar`命令将数据集解压在当前路径

  ![](E:\summerwork\week6\img\解压数据集.png)

+ 使用`python transE_simple.py`命令运行代码，原代码训练400轮停止，训练过程中autoDL给我自动关机了，所以这里改成50轮

  ![](E:\summerwork\week6\img\训练结果.png)

+ 运行`test.py`,但是报错**测试代码（test.py）试图用 JSON 格式解析实体 / 关系向量文件，但这些文件的格式不符合 JSON 规范，导致解析失败**，所以将`embedding = np.array(json.loads(embedding))`改为`embedding = np.array(ast.literal_eval(embedding)) `

  但是还是错误，检查后发现文件格式不对应在列表中去掉np.float64，使用

  ```bash
  sed -i 's/np.float64(//g' ../res/relation_50dim_batch400
  sed -i 's/)//g' ../res/relation_50dim_batch400
  ```

  测试结果：

  ![](E:\summerwork\week6\img\测试结果1.png)

  ![](E:\summerwork\week6\img\测试结果2.png)

  

## 代码解读与思考

+ 代码整体分为**数据加载**、**距离计算**、**TransE 模型核心**、**训练流程**四部分

+ `data_loader`读取知识图谱数据（训练集、实体 - ID 映射、关系 - ID 映射），将原始三元组转换为 ID 索引形式

+ **距离计算**：`distanceL1`和`distanceL2`函数，计算 “头实体 + 关系” 与 “尾实体” 的向量距离

+ #### TransE 模型核心：

  + **初始化**（`__init__`方法）：接收超参数：嵌入维度（`embedding_dim`，如 50）、学习率（`learning_rate`，如 0.01）、边际值（`margin`，如 1）、距离类型（`L1`，布尔值，选择 L1/L2 距离）。

  + **向量初始化**（`emb_initialize`方法）：为实体和关系随机初始化低维向量，并进行 L2 归一化。

  + **训练流程**（`train`方法）：迭代训练模型，通过批量梯度下降优化向量。

  + **负样本生成**（`Corrupt`方法）：生成 “错误” 三元组，用于对比学习（让正确三元组的距离小于错误三元组）。

  + **参数更新**（`update_embeddings`方法）：通过梯度下降更新实体和关系向量，最小化损失函数。

  + **损失函数**（`hinge_loss`方法）：实现 TransE 的 hinge 损失：`max(0, d(h, r, t) - d(h', r, t') + margin)`，其中`d`为距离函数。当正确三元组的距离比错误三元组大`margin`以上时，损失为正，需要优化；否则损失为 0，无需更新。

+ 知识图谱嵌入基本流程：数据加载→向量初始化→正负样本对比→梯度更新

  