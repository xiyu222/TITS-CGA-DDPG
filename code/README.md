# 代码相关内容解释
本文档用于解释SGRA-PERs算法的相关代码文件。该算法利用随机博弈的方式，实现单MEC服务器上计算卸载任务的动态平衡，从而实现资源的最优分配。

## 相关环境的安装
下载代码后，进入code目录，切换到相应的虚拟环境，在控制台中运行
``` shell
pip install -r requirements.txt
```
即可完成相关环境的安装

请注意torch的版本问题，请依照cuda版本安装对应的torch版本

同时请注意将`NashRL.py`文件中最后生成的结果的保存路径改成本地对应的路径


## train.py
代码运行的入口，直接启动该文件即可运行算法，修改`run_Nash_Agent()`函数中的`MAX_EPISOIDE`参数可以修改算法运行的最大回合数

## simulation_lib.py
该文件定义了MEC相关的环境

- `State`类：通过该类中的两个方法可以获得归一化后的state和非归一化的state

- `Device`类：该类定义了MEC系统中的SD相关参数，例如卸载任务量大小、本地设备计算能力等

- `MarketSimulator`类：该类为主要的环境模拟内容
  - `reset` : 初始化环境的状态
  - `step`: 核心函数，参数`x`表示当前action，agent与环境进行交互，计算每个回合中的处理时延和消耗的能量，获得相应的奖励，进入下一个状态，即MDP的过程

## NashRL.py
程序运行的主体，定义了外层大循环  
每一个循环中，每个智能体会根据当前的系统状态选择一个动作，根据该动作，去与环境进行交互


## nashRL_netlib.py
定义了系统训练过程中action网络和critic两个网络的相关设定

## NashAgent_lib.py
核心文件

该文件中定义了以下几个部分
- `PreExperenicePool`类：优先经验回放取样经验池。在该经验池中，通过sumTree的数据结构，能够根据每一个transition的优先级对其进行排序，并在取样的时候，优先取样取到优先级更高的transition。
  - sample_one：取一个样本
  - sample_batch：取一批样本

- `NashNN`类：最为核心的类，该类实例化的时候就创建一个nash agent，实例化的时候为每个agent创建`action_net`和`value_net`,并且定义对应的优化器、经验回放池
  - `train`：从回放池中获取transition，根据该transition，获取网络对应的loss，根据loss更新网络
  - `predict_action`：根据当前的状态，通过action_net预测对应的action，表示随机博弈的过程
  - `comput_loss_q`：从回放池中sample一个transition，获取相对应的state、action、reward、next_state,计算对应的q值，预测next_action和对应的奖励，根据next_action计算对应的target_q值，更新sumtree，计算loss，返回loss
