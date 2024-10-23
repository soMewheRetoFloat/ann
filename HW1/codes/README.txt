# 对代码做的其他修改

## loss.py
+ 添加了一个 `softmax` 函数方便所有的框架计算。
+ 添加了一个 `print_line` 函数用于调试。

## layers.py
+ 在 `Selu` 中添加了一个 `params` 字典存储两个参数。

## run_mlp.py solve_net.py
+ 进行了大幅改造，使得参数可以通过 `argparse` 传入，支持一次把 12 种模型参数组合都跑完，并且引入了 `wandb` 库进行画图。

## run_colab.ipynb
+ 添加了库的装入。
