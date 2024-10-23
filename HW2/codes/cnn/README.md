代码的其他修改：
+ 新增了 `train_net.py` 进行多参数测试，可以通过修改它获取所有文中出现过的训练结果。目前所有进行过的实验的参数都在其中有所记录；
+ `main.py` 添加了 `wandb` 进行图样绘制，并且对训练集的 `shuffle` 函数进行了实验性改动，非额外要求的的结果可以通过还原 `shuffle` 并且修改；对于 cnn block 结构的修改进行的实验，可以通过 `train_net.py` 修改输入的 `--struct_order` 进行复现；
+ `model.py` 添加了 `ConvBlock` 类和一些 `Model.__init__` 的新参数，以便更方便地进行实验。
