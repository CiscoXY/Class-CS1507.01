# <center>实验报告</center>
<p align = 'right'>夏远林-PB19020632</p>
## Task1

可以从网络结构这张图片中看出，整个网络共计**4层**  

* 第一层:Sequential[layer1],包含2个BasicBlock，输出为2个64x64x2x2的tensor  

* 第二层:Sequential[layer2],包含2个BasicBlock，输出为2个64x128x1x1的tensor  

* 第三层:Sequential[layer3],包含2个BasicBlock，输出为2个64x256x1x1的tensor  

* 第四层:Sequential[layer4],包含2个BasicBlock，输出为64x512x1x1的tensor

* 最后经过Linear(fc)的处理，输出为64x200的一个output

## Task2(观察曲线变化)
$$
\text{loss}
    (x, y) =
    \begin{cases}
        1 - \cos(x_1, x_2), & \text{if } y = 1 \\
    \max(0, \cos(x_1, x_2) - \text{margin}), & \text{if } y = -1
\end{cases}
$$