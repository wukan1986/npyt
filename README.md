# NPY file format with Tail

带小尾巴的`NPY`文件格式

## 项目特点

1. 与`NPY`文件格式兼容, 可以用原生的`np.load`函数加载
2. 支持memmap模式, 可以跨进程一读多写
3. 支持调整文件大小

## 安装

```bash
pip install npyt
```

## 使用

```python
import numpy as np

from npyt import NPYT

arr = np.array([[1, 2, 3], [4, 5, 6]])

# 创建文件
nt1 = NPYT("f2.npy", max_length=9, mode="w+").save(arr, end=0).load()
nt2 = NPYT("f2.npy", mode="r").load()
nt3 = NPYT("f2.npy", mode="r").load()

nt1.append(arr)
print(nt2.data())

nt1.append(arr[0:1])
print(nt3.data())
```

## 项目背景

本人需要一种准实时的行情存储方式，考虑`arrow`或`np.memmap()`

1. `arrow`，列式存储，并不适合行情数据。因为要将新数据写入到不同的位置
2. `np.memmap()`，内存映射，行式存储，适合行情数据。但还是有不足
    1. 需另行维护`dtype`，`shape`等信息
    2. 文件大小随数据量而增大，不能动态调整大小

所以如果初始时创建一个大文件，然后维护一个标记，用来记录数据的位置，就能实现数据增长了。

标记是放在同一文件，还是放在不同文件呢？

最开始是放在不同文件，这样代码实现方便，但是要维护两个文件。
后来发现`np.load`函数支持`mmap_mode`参数，可以直接加载内存映射文件，并且还是直接带了`dtype`和`shape`信息。为何不直接用呢？

只要把额外信息放在`NPY`文件的尾部就可以了。

## 如何实现修改文件大小而不移动数据

`NPY`文件头有`shape`信息字符串，例如；

```text
(20, 3)
(200, 3)
```

很明显，这两个字符串的长度是不一样的。而`NPYT`项目修改了此部分，让字符串长度一致，例如；

```text
(         20,          3)
(        200,          3)
```

这样就可以直接修改数据大小而不用移动数据区。
故`np.save`保存的文件'.npy'无法用`NPYT`来修改大小，但反过来`NPYT.save`保存的文件可以用`np.load`来读取。