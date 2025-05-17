# NPY file format with Tail

带小尾巴的`NPY`文件格式

## 项目特点

1. 与`NPY`文件格式兼容, 可以用原生的`np.load`函数加载
2. 支持`memmap`模式, 可以跨进程一写多读。`append`、`pop`
3. 支持调整文件大小`resize`，但不移动数据
4. 支持`tell`、`seek`、`rewind`、`read`等操作
5. 支持无限写入`NPY8`

## 安装

```bash
pip install npyt
```

## 使用

```python
import numpy as np

from npyt import NPYT

arr = np.array([1, 2, 3, 4, 5, 6])

file = "tmp.npy"
# 创建文件
nt1 = NPYT(file).save(arr, capacity=10, end=0).load(mmap_mode="r+")
# 只读加载文件
nt2 = NPYT(file).load(mmap_mode="r")
nt3 = NPYT(file).load(mmap_mode="r")

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
后来发现`np.load`函数支持`mmap_mode`参数，可以直接加载内存映射文件，并且还是直接带了`dtype`和`shape`信息。为何不直接用呢？ 只要把额外信息放在`NPY`文件的尾部就可以了。

## 如何实现修改文件大小而不移动数据

`NPY`文件头有`shape`信息的字符串，例如；

```text
(20, 3)
(200, 3)
```

很明显，这两个字符串的长度是不一样的。而`NPYT`项目修改了此部分，让字符串长度一致，例如；

```text
(                    20, 3)
(                   200, 3)
```

这样就可以直接修改数据大小而不用移动数据区。

## 兼容性

1. `np.load`可以打开`NPYT.save`保存的文件。`NPYT.load`可以打开`np.save`保存的文件, 取原始数据`NPYT._raw()`
2. `npy`文件大小不可修改，`NPYT.resize`文件大小可以修改
3. `np.load`后`dtype`缺失`align`属性。`NPYT.load`后`dtype`还原了`align`属性。(numpy 2.2.5)
    - 当`array`要传给`numba.jit`函数，函数中需要对`array`进行修改，由于`align`属性缺失，可能导致修改时出现数据复制，复制出来的对象是只读
    - https://github.com/numpy/numpy/issues/28973

## 无限写入模式NPY8

最开始提供了`RingBuffer`模式，但是编写过于复杂，还无法零拷贝，所以取消了。现在提供的是无限写入模式`NPY8`，可以`7*24`写入数据

本质是创建一个文件夹和一个`.lock`文件，通过`.lock`来维护文件夹中最新的几个`NPYT`文件。

## 优化建议

`NPY8.tail`是跨文件的，如果能大概率取的是一个文件而不是多个文件，就能减少拷贝

1. 返回的是`np.ndarray`列表，一个个按需使用比`concat`后使用更好
2. `capacity_per_file`设置得大一些，越大越能减少跨文件的概率。例如：
    - capacity=100,tail=100,则99%的概率跨文件
    - capacity=100,tail=50,则49%的概率跨文件
    - capacity=50,tail=1,则0%的概率跨文件
    - (tail-1)/capacity 跨文件的概率

`tail`由自己的策略所决定，根据用户能接受的概率，文件大小，记录时长来设置`capacity_per_file`

