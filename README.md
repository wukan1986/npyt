# NPY file format with Tail

带小尾巴的`NPY`文件格式

## 项目特点

1. 与`NPY`文件格式兼容, 可以用原生的`np.load`函数加载
2. 支持`memmap`模式, 可以跨进程一写多读。`append`、`pop`
3. 支持调整文件大小`resize`，但不移动数据
4. 支持`tell`、`seek`、`rewind`、`read`等操作

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

nt1.append(arr, ringbuffer=False, bulk=False)
print(nt2.data())

nt1.append(arr[0:1], ringbuffer=False, bulk=False)
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

## 环形缓冲区RingBuffer

本项目支持两种用法：

1. 普通缓冲区。`NPYT`
2. 环形缓冲区。`NPYT_RB`,多了`append2`和`pop2`

环形一定会出现`start>end`的情况, 如果整块取出数据，会出现要拼接首尾两段数据的情况，这会导致数据复制，降低性能。

## 优化建议

1. 优先使用`NPYT`，并且`append(ringbuffer=False)`，使用普通缓存区
2. 使用`append(bulk=True)`，防止输入数据被切片到不连续的缓存区。注意：
    - `bulk=True`，失败时返回值等于`len(array)`，成功返回值`0`
    - `bulk=Flase`，返回值范围`0~len(array)`。`>0`时还有数据没插入，一定不能忘了
3. `NPYT.save(arr, capacity=?)` 分配2n以上的空间
    - 环形缓冲区时，减少成环概率
    - 普通缓冲区时，减少重置指针概率