## DAG 计算图执行引擎

VT-Transformer 采用了一种宏语言来描述神经网络的计算图。这个宏语言采用堆栈式语法，执行效率非常高，同时又具备相当的灵活性，避免了直接用 C++ 语言来进行计算图的硬编码。

### 简单的语法

DAG 宏语言的语法实现参考 tensortype/dag.cpp ，包括四种元类型，分别是：数字、字符串、张量指针以及函数（又叫做 word），运行时包括一个全局的堆栈和一个全局的Hash。

比如代码：

```
3.14 2 * 
```

其中3.14，2都是数字型的元类型，分别压入堆栈，但解析到'*'的时候，表示一个函数，这个函数的实现从堆栈取出两个数，执行乘法之后，将结果压回栈。
因此执行完上面的代码之后，堆栈的顶部将是数字类型，其值为 6.28 。

只有两个 Word 可以操作全局 Hash，分别是 读函数 '@'，以及写函数 '!'，两个函数分别从栈上获取索引（必须是字符串）和值，读写操作 Hash ，一个简单例子如下：

```
3.14 'pi' !
2 'pi' @ *
```

第一行代码，将 'pi' 作为索引，3.14 作为值写入到全局 Hash 中。第二行，从 hash 中读出 'pi' 对应的值，并且把值压入到栈顶。

### 定义宏函数

可以通过 %def  ...  %end 的方式，定义宏函数，DAG 解析执行的时候，宏函数直接展开执行，其内部实现并没有这样的实际函数。下面是一个简单的例子：

```
%def double_add_one
2.0 * 1.0 +
%end 

3.14 double_add_one double_add_one
```

这个例子里面，定义了一个宏函数，执行乘二加一的操作。最后一行代码，执行的时候，会按照定义的方式，替换展开的方式，实际上的代码如下：

```
3.14     2.0 * 1.0 +      2.0 * 1.0 +
```

### 定义函数

真正的函数，需要用户提供一个可执行的函数入口，即通过 vt::Enviroment 的API 增加，比如下面的例子，定义了一个的 正玄 函数。

```
struct MathSine : public NativeWord {
    void run(Stack& stack) override {
        float angle = stack.pop_number();
        float value = sin(angle);
        stack.push_number(value);
    }
    NWORD_CREATOR_DEFINE_LR(MathSine)
};

// env is a refrence of vt::Envrioment
env.insert_native_word("^", MathSine::creator );

```

每一个函数参数，只有一个堆栈对象，函数只能通过堆栈传值和返回值。

vt-transformer 提供了相当的多基本功能函数，包括最基本堆栈操作，数学计算，字符串操作，终端打印等等，具体参考 tensortype/dag.cpp ，用户可以利用这些函数构建自己的宏定义函数，简化 dag 文件的开发。




