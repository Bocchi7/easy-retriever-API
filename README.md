# easy-retriever-API

## 概述

这个仓库提供了简易的API用于文本检索。支持BM25检索和稠密向量检索。BM25算法使用Elasticsearch实现。稠密向量检索中的编码器和重排器使用的是BGE模型，见[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/)，对稠密向量建立索引用的也是Elasticsearch。

这里文档库和询问采用了非常简单的格式。每个文档包含字段`id`（应确保唯一性）和`text`，询问仅需包含字段`text`。计算相似度时都是直接基于`text`计算的。如果您有附加的信息，可以在外部以`id`为唯一标识符存储，仅用该API来进行检索工作；或者您可以修改代码，比如增加`title`字段，检索时尝试同时使用`text`和`title`的信息（个人观点，如果这样同时考虑`text`和`title`字段，但没有对格式有恰当的处理，可能损害稠密向量检索算法的性能，不利于对两者各自性能的公正评估）。

该API仅实现了非常简单的功能，包括：创建索引、向索引中添加文档、基于BM25或稠密向量检索文档。对于更加细致的功能，您可以在原来的基础上自己实现。

## 安装

首先根据`requirements.txt`配置环境。

接下来需要安装Elasticsearch服务端。从官网上下载压缩包，放置于`data/`目录下，然后解压。在此之后您可以删除压缩文件。
```bash
cd data
wget -O elasticsearch-8.15.1.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.15.1-linux-x86_64.tar.gz  
tar zxvf elasticsearch-8.15.1.tar.gz
```

安装完客户端后，还需要对它做些细致的调整。接着之前的操作，进入Elasticsearch服务端目录：
```bash
cd elasticsearch-8.15.1
```

Elasticsearch8.x与Elasticsearch7.x相比的一个重大不同是默认情况下会有各种安全性验证。因为很麻烦所以作者选择将安全性验证关掉，具体做法是：打开`config/elasticsearch.yml`，将所有`enabled: `后的`true`改成`false`。如果您有在安全方面的需求，您需要对本仓库代码进行一点修改，使之适应安全性验证。

Elasticsearch中内置的分词器不支持中文。为了使之支持中文，本仓库采用了[ik分词器](https://github.com/infinilabs/analysis-ik)插件。安装操作如下：
```bash
bin/elasticsearch-plugin install https://get.infini.cloud/elasticsearch/analysis-ik/8.15.1
```

关于这个插件具体该怎么用，您可以去对应仓库查看教程。本仓库提供了一种简单的使用方法，您可以在（项目根目录下）`config/chinese.json`中看到相关的配置，但作者建议若不是很了解则暂时不要改动它。

以上Elasticsearch服务端就安装完成了，接着启动服务端：（如果之前你有些步骤不小心跳过了然后不得不重新补上，那么服务端记得重启）
```bash
bin/elasticsearch
```
如果需要，您可以用`nohup`指令让服务端在后台运行。这样做的好处是和服务端交互的命令行窗口被关闭后程序仍能保持运行，坏处是中断服务端程序不是很方便，您将需要手动找到相应进程号并中断它。
```bash
nohup bin/elasticsearch &
```

然后您可能还需要安装BGE模型用于稠密向量检索，作者尝试过的模型有：
1. [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)：英文的文本编码器模型。用于检索阶段中生成文本的向量表示。
2. [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5)：中文的文本编码器模型。用于检索阶段中生成文本的向量表示。
3. [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)：同时支持中文和英文的交叉编码模型。用于重排阶段中生成文档与询问的相关性分数。

如果您有更多的需求，可以尝试使用别的模型，比如对特定数据集微调后的BGE模型。

如果您的网络不好，那您将不得不将模型下载到本地（如果不清楚要下载哪些文件，您可以将模型所在的整个目录保持原目录结构下载下来）。下载到本地之后，在配置文件中（如`config/chinese.json`），您将要分别把`embedding_model_name_or_path`和`reranker_model_name_or_path`对应到模型所在文件夹的地址。


## 运行

API的核心代码已经封装在了`src/utils.py`的`IndexWrapper`类中，作者建议在`IndexWrapper`的基础上实现您的需求，您可以直接查看代码以了解各函数的参数和返回值。另外可以直接调用`src/build.py`和`src/search.py`分别实现建立索引和查询的操作，这两个脚本都是对`IndexWrapper`特定功能的简单封装。

默认的配置文件在`config/`中，您可以直接使用它们，或者视情况修改。其中`config/chinese.json`中，`settings`属性和`language`属性（涉及到ik分词器的调用）不要随意更改，若不了解其作用。配置中其它部分的介绍：

| 属性                         | 描述                                                         | 备注                                                         |
| ---------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| use_dense                    | 是否启用稠密向量检索。                                       | 若否，不会加载相关模型。                                     |
| batch_size                   | 进行模型推理时的批大小。                                     | 文档集合预处理和查询处理、编码器模型和交叉编码器模型都用了相同的batch_size。若有需要您可以改动。 |
| chunk_size                   | 程序中有时会分块地操作（例如分块读入文档集合），防止大量数据在内存中堆积而没有及时得到处理。这个表示块大小。 |                                                              |
| thread_count                 | 使用`elasticsearch.helpers.parallel_bulk`对Elasticsearch数据库中并发地批量执行操作（这里是添加文档）时，所用的线程数。 |                                                              |
| embedding_model_name_or_path | 编码器模型的名字或路径。                                     | 如果是名字（比如`BAAI/bge-base-en-v1.5`）则尝试从网上加载模型 |
| reranker_model_name_or_path  | 交叉编码器模型的名字或路径。                                 | 如果是名字（比如`BAAI/bge-reranker-base`则尝试从网上加载模型 |


这里提供了一个小规模的数据集供您上手操作。数据集取自[STARD](https://github.com/oneal2000/STARD)，采用了其中全部的文档和部分查询，并整理成了适合于本仓库代码操作的格式。数据集见`/data/STARD`。以下是在此数据集基础上的简单操作。

您可以快捷地运行`src/build.py`以建立索引：
```bash
cd src
python build.py --args_path ../args/build_args/build_stard.json
```
然后运行`src/search.py`进行BM25检索：
```bash
python search.py --args_path ../args/search_args/search_stard_BM25.json > output_BM25.txt
```
或者稠密向量检索：
```bash
python search.py --args_path ../args/search_args/search_stard_dense.json > output_dense.txt
```
`src/search.py`的输出格式是`json`格式下的列表套列表套对象。最外层列表是顺序显示各询问的输出结果，每个输出结果是表示一系列返回的文档的列表，列表中每个元素是个包含`id`、`text`、`score`属性的对象表示相应文档的信息。其中`score`属性表示相关性得分。每个询问的输出结果已经根据相关性得分倒序排序。

`args/`下的各文件表示这些脚本运行的参数。您运行`src/build.py`和`src/search.py`时也可以直接在命令行中输入各参数。这里解读一下各参数的含义。

`src/build.py`用到的参数文件中各属性如下：
| 属性        | 描述                                                         | 备注 |
| ----------- | ------------------------------------------------------------ | ---- |
| data_path   | 文档集合文件路径。默认为None（意味着创建的索引中未加入文档） |      |
| index_name  | 索引名字。                                                   |      |
| config_path | 配置文件路径。                                               |      |
| server      | 服务端的URL。默认为`http://localhost:9200`。                 |      |
| override    | 如果有相同名字的索引存在，是否覆盖。                         |      |

`src/search.py`用到的参数文件中各属性如下：
| 属性                  | 描述                                                         | 备注                                                         |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| query_path            | 询问集合文件路径。                                           |                                                              |
| index_name            | 索引名字。                                                   |                                                              |
| config_path           | 配置文件路径。                                               |                                                              |
| method                | 采用的算法。只能是BM25或dense。                              | 如果对应配置文件中use_dense=False，则不能采用dense。         |
| server                | 服务端的URL。默认为`http://localhost:9200`。                 |                                                              |
| rerank                | 是否启用重排器。                                             | 只有method=="dense"时才有关系。                              |
| filtered_count        | 最终返回的文档数量。                                         |                                                              |
| coarse_filtered_count | 在稠密向量检索中，检索阶段先检索出coarse_filtered_count个结果，这些结果重排后保留前filtered_count个结果。 | 只有method=="dense"且rerank==True时才有关系。默认等于filtered_count。 |
| kNN_num_candidates    | k近邻算法时的候选数目。                                      | 近似k近邻居算法中，部分向量会被筛选出，然后在这些向量中选取相关度最大的k个。kNN_num_candidates就是一开始被筛选出的部分向量的个数。 |

关于文档集合和询问集合的文件格式：两者都是`jsonl`文件格式，即是有若干行，每一行都是`json`对象。
- 文档集合中每行的`json`对象至少有`id`和`text`两个属性。
- 询问集合中`json`对象至少有`text`属性。

您可以查看`data/STARD`下具体的例子，其中`psgs.jsonl`是文档集合，`eval.jsonl`是询问集合（这里多了些属性，但不重要）。

## 结语

如果有bug欢迎指出。另外本仓库的目的就是提供一个简易API，但作者本人可能对它简易的程度没有清晰的认知。如果您顺着本教程操作且略读过代码后仍然不清楚如何调用，欢迎提意见。
