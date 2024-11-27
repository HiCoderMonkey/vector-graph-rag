# vector-graph-rag
vector-graph-rag 提供一种基于图向量特征检索的RAG方式。
![architecture_diagram.png](docs%2Fimgs%2Farchitecture_diagram.png)

## neo4j
neo4j可以在官网申请一个免费的实例

## 抽取图谱
### ne04j 数据库 （官网可以免费申请一个实例）
https://console.neo4j.io/

### 使用 llm graph builder 抽取知识
https://llm-graph-builder.neo4jlabs.com/
使用维基百科的武功山的资料抽取了一个武功山的知识图谱
![wugons_kg.png](docs%2Fimgs%2Fwugons_kg.png)

## 预处理，提取图谱特征，转为向量，保存到 milvus
### 设置环境
```shell
pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
export PYTHONPATH="backend/src"
```
### 执行提取程序
```shell
# 安装依赖
cd backend/bin
```