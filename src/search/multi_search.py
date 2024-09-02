from pymilvus import connections, Collection
from pymilvus import WeightedRanker
from pymilvus import AnnSearchRequest

# ref: https://zhuanlan.zhihu.com/p/713975401
# Connect to Milvus
connections.connect(
    host="192.168.3.5",  # Replace with your Milvus server IP
    port="19530",
    db_name='ph_dsc'
)

collection = Collection(name='hybrid_search')
collection.load()

# 单独搜索第一个字段
query_filmVector = [
    [0.8896863042430693, 0.370613100114602, 0.23779315077113428, 0.38227915951132996, 0.5997064603128835]]
search_param_1 = {
    "data": query_filmVector,  # 要查询的向量,支持多个
    "anns_field": "filmVector",  # 向量字段名
    "param": {
        "metric_type": "L2",  # 和创建collection时的类型一致
        "params": {"nprobe": 10}
    },
    "limit": 2  # 返回几条数据
}
request_1 = AnnSearchRequest(**search_param_1)

# 单独搜索第二个字段
query_posterVector = [
    [0.02550758562349764, 0.006085637357292062, 0.5325251250159071, 0.7676432650114147, 0.5521074424751443]]
search_param_2 = {
    "data": query_posterVector,  # Query vector
    "anns_field": "posterVector",  # Vector field name
    "param": {
        "metric_type": "L2",  # This parameter value must be identical to the one used in the collection schema
        "params": {"nprobe": 10}
    },
    "limit": 2  # Number of search results to return in this AnnSearchRequest
}
request_2 = AnnSearchRequest(**search_param_2)

reqs = [request_1, request_2]

# 第二步 配置一个重排序策略，用于将上面单个向量的搜索结果重新按相似度排序
# 目前支持WeightedRanker权重排序列和RRFRanker倒数排序融合
# 详情见：https://milvus.io/docs/reranking.md
rerank = WeightedRanker(0.8, 0.2)

# 第三步 执行搜索
res = collection.hybrid_search(
    reqs,  # List of AnnSearchRequests created in step 1
    rerank,  # Reranking strategy specified in step 2
    limit=2  # Number of final search results to return
)
print(res)
