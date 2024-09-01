import random

from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema


# ref: https://zhuanlan.zhihu.com/p/713975401
client = MilvusClient(uri="http://192.168.3.5:19530", db_name="ph_dsc")
index_params = MilvusClient.prepare_index_params()

if not client.has_collection("hybrid_search"):
    # filmVector 和posterVector 是两个向量字段
    fields = [
        FieldSchema(name="film_id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="filmVector", dtype=DataType.FLOAT_VECTOR, dim=5),  # Vector field for film vectors
        FieldSchema(name="posterVector", dtype=DataType.FLOAT_VECTOR, dim=5)]  # Vector field for poster vectors

    schema = CollectionSchema(fields=fields, enable_dynamic_field=False)

    client.create_collection(
        collection_name="hybrid_search",
        schema=schema
    )

    # 为2个向量字段 创建索引
    index_params.add_index(
        field_name="filmVector",
        metric_type="L2",
        index_type="IVF_FLAT",
        index_name="filmVectorIndex",
        params={"nlist": 128},
    )

    index_params.add_index(
        field_name="posterVector",
        metric_type="L2",
        index_type="IVF_FLAT",
        index_name="posterVectorIndex",
        params={"nlist": 128},
    )

    client.create_index(
        collection_name="hybrid_search",
        index_params=index_params
    )

entities = []

# 先创建1000条数据，其中的向量随机生成
for _ in range(1000):
    # generate random values for each field in the schema
    film_id = random.randint(1, 1000)
    film_vector = [random.random() for _ in range(5)]
    poster_vector = [random.random() for _ in range(5)]

    # create a dictionary for each entity
    entity = {
        "film_id": film_id,
        "filmVector": film_vector,
        "posterVector": poster_vector
    }

    # add the entity to the list
    entities.append(entity)

client.insert(
    collection_name="hybrid_search",
    data=entities
)




