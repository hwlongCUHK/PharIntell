from numba import jit
import json5
import numpy as np
import os
import time
from phar_embedding import get_sentence_transformer
from phar_post_emb import ensure_embeddings_for_posts, load_folder_emb_map
from qwen_agent.tools.base import BaseTool, register_tool

from .folder_name_resolve import resolve_folder_name

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


@jit(nopython=True)
def vector_similarity_1(avg_vector, vector):
    oneSimilarity = avg_vector @ vector.T

    return oneSimilarity


class Single_Pass_Cluster(object):
    def __init__(self, theta=0.5):
        # Clustering uses precomputed query_emb; live encode path shares phar_embedding singleton.
        self.theta = theta

    @property
    def bge_model(self):
        return get_sentence_transformer()


    def get_bge_vector_representation(self, word_segmentation, model):
        embeddings = model.encode(word_segmentation, normalize_embeddings=True)
        return embeddings


    def getBGEMaxSimilarity(self, dictTopic, vector):
        # Calculate the text similarity between the newly entered document and existing documents. Here, cosine similarity is used.

        maxValue = 0
        maxIndex = -1
        # print(dictTopic)
        for k, cluster in dictTopic.items():
            avgVector = cluster['avgVector']
            # avgVector = np.mean(cluster, axis=0)
            # vector = vector.astype(avgVector.dtype)
            oneSimilarity = vector_similarity_1(avgVector, vector)

            if oneSimilarity > maxValue:
                maxValue = oneSimilarity
                maxIndex = k
        return maxIndex, maxValue

    def single_pass(self, corpus, data_title, data_content, data_ocr, theta, unique_id, like_count, dictTopic=None, clusterTopic=None, start_cluster_id=1):
        if dictTopic == None:
            dictTopic = {}
        if clusterTopic == None:
            clusterTopic = {}
            numTopic = start_cluster_id
        else:
            # int_numbers = [int(num) for num in str_numbers]
            numTopic = max(int(num) for num in list(dictTopic.keys())) + 1
        cnt = 1
        for vector, source_title, source_content, source_ocr, unique_id, like_count in zip(corpus, data_title, data_content, data_ocr, unique_id, like_count):
            if numTopic == start_cluster_id:
                dictTopic[numTopic] = {}
                dictTopic[numTopic]['vectors'] = []
                dictTopic[numTopic]['vectors'].append(vector)
                dictTopic[numTopic]['avgVector'] = vector

                clusterTopic[numTopic] = []
                clusterTopic[numTopic].extend([{'unique_id': unique_id, 'like_count': like_count, 'title': source_title, 'content': source_content, 'ocr': source_ocr}])
                
                lastTopic = {'unique_id': unique_id,'cluster_id': numTopic}
                numTopic += 1
            else:
                maxIndex, maxValue = self.getBGEMaxSimilarity(dictTopic, vector)
                # 以第一篇文档为种子，建立一个主题，将给定语句分配到现有的、最相似的主题中
                if maxValue > theta:
                    dictTopic[maxIndex]['vectors'].append(vector)
                    dictTopic[maxIndex]['avgVector'] = np.mean(dictTopic[maxIndex]['vectors'], axis=0)
                    clusterTopic[maxIndex].extend([{'unique_id': unique_id, 'like_count': like_count,  'title': source_title, 'content': source_content, 'ocr': source_ocr}])
                    lastTopic = {'unique_id': unique_id, 'cluster_id': maxIndex}

                # 或者创建一个新的主题
                else:
                    dictTopic[numTopic] = {}
                    dictTopic[numTopic]['vectors'] = []
                    dictTopic[numTopic]['vectors'].append(vector)
                    dictTopic[numTopic]['avgVector'] = vector

                    clusterTopic[numTopic] = []
                    clusterTopic[numTopic].extend([{'unique_id': unique_id, 'like_count': like_count, 'title': source_title, 'content': source_content, 'ocr': source_ocr}])
                    lastTopic = {'unique_id': unique_id, 'cluster_id': maxIndex}
                    numTopic += 1
            cnt += 1

        return dictTopic, clusterTopic, lastTopic

    def fit_transform(self, theta=0.5, dictT=None, clusterT=None, query=None, query_emb=None, start_cluster_id=1):

        # Synthesize the above functions to get the final clustering results: including cluster labels, number of each cluster, key topic words and key sentences
        start_time1 = time.time()
        data_title = [query['标题']]
        data_content = [query['内容']]
        data_ocr = [query['OCR']]
        # 内容 OCR unique_id
        # datMat = [query['title']]
        # datMat_ori = [query['content']]
        # datMat_rewrite = [query['ocr']]
        if 'like_count' in query.keys():
            like_count = [query['like_count']]
        else:
            like_count = [0]
        # 
        unique_id = [query['unique_id']]


        # 得到文本数据的空间向量表示
        corpus_bge = query_emb#self.get_bge_vector_representation(data_content, self.bge_model)
        dictTopic, clusterTopic,lastTopic = self.single_pass(corpus_bge, data_title, data_content, data_ocr, theta, unique_id, like_count, dictT, clusterT, start_cluster_id)
        # 按聚类语句数量对聚类结果进行降序排列，找到重要的聚类群
        return dictTopic, clusterTopic,lastTopic
    
    
@register_tool('topic_clustering')
class TopicClustering(BaseTool):
    # `description` is used to tell the agent the function of the tool.
    description = "社交媒体话题聚类服务，输入文件夹名称，将对应文件夹中的贴文进行聚类。"
    # `parameters` tells the agent what input parameters the tool has.
    parameters = [{
        'name': 'folder_name',
        'type': 'string',
        'description': '文件夹名称',
        'required': True
    }]
    
    single_pass_cluster = Single_Pass_Cluster()
    data_cluster = {}
    data_vector = {}
    
    def call(self, params: str, data_folder, **kwargs):
        params_json = json5.loads(params)
        folder_name = resolve_folder_name(data_folder, params_json["folder_name"])
        parts = folder_name.split("_", 2)
        if len(parts) != 3:
            raise ValueError(
                f"文件夹名称需为 corpus_start_time_end_time（仅含两个下划线分隔符），收到: {folder_name}"
            )
        location, start_time, end_time = parts
        cluster_name = f"{location}_{start_time}_{end_time}_cluster"
        if hasattr(data_folder, "folder_has_no_posts") and data_folder.folder_has_no_posts(folder_name):
            return data_folder.EMPTY_FOLDER_REPLY
        posts = data_folder.data_folders[folder_name]
        posts_idx = data_folder.data_folders[f"{folder_name}_idx"]
        if not posts:
            if hasattr(data_folder, "mark_folder_empty"):
                data_folder.mark_folder_empty(folder_name)
                data_folder.mark_folder_empty(cluster_name)
            return data_folder.EMPTY_FOLDER_REPLY
        posts_emb = load_folder_emb_map(folder_name)
        model = get_sentence_transformer()
        posts_emb, emb_note = ensure_embeddings_for_posts(
            [str(x) for x in posts_idx],
            posts,
            posts_emb,
            model=model,
            persist_folder_name=folder_name,
        )

        dictT = None
        clusterT = None
        skipped = 0
        for item_idx, item in zip(posts_idx, posts):
            uid = str(item_idx)
            if uid not in posts_emb:
                skipped += 1
                continue
            dictT, clusterT, lastTopic = self.single_pass_cluster.fit_transform(
                theta=0.5,
                dictT=dictT,
                clusterT=clusterT,
                query=item,
                query_emb=np.array(posts_emb[uid])[None, :],
            )

        if not clusterT:
            return (
                emb_note
                + "聚类失败：没有可用向量。"
                + (f"（跳过 {skipped} 条）" if skipped else "")
            )

        cluster_folder = f"{location}_{start_time}_{end_time}_cluster"
        ordered_keys = sorted(clusterT.keys(), key=lambda k: int(k))
        data_folder.data_folders[cluster_folder] = [clusterT[k] for k in ordered_keys]
        data_folder.data_folders[f"{cluster_folder}_key"] = ordered_keys
        data_folder.show_funcs[cluster_folder] = self.show

        skip_note = f"（聚类时跳过 {skipped} 条无向量帖）\n" if skipped else ""
        k = len(ordered_keys)
        return (
            emb_note
            + skip_note
            + f"文件夹'{folder_name}'中的帖子已经完成聚类，共 **K={k}** 个话题簇，"
            f"结果存放在 '{cluster_folder}'。\n"
            f"下一步必须用 data_folder 读取该 _cluster 文件夹（不可跳过）。"
            f"start_idx/end_idx 为**簇编号**（半开区间），不是帖序号；每个簇内最多展示 5 条帖。"
            f"最终 JSON 条目数应为 1–min(K, 8)，按簇的异质化程度决定，**勿固定凑满 5 条**。"
            f"示例：读前 3 个簇 → start_idx=0, end_idx=3；读全部簇 → end_idx={k}。"
        )
    
    @staticmethod
    def show(clusters: list, cluster_start: int, cluster_end: int) -> str:
        """clusters: list of clusters; each cluster is a list of post dicts."""
        if not clusters:
            return "没有找到符合条件的帖子。"

        n_clusters = len(clusters)
        cluster_start = max(0, int(cluster_start))
        cluster_end = min(int(cluster_end), n_clusters)
        if cluster_start >= cluster_end:
            return f"簇索引无效：共 {n_clusters} 个簇，请求区间 [{cluster_start}, {cluster_end})。"

        from phar_tool_limits import truncate_post_field

        header = (
            f"共 {n_clusters} 个话题簇；本次展示簇 {cluster_start}–{cluster_end - 1} "
            f"（data_folder 参数 start_idx={cluster_start}, end_idx={cluster_end}）。"
            f"每个簇内最多展示 5 条帖。\n\n"
        )
        result = header
        for ci in range(cluster_start, cluster_end):
            cluster_posts = clusters[ci]
            if not cluster_posts:
                result += f"=== 簇 {ci}（空）===\n\n"
                continue
            show_n = min(5, len(cluster_posts))
            result += f"=== 簇 {ci}（簇内 {len(cluster_posts)} 帖，展示前 {show_n} 条）===\n"
            for pi in range(show_n):
                post = cluster_posts[pi]
                result += f"  帖 {pi + 1}:\n"
                for key, value in post.items():
                    if key in ["id", "source", "pic_num", "pictures", "video", "rca_list"]:
                        continue
                    text = value if isinstance(value, str) else str(value)
                    if key in ("content", "title", "ocr", "内容", "标题"):
                        text = truncate_post_field(text)
                    result += f"  ***{key}***: {text}\n"
                result += "\n"
        return result.strip()
        