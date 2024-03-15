import sys

sys.path += ['./']
import os
import faiss
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer


def index_retrieve(index, query_embeddings, topk, batch=None):
    """
    retrieve top k neighbours using the efficient index
    @param index: a faiss index
    @param query_embeddings: a list of encoded queries
    @param topk: top k neighbours
    @param batch: size of batch
    @return:
    """
    print("Query Num", len(query_embeddings))
    start = timer()
    if batch is None:
        # one go
        scores, nearest_neighbors = index.search(query_embeddings, topk)
    else:
        # batched
        query_offset_base = 0
        pbar = tqdm(total=len(query_embeddings))
        nearest_neighbors,scores = [] ,[]

        while query_offset_base < len(query_embeddings):
            batch_query_embeddings = query_embeddings[query_offset_base:query_offset_base + batch]
            batch_scores, batch_nn = index.search(batch_query_embeddings, topk)
            nearest_neighbors.extend(batch_nn.tolist())
            scores.extend(batch_scores.tolist())
            query_offset_base += len(batch_query_embeddings)
            pbar.update(len(batch_query_embeddings))
        pbar.close()

    elapsed_time = timer() - start
    elapsed_time_per_query = 1000 * elapsed_time / len(query_embeddings)
    print(f"Elapsed Time: {elapsed_time:.1f}s, Elapsed Time per query: {elapsed_time_per_query:.1f}ms")
    return nearest_neighbors,scores


def construct_flatindex_from_embeddings(embeddings, ids=None):
    """
    construct a flat index (no compression)
    @param embeddings: embeddings for documents to index
    @param ids: id for document
    @return:
    """
    dim = embeddings.shape[1]
    print('embedding shape: ' + str(embeddings.shape))
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    if ids is not None:
        ids = ids.astype(np.int64)
        print(ids.shape, ids.dtype)
        index = faiss.IndexIDMap2(index)
        index.add_with_ids(embeddings, ids)
    else:
        index.add(embeddings)
    return index


gpu_resources = []


def convert_index_to_gpu(index, faiss_gpu_index, useFloat16=False):
    """
    converting an faiss index into gpu or multiple gpus

    @param index: index instance
    @param faiss_gpu_index: gpu list or index
    @param useFloat16: bool
    @return:
    """
    if type(faiss_gpu_index) == list and len(faiss_gpu_index) == 1:
        faiss_gpu_index = faiss_gpu_index[0]
        """
         working with a single gpu
         """
    if isinstance(faiss_gpu_index, int):
        res = faiss.StandardGpuResources()
        res.setTempMemory(512 * 1024 * 1024)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = useFloat16
        index = faiss.index_cpu_to_gpu(res, faiss_gpu_index, index, co)
    else:
        """
        for a list of gpus, distribute the index
        """
        global gpu_resources
        if len(gpu_resources) == 0:
            import torch
            for i in range(torch.cuda.device_count()):
                res = faiss.StandardGpuResources()
                res.setTempMemory(256 * 1024 * 1024)
                gpu_resources.append(res)

        assert isinstance(faiss_gpu_index, list)
        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = useFloat16
        for i in faiss_gpu_index:
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)

    return index


gpu_resources = []


def adore_load_index(passage_embeddings, faiss_gpu_index):
    """
    @param passage_embeddings: inferred document embeddings from STAR encoder
    @param faiss_gpu_index: a list of gpus to place the index
    @return: the index placed on gpu/s if possible
    """
    dim = passage_embeddings.shape[1]
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(passage_embeddings)
    if faiss_gpu_index:
        if len(faiss_gpu_index) == 1:
            res = faiss.StandardGpuResources()
            res.setTempMemory(128 * 1024 * 1024)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = False
            index = faiss.index_cpu_to_gpu(res, faiss_gpu_index, index, co)
        else:
            global gpu_resources
            import torch
            for i in range(torch.cuda.device_count()):
                res = faiss.StandardGpuResources()
                res.setTempMemory(128 * 1024 * 1024)
                gpu_resources.append(res)

            assert isinstance(faiss_gpu_index, list)
            vres = faiss.GpuResourcesVector()
            vdev = faiss.IntVector()
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            for i in faiss_gpu_index:
                vdev.push_back(i)
                vres.push_back(gpu_resources[i])
            index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)

    return index
