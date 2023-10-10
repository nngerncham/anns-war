import faiss
from testbed import load_base, load_query, K, D, NUM_THREADS, evaluate, TestResult
from psutil import Process
from os import getpid
from time import time_ns

index_directory = "/home/nawat/muic/senior/anns-war/indices/faiss/index.bin"


def build_faiss():
    index = faiss.IndexHNSWFlat(D, 128)
    index.hnsw.efConstruction = 512
    faiss.omp_set_num_threads(NUM_THREADS)
    data = load_base()

    proc = Process(getpid())
    start_mem = proc.memory_info().rss
    start_time = time_ns()

    index.add(data)

    end_time = time_ns()
    end_mem = proc.memory_info().rss

    # print(f"\n== FAISS-based HNSW ==\n"
    #       f"Time taken: {(end_time - start_time) / 1e9 / 60} minutes\n"
    #       f"Memory difference: {(end_mem - start_mem) / 1e9} GB\n"
    #       f"Memory at end: {end_mem / 1e9} GB")

    faiss.write_index(index, index_directory)
    return (end_time - start_time) / 1e9 / 60, (end_mem - start_mem) / 1e9, end_mem / 1e9


def search_faiss():
    index = faiss.read_index(index_directory)
    index.hnsw.efSearch = 128

    queries, gts = load_query()
    start_time = time_ns()
    _, res = index.search(queries, K)
    end_time = time_ns()

    recall = evaluate(res, gts)
    # print(f"== HNSW Search ==\n"
    #       f"Time taken: {(end_time - start_time) / 1e9 / 60} minutes\n"
    #       f"Recall: {recall}")
    return (end_time - start_time) / 1e9 / 60, recall


def test_faiss():
    build_time, build_mem_diff, build_mem_final = build_faiss()
    search_time, recall = search_faiss()
    return TestResult(
        build_time_minutes=build_time,
        mem_diff_gb=build_mem_diff,
        mem_final_gb=build_mem_final,
        search_time_minutes=search_time,
        recall=recall,
    )


if __name__ == '__main__':
    build_faiss()
    search_faiss()
