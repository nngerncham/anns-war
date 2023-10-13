import hnswlib
from testbed import load_base, NUM_THREADS, D, K, load_query, evaluate, TestResult
from psutil import Process
from os import getpid
from time import time_ns

index_directory = "/home/nawat/muic/senior/anns-war/indices/hnswlib/index.bin"


def build_hnswlib():
    index = hnswlib.Index("l2", D)
    data = load_base()
    index.init_index(data.shape[0], M=128, ef_construction=256)
    index.set_num_threads(NUM_THREADS)

    proc = Process(getpid())
    start_mem = proc.memory_info().rss
    start_time = time_ns()

    index.add_items(data)

    end_time = time_ns()
    end_mem = proc.memory_info().rss

    # print(f"== HNSW Build ==\n"
    #       f"Time taken: {(end_time - start_time) / 1e9 / 60} minutes\n"
    #       f"Memory difference: {(end_mem - start_mem) / 1e9} GB\n"
    #       f"Memory at end: {end_mem / 1e9} GB")

    index.save_index(index_directory)
    return (end_time - start_time) / 1e9 / 60, (end_mem - start_mem) / 1e9, end_mem / 1e9


def search_hnswlib():
    index = hnswlib.Index("l2", D)
    index.load_index(index_directory)
    index.set_num_threads(NUM_THREADS)

    queries, gts = load_query()
    start_time = time_ns()
    res, _ = index.knn_query(queries, K)
    end_time = time_ns()

    recall = evaluate(res, gts)
    # print(f"== HNSW Search ==\n"
    #       f"Time taken: {(end_time - start_time) / 1e9 / 60} minutes\n"
    #       f"Recall: {recall}")
    return (end_time - start_time) / 1e9 / 60, recall


def test_hnsw():
    build_time, build_mem_diff, build_mem_final = build_hnswlib()
    search_time, recall = search_hnswlib()
    return TestResult(
        build_time_minutes=build_time,
        mem_diff_gb=build_mem_diff,
        mem_final_gb=build_mem_final,
        search_time_minutes=search_time,
        recall=recall,
    )


if __name__ == '__main__':
    build_hnswlib()
    search_hnswlib()
