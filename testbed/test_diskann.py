import diskannpy
import numpy as np
from psutil import Process
from os import getpid, remove
from time import time_ns
from testbed import load_base, load_query, NUM_THREADS, K, D, evaluate, TestResult
from pathlib import Path

index_directory = "/home/nawat/muic/senior/anns-war/indices/diskann"


def build_diskann():
    index_path = Path(index_directory)
    if index_path.exists():
        for file in index_path.iterdir():
            if file.is_file():
                remove(file)

    data = load_base()

    proc = Process(getpid())
    start_time = time_ns()

    diskannpy.build_memory_index(data,
                                 distance_metric="l2",
                                 index_directory=index_directory,
                                 complexity=125,
                                 graph_degree=70,
                                 alpha=2.0,
                                 num_threads=NUM_THREADS)

    end_time = time_ns()

    start_mem = proc.memory_info().rss
    index = diskannpy.StaticMemoryIndex(index_directory,
                                        distance_metric="l2",
                                        num_threads=NUM_THREADS,
                                        initial_search_complexity=125,
                                        vector_dtype=np.float32)
    end_mem = proc.memory_info().rss

    # print(f"\n== DiskANN ==\n"
    #       f"Time taken: {(end_time - start_time) / 1e9 / 60} minutes\n"
    #       f"Memory difference: {(end_mem - start_mem) / 1e9} GB\n"
    #       f"Memory at end: {end_mem / 1e9} GB\n")
    return (end_time - start_time) / 1e9 / 60, (end_mem - start_mem) / 1e9, end_mem / 1e9


def search_diskann():
    queries, gts = load_query()
    index = diskannpy.StaticMemoryIndex(index_directory,
                                        distance_metric="l2",
                                        num_threads=NUM_THREADS,
                                        initial_search_complexity=125,
                                        vector_dtype=np.float32)

    queries, gts = load_query()
    start_time = time_ns()
    res, _ = index.batch_search(queries, K, 125, NUM_THREADS)
    end_time = time_ns()

    recall = evaluate(res, gts)
    # print(f"\n== DiskANN Search ==\n"
    #       f"Time taken: {(end_time - start_time) / 1e9 / 60} minutes\n"
    #       f"Recall: {recall}\n")
    return (end_time - start_time) / 1e9 / 60, recall


def test_diskann():
    build_time, build_mem_diff, build_mem_final = build_diskann()
    search_time, recall = search_diskann()
    return TestResult(
        build_time_minutes=build_time,
        mem_diff_gb=build_mem_diff,
        mem_final_gb=build_mem_final,
        search_time_minutes=search_time,
        recall=recall,
    )


if __name__ == '__main__':
    build_diskann()
    search_diskann()
