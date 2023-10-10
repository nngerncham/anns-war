import hnswlib
from testbed import load_base, NUM_THREADS
from psutil import Process
from os import getpid
from time import time_ns


def build_hnswlib():
    index = hnswlib.Index("l2", 128)
    data = load_base()
    index.init_index(data.shape[0], M=128, ef_construction=500)
    index.set_num_threads(NUM_THREADS)

    proc = Process(getpid())
    start_mem = proc.memory_info().rss
    start_time = time_ns()

    index.add_items(data)

    end_time = time_ns()
    end_mem = proc.memory_info().rss

    print(f"== FAISS-based HNSW ==\n"
          f"Time taken: {(end_time - start_time) / 1e9 / 60} minutes\n"
          f"Memory difference: {(end_mem - start_mem) / 1e9} GB\n"
          f"Memory at end: {end_mem / 1e9} GB")

    index.save_index("/home/nawat/muic/senior/anns-war/indices/hnswlib/index.bin")


if __name__ == '__main__':
    build_hnswlib()
