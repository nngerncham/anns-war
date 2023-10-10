import faiss
from testbed import load_base, NUM_THREADS
from psutil import Process
from os import getpid
from time import time_ns


def build_faiss():
    index = faiss.IndexHNSWFlat(128, 128)
    index.hnsw.efConstruction = 512
    faiss.omp_set_num_threads(NUM_THREADS)
    data = load_base()

    proc = Process(getpid())
    start_mem = proc.memory_info().rss
    start_time = time_ns()

    index.add(data)

    end_time = time_ns()
    end_mem = proc.memory_info().rss

    print(f"== HNSW ==\n"
          f"Time taken: {(end_time - start_time) / 1e9 / 60} minutes\n"
          f"Memory difference: {(end_mem - start_mem) / 1e9} GB\n"
          f"Memory at end: {end_mem / 1e9} GB")

    faiss.write_index(index, "/home/nawat/muic/senior/anns-war/indices/faiss/index.bin")


if __name__ == '__main__':
    build_faiss()
