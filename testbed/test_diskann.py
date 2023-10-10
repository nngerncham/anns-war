import diskannpy
import numpy as np
from psutil import Process
from os import getpid, remove
from time import time_ns
from testbed import load_base, NUM_THREADS
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

    print(f"== DiskANN ==\n"
          f"Time taken: {(end_time - start_time) / 1e9 / 60} minutes\n"
          f"Memory difference: {(end_mem - start_mem) / 1e9} GB\n"
          f"Memory at end: {end_mem / 1e9} GB")


if __name__ == '__main__':
    build_diskann()
