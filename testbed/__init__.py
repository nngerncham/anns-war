import numpy as np

# BASE_SOURCE = "/home/nawat/muic/senior/anns-war/data/siftsmall/siftsmall"
BASE_SOURCE = "/home/nawat/muic/senior/anns-war/data/sift/sift"
DATA_SOURCE = BASE_SOURCE + "_base.fvecs"
QUERY_SOURCE = BASE_SOURCE + "_query.fvecs"
D = 128

NUM_THREADS = 8


def load_base():
    data = np.fromfile(DATA_SOURCE, dtype=np.float32)
    return data.reshape(-1, D + 1)[:, 1:]


def load_query():
    data = np.fromfile(QUERY_SOURCE, dtype=np.float32)
    return data.reshape(-1, D + 1)[:, 1:]
