from testbed.test_diskann import build_diskann
from testbed.test_faiss import build_faiss
from testbed.test_hnswlib import build_hnswlib

from sys import stderr

if __name__ == '__main__':
    print("=== testing DiskANN ===", file=stderr)
    build_diskann()

    print("=== testing FAISS ===", file=stderr)
    build_faiss()

    print("=== testing HNSW ===", file=stderr)
    build_hnswlib()
