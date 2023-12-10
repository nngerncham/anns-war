import json

from testbed import average
from testbed.test_diskann import test_diskann
from testbed.test_faiss import test_faiss
from testbed.test_hnswlib import test_hnsw

REPS = 5

if __name__ == '__main__':
    total_results = {}
    #  print("=== testing HNSW ===")
    hnsw_results = [test_hnsw() for _ in range(REPS)]
    hnsw_avg = average(hnsw_results)
    total_results["hnsw"] = hnsw_avg

    #  print("=== testing FAISS ===")
    faiss_results = [test_faiss() for _ in range(REPS)]
    faiss_avg = average(faiss_results)
    total_results["faiss"] = faiss_avg

    #  print("=== testing DiskANN ===")
    diskann_results = [test_diskann() for _ in range(REPS)]
    diskann_avg = average(diskann_results)
    total_results["diskann"] = diskann_avg

    with open("results-sist1m-run2.json", "w") as fp:
        json.dump(total_results, fp)
