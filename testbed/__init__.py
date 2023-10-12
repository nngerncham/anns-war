import numpy as np
from dataclasses import dataclass
from dataclasses_serialization.json import JSONSerializer

# BASE_SOURCE = "/home/nawat/muic/senior/anns-war/data/siftsmall/siftsmall"
# BASE_SOURCE = "/home/nawat/muic/senior/anns-war/data/sift/sift"
BASE_SOURCE = "/home/nawat/muic/senior/anns-war/data/gist/gist"
DATA_SOURCE = BASE_SOURCE + "_base.fvecs"
QUERY_SOURCE = BASE_SOURCE + "_query.fvecs"
GT_SOURCE = BASE_SOURCE + "_groundtruth.ivecs"

NUM_THREADS = 8
K = 100
D = 960


def load_base():
    data = np.fromfile(DATA_SOURCE, dtype=np.float32)
    return data.reshape(-1, D + 1)[:, 1:]


def load_query():
    queries = np.fromfile(QUERY_SOURCE, dtype=np.float32)
    ground_truth = np.fromfile(GT_SOURCE, dtype=np.int32)
    shaped_gt = ground_truth.reshape(-1, K + 1)[:, 1:]
    sets_of_gt = [set(gt) for gt in shaped_gt]
    return queries.reshape(-1, D + 1)[:, 1:], sets_of_gt


def evaluate(results, gts):
    sets_of_results = [set(result) for result in results]
    intersection_sizes = np.array([len(a & b) for a, b in zip(sets_of_results, gts)])
    return np.mean(intersection_sizes / np.repeat(K, len(gts)))


@dataclass
class TestResult:
    build_time_minutes: float
    search_time_minutes: float
    mem_diff_gb: float
    mem_final_gb: float
    recall: float


def average(test_results: list[TestResult]):
    build_times = [result.build_time_minutes for result in test_results]
    search_times = [result.search_time_minutes for result in test_results]
    mem_diffs = [result.mem_diff_gb for result in test_results]
    mem_finals = [result.mem_final_gb for result in test_results]
    recalls = [result.recall for result in test_results]

    return JSONSerializer.serialize(TestResult(
        build_time_minutes=np.mean(build_times),
        search_time_minutes=np.mean(search_times),
        mem_diff_gb=np.mean(mem_diffs),
        mem_final_gb=np.mean(mem_finals),
        recall=np.mean(recalls)
    ))
