use std::time::Duration;

#[allow(dead_code)]
struct TestResult {
    memory_usage: usize,
    index_construction_time: Duration,
    search_time: Duration,
    recall: f64,
}
