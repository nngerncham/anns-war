use std::{fs::File, io::Read, time::Duration};

const VEC_SIZE: usize = 4;

#[allow(dead_code)]
struct TestResult {
    memory_usage: usize,
    index_construction_time: Duration,
    search_time: Duration,
    recall: f64,
}

#[allow(dead_code, unused_must_use)]
pub fn load_fvecs(file_name: &str, d: usize, n: usize) -> Result<Vec<f32>, std::io::Error> {
    let chunk_size = VEC_SIZE + VEC_SIZE * d;
    let total_bytes = chunk_size * n;
    let mut buffer = vec![0u8; total_bytes];

    let mut file = File::open(file_name)?;
    file.read_exact(&mut buffer);

    let result = buffer
        .chunks(chunk_size)
        .map(|bin_of_vec| {
            let (_, vec_values) = bin_of_vec.split_at(VEC_SIZE);
            vec_values
                .chunks(VEC_SIZE)
                .map(|byte_slice| byte_slice.try_into().expect("Slice with wrong length!"))
                .map(|arr| f32::from_le_bytes(arr))
                .collect::<Vec<f32>>()
        })
        .flatten()
        .collect();
    Ok(result)
}
