use std::collections::HashMap;
use pyo3::prelude::*;

#[pyclass]
pub struct PyETokenizer {
    special_ids: Vec<i32>,
    merges: HashMap<(i32, i32), i32>,
}

#[pymethods]
impl PyETokenizer {
    #[new]
    fn new(special_ids: Vec<i32>, merges: HashMap<(i32, i32), i32>) -> Self {
        PyETokenizer {
            special_ids,
            merges,
        }
    }

    fn encode(&self, py: Python<'_>, initial_ids: Vec<i32>) -> PyResult<Vec<i32>> {
        py.allow_threads(|| {
            let mut ids = initial_ids;
            
            while ids.len() >= 2 {
                // Get valid stats (pairs and their counts)
                let mut stats: HashMap<(i32, i32), usize> = HashMap::new();
                let mut i = 0;
                
                while i < ids.len() - 1 {
                    let pair = (ids[i], ids[i + 1]);
                    if !self.special_ids.contains(&pair.0) && !self.special_ids.contains(&pair.1) {
                        *stats.entry(pair).or_default() += 1;
                    }
                    i += 1;
                }

                // Find valid pair with lowest merge index
                let mut best_pair = None;
                let mut best_merge_idx = i32::MAX;

                for (&pair, _) in stats.iter() {
                    if let Some(&merge_idx) = self.merges.get(&pair) {
                        if merge_idx < best_merge_idx {
                            best_merge_idx = merge_idx;
                            best_pair = Some(pair);
                        }
                    }
                }

                // Break if no valid merges found
                let (pair, merge_idx) = match best_pair {
                    Some(pair) => (pair, self.merges[&pair]),
                    None => break,
                };

                // Perform the merge
                let mut new_ids = Vec::with_capacity(ids.len());
                let mut i = 0;
                while i < ids.len() {
                    if i + 1 < ids.len() && ids[i] == pair.0 && ids[i + 1] == pair.1 {
                        new_ids.push(merge_idx);
                        i += 2;
                    } else {
                        new_ids.push(ids[i]);
                        i += 1;
                    }
                }
                ids = new_ids;
            }
            
            Ok(ids)
        })
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_tokenizer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyETokenizer>()?;
    Ok(())
} 
