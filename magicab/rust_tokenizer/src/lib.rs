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

    fn _add_tokens(
        &self,
        py: Python<'_>,
        vocab: HashMap<i32, String>,
        merges: HashMap<(i32, i32), i32>,
        tokens_to_group: Vec<Vec<i32>>,
        group_positions: Option<Vec<Vec<(usize, usize)>>>,
        in_place: bool,
    ) -> PyResult<(HashMap<i32, String>, HashMap<(i32, i32), i32>, Vec<i32>, Vec<(i32, i32)>, Vec<(usize, usize)>)> {
        py.allow_threads(|| {
            // Create deep copies
            let mut vocab = vocab.clone();
            let mut merges = merges.clone();
            
            let mut eom_tokens = Vec::new();
            let mut pair_token_groups = Vec::new();
            let mut pair_token_positions = Vec::new();

            for (group_idx, token_group) in tokens_to_group.iter().enumerate() {
                if token_group.len() == 1 {
                    continue;
                }

                let length = token_group.len() - 1;
                let mut l = 0;
                let mut r = 1;
                let mut prefix_token_idx = token_group[l];
                let mut prefix_token = vocab[&prefix_token_idx].clone();

                while l < r {
                    let curr_token_idx = token_group[r];
                    let curr_token = vocab[&curr_token_idx].clone();
                    let new_token = format!("{}{}", prefix_token, curr_token);

                    let prefix_token_idx = if let Some(&existing_idx) = vocab
                        .iter()
                        .find(|(_, v)| **v == new_token)
                        .map(|(k, _)| k) 
                    {
                        existing_idx
                    } else {
                        let new_idx = *vocab.keys().max().unwrap_or(&0) + 1;
                        vocab.insert(new_idx, new_token.clone());
                        merges.insert((prefix_token_idx, curr_token_idx), new_idx);

                        if in_place {
                            println!(" :: Add new token {}  Id: {}", new_token, new_idx);
                        }

                        eom_tokens.push(curr_token_idx);
                        pair_token_groups.push((prefix_token_idx, curr_token_idx));

                        if let Some(positions) = &group_positions {
                            if let Some(group_pos) = positions.get(group_idx) {
                                pair_token_positions.push((group_pos[l], group_pos[r]));
                            }
                        }

                        new_idx
                    };

                    // Update pointers
                    l += 1;
                    r = (r + 1).min(length);

                    // Update prefix token
                    prefix_token = new_token;
                }
            }

            Ok((vocab, merges, eom_tokens, pair_token_groups, pair_token_positions))
        })
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_tokenizer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyETokenizer>()?;
    Ok(())
} 
