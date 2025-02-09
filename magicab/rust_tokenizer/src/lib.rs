use std::collections::HashMap;
use std::collections::HashSet;
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

#[pyfunction]
fn filter_tokens(remove_tokens: Vec<Vec<i64>>, leaf_tokens: Vec<i64>) -> Vec<Vec<i64>> {
    let leaf_set: HashSet<_> = leaf_tokens.into_iter().collect();
    
    remove_tokens
        .into_iter()
        .map(|row| {
            row.into_iter()
                .filter(|&token| leaf_set.contains(&token))
                .collect()
        })
        .collect()
}

#[pyfunction]
fn _add_tokens(
    vocab: HashMap<i64, String>,
    merges: HashMap<(i64, i64), i64>,
    tokens_to_group: Vec<Vec<i64>>,
    group_positions: Option<Vec<Vec<(usize, usize)>>>
) -> PyResult<(
    HashMap<i64, String>,
    HashMap<(i64, i64), i64>,
    Vec<i64>,
    Vec<(i64, i64)>,
    Vec<(usize, usize)>
)> {
    // Calculate estimated new entries for capacity pre-allocation
    let estimated_size: usize = tokens_to_group.iter()
        .map(|group| group.len().saturating_sub(1))
        .sum();
    
    // Pre-allocate all containers with capacity
    let mut vocab = vocab;
    let mut merges = merges;
    vocab.reserve(estimated_size);
    merges.reserve(estimated_size);
    
    let mut eom_tokens = Vec::with_capacity(estimated_size);
    let mut pair_token_groups = Vec::with_capacity(estimated_size);
    let mut pair_token_positions = Vec::with_capacity(
        if group_positions.is_some() { estimated_size } else { 0 }
    );
    
    // Find next available index
    let mut next_idx = vocab.keys().max().unwrap_or(&0) + 1;
    
    // Create reverse lookup for faster token existence checks
    let mut token_to_idx: HashMap<String, i64> = HashMap::with_capacity(vocab.len());
    for (idx, token) in vocab.iter() {
        token_to_idx.insert(token.clone(), *idx);
    }

    for (group_idx, token_group) in tokens_to_group.iter().enumerate() {
        if token_group.len() <= 1 {
            continue;
        }
        
        let length = token_group.len() - 1;
        let mut l = 0;
        let mut r = 1;
        
        let mut prefix_token_idx = token_group[l];
        let prefix_token = vocab.get(&prefix_token_idx)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Token not found in vocab"))?;
        let mut current_token = String::with_capacity(prefix_token.len() * 2);
        current_token.push_str(prefix_token);
        
        while l < r {
            let curr_token_idx = token_group[r];
            let curr_token = vocab.get(&curr_token_idx)
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Token not found in vocab"))?;
            
            // Reuse string buffer
            let saved_len = current_token.len();
            current_token.push_str(curr_token);
            
            if let Some(&existing_idx) = token_to_idx.get(&current_token) {
                prefix_token_idx = existing_idx;
            } else {
                // Insert new token
                vocab.insert(next_idx, current_token.clone());
                token_to_idx.insert(current_token.clone(), next_idx);
                merges.insert((prefix_token_idx, curr_token_idx), next_idx);
                
                eom_tokens.push(curr_token_idx);
                pair_token_groups.push((next_idx, curr_token_idx));
                
                if let Some(ref positions) = group_positions {
                    if let Some(group_pos) = positions.get(group_idx) {
                        if let (Some(&pos_l), Some(&pos_r)) = (group_pos.get(l), group_pos.get(r)) {
                            pair_token_positions.push(pos_r);
                        }
                    }
                }
                
                prefix_token_idx = next_idx;
                next_idx += 1;
            }
            
            // Truncate string buffer back to prefix for next iteration
            current_token.truncate(saved_len);
            l += 1;
            r = (r + 1).min(length);
        }
    }
    
    Ok((vocab, merges, eom_tokens, pair_token_groups, pair_token_positions))
}


fn add_to_groups(
    curr_group: &[i64],
    curr_group_positions: &[i64],
    natural_groups: &mut Vec<Vec<i64>>,
    natural_group_positions: &mut Vec<Vec<i64>>,
) {
    // Only add groups that have at least 2 tokens
    if curr_group.len() >= 2 {
        natural_groups.push(curr_group.to_vec());
        natural_group_positions.push(curr_group_positions.to_vec());
    }
}


#[pyfunction]
pub fn detect_group_token(
    token_loss: Vec<Vec<f64>>,
    token_ids: Vec<Vec<i64>>,
    perplexity_threshold: Vec<f64>,
    char_token_mask: Option<Vec<Vec<bool>>>,
) -> PyResult<(Vec<Vec<Vec<i64>>>, Vec<Vec<Vec<i64>>>, Vec<Vec<bool>>, Vec<Vec<(i64, i64, String, String)>>)> {
    let mut natural_groups: Vec<Vec<Vec<i64>>> = Vec::new();
    let mut natural_group_positions: Vec<Vec<Vec<i64>>> = Vec::new();
    let mut formatted_groups: Vec<Vec<(i64, i64, String, String)>> = Vec::new();
    let char_token_mask = char_token_mask.unwrap_or_else(|| vec![vec![true; token_ids[0].len()]; token_ids.len()]);
    
    // Initialize group_token_mask with all false
    let mut group_token_mask = vec![vec![false; token_ids[0].len()]; token_ids.len()];
    
    for (seq_idx, ((seq_loss, seq_ids), seq_mask)) in token_loss.iter()
        .zip(token_ids.iter())
        .zip(char_token_mask.iter())
        .enumerate() 
    {
        let mut seq_formatted_groups = Vec::new();
        let mut current_group = Vec::new();
        let mut current_token_group = Vec::new();
        let mut prev_loss = seq_loss[0];
        let mut group_counter = 1;
        
        // Initialize sequence-specific groups
        let mut natural_groups_row: Vec<Vec<i64>> = Vec::new();
        let mut natural_group_positions_row: Vec<Vec<i64>> = Vec::new();

        // Get the threshold for this sequence
        let row_perplexity_threshold = perplexity_threshold.get(seq_idx)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Perplexity threshold list length doesn't match input sequences"
            ))?;
        
        // Add first position if it's a valid char token
        if seq_mask[0] {
            current_group.push(0i64);
            current_token_group.push(seq_ids[0]);
        }
        
        for (pos, ((&loss, &token_id), &is_char)) in seq_loss.iter()
            .zip(seq_ids.iter())
            .zip(seq_mask.iter())
            .enumerate()
            .skip(1) 
        {
            if !is_char {
                if current_group.len() >= 2 {
                    // Mark all positions in the current group as true in group_token_mask
                    for &pos in &current_group {
                        group_token_mask[seq_idx][pos as usize] = true;
                    }
                    natural_groups_row.push(current_token_group.clone());
                    natural_group_positions_row.push(current_group.clone());
                    let start = *current_group.first().unwrap() as i64;
                    let end = *current_group.last().unwrap() as i64 + 1;
                    seq_formatted_groups.push((start, end, group_counter.to_string(), "green".to_string()));
                    group_counter += 1;
                }
                current_group.clear();
                current_token_group.clear();
            } else {
                if loss < *row_perplexity_threshold && loss <= prev_loss {
                    current_group.push(pos as i64);
                    current_token_group.push(token_id);
                } else {
                    if current_group.len() >= 2 {
                        // Mark all positions in the current group as true in group_token_mask
                        for &pos in &current_group {
                            group_token_mask[seq_idx][pos as usize] = true;
                        }
                        natural_groups_row.push(current_token_group.clone());
                        natural_group_positions_row.push(current_group.clone());
                        let start = *current_group.first().unwrap() as i64;
                        let end = *current_group.last().unwrap() as i64 + 1;
                        seq_formatted_groups.push((start, end, group_counter.to_string(), "green".to_string()));
                        group_counter += 1;
                    }
                    current_group.clear();
                    current_token_group.clear();
                    current_group.push(pos as i64);
                    current_token_group.push(token_id);
                }
            }
            prev_loss = loss;
        }
        
        // Check final group
        if current_group.len() >= 2 {
            for &pos in &current_group {
                group_token_mask[seq_idx][pos as usize] = true;
            }
            natural_groups_row.push(current_token_group);
            natural_group_positions_row.push(current_group.clone());
            let start = *current_group.first().unwrap() as i64;
            let end = *current_group.last().unwrap() as i64 + 1;
            seq_formatted_groups.push((start, end, group_counter.to_string(), "green".to_string()));
        }
        
        // Add sequence groups to overall results
        natural_groups.push(natural_groups_row);
        natural_group_positions.push(natural_group_positions_row);
        formatted_groups.push(seq_formatted_groups);
    }

    Ok((natural_groups, natural_group_positions, group_token_mask, formatted_groups))
}


/// A Python module implemented in Rust.
#[pymodule]
fn rust_tokenizer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyETokenizer>()?;
    m.add_function(wrap_pyfunction!(filter_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(detect_group_token, m)?)?;
    Ok(())
} 
