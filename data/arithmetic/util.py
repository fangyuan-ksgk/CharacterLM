import numpy as np 
import itertools
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm 
from magicab import save_sequences_for_memmap

np.random.seed(42)

def generate_number_weight(): 
    # 0 - 9 weightage 
    unorm_weight = np.random.uniform(0,1,9)
    return unorm_weight / np.sum(unorm_weight)

def generate_digits_weight(max_digit=3): 
    unorm_weight = np.random.uniform(0,1,max_digit)
    return unorm_weight / np.sum(unorm_weight)

def sample_number(n_weight, d_weight): 
    digits = np.random.choice(range(1, len(d_weight)+1), size=1, p=d_weight)
    # digits multiplication 
    num_digits = digits[0]
    val_str = ""
    for i in range(num_digits): 
        if i == num_digits - 1: 
            n = np.random.choice(range(1, len(n_weight)), size=1, p=n_weight[1:]/(np.sum(n_weight[1:])))[0]
        else: 
            n = np.random.choice(range(len(n_weight)), size=1, p=n_weight)[0]
        val_str = str(n) + val_str  
    return int(val_str)

def reverse_distribution(weight):
    epsilon = 1e-10
    reversed_weight = [1.0 / (w + epsilon) for w in weight]    
    total = sum(reversed_weight)
    reversed_weight = [w / total for w in reversed_weight]
    return reversed_weight


def sample_number_batch(n_weight, d_weight, batch_size=32): 
    # Sample the number of digits for each element in the batch
    digits_batch = np.random.choice(range(1,len(d_weight)+1), size=batch_size, p=d_weight)
    
    # Initialize array to store the generated numbers
    result_batch = np.zeros(batch_size, dtype=int)
    
    # Generate each number in the batch
    for batch_idx in tqdm(range(batch_size), desc="Generating numbers"):
        num_digits = digits_batch[batch_idx]
        val_str = ""
        
        # Handle special case of 0 digits
        if num_digits == 0:
            result_batch[batch_idx] = 0
            continue
            
        for i in range(num_digits):
            if i == num_digits - 1:  # Leftmost digit (can't be zero)
                n = np.random.choice(range(1, len(n_weight)), size=1, 
                                    p=n_weight[1:]/(np.sum(n_weight[1:])))[0]
            else:
                n = np.random.choice(range(len(n_weight)), size=1, p=n_weight)[0]
            val_str = str(n) + val_str
            
        result_batch[batch_idx] = int(val_str)
    
    return result_batch


def collect_digit_arithmetic_data(numbers): 
    pairs = list(itertools.combinations(numbers, 2))
    multiplications = [a * b for a, b in pairs]
    additions = [a + b for a, b in pairs]
    data = {"mul_input": [], "add_input": [], "mul_output": [], "add_output": []}
    for pair, mul, add in tqdm(zip(pairs, multiplications, additions), desc="Collecting integer pairs", total=len(pairs)): 
        mul_input = f"{pair[0]} * {pair[1]} = "
        add_input = f"{pair[0]} + {pair[1]} = "
        mul_output = f"{mul}"
        add_output = f"{add}"
        data["mul_input"].append(mul_input)
        data["add_input"].append(add_input)
        data["mul_output"].append(mul_output)
        data["add_output"].append(add_output)
    return data


def plot_digit_distribution(numbers): 

    digit_groups = {}
    for num in numbers:
        num_digits = len(str(num))
        if num_digits not in digit_groups:
            digit_groups[num_digits] = []
        digit_groups[num_digits].append(num)

    plt.figure(figsize=(15, 4 * len(digit_groups)))

    for i, (digit_len, nums) in enumerate(sorted(digit_groups.items())):
        plt.subplot(len(digit_groups), 1, i+1)
        plt.hist(nums, bins=30, alpha=0.7, color='skyblue')
        plt.title(f'Distribution of {digit_len}-digit Numbers (n={len(nums)})')
        plt.xlabel('Number Value')
        plt.ylabel('Frequency')
        
        plt.text(0.98, 0.95, f'Mean: {np.mean(nums):.1f}\nMedian: {np.median(nums):.1f}',
                transform=plt.gca().transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()
    
    
def _preprocess_arithmetoc_data(data: dict, tokenizer, save_dir: str, name: str): 
    
    # batch conversation 
    conv_datas = [
        {"user": data["mul_input"] + data["add_input"]},
        {"assistant": data["mul_output"] + data["add_output"]},
    ]
    conv_texts = tokenizer.prepare_pt_conversation_data(conv_datas)

    # encoding 
    encoded_ids = tokenizer.encode_with_chunking(conv_texts, mode='multiprocessing') # Issue : this combines all conversation into a long list --> not desirable with no clear split (!)


    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}.bin")
    
    print(f"Saving {len(encoded_ids)} {name} examples to {save_path}")
    save_sequences_for_memmap(encoded_ids, save_path)
    
    
def _prepare_arithmetic_data(n_weight, d_weight, batch_size, save_dir, name, tokenizer): 
    numbers = sample_number_batch(n_weight, d_weight, batch_size=batch_size)
    data = collect_digit_arithmetic_data(numbers) 
    _preprocess_arithmetoc_data(data, tokenizer, save_dir, name)
    
    
    
def prepare_arithmetic_pt_data(
    save_dir, 
    tokenizer, 
    train_size, 
    val_size, 
    max_digits
): 
    
    # Set random seed for reproducibility
    np.random.seed(42)
    # skewed distribution of per-digit values and number of digits
    max_digits = 6
    n_weight = generate_number_weight()
    d_weight = generate_digits_weight(max_digits)

    train_name = "train"
    train_size = 10000
    _prepare_arithmetic_data(n_weight, d_weight, train_size, save_dir, train_name, tokenizer)

    reversed_d_weight = reverse_distribution(d_weight)
    reversed_n_weight = reverse_distribution(n_weight)

    val_size = 100
    val_name = "ood-numer"
    _prepare_arithmetic_data(reversed_n_weight, d_weight, val_size, save_dir, val_name, tokenizer)
    val_name = "ood-digit"
    _prepare_arithmetic_data(n_weight, reversed_d_weight, val_size, save_dir, val_name, tokenizer)
    val_name = "ood-both"
    _prepare_arithmetic_data(reversed_n_weight, reversed_d_weight, val_size, save_dir, val_name, tokenizer)
    