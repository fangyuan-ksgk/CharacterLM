from magicab.etoken import TokenTrie 
from magicab import ETokenizer 
import time
import warnings
import asyncio 
from magicab.etoken import chunk_text, _encode_chunks
from multiprocessing import Pool, cpu_count, freeze_support

# Create tokenizer outside the multiprocessing section
tokenizer = ETokenizer(mode="byte")

def _single_encoding(args): 
    text, chunk_size = args
    # Create tokenizer inside the process to avoid sharing across processes
    process_tokenizer = ETokenizer(mode="byte")
    chunks = chunk_text(text, chunk_size)
    return _encode_chunks(chunks, process_tokenizer, chunk_size)

def multiprocessing_encoding(texts, chunk_size=512, num_processes=None):
    if num_processes is None:
        num_processes = min(cpu_count(), 8)  # Use CPU count but cap at reasonable max
    
    args = [(text, chunk_size) for text in texts]
    with Pool(num_processes) as p:
        return p.map(_single_encoding, args)
    
def wrap_mp_encoding(texts, chunk_size=512, num_processes=None):
    return multiprocessing_encoding(texts, chunk_size, num_processes)

# Move all test code inside this if statement
if __name__ == '__main__':
    # This is required for Windows compatibility
    freeze_support()
    
    # test on encoding speed
    texts = ["I am super duper"*1000, "Hey how is it going?"*1000, "I am not super duper"*1000] * 1000

    start = time.time()
    results = asyncio.run(tokenizer.encode_with_chunking(texts, batch_size=1000, mode='parallel'))
    end = time.time()
    time_parallel = end - start
    print(f"Time for encoding (parallel): {time_parallel} seconds")

    start = time.time()
    results = tokenizer.encode_with_chunking(texts, batch_size=1000, mode='sequential')
    end = time.time()
    time_sequential = end - start
    print(f"Time for encoding (sequential): {time_sequential} seconds")

    # Test multiprocessing performance
    start = time.time()
    results_mp = tokenizer.encode_with_chunking(texts, batch_size=1000, mode='multiprocessing')
    end = time.time()
    time_multiprocessing = end - start
    print(f"Time for encoding (multiprocessing): {time_multiprocessing} seconds")
    print(f"Speed comparison - Parallel is {time_multiprocessing/time_parallel:.2f}x multiprocessing")
    print(f"Speed comparison - Sequential is {time_multiprocessing/time_sequential:.2f}x multiprocessing")
    
    
    print("Multiprocessing encoding results: ", results_mp)