import os, shutil 
from glob import glob 

def move_experiment(run_name, max_iter: int = 8): 
    """ 
    Moving experiment results from checkpoint to experiment directory
    """
    for i in range(9): 
        if i == 0: 
            name = "base"
        else:
            name = f"increase_iter{i}"
        from_dir = f"checkpoint/{run_name}/{name}"
        target_dir = f"experiment/{run_name}/{name}"
        os.makedirs(target_dir, exist_ok=True)
        pkl_files = glob(from_dir+"/info*.pkl")
        tokenizer_files = glob(from_dir+"/*.json")
        if not pkl_files or not tokenizer_files: 
            break 
    
        for file in pkl_files + tokenizer_files: 
            shutil.copyfile(file, target_dir + "/" + file.split("/")[-1])