import argparse
from util import prepare_alpaca_data
from magicab import ETokenizer 

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Prepare Alpaca dataset for training")
    parser.add_argument("--model_dir", type=str, default="checkpoint/gpt_tiny/increase_iter6",
                        help="Directory containing the model and tokenizer")
    args = parser.parse_args()

    # Get model directory from arguments
    model_dir = args.model_dir

    # Load tokenizer and prepare data
    tokenizer_path = f"{model_dir}/tokenizer.json"
    tokenizer = ETokenizer.load(tokenizer_path)
    prepare_alpaca_data(tokenizer)

if __name__ == "__main__":
    main()