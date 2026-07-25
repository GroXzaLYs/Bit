import os
import torch
import argparse
from bit.config import BITConfig
from bit.model import BitModel
from bit.tokenizer import BITTokenizer
from bit.inference import BITInference

def main():
    parser = argparse.ArgumentParser(description="BIT - Production Chat")
    parser.add_argument("--model", type=str, default="bit_model.pt", help="Path to model checkpoint")
    args = parser.parse_args()

    config = BITConfig(model_path=args.model)
    
    if not os.path.exists(config.model_path):
        print(f"Model {config.model_path} not found.")
        return

    print("Loading BIT...")
    tokenizer = BITTokenizer(config)
    checkpoint = torch.load(config.model_path, map_location=config.device)
    
    # Ensure config vocab size matches checkpoint if available
    if "config" in checkpoint:
        model_config = checkpoint["config"]
        config.vocab_size = model_config.vocab_size
        config.n_embed = model_config.n_embed
        config.n_head = model_config.n_head
        config.n_layer = model_config.n_layer
        config.block_size = model_config.block_size

    model = BitModel(config).to(config.device)
    model.load_state_dict(checkpoint["model_state"])
    inference = BITInference(model, tokenizer, config)

    print("BIT siap. Ketik '/exit' untuk keluar.")
    
    temperature = 0.8
    max_tokens = 100
    top_p = 0.95
    top_k = 50

    while True:
        try:
            user_input = input("Kamu: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "/exit":
            break
            
        if user_input.startswith("/temp"):
            temperature = float(user_input.split()[1])
            print(f"Temperature set to {temperature}")
            continue

        response = inference.generate(
            user_input, 
            max_new_tokens=max_tokens, 
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        print(f"\nBIT: {response}\n")

if __name__ == "__main__":
    main()
