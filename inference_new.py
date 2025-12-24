import torch
import tiktoken

def load_lora_weights(model, lora_weights_path):
    """Load LoRA weights into the model"""
    lora_state_dict = torch.load(lora_weights_path)
    
    # Load only LoRA parameters
    model_state_dict = model.state_dict()
    model_state_dict.update(lora_state_dict)
    model.load_state_dict(model_state_dict)
    
    print(f"Loaded LoRA weights from {lora_weights_path}")
    return model

def generate_answer(model, enc, question, max_tokens=100, device='cuda'):
    """Generate answer for a question"""
    
    # Format the prompt (same format as training)
    prompt = f"Question: {question}\nAnswer:"
    
    # Tokenize
    tokens = enc.encode_ordinary(prompt)
    context = torch.tensor(tokens).unsqueeze(0).to(device)
    
    # Generate
    model.eval()
    with torch.no_grad():
        output = model.generate(context, max_tokens)
    
    # Decode
    generated_text = enc.decode(output.squeeze().tolist())
    
    # Extract only the answer part
    answer_start = generated_text.find("Answer:") + len("Answer:")
    answer = generated_text[answer_start:].strip()
    
    return answer

# Example usage
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Load your base model (uncomment and adjust for your model)
    from slm_arc import GPT, GPTConfig
    config = GPTConfig(
        vocab_size=50257,
        block_size=128,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.1,
        bias=True
    )
    model = GPT(config)
    model.load_state_dict(torch.load('models/best_model_params.pt'))
    
    # Apply LoRA architecture (must match training)
    from fine_tune_new import apply_lora_to_model
    model = apply_lora_to_model(model, rank=8, alpha=16)
    
    # Load LoRA weights
    model = load_lora_weights(model, 'lora_weights.pt')
    model = model.to(device)
    
    # Test inference
    question = "courage"
    
    answer = generate_answer(model, enc, question, max_tokens=100, device=device)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()