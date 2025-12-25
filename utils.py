import torch
import torch.nn as nn
import math
import torch
import tiktoken
from slm_arc import GPT, GPTConfig


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling

class LinearWithLoRA(nn.Module):
    def __init__(self, linear_layer, rank=8, alpha=16):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank,
            alpha=alpha
        )
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)

def apply_lora_to_model(model, rank=8, alpha=16, target_modules=["c_attn", "c_proj"]):
    """
    Apply LoRA to your model. 
    By default targets c_attn and c_proj in CausalSelfAttention blocks.
    """
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                *parent_path, attr_name = name.split('.')
                parent = model
                for p in parent_path:
                    parent = getattr(parent, p)
                setattr(parent, attr_name, LinearWithLoRA(module, rank=rank, alpha=alpha))
    return model



def load_lora_weights(model, lora_weights_path, device='cpu'):
    """Load LoRA weights into the model"""
    lora_state_dict = torch.load(lora_weights_path, map_location=device)  # Add map_location
    
    # Load only LoRA parameters
    model_state_dict = model.state_dict()
    model_state_dict.update(lora_state_dict)
    model.load_state_dict(model_state_dict)
    
    print(f"Loaded LoRA weights from {lora_weights_path}")
    return model

def generate_answer(model, enc, question, max_tokens=20, device='cpu'):
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

def initialize_model():
    """Initialize model once at startup"""
    global model, enc, device
    
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Load tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Load base model
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
    model.load_state_dict(torch.load('models/best_model_params.pt', map_location=device))
    
    # Apply LoRA architecture
    model = apply_lora_to_model(model, rank=8, alpha=16)
    
    # Load LoRA weights
    model = load_lora_weights(model, 'lora_weights.pt', device=device)
    model = model.to(device)
    model.eval()
    
    print("Model initialized successfully!")
