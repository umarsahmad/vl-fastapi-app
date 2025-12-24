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



