# Attempt to build spline causal LM  
from __future__ import annotations
import math

import torch
from torch import nn, Tensor, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils._pytree import tree_flatten, tree_unflatten

from einops import rearrange, repeat, pack, unpack

def exists(v):
    return v is not None

# cubic b-spline 
class BSpline(Module): 
    def __init__(
        self,
        learned = False
    ):
        super().__init__()

        matrix = tensor([
            [-1,  3, -3,  1],
            [ 3, -6,  3,  0],
            [-3,  0,  3,  0],
            [ 1,  4,  1,  0]
        ]) / 6

        self.coeff = nn.Parameter(matrix, requires_grad = learned)

    def forward(
        self,
        control_points: Tensor,
        num_times: int,
        lens: Tensor | None = None
    ):
        batch, device = control_points.shape[0], control_points.device
        assert control_points.shape[1] == 4

        # uniform times from 0 - 1

        if exists(lens):
            times = torch.arange(num_times, device = device, dtype = torch.float)
            times = rearrange(times, 't -> 1 t') / rearrange(lens - 1, 'b -> b 1')
            times = times.clamp(max = 1.)
            times = rearrange(times, 'b t -> b t')
        else:
            times = torch.linspace(0, 1, num_times, device = device)
            times = repeat(times, 't -> b t', b = batch)

        # following https://en.wikipedia.org/wiki/B-spline
        # open an issue if you see some obvious error

        powers = torch.arange(4, device = device).flip(dims = (0,))

        times = rearrange(times, '... -> ... 1') ** powers

        return times @ self.coeff @ control_points
    
from model import * 


class ControlPointPredictor(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        # Smaller config for efficiency
        predictor_config = GPTConfig(
            n_layer=config.n_layer // 3,  # Fewer layers (e.g., 4 if main has 12)
            n_head=config.n_head // 2,    # Fewer heads
            n_embd=config.n_embd // 2,    # Smaller dimension
            block_size=config.block_size,
            dropout=config.dropout,
            bias=config.bias
        )
        
        self.to_predictor_dim = nn.Linear(config.n_embd, predictor_config.n_embd)
        
        # Learnable initial control token embeddings
        self.control_tokens = nn.Parameter(torch.randn(4, predictor_config.n_embd))
        
        self.mini_transformer = nn.ModuleList([
            Block(predictor_config) for _ in range(predictor_config.n_layer)
        ])
        
        # Project back to full embedding dim and split into control points
        self.to_model_dim = nn.Linear(predictor_config.n_embd, config.n_embd)
        
    def forward(self, x):
        # x shape: (batch, seq_len, n_embd)
        b = x.shape[0]
        
        x = self.to_predictor_dim(x)
        
        latent_control_emb = repeat(self.control_tokens, 'n d -> b n d', b = b)
        x = torch.cat((latent_control_emb, x), dim = 1)
        
        # Pass through mini transformer
        for block in self.mini_transformer:
            x = block(x)
            
        # Extract control points from control token positions 
        latent_control_points = x[:, :4, :]
        
        # Reshape to separate control points
        control_points = self.to_model_dim(latent_control_points)

        return control_points  # (batch, seq_len, 4, n_embd)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

@dataclass
class SplineGPTConfig:
    block_size: int = 1024 # sequence length ?
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    spline_control_layers: list[int] = None
    
class SplineGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.control_layers = config.spline_control_layers

        self.spline_transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd), # what is 'block size', is this position embedding?
            control_predictor = ControlPointPredictor(config), # control point predictor
            spline = BSpline(), # b-spline constructor
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.spline_transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("Total number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        print("Control predictor number of parameters: %.2fM" % (self.spline_transformer.control_predictor.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.spline_transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, reduction='mean', return_representation: bool = False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.spline_transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.spline_transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        
        # B-Spline embeddings 
        control_points = self.spline_transformer.control_predictor(tok_emb)
        spline_emb = self.spline_transformer.spline(
            control_points, 
            num_times = t # embeddings for each position
        ) 
        
        if self.control_layers is None: 
            x = tok_emb + pos_emb + spline_emb
        else: 
            x = tok_emb + pos_emb
            
        x = self.spline_transformer.drop(x)
        
        # Pass through transformer blocks
        for i, block in enumerate(self.spline_transformer.h):
            if self.control_layers is not None and i in self.control_layers:
                x = x + spline_emb
            x = block(x)
            
        x = self.spline_transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1, reduction=reduction)
            if reduction == 'none': 
                loss = loss.view(b, t)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        if return_representation: 
            return logits, loss, x
        else: 
            return logits, loss
        

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.spline_transformer.wpe.weight = nn.Parameter(self.spline_transformer.wpe.weight[:block_size])
        for block in self.spline_transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, checkpoint_path):
        """
        Load a model from a checkpoint file containing state dict and config
        """
        print(f"Loading weights from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get config from checkpoint
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = cls(gptconf)
        
        # Clean up state dict if needed (e.g., remove DDP wrapper prefixes)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                
        # Load weights
        model.load_state_dict(state_dict)
        return model

    @classmethod
    def load_model(cls, checkpoint, device):
        """Alias for from_pretrained for backwards compatibility"""
        if isinstance(checkpoint, str):
            return cls.from_pretrained(checkpoint)
        else:
            # Handle direct checkpoint dict input
            gptconf = GPTConfig(**checkpoint['model_args'])
            model = cls(gptconf)
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    