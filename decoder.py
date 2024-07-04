from utils import *
from causal_attention import CausalAttention
from cross_attention import CrossAttention
from mlp import MLP

class Decoder(nn.Module):
    def __init__ (self, config):
        super().__init__()
        self.embed_dim=config.embed_dim
        self.ln1=nn.LayerNorm(self.embed_dim)
        self.attn=CausalAttention(config)
        self.ln2=nn.LayerNorm(self.embed_dim)
        self.mlp=MLP(config)
        self.ln3=nn.LayerNorm(config.embed_dim)
        self.cross_attn=CrossAttention(config)
    
    def forward(self, x, enc_out):
        x=x+self.attn(self.ln1(x))
        x=x+self.cross_attn(self.ln2(x), enc_out, enc_out)
        x=x+self.mlp(self.ln3(x))
        return x