from utils import *
class CausalAttention(nn.Module):
    def __init__ (self, config):
        super().__init__()
        self.embed_dim=config.embed_dim
        self.num_heads=config.num_heads
        self.head_size=self.embed_dim//self.num_heads
        self.seq_len=config.seq_len
        self.attn=nn.Linear(self.embed_dim, self.head_size*self.num_heads*3, bias=True)
        self.scale=self.head_size**-0.5
        self.register_buffer('mask', torch.tril(torch.ones(1,1,self.seq_len, self.seq_len)))
        self.proj=nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.attn_dropout=nn.Dropout(config.attn_dropout)
        self.resid_dropout=nn.Dropout(config.resid_dropout)
        
    def forward(self,x):
        i,j,k=x.shape
        q,k,v=self.attn(x).chunk(3, dim=-1)
        q=q.view(i, j, self.num_heads, self.head_size).permute(0,2,1,3)
        k=k.view(i, j, self.num_heads, self.head_size).permute(0,2,1,3)
        v=v.view(i, j, self.num_heads, self.head_size).permute(0,2,1,3)
        qk_t=q@k.transpose(-2,-1)*self.scale
        qk_t=qk_t.masked_fill(self.mask[:,:,:j, :j]==0, float('-inf'))
        qk_t=F.softmax(qk_t, dim=-1)
        weights=self.attn_dropout(qk_t)
        attention=weights@v
        attention=attention.permute(0,2,1,3).contiguous().view(i,j,k)
        out=self.proj(attention)
        out=self.resid_dropout(attention)
        
        return out