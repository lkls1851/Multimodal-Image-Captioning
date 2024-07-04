from utils import *

class MLP(nn.Module):
    def __init__ (self, config):
        super().__init__()
        self.embed_dim=config.embed_dim
        self.mlp_ratio=config.mlp_ratio
        self.mlp_dropout=config.mlp_dropout
        self.fc=nn.Linear(self.embed_dim, self.embed_dim*self.mlp_ratio)
        self.proj=nn.Linear(self.embed_dim*self.mlp_ratio, self.embed_dim)
        self.activation=nn.GELU()
        self.dropout=nn.Dropout(self.mlp_dropout)
    def forward(self, x):
        x=self.fc(x)
        x=self.activation(x)
        x=self.proj(x)
        x=self.dropout(x)
        return x