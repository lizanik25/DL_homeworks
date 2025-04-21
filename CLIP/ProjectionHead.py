import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=256, dropout=0.1):
        super().__init__()

        """
        Here you should write a simple 2-layer MLP consisting:
        2 Linear layers, GELU activation, Dropout, and LayerNorm. 
        Do not forget to send a skip-connection right after projection and before LayerNorm.
        The whole structure should be in the following order:
        [Linear, GELU, Linear, Dropout, Skip, LayerNorm]
        """
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.projection_2 = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        """
        Perform forward pass, do not forget about skip-connections.
        """
        proj = self.projection(x)
        x = self.dropout(self.projection_2(self.gelu(proj)))
        x = self.layer_norm(x + proj) 
        
        return x
