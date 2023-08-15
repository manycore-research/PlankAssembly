# Copyright (c) Manycore Tech Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.x_embed = nn.Embedding(50, num_pos_feats)
        self.y_embed = nn.Embedding(50, num_pos_feats)
        self.z_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.x_embed.weight)
        nn.init.uniform_(self.y_embed.weight)
        nn.init.uniform_(self.z_embed.weight)

    def forward(self, features):
        nx, ny, nz = features.shape[2:]

        x = torch.arange(nx, device=features.device)
        y = torch.arange(ny, device=features.device)
        z = torch.arange(nz, device=features.device)

        x_emb = self.x_embed(x)
        y_emb = self.y_embed(y)
        z_emb = self.z_embed(z)

        x_emb = x_emb.unsqueeze(1).unsqueeze(1).repeat(1, ny, nz, 1)
        y_emb = y_emb.unsqueeze(1).unsqueeze(0).repeat(nx, 1, nz, 1)
        z_emb = z_emb.unsqueeze(0).unsqueeze(0).repeat(nx, ny, 1, 1)

        pos_embeds = (x_emb + y_emb + z_emb).permute(3, 0, 1, 2).unsqueeze(0).repeat(features.shape[0], 1, 1, 1, 1)

        return pos_embeds


if __name__ == "__main__":
    x = torch.rand((4, 12, 5, 5, 5))
    # pos = PositionEmbeddingSine()
    # pos_embeding = pos(x)
    # print(pos_embeding.size())

    pos = PositionEmbeddingLearned()
    pos_embeding = pos(x)
    print(pos_embeding.size())

    