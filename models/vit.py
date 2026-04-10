import torch
import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 embed_dim=160, depth=8, num_heads=8, num_classes=10):
        super().__init__()

        self.patch_size = patch_size

        # Number of patches
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        # Patch embedding
        self.patch_embedding = nn.Linear(self.patch_dim, embed_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Positional embedding
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim)
        )

        self.dropout = nn.Dropout(0.1)

        # Transformer (FIX: batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def create_patches(self, x):
        B, C, H, W = x.shape
        P = self.patch_size

        patches = x.unfold(2, P, P).unfold(3, P, P)
        patches = patches.contiguous().view(B, C, -1, P, P)
        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.flatten(2)

        return patches

    def forward(self, x):
        B = x.shape[0]

        # CREATE PATCHES (FIX)
        patches = self.create_patches(x)

        x = self.patch_embedding(patches)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
        x = x + self.position_embedding

        x = self.dropout(x)

        x = self.transformer(x)

        # Take CLS token output
        x = x[:, 0]

        return self.mlp_head(x)