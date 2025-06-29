import torch
import torch.nn as nn
from models.network.net_stage1 import net_stage1
from util import read_yaml

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, reduction_factor: int , attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // reduction_factor),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d_model // reduction_factor, d_model),
            nn.Dropout(0.5)
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class FAFormer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, reduction_factor: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, reduction_factor, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        out = {}
        for idx, layer in enumerate(self.resblocks.children()):
            x = layer(x)
            out['layer'+str(idx)] = x[0] # shape:LND. choose cls token feature
        return out, x

class net_stage2(nn.Module):
    def __init__(self, opt, dim=768, drop_rate=0.5, output_dim=1, train=True):
        super(net_stage2, self).__init__()

        self.backbone = net_stage1()
        if train:
            model_load = torch.load(opt.intermediate_model_path)
            self.backbone.load_state_dict(model_load['model_state_dict'])
            print(f"LOAD {opt.intermediate_model_path}!!!!!!")

        params = []
        for name, p in self.backbone.named_parameters():
            if name == "fc.weight" or name == "fc.bias":
                params.append(p)
            else:
                p.requires_grad = False
        # print(params)

        self.transformer = FAFormer(dim, layers=opt.FAFormer_layers, heads=opt.FAFormer_head, reduction_factor=opt.FAFormer_reduction_factor)
        self.cls_token = nn.Parameter(torch.zeros([dim]))
        self.ln_post = nn.LayerNorm(dim)

        self.fc = nn.Sequential(
            nn.Dropout(drop_rate),
            torch.nn.Linear(dim, output_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize the cls_token
        nn.init.normal_(self.cls_token, std=0.02)
        for m in self.transformer.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.size()
        _, cls_tokens = self.backbone(x)

        cls_tokens = torch.stack(cls_tokens, dim=1)
        cls = self.cls_token.view(1, 1, -1).repeat(B, 1, 1)
        x = torch.cat([cls, cls_tokens], dim=1)

        x = x.permute(1, 0, 2)  # NLD -> LND
        out, x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])
        #output = x
        #x = torch.cat([feature, x], dim=1)
        result = self.fc(x)
        return result


