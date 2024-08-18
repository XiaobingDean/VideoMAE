import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from functools import partial
from modeling_finetune import Block, Mlp, RayEncoder, Transformer, SlotAttention, PositionalEncoding
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model

class PatchEmbedWithPose(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, pose_dim=6):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans + pose_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, pose, direction):
        B, C, H, W = x.shape
        pose_direction = torch.cat((pose, direction), dim=1).unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
        pose_direction = pose_direction.expand(B, -1, H, W)  # Match image dimensions
        x = torch.cat((x, pose_direction), dim=1)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class PretrainVisionTransformerEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, use_checkpoint=False,
                 use_learnable_pos_emb=False, pose_dim=6):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedWithPose(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, pose_dim=pose_dim)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)
        else:
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, pose, direction):
        x = self.patch_embed(x, pose, direction)
        B, _, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x, pose, direction):
        x = self.forward_features(x, pose, direction)
        return x

class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer Decoder with support for patch or hybrid CNN input stage """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=None, num_patches=196, use_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 3 * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens to predict pixels
        else:
            x = self.head(self.norm(x))

        return x

class PretrainVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0,
                 encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                 decoder_num_classes=1536, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0., use_learnable_pos_emb=False,
                 use_checkpoint=False, num_classes=0, in_chans=0):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim, depth=encoder_depth, num_heads=encoder_num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values,
            use_checkpoint=use_checkpoint, use_learnable_pos_emb=use_learnable_pos_emb)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, embed_dim=decoder_embed_dim, depth=decoder_depth,
            num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
            init_values=init_values, use_checkpoint=use_checkpoint)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)
        trunc_normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask):
        _, _, T, _, _ = x.shape
        x_vis = self.encoder(x, mask)
        x_vis = self.encoder_to_decoder(x_vis)
        B, N, C = x_vis.shape
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
        x = self.decoder(x_full, pos_emd_mask.shape[1])
        return x

