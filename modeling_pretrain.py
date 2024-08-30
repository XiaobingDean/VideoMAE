import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from functools import partial
from modeling_finetune import Block, PatchEmbed, get_sinusoid_encoding_table, RayEncoder
from timm.models.layers import trunc_normal_

class PretrainVisionTransformerEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=6, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, use_checkpoint=False,
                 use_learnable_pos_emb=False, pos_octaves=8, ray_octaves=4):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.ray_encoder = RayEncoder(pos_octaves=pos_octaves, ray_octaves=ray_octaves)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
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
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, camera_pos, rays, mask):
        B, C, H, W = x.shape
        ray_encoded = self.ray_encoder(camera_pos, rays)

        combined_input = torch.cat((x, ray_encoded), dim=1)

        x = self.patch_embed(combined_input)

        B, _, _ = x.size()
        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        x_vis = x[~mask].reshape(B, -1, self.embed_dim)

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, camera_pos, rays, mask):
        return self.forward_features(x, camera_pos, rays, mask)


class PretrainVisionTransformerDecoder(nn.Module):
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=None, num_patches=196, use_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint

        self.mlp_ray = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
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

    def forward(self, x, rays, return_token_num, mask_token=None, pos_embed=None):
        B, N, C = x.shape

        ray_mapped = self.mlp_ray(rays) 
        x = x + ray_mapped  

        if mask_token is not None and pos_embed is not None:
            expand_pos_embed = pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
            pos_emd_mask = expand_pos_embed[:, -return_token_num:]
            x = torch.cat([x, mask_token + pos_emd_mask], dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.head(self.norm(x))
        return x

class PretrainVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0,
                 encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12, decoder_num_classes=1536,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 init_values=0., use_learnable_pos_emb=False, use_checkpoint=False, num_classes=0,
                 pos_octaves=8, ray_octaves=4):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim, depth=encoder_depth, num_heads=encoder_num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values,
            use_checkpoint=use_checkpoint, use_learnable_pos_emb=use_learnable_pos_emb, pos_octaves=pos_octaves,
            ray_octaves=ray_octaves)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, embed_dim=decoder_embed_dim, depth=decoder_depth,
            num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer,
            init_values=init_values, use_checkpoint=use_checkpoint)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)
        trunc_normal_(self.mask_token, std=.02)

    def forward(self, x, camera_pos, rays, mask):
        x_vis = self.encoder(x, camera_pos, rays, mask)  # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]
        B, N, C = x_vis.shape

        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)

        x = self.decoder(x_full, rays, pos_emd_mask.shape[1])
        return x
    
def pretrain_videomae_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    # model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
