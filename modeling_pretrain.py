import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from functools import partial
from modeling_finetune import Block, _cfg, PatchEmbedWithPose, get_sinusoid_encoding_table
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

class PositionalEncoding(nn.Module):
    def __init__(self, num_octaves=8, start_octave=0):
        super().__init__()
        self.num_octaves = num_octaves
        self.start_octave = start_octave

    def forward(self, coords, rays=None):
        batch_size, num_points, dim = coords.shape
        octaves = torch.arange(self.start_octave, self.start_octave + self.num_octaves).float().to(coords)
        multipliers = 2 ** octaves * math.pi
        coords = coords.unsqueeze(-1)
        while len(multipliers.shape) < len(coords.shape):
            multipliers = multipliers.unsqueeze(0)
        scaled_coords = coords * multipliers
        sines = torch.sin(scaled_coords).reshape(batch_size, num_points, dim * self.num_octaves)
        cosines = torch.cos(scaled_coords).reshape(batch_size, num_points, dim * self.num_octaves)
        return torch.cat((sines, cosines), -1)

class RayEncoder(nn.Module):
    def __init__(self, pos_octaves=8, pos_start_octave=0, ray_octaves=4, ray_start_octave=0):
        super().__init__()
        self.pos_encoding = PositionalEncoding(num_octaves=pos_octaves, start_octave=pos_start_octave)
        self.ray_encoding = PositionalEncoding(num_octaves=ray_octaves, start_octave=ray_start_octave)

    def forward(self, pos, rays):
        batchsize, height, width, dims = rays.shape
        pos_enc = self.pos_encoding(pos.unsqueeze(1))
        pos_enc = pos_enc.view(batchsize, pos_enc.shape[-1], 1, 1).repeat(1, 1, height, width)
        rays = rays.flatten(1, 2)
        ray_enc = self.ray_encoding(rays)
        ray_enc = ray_enc.view(batchsize, height, width, ray_enc.shape[-1]).permute((0, 3, 1, 2))
        x = torch.cat((pos_enc, ray_enc), 1)
        return x

class SRTConvBlock(nn.Module):
    def __init__(self, idim, hdim=None, odim=None):
        super().__init__()
        if hdim is None:
            hdim = idim
        if odim is None:
            odim = 2 * hdim
        conv_kwargs = {'bias': False, 'kernel_size': 3, 'padding': 1}
        self.layers = nn.Sequential(
            nn.Conv2d(idim, hdim, stride=1, **conv_kwargs),
            nn.ReLU(),
            nn.Conv2d(hdim, odim, stride=2, **conv_kwargs),
            nn.ReLU())

    def forward(self, x):
        return self.layers(x)

class ImprovedSRTEncoder(nn.Module):
    def __init__(self, num_conv_blocks=3, num_att_blocks=5, pos_start_octave=0):
        super().__init__()
        self.ray_encoder = RayEncoder(pos_octaves=15, pos_start_octave=pos_start_octave, ray_octaves=15)
        conv_blocks = [SRTConvBlock(idim=183, hdim=96)]
        cur_hdim = 192
        for i in range(1, num_conv_blocks):
            conv_blocks.append(SRTConvBlock(idim=cur_hdim, odim=None))
            cur_hdim *= 2
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.per_patch_linear = nn.Conv2d(cur_hdim, 768, kernel_size=1)
        self.transformer = Transformer(768, depth=num_att_blocks, heads=12, dim_head=64, mlp_dim=1536, selfatt=True)

    def forward(self, images, camera_pos, rays):
        batch_size, num_images = images.shape[:2]
        x = images.flatten(0, 1)
        camera_pos = camera_pos.flatten(0, 1)
        rays = rays.flatten(0, 1)
        ray_enc = self.ray_encoder(camera_pos, rays)
        x = torch.cat((x, ray_enc), 1)
        x = self.conv_blocks(x)
        x = self.per_patch_linear(x)
        x = x.flatten(2, 3).permute(0, 2, 1)
        patches_per_image, channels_per_patch = x.shape[1:]
        x = x.reshape(batch_size, num_images * patches_per_image, channels_per_patch)
        x = self.transformer(x)
        return x

class SlotAttention(nn.Module):
    def __init__(self, num_slots, input_dim=768, slot_dim=1536, hidden_dim=3072, iters=3, eps=1e-8, randomize_initial_slots=False):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.scale = slot_dim ** -0.5
        self.slot_dim = slot_dim
        self.randomize_initial_slots = randomize_initial_slots
        self.initial_slots = nn.Parameter(torch.randn(num_slots, slot_dim))
        self.eps = eps
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(input_dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )
        self.norm_input   = nn.LayerNorm(input_dim)
        self.norm_slots   = nn.LayerNorm(slot_dim)
        self.norm_pre_mlp = nn.LayerNorm(slot_dim)

    def forward(self, inputs):
        batch_size, num_inputs, dim = inputs.shape
        inputs = self.norm_input(inputs)
        if self.randomize_initial_slots:
            slot_means = self.initial_slots.unsqueeze(0).expand(batch_size, -1, -1)
            slots = torch.distributions.Normal(slot_means, 1).rsample()
        else:
            slots = self.initial_slots.unsqueeze(0).expand(batch_size, -1, -1)
        k, v = self.to_k(inputs), self.to_v(inputs)
        for _ in range(self.iters):
            slots_prev = slots
            norm_slots = self.norm_slots(slots)
            q = self.to_q(norm_slots)
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bjd,bij->bid', v, attn)
            slots = self.gru(updates.flatten(0, 1), slots_prev.flatten(0, 1))
            slots = slots.reshape(batch_size, self.num_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_pre_mlp(slots))
        return slots

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
