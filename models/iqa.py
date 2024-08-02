import torch
import torch.nn as nn
import timm
from models.transformer_decoder import Block_dec
from timm.models.vision_transformer import Block
from models.swin import SwinTransformer
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from functools import partial


class TABlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__() 
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x


class SaveOutput:
    def __init__(self):
        self.outputs = []
    
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    
    def clear(self):
        self.outputs = []






class degradation_Encoder(nn.Module):
    def __init__(self):
        super(degradation_Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            # nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveAvgPool2d((8,12)),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        fea = self.E(x)
        fea = rearrange(fea, 'b c h w -> b h w c')
        out = self.mlp(fea)
        return fea, out


class contrastive_loss(nn.Module):
    def __init__(self, queue_size = 256):
        super().__init__()

        self.device = torch.device('cuda')
        self.queue_size = queue_size
        self.n_tokens = 8
        print(f'queue_size: {self.queue_size}', '\n', '-' * 40, '\n')
        self.register_buffer('style_queue', torch.randn((256*96, self.queue_size)))
        self.ce_loss = nn.CrossEntropyLoss()

    def push_to_tensor_alternative(self, x):

        self.style_queue = torch.cat((self.style_queue[:, 1:], x.permute(1, 0).detach()), dim=1)
 
    def infoNCE(self, style_feats, idx, batch_size, T=0.2):
        a, p = style_feats[[idx],:], style_feats[[batch_size+idx],:]
        # --- infoNCE ---
        logits_pos = torch.einsum('bs, bs -> b', [a, p]).unsqueeze(-1)
        logits_neg = torch.einsum('bs, sq -> bq', [a, self.style_queue])
        logits = torch.cat([logits_pos, logits_neg], dim=1)
        logits /= T
        loss = self.ce_loss(logits, torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device))
        # --- infoNCE ---
        return loss

    def forward(self, style_q):
        b = style_q.size(dim = 0)
        batch_size = b//2
        style_q = rearrange(style_q, 'b h w c -> b (h w c) ')
        # - contrastive loss
        style_q = F.normalize(style_q, dim=-1)
        loss = self.infoNCE(style_q, 0, batch_size)
        self.push_to_tensor_alternative(style_q[[0],:])
        
        return loss



class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 1,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim



class IQA(nn.Module):
    def __init__(self, embed_dim=72, num_outputs=1, patch_size=8, drop=0.1, 
                    depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                    img_size=224, num_tab=2, scale=0.8, freq = 500, multires=10, i_embed=0, use_landmark=False, add_mlp = False, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        self.embed_fn, self.input_ch = get_embedder(multires, i_embed)
        self.use_landmark = use_landmark
        self.add_mlp = add_mlp

        if self.patch_size == 16:
            self.vit = timm.create_model('vit_base_patch16_384', pretrained=True)
        else:
            self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)

        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        self.tablock1 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock1.append(tab)

        self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        self.tablock2 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock2.append(tab)

        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.swintransformer2 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim // 2,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )
        depths = [3, 4, 6, 3]
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]  # stochastic depth decay rule
        self.decoder = nn.ModuleList([Block_dec(
                                    dim=256, num_heads=8, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                    drop=0., attn_drop=0., drop_path=dpr[i], 
                                    norm_layer=partial(nn.LayerNorm, eps=1e-6), sr_ratio=1)
                                    for i in range(depths[0])])


        self.conv_decoder = nn.Conv2d(384, 256, 1, 1, 0)


        self.testmlp = nn.Sequential(
            nn.Linear(512 // 2, 512 // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(512 // 2, 512 // 2),
            nn.ReLU()
        )
        input_dim = 10500
        
        if self.patch_size == 8:
            mlp_dim = 784
        else:
            mlp_dim = 576
        
        self.embed_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(input_dim, mlp_dim),
            nn.Sigmoid()
        )
        if self.use_landmark:
            addtional_channel = 1
        else:
            addtional_channel = 0

        self.fc_score = nn.Sequential(
            nn.Linear((512 // 2 + addtional_channel), (512 // 2 + addtional_channel)),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear((512 // 2 + addtional_channel), num_outputs),
            nn.ReLU()
        )

        self.fc_weight = nn.Sequential(
            nn.Linear((512 // 2 + addtional_channel), (512 // 2 + addtional_channel)),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear((512 // 2 + addtional_channel), num_outputs),
            nn.Sigmoid()
        )



    def extract_feature(self, save_output):
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x

    def forward(self, x, style_q = None, landmark = None):
        _x = self.vit(x)
        x = self.extract_feature(self.save_output)
        self.save_output.outputs.clear()
        # stage 1
        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)

        for tab in self.tablock1:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)
        x = self.swintransformer1(x)

        # stage2
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock2:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv2(x)
        x = self.swintransformer2(x)
        x = self.conv_decoder(x)
        b, c, h, w = x.shape

        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)

        style_q = rearrange(style_q, 'b h w c -> b (h w) c')

        for blk in self.decoder:
            x = blk(x, h, w, style_q)

        if self.add_mlp:
            x = self.testmlp(x)   # test

        if self.use_landmark:
            embedded = self.embed_fn(landmark)
            embedded = self.embed_mlp(embedded.float())
            x = torch.cat((x, embedded.unsqueeze(-1)), -1)        

        score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
        return score
