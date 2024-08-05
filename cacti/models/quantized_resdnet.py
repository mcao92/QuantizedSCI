from torch import nn 
import torch 
import einops
from cacti.models.Quant import *
from cacti.models._quant_base import *

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class quantized_TimesAttention3D(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., wa_bits_w = 4, att_bits_w = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.moveq = LearnableBias(self.num_heads)
        self.movek = LearnableBias(self.num_heads)

        self.qkv = LinearQ(dim, (dim//2) * 3, bias=qkv_bias, nbits_w=wa_bits_w) 
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = LinearQ(dim//2, dim, nbits_w=wa_bits_w)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_act = ActQ(in_features=self.num_heads, nbits_a=wa_bits_w)
        self.k_act = ActQ(in_features=self.num_heads, nbits_a=wa_bits_w)
        self.v_act = ActQ(in_features=self.num_heads, nbits_a=wa_bits_w)
        self.attn_act = ActQ(in_features=self.num_heads, nbits_a=att_bits_w)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        C = C//2
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  

        q = self.moveq(q)
        k = self.movek(k)

        q = self.q_act(q)
        k = self.k_act(k)
        v = self.v_act(v)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        attn = self.attn_act(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class quantized_CFormerBlock(nn.Module):
    def __init__(self,dim, wa_bits_w = 8, att_bits_w = 8, ffn_bits_w = 8):
        super().__init__()
        self.scb = nn.Sequential(
            Conv3dQ(dim, dim, (1,3,3), padding=(0,1,1), nbits_w=wa_bits_w),
            nn.LeakyReLU(inplace=True),
            Conv3dQ(dim, dim, (1,3,3), padding=(0,1,1), nbits_w=wa_bits_w),
        )
        self.tsab = quantized_TimesAttention3D(dim,num_heads=4, wa_bits_w = wa_bits_w, att_bits_w = att_bits_w)
        self.ffn = nn.Sequential(
            Conv3dQ(dim,dim,3,1,1, nbits_w=ffn_bits_w),
            nn.LeakyReLU(inplace=True),
            Conv3dQ(dim,dim,1, nbits_w=ffn_bits_w)
        )
    def forward(self,x):
        _,_,_,h,w = x.shape
        scb_out = self.scb(x)
        tsab_in = einops.rearrange(x,"b c d h w->(b h w) d c")
        tsab_out = self.tsab(tsab_in)
        tsab_out = einops.rearrange(tsab_out,"(b h w) d c->b c d h w",h =h,w=w)
        ffn_in = scb_out+tsab_out+x
        ffn_out = self.ffn(ffn_in)+ffn_in
        return ffn_out

class quantized_ResDNetBlock(nn.Module):
    def __init__(self,dim,group_num, wa_bits_w = 8, att_bits_w = 8, ffn_bits_w = 8):
        super().__init__()
        self.cformer_list = nn.ModuleList()
        self.group_num = group_num
        group_dim = dim//group_num
        self.dense_conv = nn.ModuleList()
        for i in range(group_num):
            self.cformer_list.append(quantized_CFormerBlock(group_dim, wa_bits_w = wa_bits_w, att_bits_w = att_bits_w, ffn_bits_w = ffn_bits_w))
            if i > 0:
                self.dense_conv.append(
                    nn.Sequential(
                        Conv3dQ(group_dim*(i+1),group_dim,1, nbits_w=wa_bits_w),
                        nn.LeakyReLU(inplace=True),
                    )
                )
        self.last_conv = Conv3dQ(dim,dim,1, nbits_w=wa_bits_w)

    def forward(self, x):
        input_list = torch.chunk(x,chunks=self.group_num,dim=1)
        cf_in = input_list[0]
        out_list = []
        cf_out = self.cformer_list[0](cf_in)
        out_list.append(cf_out)
        for i in range(1,self.group_num):
            in_list = out_list.copy()
            in_list.append(input_list[i])
            cf_in = torch.cat(in_list,dim=1)
            cf_in = self.dense_conv[i-1](cf_in)
            cf_out = self.cformer_list[i](cf_in)
            out_list.append(cf_out)
        out = torch.cat(out_list,dim=1)
        out = self.last_conv(out)
        out = x + out
        return out
