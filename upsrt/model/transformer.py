import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from xformers.ops import memory_efficient_attention

# A lot of the transformer code was taken from:
# https://pytorch.org/docs/1.10/_modules/torch/nn/modules/transformer.html

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, x):
        for mod in self.layers:
            x = mod(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, x, memory):
        for mod in self.layers:
            x = mod(x, memory)
        return x

class TransformerEncoderBlock(nn.Module):

    def __init__(
        self, attn_type, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation=F.relu,
        layer_norm_eps=1e-5, device=None, dtype=None
    ):

        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.d_model = d_model
        self.nhead = nhead

        if attn_type == "default":
            self._default_mha = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True,
                **factory_kwargs
            )
            self.self_attn = self._default_mha_wrapper

        elif attn_type == "xformers":
            self.to_q = nn.Linear(d_model, d_model, bias = False)
            self.to_k = nn.Linear(d_model, d_model, bias = False)
            self.to_v = nn.Linear(d_model, d_model, bias = False)
            self.to_out = nn.Linear(d_model, d_model, bias = False)
            self.self_attn = self._xformers_mha_wrapper

        else:
            raise ValueError(f"Unsupported attn_type: {attn_type}")

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderBlock, self).__setstate__(state)

    def forward(self, x):
        x = x + self._sa_block(self.norm1(x))
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self, x):
        x = self.self_attn(q=x, k=x, v=x)
        x = self.dropout1(x)
        return x

    def _ff_block(self, x):
        x = self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))
        return x

    def _default_mha_wrapper(self, q, k, v):

        output = self._default_mha(
            query=q, key=k, value=v,
            attn_mask=None, key_padding_mask=None, need_weights=False
        )[0]
        return output

    def _xformers_mha_wrapper(self, q, k, v):

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        output = memory_efficient_attention(query=q, key=k, value=v)
        output = self._join_heads(output)

        output = self.to_out(output)
        return output

    def _split_heads(self, x):
        s = x.shape
        return torch.reshape(x, (s[0], s[1], self.nhead, -1))

    def _join_heads(self, x):
        s = x.shape
        return torch.reshape(x, (s[0], s[1], s[2]*s[3]))

class TransformerDecoderBlock(nn.Module):

    def __init__(
        self, attn_type, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation=F.relu,
        layer_norm_eps=1e-5, device=None, dtype=None
    ):

        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.d_model = d_model
        self.nhead = nhead

        if attn_type == "default":
            self._default_mha = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True,
                **factory_kwargs
            )
            self.cross_attn = self._default_mha_wrapper

        elif attn_type == "xformers":
            self.to_q = nn.Linear(d_model, d_model, bias = False)
            self.to_k = nn.Linear(d_model, d_model, bias = False)
            self.to_v = nn.Linear(d_model, d_model, bias = False)
            self.to_out = nn.Linear(d_model, d_model, bias = False)
            self.cross_attn = self._xformers_mha_wrapper

        else:
            raise ValueError(f"Unsupported attn_type: {attn_type}")

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderBlock, self).__setstate__(state)

    def forward(self, x, memory):
        x = x + self._ca_block(self.norm1(x), memory)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _ca_block(self, x, memory):
        x = self.cross_attn(q=x, k=memory, v=memory)
        x = self.dropout1(x)
        return x

    def _ff_block(self, x):
        x = self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))
        return x

    def _default_mha_wrapper(self, q, k, v):

        output = self._default_mha(
            query=q, key=k, value=v,
            attn_mask=None, key_padding_mask=None, need_weights=False
        )[0]
        return output

    def _xformers_mha_wrapper(self, q, k, v):

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        output = memory_efficient_attention(query=q, key=k, value=v)
        output = self._join_heads(output)

        output = self.to_out(output)
        return output

    def _split_heads(self, x):
        s = x.shape
        return torch.reshape(x, (s[0], s[1], self.nhead, -1))

    def _join_heads(self, x):
        s = x.shape
        return torch.reshape(x, (s[0], s[1], s[2]*s[3]))


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
