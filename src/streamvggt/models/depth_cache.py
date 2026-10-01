"""Opt-in historical-data allocation; all native blocks and heads still execute."""
from types import MethodType

import torch
import torch.nn.functional as F


def storage_bytes(values):
    """Actual unique tensor storage, including views and positional metadata."""
    seen = {}
    def visit(value):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            seen[(value.device, storage.data_ptr())] = storage.nbytes()
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
    visit(values)
    return sum(seen.values())


class DepthCache:
    def __init__(self, aggregator, mode='native', omitted=(), capacity=32):
        assert not aggregator.training and aggregator.aa_block_size == 1
        assert mode in ('native', 'omit', 'uniform_special')
        assert capacity > 0
        self.model, self.mode = aggregator, mode
        self.omitted = tuple(sorted(omitted))
        assert len(set(self.omitted)) == len(self.omitted)
        assert all(0 <= i < aggregator.depth for i in self.omitted)
        self.capacity = capacity
        self.originals = [block.attn.forward for block in aggregator.global_blocks]
        for layer, block in enumerate(aggregator.global_blocks):
            def forward(attention, x, pos=None, attn_mask=None, past_key_values=None,
                        use_cache=False, layer=layer):
                assert use_cache and attn_mask is None and attention.fused_attn
                assert not attention.training and x.shape[0] == 1
                batch, count, channels = x.shape
                assert pos.shape == (batch, count, 2) and pos.dtype == torch.long
                q, k, v = attention.qkv(x).reshape(
                    batch, count, 3, attention.num_heads, attention.head_dim
                ).permute(2, 0, 3, 1, 4).unbind(0)
                if past_key_values is not None:
                    assert past_key_values[0].ndim == 5 and past_key_values[0].shape[2] == 1
                    k = torch.cat((past_key_values[0].flatten(2, 3), k), 2)
                    v = torch.cat((past_key_values[1].flatten(2, 3), v), 2)
                    key_pos = torch.cat((self.positions[layer], pos), 1)
                else:
                    key_pos = pos
                self.positions[layer] = key_pos
                self.ids[layer] = torch.cat((self.ids[layer],
                    torch.arange(count, device=x.device) + self.frame * count))
                self.calls.append(layer)
                pair = (k.unsqueeze(2), v.unsqueeze(2))
                q, keys = attention.q_norm(q), attention.k_norm(k)
                q = attention.rope(q, pos)
                keys = attention.rope(keys, key_pos)
                result = F.scaled_dot_product_attention(q, keys, v, dropout_p=0.)
                result = result.transpose(1, 2).reshape(batch, count, channels)
                return attention.proj_drop(attention.proj(result)), pair
            block.attn.forward = MethodType(forward, block.attn)
        self.reset()

    def reset(self):
        self.cache = [None] * self.model.depth
        self.positions = [None] * self.model.depth
        device = self.model.camera_token.device
        self.ids = [torch.empty(0, dtype=torch.long, device=device) for _ in self.cache]
        self.frames = []
        self.calls = []

    def forward(self, image, frame):
        self.frame = frame
        self.calls = []
        outputs, special, self.cache = self.model(image, past_key_values=self.cache,
                                                 use_cache=True, past_frame_idx=frame)
        assert self.calls == list(range(self.model.depth))
        self.tokens = outputs[0].shape[2]
        assert special == self.model.patch_start_idx
        return outputs, special

    def retain(self, admit):
        """Filter only after the current query; fixed per-frame equal-row budgets."""
        if admit:
            self.frames = (self.frames + [self.frame])[-self.capacity:]
        total = (self.model.depth - len(self.omitted)) * self.tokens
        base, remainder = divmod(total, self.model.depth)
        for layer, pair in enumerate(self.cache):
            ids = self.ids[layer]
            keep = torch.isin(ids // self.tokens, torch.tensor(self.frames, device=ids.device, dtype=torch.long))
            if self.mode == 'omit' and layer in self.omitted:
                keep.zero_()
            elif self.mode == 'uniform_special':
                count = base + int(layer < remainder)
                special = self.model.patch_start_idx
                assert special <= count <= self.tokens
                patch_count = count - special
                patches = special + torch.arange(patch_count, device=ids.device) * (
                    self.tokens - special) // patch_count
                selected = torch.cat((torch.arange(special, device=ids.device), patches))
                keep &= torch.isin(ids % self.tokens, selected)
            indices = keep.nonzero().flatten()
            self.cache[layer] = tuple(value.index_select(3, indices) for value in pair)
            self.positions[layer] = self.positions[layer].index_select(1, indices)
            self.ids[layer] = ids.index_select(0, indices)
        rows = sum(len(ids) for ids in self.ids)
        expected = len(self.frames) * (self.model.depth * self.tokens if self.mode == 'native' else total)
        assert rows == expected
        pair = self.cache[0][0]
        per_row = 2 * pair.shape[0] * pair.shape[1] * pair.shape[-1] * pair.element_size()
        per_row += self.positions[0].shape[0] * 2 * self.positions[0].element_size() + 8
        actual = storage_bytes((self.cache, self.positions, self.ids))
        assert actual == expected * per_row, (actual, expected * per_row)
        return dict(frame=self.frame, admitted=admit, history_frames=list(self.frames),
                    layer_rows=[len(ids) for ids in self.ids], persistent_bytes=actual,
                    kv_bytes=storage_bytes(self.cache), positions_bytes=storage_bytes(self.positions),
                    ids_bytes=storage_bytes(self.ids), dense_equivalent_bytes=
                    len(self.frames) * self.model.depth * self.tokens * per_row)

    def close(self):
        for block, original in zip(self.model.global_blocks, self.originals):
            block.attn.forward = original
