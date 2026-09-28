"""Read-only context attention probes on a dense causal reference trajectory.

Shadow K/V are derived from merged dense layer inputs, not an independently
evolving compressed model. Scores are diagnostics, never used for selection.
"""

import torch
import torch.nn.functional as F

from .research_cache import context_tokens, merge_rows, nbytes


class ContextProbe:
    def __init__(self, special, count=64, layers=(0, 4, 11, 17, 23)):
        assert special >= 2 and count >= 1 and layers
        self.special, self.count, self.layers = special, count, tuple(layers)
        self.modes = dict(spread64='drop', spatial64='spatial', appearance64='appearance')
        self.banks = {name: {} for name in self.modes}
        self.groups = {}
        self.report = {}

    def begin(self, frame, patches, positions, foreground):
        self.frame = frame
        self.groups = {}
        self.report = dict(frame=frame, conditions={})
        for name, mode in self.modes.items():
            sparse, pos, inverse = context_tokens(patches, positions, foreground, self.count, mode)
            mapping = torch.cat((torch.arange(self.special, device=patches.device),
                                 inverse + self.special))
            matches = (pos[:, :, None] == positions[:, None]).all(-1)
            assert bool((matches.sum(-1) == 1).all())
            representatives = matches[0].long().argmax(-1)
            kept = torch.cat((torch.arange(self.special, device=patches.device),
                              representatives + self.special))
            means, _ = merge_rows(patches, inverse, torch.ones_like(inverse, dtype=patches.dtype),
                                  sparse.shape[1])
            residual = patches[:, ~foreground] - means[:, inverse[~foreground]]
            variance = residual.square().sum() / patches[:, ~foreground].square().sum().clamp_min(1e-12)
            self.groups[name] = dict(mapping=mapping, kept=kept,
                positions=torch.cat((positions.new_zeros(1, self.special, 2), pos), 1))
            self.banks[name][frame] = {}
            self.report['conditions'][name] = dict(processed_patches=sparse.shape[1],
                feature_variance=float(variance), layers=[])

    def observe(self, block, tokens, pos, layer, q, keys, values, past_frames):
        if layer not in self.layers:
            return
        assert tokens.shape[0] == 1 and tokens.shape[1] == self.groups['spatial64']['mapping'].numel()
        attention = block.attn
        probes = q[:, :, :self.special]
        dense = F.scaled_dot_product_attention(probes, keys, values, dropout_p=0.)
        self.report['past_frames'] = past_frames
        for name, mode in self.modes.items():
            group = self.groups[name]
            if mode == 'drop':
                merged = tokens.index_select(1, group['kept'])
            else:
                merged, _ = merge_rows(tokens, group['mapping'], tokens.new_ones(tokens.shape[1]),
                                       group['positions'].shape[1])
            _, k, v = attention.qkv(block.norm1(merged)).reshape(
                1, -1, 3, attention.num_heads, attention.head_dim
            ).permute(2, 0, 3, 1, 4).unbind(0)
            k = attention.rope(attention.k_norm(k), group['positions'])
            # No qkv-backed views retained by the shadow banks.
            current = (k.clone(), v.clone())
            history = [self.banks[name][frame][layer] for frame in past_frames]
            compact_keys = torch.cat([pair[0] for pair in history] + [current[0]], 2)
            compact_values = torch.cat([pair[1] for pair in history] + [current[1]], 2)
            compressed = F.scaled_dot_product_attention(probes, compact_keys, compact_values, dropout_p=0.)
            # Current-only isolates incoming merging from compressed historical reads.
            current_keys = torch.cat((keys[:, :, :-tokens.shape[1]], current[0]), 2)
            current_values = torch.cat((values[:, :, :-tokens.shape[1]], current[1]), 2)
            current_only = F.scaled_dot_product_attention(probes, current_keys, current_values, dropout_p=0.)
            scores = dict(layer=layer, dense_keys=keys.shape[2], compact_keys=compact_keys.shape[2])
            for label, output in (('history_and_current', compressed), ('current_only', current_only)):
                error = (output - dense).square()
                energy = dense.square()
                scores[label] = dict(
                    camera_rse=float(error[:, :, :1].sum() / energy[:, :, :1].sum().clamp_min(1e-12)),
                    register_rse=float(error[:, :, 1:].sum() / energy[:, :, 1:].sum().clamp_min(1e-12)),
                    per_head_rse=(error.sum((0, 2, 3)) / energy.sum((0, 2, 3)).clamp_min(1e-12)).tolist())
            self.report['conditions'][name]['layers'].append(scores)
            self.banks[name][self.frame][layer] = current

    def prune(self, retained):
        for bank in self.banks.values():
            for frame in list(bank):
                if frame not in retained:
                    del bank[frame]

    def memory(self):
        return nbytes(self.banks) + nbytes(self.groups)
