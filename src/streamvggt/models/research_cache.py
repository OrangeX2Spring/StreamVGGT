"""Inference-only context merging, temporal allocation and selective refresh.

Explicit ragged raw K/V, with native normalization and RoPE applied at read time.
No trained weights or native model defaults are changed. See the parent repository's
tools/STREAM_CACHE_RESEARCH.md for the experimental contracts.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ResearchConfig:
    method: str = 'native'
    frame_budget: int = 8
    context: str = 'appearance'
    context_tokens: int = 64
    context_probe: bool = False
    context_mass: bool = False
    budget_frames: int = 8
    recent_frames: int = 2
    min_patches: int = 16
    allocation: str = 'coverage'
    refresh: str = 'selective'
    refresh_every: int = 8
    refresh_frames: int = 1
    seed: int = 0

    def __post_init__(self):
        assert self.method in ('native', 'context', 'temporal', 'refresh')
        assert self.context in ('drop', 'spatial', 'appearance', 'dense')
        assert self.allocation in ('coverage', 'fifo', 'uniform')
        assert self.refresh in ('none', 'selective', 'random', 'oracle', 'full')
        assert self.frame_budget >= 2 and self.context_tokens >= 1
        assert self.budget_frames > self.recent_frames >= 1
        assert self.min_patches >= 1 and self.refresh_every >= 1
        assert 1 <= self.refresh_frames < self.frame_budget
        assert not self.context_probe or (self.method == 'context' and self.context == 'dense')
        assert not self.context_mass or (self.method == 'context' and self.context in ('spatial', 'dense'))


def nbytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(nbytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(nbytes(item) for item in value)
    return 0


def groups(features, positions, count, appearance=False):
    """Spatial farthest-point seeds, optionally feature-aware assignment.

    Seed membership is fixed, guaranteeing exactly count nonempty groups.
    Positions are integer patch coordinates; no fictitious fractional RoPE indices.
    """
    assert features.ndim == positions.ndim == 2 and positions.shape == (len(features), 2)
    assert 1 <= count <= len(features)
    if count == len(features):
        identity = torch.arange(count, device=features.device)
        return identity, identity
    xy = positions.float()
    xy = (xy - xy.amin(0)) / (xy.amax(0) - xy.amin(0)).clamp_min(1)
    seeds = torch.empty(count, dtype=torch.long, device=features.device)
    seeds[0] = 0
    distance = (xy - xy[0]).square().sum(-1)
    for index in range(1, count):
        distance[seeds[:index]] = -1
        next_seed = distance.argmax()
        seeds[index] = next_seed
        point = xy.index_select(0, next_seed.reshape(1))
        distance = torch.minimum(distance, (xy - point).square().sum(-1))
    cost = (xy[:, None] - xy[seeds][None]).square().sum(-1)
    if appearance:
        normalized = F.normalize(features.float(), dim=-1)
        cost = cost + .25 * (1 - normalized @ normalized[seeds].T).clamp_min(0)
    assignment = cost.argmin(-1)
    assignment[seeds] = torch.arange(count, device=features.device)
    return seeds, assignment


def merge_rows(value, assignment, weights, count):
    """Weighted token means along the penultimate dimension, preserving dtype."""
    assert value.shape[-2] == len(assignment) == len(weights)
    shape = (*value.shape[:-2], count, value.shape[-1])
    result = torch.zeros(shape, dtype=torch.float32, device=value.device)
    result.index_add_(-2, assignment, value.float() * weights[:, None])
    mass = torch.zeros(count, dtype=torch.float32, device=value.device)
    mass.index_add_(0, assignment, weights.float())
    assert bool((mass > 0).all())
    return (result / mass[:, None]).to(value.dtype), mass


def context_tokens(patches, positions, foreground, count, mode):
    """Return sparse inputs, positions and dense reconstruction correspondence."""
    assert patches.ndim == 3 and patches.shape[0] == 1
    assert foreground.shape == (patches.shape[1],) and foreground.dtype == torch.bool
    if mode == 'dense' or int((~foreground).sum()) <= count:
        inverse = torch.arange(patches.shape[1], device=patches.device)
        return patches, positions, inverse
    target = foreground.nonzero().flatten()
    background = (~foreground).nonzero().flatten()
    seeds, assignment = groups(patches[0, background], positions[0, background],
                               count, mode == 'appearance')
    if mode == 'drop':
        reduced = patches[:, background[seeds]]
    else:
        reduced, _ = merge_rows(patches[:, background], assignment,
                               torch.ones(len(background), device=patches.device), count)
    sparse = torch.cat((patches[:, target], reduced), 1)
    pos = torch.cat((positions[:, target], positions[:, background[seeds]]), 1)
    inverse = torch.empty(patches.shape[1], dtype=torch.long, device=patches.device)
    inverse[target] = torch.arange(len(target), device=patches.device)
    inverse[background] = len(target) + assignment
    return sparse, pos, inverse


class ResearchCache:
    """One causal stream. Historical refresh changes future reads, never past poses."""

    def __init__(self, aggregator, config, observer=None):
        assert not aggregator.training
        assert aggregator.aa_order == ['frame', 'global'] and aggregator.aa_block_size == 1
        assert aggregator.rope is not None
        self.model, self.config = aggregator, config
        self.special = aggregator.patch_start_idx
        self.records = {}
        self.byte_budget = None
        self.generator = torch.Generator().manual_seed(config.seed)
        self.last_frame = -1
        self.observer = observer
        assert observer is None or (config.method == 'context' and config.context == 'dense')

    def _global(self, block, tokens, pos, layer, exclude, mass=None):
        attention = block.attn
        assert attention.fused_attn and not attention.training
        batch, count, channels = tokens.shape
        q, k, v = attention.qkv(block.norm1(tokens)).reshape(
            batch, count, 3, attention.num_heads, attention.head_dim
        ).permute(2, 0, 3, 1, 4).unbind(0)
        # Materialize only K/V; views of qkv would keep the unused Q allocation.
        fresh = (k.clone(memory_format=torch.contiguous_format),
                 v.clone(memory_format=torch.contiguous_format))
        old = [record for frame, record in self.records.items() if frame != exclude]
        keys = torch.cat([record['kv'][layer][0] for record in old] + [k], 2)
        values = torch.cat([record['kv'][layer][1] for record in old] + [v], 2)
        key_pos = torch.cat([record['positions'] for record in old] + [pos], 1)
        q, keys = attention.q_norm(q), attention.k_norm(keys)
        q, keys = attention.rope(q, pos), attention.rope(keys, key_pos)
        if self.observer is not None:
            self.observer.observe(block, tokens, pos, layer, q, keys, values,
                                  list(self.records))
        bias = None
        if self.config.context_mass:
            bias = torch.cat([record['mass'] for record in old] + [mass]).log().to(q.dtype)[None, None, None]
        result = F.scaled_dot_product_attention(q, keys, values, attn_mask=bias, dropout_p=0.)
        result = result.transpose(1, 2).reshape(batch, count, channels)
        tokens = tokens + block.ls1(attention.proj_drop(attention.proj(result)))
        tokens = tokens + block.ls2(block.mlp(block.norm2(tokens)))
        return tokens, fresh

    def _aggregate(self, tokens, positions, exclude=None, outputs=True, mass=None):
        cache, result = [], []
        bias = mass.log().to(tokens.dtype)[None, None, None] if self.config.context_mass else None
        for layer, (frame_block, global_block) in enumerate(
                zip(self.model.frame_blocks, self.model.global_blocks)):
            if self.config.context_mass:
                local = frame_block(tokens, pos=positions, attn_mask=bias)
            else:
                local = frame_block(tokens, pos=positions)
            tokens, pair = self._global(global_block, local, positions, layer, exclude, mass)
            cache.append(pair)
            if outputs:
                result.append(torch.cat((local, tokens), -1)[:, None])
        return result, cache

    def forward(self, image, frame, mask=None):
        assert image.ndim == 5 and image.shape[:3] == (1, 1, 3)
        assert frame == self.last_frame + 1
        self.last_frame = frame
        model = self.model
        normalized = (image - model._resnet_mean) / model._resnet_std
        patches = model.patch_embed(normalized.flatten(0, 1))
        if isinstance(patches, dict):
            patches = patches['x_norm_patchtokens']
        grid = tuple(size // model.patch_size for size in image.shape[-2:])
        positions = model.position_getter(1, *grid, device=image.device) + 1
        descriptor = F.normalize(patches[0].float().mean(0), dim=0)
        original_patches = patches
        inverse = torch.arange(patches.shape[1], device=image.device)
        if self.config.method == 'context':
            if mask is None:
                raise ValueError('Context compression requires an explicit foreground mask')
            assert mask.shape == image.shape[-2:] and mask.dtype == torch.bool
            foreground = F.max_pool2d(mask[None, None].float().to(image.device),
                                      model.patch_size, model.patch_size).flatten().bool()
            if self.observer is not None:
                self.observer.begin(frame, patches, positions, foreground)
            patches, positions, inverse = context_tokens(
                patches, positions, foreground, self.config.context_tokens, self.config.context)
        kind = int(frame != 0)
        special_tokens = torch.cat((model.camera_token[:, kind], model.register_token[:, kind]), 1)
        tokens = torch.cat((special_tokens, patches), 1)
        positions = torch.cat((torch.zeros(1, self.special, 2, device=image.device,
                                          dtype=positions.dtype), positions), 1)
        mass = torch.ones(tokens.shape[1], device=image.device)
        if self.config.method == 'context':
            mass[self.special:] = torch.bincount(inverse, minlength=patches.shape[1]).to(mass.dtype)
        output, cache = self._aggregate(tokens, positions, mass=mass)
        record = dict(kv=cache, positions=positions, descriptor=descriptor,
                      mass=mass)
        if self.config.method == 'refresh':
            # Auxiliary embeddings are real persistent storage, included in bytes.
            record['seed_tokens'] = tokens.detach().clone()
            record['last_context'] = descriptor.clone()
            record['last_refresh'] = frame
        self.records[frame] = record
        if self.byte_budget is None:
            self.byte_budget = self.config.budget_frames * self.memory()['state_bytes']
        if patches.shape[1] != original_patches.shape[1]:
            # Unmerge both frame/global features while retaining the dense encoder
            # residual. Heads run normally, so dense-head cost is not hidden.
            residual = original_patches - patches[:, inverse]
            residual = torch.cat((residual, residual), -1)[:, None]
            output = [torch.cat((row[:, :, :self.special],
                                row[:, :, self.special:][:, :, inverse] + residual), 2)
                      for row in output]
        self.event = dict(frame=frame, dense_patches=original_patches.shape[1],
                          processed_patches=patches.shape[1], refreshed_frames=[],
                          refresh_scores={}, oracle_diagnostic=self.config.refresh == 'oracle'
                          and self.config.method == 'refresh')
        if self.config.method == 'context':
            self.event['dense_to_sparse'] = inverse.tolist()
            self.event['representative_positions'] = positions[0, self.special:].tolist()
            self.event['token_mass'] = mass.tolist()
        return output

    def _compress(self, frame, count):
        record = self.records[frame]
        special = self.special
        positions = record['positions'][0, special:]
        # Carry masses when averaging previously merged tokens.
        seeds, assignment = groups(positions.float(), positions, count)
        weights = record['mass'][special:]
        for layer, pair in enumerate(record['kv']):
            reduced = [merge_rows(value[:, :, special:], assignment, weights, count)[0]
                       for value in pair]
            record['kv'][layer] = tuple(torch.cat((value[:, :, :special], compact), 2)
                                        for value, compact in zip(pair, reduced))
        _, mass = merge_rows(positions.float(), assignment, weights, count)
        record['mass'] = torch.cat((record['mass'][:special], mass))
        record['positions'] = torch.cat((record['positions'][:, :special],
                                         positions[seeds][None]), 1)

    def _novelty(self, frame):
        others = [row['descriptor'] for key, row in self.records.items() if key != frame]
        similarity = torch.stack(others) @ self.records[frame]['descriptor']
        return float((1 - similarity.max()).clamp_min(0))

    def maintain(self, frame):
        """Post-prediction maintenance; outputs for frame have already been emitted."""
        config = self.config
        actions = []
        if config.method == 'temporal':
            recent = set(sorted(self.records)[-config.recent_frames:])
            while self.memory()['state_bytes'] > self.byte_budget:
                candidates = [key for key in self.records if key not in recent]
                assert candidates, 'Byte budget cannot fit protected recent history'
                choices = []
                for key in candidates:
                    row = self.records[key]
                    patches = len(row['mass']) - self.special
                    count = max(config.min_patches, patches // 2)
                    can_merge = count < patches
                    if not can_merge and key == 0:
                        continue  # protect anchor presence, not its spatial resolution
                    score = 0.
                    if config.allocation == 'coverage':
                        token_bytes = (nbytes(row) - nbytes(row['descriptor'])) // len(row['mass'])
                        released = ((patches - count) * token_bytes if can_merge else nbytes(row))
                        score = self._novelty(key) / released
                    choices.append((score, key, count if can_merge else 0))
                assert choices, 'Budget cannot fit recent frames plus anchor summaries'
                if config.allocation == 'coverage':
                    _, victim, count = min(choices)
                elif config.allocation == 'fifo':
                    _, victim, count = min(choices, key=lambda row: row[1])
                else:
                    _, victim, count = max(choices, key=lambda row: (
                        len(self.records[row[1]]['mass']), -row[1]))
                if count:
                    self._compress(victim, count)
                    actions.append(dict(frame=victim, patches=count))
                else:
                    del self.records[victim]
                    actions.append(dict(frame=victim, patches=0))
        else:
            while len(self.records) > config.frame_budget:
                del self.records[next(key for key in self.records if key != 0)]
        self.event['allocation_actions'] = actions
        self.event['retained_frames'] = list(self.records)
        self.event['patches_per_frame'] = {str(key): len(row['mass']) - self.special
                                          for key, row in self.records.items()}
        self.event['state_budget_bytes'] = self.byte_budget if config.method == 'temporal' else None
        if self.observer is not None:
            self.observer.prune(list(self.records))
        return self.event

    def refresh(self, frame):
        config = self.config
        if (config.method != 'refresh' or config.refresh == 'none' or frame == 0
                or frame % config.refresh_every):
            return self.event
        candidates = [key for key in self.records if key != frame]
        current = self.records[frame]['descriptor']
        scores = {}
        if config.refresh == 'oracle':
            # Diagnostic ranking only: assess each independent full-frame refresh
            # against the same frozen cache. No candidate is committed while scoring.
            for key in candidates:
                row = self.records[key]
                _, candidate = self._aggregate(row['seed_tokens'], row['positions'], key, False)
                difference = sum((new.float() - old.float()).square().sum()
                                 for pair, prior in zip(candidate, row['kv'])
                                 for new, old in zip(pair, prior))
                energy = sum(old.float().square().sum() for pair in row['kv'] for old in pair)
                scores[key] = float(difference / energy.clamp_min(1e-12))
                del candidate
        elif config.refresh == 'selective':
            for key in candidates:
                row = self.records[key]
                affinity = ((current @ row['descriptor']) + 1).clamp(0, 2) / 2
                change = (1 - current @ row['last_context']).clamp_min(0)
                age = frame - row['last_refresh']
                scores[key] = float(affinity * change) * age
        if config.refresh == 'full':
            selected = candidates
        elif config.refresh == 'random':
            order = torch.randperm(len(candidates), generator=self.generator).tolist()
            selected = [candidates[index] for index in order[:config.refresh_frames]]
        else:
            selected = sorted(candidates, key=lambda key: (-scores[key], key))[:config.refresh_frames]
        # Stable oldest-first commits. Exclusion avoids duplicated historical keys.
        for key in sorted(selected):
            row = self.records[key]
            _, replacement = self._aggregate(row['seed_tokens'], row['positions'], key, False)
            row['kv'] = replacement
            row['last_context'] = current.clone()
            row['last_refresh'] = frame
        self.event['refreshed_frames'] = sorted(selected)
        self.event['refresh_scores'] = {str(key): score for key, score in scores.items()}
        return self.event

    def memory(self):
        kv = sum(nbytes(row['kv']) for row in self.records.values())
        total = nbytes(self.records)
        return dict(aggregator_bytes=kv, auxiliary_bytes=total - kv, state_bytes=total,
                    aggregator_tokens=sum(len(row['mass']) for row in self.records.values()))
