"""Offline distillation of recorded decisions into the shared policy.

A recorded world is ``[(time_s, {agent: (observation, action)})]`` in step order. Each
agent's decisions form one sequence run from a zero GRU state, as the env runner's
per-agent episode state does, so the offline logits reproduce the online ones and the
loss is plain cross-entropy toward the recorded actions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from ray.rllib.core.columns import Columns

IGNORE = -100


def _map(function, *structures):
    """``function`` over the leaves of equally nested dicts."""
    if isinstance(structures[0], dict):
        return {key: _map(function, *(s[key] for s in structures)) for key in structures[0]}
    return function(*structures)


def agent_sequences(worlds):
    """``{world: [(stacked observations [L, ...], actions [L])]}``, one per agent."""
    result = {}
    for world, steps in worlds.items():
        decisions = {}
        for _time, joint in steps:
            for agent, decision in joint.items():
                decisions.setdefault(agent, []).append(decision)
        result[world] = [
            (
                _map(lambda *steps: np.stack(steps), *(o for o, _ in rows)),
                np.array([a for _, a in rows], dtype=np.int64),
            )
            for rows in decisions.values()
        ]
    return result


def module_batch(sequences, device):
    """Pad sequences into a ``[B, T]`` module batch (last row repeated) and ``[B, T]``
    targets with padding ignored."""
    lengths = [len(actions) for _, actions in sequences]
    steps = max(lengths)

    def pad(*arrays):
        return torch.from_numpy(
            np.stack([np.concatenate([a, np.repeat(a[-1:], steps - len(a), axis=0)]) for a in arrays])
        ).to(device)

    targets = torch.full((len(sequences), steps), IGNORE, dtype=torch.int64)
    for row, (_, actions) in enumerate(sequences):
        targets[row, : len(actions)] = torch.from_numpy(actions)
    batch = {
        Columns.OBS: _map(pad, *(o for o, _ in sequences)),
        Columns.SEQ_LENS: torch.tensor(lengths),
    }
    return batch, targets.to(device)


def sequence_logits(module, batch):
    """Masked ``[B, T, A]`` logits of ``batch`` from a zero recurrent state."""
    size = batch[Columns.SEQ_LENS].shape[0]
    device = batch[Columns.SEQ_LENS].device
    initial = _map(
        lambda h: h.expand(size, *h.shape).to(device), module.get_initial_state()
    )
    output = module.forward_train({**batch, Columns.STATE_IN: initial})
    return output[Columns.ACTION_DIST_INPUTS]


def _batches(sequences, size, generator=None):
    order = (
        range(len(sequences))
        if generator is None
        else torch.randperm(len(sequences), generator=generator).tolist()
    )
    for start in range(0, len(sequences), size):
        yield [sequences[i] for i in order[start : start + size]]


def teacher_actions(module, sequences, *, batch_size, device):
    """The module's greedy actions on every sequence, per sequence."""
    result = []
    with torch.no_grad():
        for chunk in _batches(sequences, batch_size):
            batch, _targets = module_batch(chunk, device)
            greedy = sequence_logits(module, batch).argmax(-1).cpu().numpy()
            result.extend(row[: len(actions)] for row, (_, actions) in zip(greedy, chunk))
    return result


def evaluate(module, sequences, teacher, *, batch_size, device):
    """Mean cross-entropy, accuracy, and accuracy on the rows whose label differs from
    ``teacher`` (the rows distillation must change)."""
    loss = rows = correct = changed = changed_correct = 0
    index = 0
    with torch.no_grad():
        for chunk in _batches(sequences, batch_size):
            batch, targets = module_batch(chunk, device)
            logits = sequence_logits(module, batch)
            loss += F.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), ignore_index=IGNORE, reduction="sum"
            ).item()
            greedy = logits.argmax(-1).cpu().numpy()
            for row, (_, actions) in zip(greedy, chunk):
                hit = row[: len(actions)] == actions
                differs = teacher[index] != actions
                rows += len(actions)
                correct += int(hit.sum())
                changed += int(differs.sum())
                changed_correct += int((hit & differs).sum())
                index += 1
    return {
        "cross_entropy": loss / rows,
        "accuracy": correct / rows,
        "changed_rows": changed,
        "changed_accuracy": changed_correct / changed if changed else None,
    }


def distill(module, train, held_out, *, epochs, lr, batch_size, device, seed):
    """Adam on cross-entropy; returns the state dict of the epoch with the lowest held-out
    cross-entropy (epoch 0: the unchanged module) and the per-epoch history."""
    teacher = {
        name: teacher_actions(module, data, batch_size=batch_size, device=device)
        for name, data in (("train", train), ("held_out", held_out))
    }

    def measure(epoch):
        return {
            "epoch": epoch,
            **{
                name: evaluate(module, data, teacher[name], batch_size=batch_size, device=device)
                for name, data in (("train", train), ("held_out", held_out))
            },
        }

    def snapshot():
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

    history = [measure(0)]
    best, best_state = history[0], snapshot()
    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        for chunk in _batches(train, batch_size, generator):
            batch, targets = module_batch(chunk, device)
            loss = F.cross_entropy(
                sequence_logits(module, batch).flatten(0, 1),
                targets.flatten(),
                ignore_index=IGNORE,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        history.append(measure(epoch))
        if history[-1]["held_out"]["cross_entropy"] < best["held_out"]["cross_entropy"]:
            best, best_state = history[-1], snapshot()
    return best_state, best["epoch"], history
