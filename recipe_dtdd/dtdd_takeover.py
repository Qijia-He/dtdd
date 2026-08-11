"""Pure-tensor DTDD stage helpers: window divergence, first-passage trigger,
and batch splicing. No Ray / no verl imports — unit-testable on CPU.

Math contract matches SOD recipe/dtdd_cot/dtdd_cot_agent_loop.py:199-253:
  d_k = mean_t |logp_S(t) - logp_T(t)| over window k (256 toks),
  trigger = first k with d_k > tau and k >= min_prefix_turns,
  rewind cut = (k-1)*chunk, inherit cut = k*chunk.
"""

import random

import torch


def compute_dk(student_logp, teacher_logp, response_mask, chunk):
    """(B,R) logps + (B,R) mask -> (B,W) window mean |diff| (nan for empty windows)."""
    diff = (student_logp - teacher_logp).abs() * response_mask
    B, R = diff.shape
    W = (R + chunk - 1) // chunk
    pad = W * chunk - R
    if pad:
        diff = torch.nn.functional.pad(diff, (0, pad))
        response_mask = torch.nn.functional.pad(response_mask, (0, pad))
    diff = diff.view(B, W, chunk).sum(-1)
    cnt = response_mask.view(B, W, chunk).sum(-1)
    return diff / cnt.clamp(min=1), cnt


def pick_triggers(dk, cnt, cfg, epoch_seed):
    """Per-row first-passage trigger.

    Returns list of dicts: {row, window(1-based), mode} for triggered rows only.
    cfg keys: tau, min_prefix_turns, epsilon_random, random_turn_max, dry_run.
    """
    out = []
    if cfg.get("dry_run", False):
        return out
    rng = random.Random(epoch_seed)
    B, W = dk.shape
    tau = float(cfg["tau"])
    minp = int(cfg.get("min_prefix_turns", 3))
    eps = float(cfg.get("epsilon_random", 0.0))
    rmax = int(cfg.get("random_turn_max", 16))
    for i in range(B):
        n_win = int((cnt[i] > 0).sum().item())
        if n_win == 0:
            continue
        if rng.random() < eps:
            k = rng.randint(minp, rmax)
            if k <= n_win:
                out.append({"row": i, "window": k, "mode": "random"})
            continue
        for k in range(minp, n_win + 1):
            if float(dk[i, k - 1]) > tau:
                out.append({"row": i, "window": k, "mode": "divergence"})
                break
    return out


def splice_batch(batch, triggers, teacher_ids_list, chunk, pad_token_id, takeover_mode="rewind"):
    """In-place splice of teacher continuations into verl batch tensors.

    batch: TensorDict-like with input_ids (B,P+R), attention_mask (B,P+R),
    position_ids (B,P+R), responses (B,R), response_mask (B,R).
    triggers[j] corresponds to teacher_ids_list[j].
    Returns origin_mask (B,R): 1 = student token, 0 = teacher-written.
    """
    resp = batch["responses"]
    B, R = resp.shape
    P = batch["input_ids"].shape[1] - R
    origin = torch.ones_like(resp, dtype=torch.float32)
    for trg, t_ids in zip(triggers, teacher_ids_list):
        i, k = trg["row"], trg["window"]
        if takeover_mode == "restart":
            # paper-2604.13016 5.1 variant: on trigger the teacher demonstrates the WHOLE
            # trajectory from the prompt (cut=0) instead of continuing the student prefix.
            # Trigger selection is untouched, so vs the suffix arm the only difference is
            # what the teacher writes -- full demo (online targeted SFT) vs suffix.
            cut = 0
        else:
            cut = (k - 1) * chunk if (trg["mode"] == "random" or takeover_mode == "rewind") else k * chunk
        cut = min(cut, R)
        t_ids = list(t_ids)[: R - cut]
        L = cut + len(t_ids)
        new_resp = torch.full((R,), pad_token_id, dtype=resp.dtype, device=resp.device)
        new_resp[:cut] = resp[i, :cut]
        if t_ids:
            new_resp[cut:L] = torch.tensor(t_ids, dtype=resp.dtype, device=resp.device)
        resp[i] = new_resp
        batch["input_ids"][i, P:] = new_resp
        am = batch["attention_mask"][i]
        am[P:] = 0
        am[P : P + L] = 1
        batch["response_mask"][i] = 0
        batch["response_mask"][i, :L] = 1
        origin[i, cut:] = 0.0
    # position_ids: standard verl convention, cumsum over attention mask
    batch["position_ids"] = torch.clip(batch["attention_mask"].cumsum(-1) - 1, min=0)
    return origin


def compute_dk_from_diff(per_pos_diff, response_mask, chunk):
    """Generic variant: (B,R) per-position divergence -> (B,W) window means.
    Used for topk mode where diff = sum_K(|S_topk - T_on_S| * softmax(S_topk))."""
    import torch.nn.functional as F

    d = per_pos_diff * response_mask
    B, R = d.shape
    W = (R + chunk - 1) // chunk
    pad = W * chunk - R
    if pad:
        d = F.pad(d, (0, pad))
        response_mask = F.pad(response_mask, (0, pad))
    d = d.view(B, W, chunk).sum(-1)
    cnt = response_mask.view(B, W, chunk).sum(-1)
    return d / cnt.clamp(min=1), cnt
