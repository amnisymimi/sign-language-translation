#!/usr/bin/env python3
"""Week 1: manual-feature-to-Korean translation starter (Python >=3.10).

INSTALL: python -m pip install torch
RUN:     python week1_sign_translation.py
CPU:     python week1_sign_translation.py --device cpu
LONGER:  python week1_sign_translation.py --steps 100

No corpus, pretrained weights, internet access, or GPU required after installation.
Random features test the plumbing only; their loss is NOT translation performance.

REAL FEATURES (after your teammates verify landmark ordering/normalization):
  python week1_sign_translation.py --manifest data/train.jsonl --steps 100
Each JSONL line must contain:
  {"sample_id":"s1","signer_id":"p1","split":"train",
   "text":"안녕하세요","feature_path":"features/s1.json"}
Feature paths are relative to the manifest. Each feature JSON contains:
  {"manual":[[0.1,0.2],[0.3,0.4]], "manual_valid":[true,true],
   "timestamps":[0.0,0.033333]}
manual is [T,D], validity and timestamps are [T]. All samples must share D.
Optional face [T,D_face] and face_valid [T] are preserved by this loader but
NOT used by this manual-only baseline. Both streams must use shared timestamps.
Invalid detections are zero-filled and get a learned missing-frame embedding;
they are separate from padding. Reject all-invalid manual sequences.

This is a processed-feature adapter, NOT a raw NIKL JSON parser. Do not guess
landmark order, target-text fields, missing-value conventions or marker meanings.
Use the corpus documentation and inspect actual JSON before writing an adapter.

TEAM CONTRACT:
  P1: verified manual features, IDs, metadata and normalization documentation.
  P2: aligned facial features and fusion output [B,T,H] plus padding mask.
  P3: this loader, tokenizer, encoder/decoder and training smoke check.
  Leader: inventory signers before fixing splits; keep each signer in one split
  for signer-independent evaluation, and commit one split manifest. Check source
  sentence overlap separately. Fit vocabulary on TRAIN text only.

OUTPUT: checkpoint.pt, tokenizer.json, run_config.json and smoke_report.json
inside --output (default week1_output). Dummy tokenizer is a character-level
interface placeholder; replace later if needed. Inference starts with BOS and
does not use reference text. No validation score is computed in this starter.
"""

import argparse
import json
import math
import random
import unicodedata
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class CharacterTokenizer:
    """Deterministic Korean character vocabulary; NFC keeps Hangul consistent."""
    SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(self, tokens):
        if tokens[:4] != self.SPECIAL or len(tokens) != len(set(tokens)):
            raise ValueError("Invalid vocabulary")
        self.tokens = tokens
        self.stoi = {token: i for i, token in enumerate(tokens)}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = range(4)

    @staticmethod
    def normalize(text):
        return unicodedata.normalize("NFC", text.strip())

    @classmethod
    def fit(cls, texts):
        chars = sorted(set("".join(cls.normalize(t) for t in texts)))
        return cls(cls.SPECIAL + chars)

    @property
    def vocab_size(self):
        return len(self.tokens)

    def encode(self, text):
        return [self.bos_id] + [self.stoi.get(c, self.unk_id)
                               for c in self.normalize(text)] + [self.eos_id]

    def decode(self, ids):
        result = []
        for i in ids:
            i = int(i)
            if i == self.eos_id:
                break
            if i not in (self.pad_id, self.bos_id):
                result.append(self.tokens[i])
        return "".join(result)

    def save(self, path):
        path.write_text(json.dumps(self.tokens, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_sample(sample):
    x = torch.as_tensor(sample["manual"], dtype=torch.float32)
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError("manual must be nonempty [T,D]")
    n = len(x)
    valid = torch.as_tensor(sample["manual_valid"], dtype=torch.bool)
    times = torch.as_tensor(sample["timestamps"], dtype=torch.float32)
    if valid.shape != (n,) or times.shape != (n,):
        raise ValueError("manual_valid and timestamps must be [T]")
    if not valid.any() or not torch.isfinite(x[valid]).all():
        raise ValueError("Need valid, finite manual observations")
    if not torch.isfinite(times).all() or (times[1:] <= times[:-1]).any():
        raise ValueError("Timestamps must be finite and strictly increasing")
    x = x.clone()
    x[~valid] = 0.0
    sample = dict(sample, manual=x, manual_valid=valid, timestamps=times)
    if not isinstance(sample["text"], str) or not sample["text"].strip():
        raise ValueError("Need a nonempty target sentence")
    if "face" in sample:
        face = torch.as_tensor(sample["face"], dtype=torch.float32)
        fv = torch.as_tensor(sample["face_valid"], dtype=torch.bool)
        if face.ndim != 2 or face.shape[0] != n or face.shape[1] < 1 or fv.shape != (n,):
            raise ValueError("Face stream must align with manual timestamps")
        if not torch.isfinite(face[fv]).all():
            raise ValueError("Nonfinite valid face features")
        face = face.clone()
        face[~fv] = 0
        sample.update(face=face, face_valid=fv)
    return sample


class SignDataset(Dataset):
    def __init__(self, samples):
        self.samples = [validate_sample(s) for s in samples]
        if not self.samples:
            raise ValueError("Empty dataset")
        ids = [s["sample_id"] for s in self.samples]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate sample IDs")
        self.manual_dim = self.samples[0]["manual"].shape[1]
        if any(s["manual"].shape[1] != self.manual_dim for s in self.samples):
            raise ValueError("Manual feature dimension varies across samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    @classmethod
    def from_manifest(cls, path):
        path = Path(path)
        samples = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError(f"Line {number}: training manifest must contain only train rows")
            features = json.loads((path.parent / row["feature_path"]).read_text(encoding="utf-8"))
            samples.append({**features, **{k: row[k] for k in
                           ("sample_id", "signer_id", "text", "split")}})
        return cls(samples)


def dummy_dataset(seed, manual_dim):
    generator = torch.Generator().manual_seed(seed)
    texts = ["안녕하세요", "병원에 가고 싶어요", "얼마예요", "감사합니다"]
    samples = []
    for i, (text, length) in enumerate(zip(texts, [40, 55, 30, 47])):
        valid = torch.ones(length, dtype=torch.bool)
        valid[5:7] = False  # A missing detection is not batch padding.
        samples.append(dict(sample_id=f"dummy_{i}", signer_id=f"dummy_signer_{i}",
                            split="train", text=text,
                            manual=torch.randn(length, manual_dim, generator=generator),
                            manual_valid=valid, timestamps=torch.arange(length) / 30))
    return SignDataset(samples)


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples):
        lengths = torch.tensor([len(s["manual"]) for s in samples])
        padding = torch.arange(int(lengths.max()))[None, :] >= lengths[:, None]
        return dict(
            manual=pad_sequence([s["manual"] for s in samples], batch_first=True),
            manual_valid=pad_sequence([s["manual_valid"] for s in samples], batch_first=True),
            timestamps=pad_sequence([s["timestamps"] for s in samples], batch_first=True),
            src_padding_mask=padding,
            target_ids=pad_sequence([torch.tensor(self.tokenizer.encode(s["text"]))
                                     for s in samples], batch_first=True,
                                    padding_value=self.tokenizer.pad_id),
            sample_ids=[s["sample_id"] for s in samples],
        )


class PositionalEncoding(nn.Module):
    def __init__(self, hidden, max_length=8192):
        super().__init__()
        if hidden % 2:
            raise ValueError("Use an even hidden dimension")
        position = torch.arange(max_length).float().unsqueeze(1)
        scale = torch.exp(torch.arange(0, hidden, 2).float() * (-math.log(10000) / hidden))
        pe = torch.zeros(max_length, hidden)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * scale), torch.cos(position * scale)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        if x.shape[1] > self.pe.shape[1]:
            raise ValueError("Sequence exceeds positional encoding capacity")
        return x + self.pe[:, :x.shape[1]].to(dtype=x.dtype)


class SignTranslator(nn.Module):
    def __init__(self, manual_dim, vocab_size, pad_id=0, hidden=128,
                 heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.hidden = hidden
        self.manual_projection = nn.Linear(manual_dim, hidden)
        self.missing_frame = nn.Parameter(torch.zeros(hidden))
        self.position = PositionalEncoding(hidden)
        enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout,
                                         activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        dec = nn.TransformerDecoderLayer(hidden, heads, hidden * 4, dropout,
                                         activation="gelu", batch_first=True)
        self.decoder = nn.TransformerDecoder(dec, layers)
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=pad_id)
        self.output = nn.Linear(hidden, vocab_size)

    def encode(self, manual, src_padding_mask, manual_valid):
        x = self.manual_projection(manual)
        x = x + (~manual_valid).unsqueeze(-1).to(x.dtype) * self.missing_frame
        return self.encoder(self.position(x), src_key_padding_mask=src_padding_mask)

    def decode(self, memory, target_input, src_padding_mask):
        length = target_input.shape[1]
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool,
                                      device=target_input.device), diagonal=1)
        target = self.position(self.embedding(target_input) * math.sqrt(self.hidden))
        decoded = self.decoder(target, memory, tgt_mask=causal,
                               tgt_key_padding_mask=target_input.eq(self.pad_id),
                               memory_key_padding_mask=src_padding_mask)
        return self.output(decoded)

    def forward(self, manual, target_input, src_padding_mask, manual_valid):
        return self.decode(self.encode(manual, src_padding_mask, manual_valid),
                           target_input, src_padding_mask)

    @torch.no_grad()
    def generate(self, manual, src_padding_mask, manual_valid, bos_id, eos_id, max_tokens=40):
        was_training = self.training
        self.eval()
        try:
            memory = self.encode(manual, src_padding_mask, manual_valid)
            ids = torch.full((len(manual), 1), bos_id, device=manual.device, dtype=torch.long)
            done = torch.zeros(len(manual), dtype=torch.bool, device=manual.device)
            for _ in range(max_tokens):
                scores = self.decode(memory, ids, src_padding_mask)[:, -1].clone()
                scores[:, [self.pad_id, bos_id]] = -torch.inf
                next_ids = scores.argmax(-1)
                next_ids = torch.where(done, torch.full_like(next_ids, eos_id), next_ids)
                ids = torch.cat([ids, next_ids[:, None]], dim=1)
                done |= next_ids.eq(eos_id)
                if done.all():
                    break
            return ids
        finally:
            self.train(was_training)


def run(args):
    if args.steps < 1 or args.batch_size < 1 or args.manual_dim < 1 or args.lr <= 0:
        raise ValueError("steps, batch size, manual dimension and learning rate must be positive")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    dataset = SignDataset.from_manifest(args.manifest) if args.manifest else dummy_dataset(args.seed, args.manual_dim)
    tokenizer = CharacterTokenizer.fit([s["text"] for s in dataset.samples])
    for s in dataset.samples:
        assert tokenizer.decode(tokenizer.encode(s["text"])) == tokenizer.normalize(s["text"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=Collator(tokenizer), num_workers=0)
    model_config = dict(manual_dim=dataset.manual_dim, vocab_size=tokenizer.vocab_size,
                        pad_id=tokenizer.pad_id, hidden=128, heads=4, layers=2,
                        dropout=0.1)
    model = SignTranslator(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    iterator = iter(loader)
    losses = []
    print(f"Device: {device}; samples: {len(dataset)}; vocabulary: {tokenizer.vocab_size}")
    print("Mode:", "processed real features" if args.manifest else "DUMMY FEATURES (pipeline check only)")
    model.train()
    first_batch = None
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        target_input, labels = batch["target_ids"][:, :-1], batch["target_ids"][:, 1:]
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["manual"], target_input, batch["src_padding_mask"], batch["manual_valid"])
        assert logits.shape == (*labels.shape, tokenizer.vocab_size)
        assert torch.isfinite(logits).all(), "Nonfinite logits"
        loss = criterion(logits.reshape(-1, tokenizer.vocab_size), labels.reshape(-1))
        assert torch.isfinite(loss), "Nonfinite loss"
        before = model.manual_projection.weight.detach().clone()
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"Missing gradient: {name}"
            assert torch.isfinite(parameter.grad).all(), f"Nonfinite gradient: {name}"
        # Both ends of the encoder-decoder must participate in learning.
        for parameter in (model.manual_projection.weight, model.output.weight):
            assert parameter.grad.abs().sum() > 0, "Zero critical gradient"
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        assert not torch.equal(before, model.manual_projection.weight), "Parameters did not update"
        losses.append(float(loss.detach()))
        if first_batch is None:
            first_batch = batch
            print("Manual:", tuple(batch["manual"].shape), "targets:", tuple(batch["target_ids"].shape),
                  "logits:", tuple(logits.shape))
            print("PASS: finite forward/loss/gradients and encoder parameter update")
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == args.steps:
            print(f"Step {step + 1:3d}/{args.steps}: loss={loss.item():.4f}, gradient_norm={grad_norm.item():.4f}")

    # Behavioral checks: causal masking and padding invariance in eval mode.
    model.eval()
    b = first_batch
    ti = b["target_ids"][:, :-1]
    with torch.no_grad():
        original = model(b["manual"], ti, b["src_padding_mask"], b["manual_valid"])
        extended_manual = torch.cat([b["manual"], torch.randn(len(ti), 3, dataset.manual_dim, device=device)], 1)
        extended_pad = torch.cat([b["src_padding_mask"], torch.ones(len(ti), 3, dtype=torch.bool, device=device)], 1)
        extended_valid = torch.cat([b["manual_valid"], torch.zeros(len(ti), 3, dtype=torch.bool, device=device)], 1)
        padded = model(extended_manual, ti, extended_pad, extended_valid)
        padding_error = float((original - padded).abs().max())
        assert torch.allclose(original, padded, atol=2e-4, rtol=2e-4), "Source padding affects predictions"
        changed = ti.clone()
        changed[:, -1] = tokenizer.unk_id
        future = model(b["manual"], changed, b["src_padding_mask"], b["manual_valid"])
        assert torch.allclose(original[:, :-1], future[:, :-1], atol=2e-4, rtol=2e-4), "Future target leakage"
    print("PASS: source padding invariance and causal decoder masking")
    generated = model.generate(b["manual"], b["src_padding_mask"], b["manual_valid"], tokenizer.bos_id, tokenizer.eos_id)
    print("Unscored example prediction:", repr(tokenizer.decode(generated[0].tolist())))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output / "tokenizer.json")
    assert CharacterTokenizer.load(output / "tokenizer.json").tokens == tokenizer.tokens
    torch.save(dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                    model_config=model_config, step=args.steps, tokenizer_tokens=tokenizer.tokens), output / "checkpoint.pt")
    # Reload the checkpoint to verify that inference can be reproduced.
    saved = torch.load(output / "checkpoint.pt", map_location=device, weights_only=True)
    restored = SignTranslator(**saved["model_config"]).to(device)
    restored.load_state_dict(saved["model_state"])
    restored.eval()
    with torch.no_grad():
        reloaded = restored(b["manual"], ti, b["src_padding_mask"], b["manual_valid"])
    assert torch.allclose(original, reloaded, atol=1e-6, rtol=1e-5)
    config = dict(vars(args), model=model_config, torch_version=str(torch.__version__), device_used=str(device))
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = dict(mode="real_processed_features" if args.manifest else "dummy", steps=args.steps,
                  first_loss=losses[0], last_loss=losses[-1], losses=losses,
                  finite_gradients=True, parameters_updated=True, padding_invariance=True,
                  causal_mask=True, checkpoint_reload=True, padding_max_abs_error=padding_error,
                  evaluation_score=None)
    (output / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PASS: tokenizer round trip and checkpoint reload")
    print("Week 1 smoke check complete. Outputs:", output.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", help="Training-only processed-feature JSONL; omit for dummy data")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--manual-dim", type=int, default=12, help="Dummy feature dimension only")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="week1_output")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
