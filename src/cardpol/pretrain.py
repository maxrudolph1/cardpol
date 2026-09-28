"""Stage 3: action-free representation learning from multi-policy video.

Two objectives, both supervised only by *which policy* produced a clip:

  acroaf  CardPol. A discriminator sees the embeddings of two frame stacks of the
          same trajectory, g ~ U{1..K} steps apart, and classifies the policy.
  polcon  Policy contrastive (trajectory-agnostic). Supervised-contrastive
          (SupCon) loss on single frame stacks; same-policy stacks are positives.

A detached MLP probe that regresses the true state from the embedding is
trained alongside (it does not shape the encoder) and reported as state_loss.

    python -m cardpol.pretrain --method acroaf --domain cheetah --setting easy_video

Writes runs/pretrain/<method>/<domain>_<setting>/seed<seed>/{encoder.pt, config.json, *_metrics.csv}.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cardpol.common import DATA_DIR, FRAME_STACK, PRETRAIN_CKPTS, RUNS_DIR, SETTINGS, STATE_DIM, TASK, rollout_path
from cardpol.data import MultiPolicyDataset, augment, make_loader, to_unit_range
from cardpol.models import MLP, Encoder

OUTPUT_NORM = {'acroaf': 'acro', 'polcon': 'none'}
# Epochs of the paper encoders: 4, except 3 for walker clean / static / easy_video.
PAPER_EPOCHS = {('walker', 'clean'): 3, ('walker', 'static'): 3, ('walker', 'easy_video'): 3}


def supcon_loss(z, labels, temperature):
    """Supervised contrastive loss (Khosla et al., 2020) over L2-normalized embeddings z (B, d)."""
    logits = z @ z.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    not_self = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
    log_prob = logits - torch.log((torch.exp(logits) * not_self).sum(1, keepdim=True) + 1e-12)
    positives = (labels[None] == labels[:, None]) & not_self
    num_pos = positives.sum(1)
    mean_log_prob_pos = (positives * log_prob).sum(1) / num_pos.clamp(min=1)
    return -mean_log_prob_pos[num_pos > 0].mean()


class Pretrainer:
    def __init__(self, args, num_policies, state_mean, state_std, device):
        self.args, self.device = args, device
        self.encoder = Encoder(3 * FRAME_STACK, args.feature_dim, OUTPUT_NORM[args.method]).to(device)
        self.probe = MLP(args.feature_dim, STATE_DIM[args.domain], hidden_dim=128, num_layers=2).to(device)
        params = list(self.encoder.parameters()) + list(self.probe.parameters())
        if args.method == 'acroaf':
            self.discriminator = MLP(2 * args.feature_dim, num_policies, hidden_dim=512, num_layers=2).to(device)
            params += list(self.discriminator.parameters())
        self.optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        self.state_mean = torch.as_tensor(state_mean, device=device)
        self.state_std = torch.as_tensor(state_std, device=device)

    def modules(self):
        return [self.encoder, self.probe] + ([self.discriminator] if self.args.method == 'acroaf' else [])

    def losses(self, batch, train):
        fs = FRAME_STACK
        frames = to_unit_range(batch['frames'].to(self.device, non_blocking=True))
        if train and self.args.augment:
            frames = augment(frames)
        labels = batch['label'].to(self.device)
        state = (batch['state'][:, fs - 1].to(self.device) - self.state_mean) / self.state_std
        B = len(frames)
        t0 = frames[:, :fs].flatten(1, 2)  # (B, 3*fs, 84, 84)
        metrics = {}
        if self.args.method == 'acroaf':
            # second frame stack starts g ~ U{1..K} steps after the first
            gap = torch.randint(1, self.args.K + 1, (B, 1), device=self.device)
            rows = torch.arange(B, device=self.device)[:, None]
            tk = frames[rows, gap + torch.arange(fs, device=self.device)].flatten(1, 2)
            z = self.encoder(torch.cat([t0, tk]))
            z0, zk = z[:B], z[B:]
            logits = self.discriminator(torch.cat([z0, zk], dim=-1))
            loss = F.cross_entropy(logits, labels)
            metrics['accuracy'] = (logits.argmax(1) == labels).float().mean().item()
        else:
            z0 = self.encoder(t0)
            loss = supcon_loss(F.normalize(z0, dim=-1), labels, self.args.temperature)
        state_loss = F.mse_loss(self.probe(z0.detach()), state)
        metrics.update(loss=loss.item(), state_loss=state_loss.item())
        return loss + state_loss, metrics

    def train_step(self, batch):
        total, metrics = self.losses(batch, train=True)
        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()
        return metrics

    @torch.no_grad()
    def validate(self, loader, num_batches):
        for m in self.modules():
            m.eval()
        rows = [self.losses(batch, train=False)[1] for _, batch in zip(range(num_batches), loader)]
        for m in self.modules():
            m.train()
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


class CsvLog:
    def __init__(self, path):
        self.file = open(path, 'w', newline='')
        self.writer = None

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--method', choices=list(OUTPUT_NORM), required=True)
    p.add_argument('--domain', choices=list(TASK), required=True)
    p.add_argument('--setting', choices=list(SETTINGS), required=True)
    p.add_argument('--ckpts', type=int, nargs='+', default=None, help='policies to learn from (default: paper set)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=None, help='default: the paper recipe (4, or 3 for some walker settings)')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--feature-dim', type=int, default=256)
    p.add_argument('--K', type=int, default=10, help='acroaf: max gap in steps between the two frame stacks')
    p.add_argument('--temperature', type=float, default=0.1, help='polcon: SupCon temperature')
    p.add_argument('--no-augment', dest='augment', action='store_false', help='disable crop + color jitter')
    p.add_argument('--val-every', type=int, default=100, help='steps between validations')
    p.add_argument('--val-batches', type=int, default=10)
    p.add_argument('--max-steps', type=int, default=None, help='stop early (for smoke tests)')
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--data-dir', type=Path, default=DATA_DIR)
    p.add_argument('--out', type=Path, default=None, help='default: runs/pretrain/<method>/<domain>_<setting>/seed<seed>')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--wandb', action='store_true', help='also log to Weights & Biases (project cardpol)')
    args = p.parse_args()

    args.ckpts = args.ckpts or PRETRAIN_CKPTS[args.domain]
    args.epochs = args.epochs or PAPER_EPOCHS.get((args.domain, args.setting), 4)
    out = args.out or RUNS_DIR / 'pretrain' / args.method / f'{args.domain}_{args.setting}' / f'seed{args.seed}'
    if (out / 'encoder.pt').exists() and not args.overwrite:
        print(f'{out}/encoder.pt exists, skipping (pass --overwrite to retrain)')
        return
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    horizon = args.K + FRAME_STACK if args.method == 'acroaf' else FRAME_STACK
    paths = {split: [rollout_path(args.domain, args.setting, split, c, args.data_dir) for c in args.ckpts]
             for split in ('train', 'val')}
    train_set = MultiPolicyDataset(paths['train'], horizon)
    val_set = MultiPolicyDataset(paths['val'], horizon)
    train_loader = make_loader(train_set, args.batch_size, args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, min(args.num_workers, 2))
    trainer = Pretrainer(args, len(args.ckpts), train_set.state_mean, train_set.state_std, device)

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(output_norm=OUTPUT_NORM[args.method], frame_stack=FRAME_STACK, horizon=horizon)
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    if args.wandb:
        import wandb
        wandb.init(project='cardpol', group=f'pretrain_{args.method}', name=out.parent.name, config=config)
    train_log, val_log = CsvLog(out / 'train_metrics.csv'), CsvLog(out / 'val_metrics.csv')
    print(f'{len(train_set)} train windows, {len(train_loader)} steps/epoch x {args.epochs} epochs')

    step, start = 0, time.time()
    for epoch in range(args.epochs):
        for batch in train_loader:
            metrics = trainer.train_step(batch)
            step += 1
            if step % 10 == 0:
                train_log.write({'step': step, 'epoch': epoch, **metrics})
            if (step - 1) % args.val_every == 0:
                val = trainer.validate(val_loader, args.val_batches)
                val_log.write({'step': step, 'epoch': epoch, **val})
                print(f'epoch {epoch} step {step} ({time.time() - start:.0f}s) train {metrics} val {val}', flush=True)
                if args.wandb:
                    wandb.log({**{f'train/{k}': v for k, v in metrics.items()},
                               **{f'val/{k}': v for k, v in val.items()}}, step=step)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    val = trainer.validate(val_loader, args.val_batches)
    val_log.write({'step': step, 'epoch': args.epochs - 1, **val})
    print(f'final val {val}')
    torch.save(trainer.encoder.state_dict(), out / 'encoder.pt')
    torch.save({name: m.state_dict() for name, m in zip(['encoder', 'probe', 'discriminator'], trainer.modules())},
               out / 'models.pt')
    print(f'saved {out}/encoder.pt')


if __name__ == '__main__':
    main()
