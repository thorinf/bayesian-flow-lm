import argparse
import os

import torch
from bayesian_flow_torch import BayesianFlow
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import SentencePieceTokenizer, TextDataset, Collate
from model import SimplexTransformerModel
from utils import append_dims, count_parameters, cosine_decay_with_warmup, update_model_ema

torch.set_float32_matmul_precision('high')


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--epochs', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-decs', '--decay_steps', type=int, default=1e6)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-acc', '--accumulation_steps', type=int, default=1)

    parser.add_argument('-edim', '--embedding_dim', type=int, default=128)
    parser.add_argument('-mdim', '--model_dim', type=int, default=1024)
    parser.add_argument('-numl', '--num_layers', type=int, default=8)
    parser.add_argument('-numh', '--num_heads', type=int, default=8)
    parser.add_argument('-do', '--dropout_prob', type=float, default=0.1)
    parser.add_argument('-ld', '--layerdrop_prob', type=float, default=0.0)

    parser.add_argument('-ckpt', '--checkpoint', type=str, required=True)
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-spm', '--spm_model', type=str, required=True)
    parser.add_argument('-cl', '--crop_length', type=int, default=64)
    parser.add_argument('-ngen', '--num_examples', type=int, default=8)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    tokenizer = SentencePieceTokenizer(args.spm_model)

    model = SimplexTransformerModel(
        num_classes=len(tokenizer),
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_prob=args.dropout_prob,
        layerdrop_prob=args.layerdrop_prob,
    )
    model.to(device)

    if os.path.exists(args.checkpoint):
        print(f"Restoring Checkpoint: {args.checkpoint}.")
        checkpoint = torch.load(args.checkpoint)
    else:
        print(f"Starting new training run: {args.checkpoint}.")
        checkpoint = {}

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    num_params = count_parameters(model)
    print(f"Total number of parameters: {num_params:,}")

    ema_model = SimplexTransformerModel(
        num_classes=len(tokenizer),
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_prob=args.dropout_prob,
        layerdrop_prob=args.layerdrop_prob,
    )
    ema_model.to(device)

    if 'ema_model_state_dict' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    else:
        ema_model.load_state_dict(model.state_dict())

    bayesian_flow = BayesianFlow(num_classes=len(tokenizer), beta=3.0)

    ema_model.eval()
    probs = bayesian_flow.discrete_data_sample(
        ema_model,
        size=(8, args.crop_length),
        num_steps=100,
        device=device
    )
    output_ids = probs.argmax(-1).tolist()
    decoded = tokenizer.decode(output_ids)
    [print(text) for text in decoded]

    dataset = TextDataset(path=args.data_path, tokenizer=tokenizer)
    collate = Collate(
        crop_length=args.crop_length,
        eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id,
        length_includes_pad=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        collate_fn=collate
    )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    if 'optimizer_state_dict' in checkpoint:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

    global_step = checkpoint.get('global_step', 0)
    print(f"Number of completed training steps: {global_step}")
    lr_lambda = lambda step: cosine_decay_with_warmup(step, args.learning_rate, 10000, args.decay_steps)

    for ep in range(0, args.epochs):
        model.train()
        pbar = tqdm(dataloader)
        pbar.set_description(f"epoch: {ep}")

        for idx, (ids, lengths, conditional_mask) in enumerate(pbar):

            ids, lengths, conditional_mask = ids.to(device), lengths.to(device), conditional_mask.to(device)

            length_mask = torch.lt(torch.arange(ids.shape[1], device=device).unsqueeze(0), lengths.unsqueeze(1))

            loss = bayesian_flow.discrete_data_continuous_loss(
                model=model,
                target=ids,
                reduction='none',
                length_mask=length_mask,
                conditional_mask=conditional_mask,
                conditional_ids=ids
            )

            loss_mask = torch.logical_and(length_mask, torch.logical_not(conditional_mask))
            loss = (loss * append_dims(loss_mask, loss.ndim)).sum() / loss_mask.sum()

            (loss / args.accumulation_steps).backward()

            if ((idx + 1) % args.accumulation_steps == 0) or (idx + 1 == len(dataloader)):
                optim.param_groups[0]['lr'] = lr_lambda(global_step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad()
                update_model_ema(model, ema_model, 0.95)
                global_step += 1

            metrics = {
                "loss": loss.item()
            }
            pbar.set_postfix(metrics)

            if ((idx + 1) % 1000 == 0) or (idx + 1 == len(dataloader)):
                checkpoint = {
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'ema_model_state_dict': ema_model.state_dict(),
                    'optimizer_state_dict': optim.state_dict()
                }
                torch.save(checkpoint, args.checkpoint)

                ema_model.eval()
                probs = bayesian_flow.discrete_data_sample(
                    ema_model,
                    size=(8, args.crop_length),
                    num_steps=100,
                    device=device
                )
                output_ids = probs.argmax(-1).cpu().tolist()
                decoded = tokenizer.decode(output_ids)
                [print(text) for text in decoded]
                model.train()


if __name__ == "__main__":
    train()
