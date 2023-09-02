import argparse
import os
from logging import getLogger

from bayesian_flow_torch import BayesianFlow

from data import SentencePieceTokenizer, TextDataset
from model import SimplexTransformerModel
from trainer import Trainer
from utils import get_text
from monitoring import get_initialised_logger

logger = getLogger()


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--epochs', type=int, default=100)
    parser.add_argument('-bsz', '--batch_size', type=int, default=128)
    parser.add_argument('-acc', '--accumulation_steps', type=int, default=1)
    parser.add_argument('-svi', '--log_interval', type=int, default=100)
    parser.add_argument('-lgi', '--save_interval', type=int, default=1000)
    parser.add_argument('-smi', '--sample_interval', type=int, default=10000)

    parser.add_argument('-mdim', '--model_dim', type=int, default=1024)
    parser.add_argument('-nl', '--num_layers', type=int, default=8)
    parser.add_argument('-nh', '--num_heads', type=int, default=8)
    parser.add_argument('-dop', '--dropout_prob', type=float, default=0.1)
    parser.add_argument('-ldp', '--layerdrop_prob', type=float, default=0.0)

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-wus', '--warmup_steps', type=int, default=1e5)
    parser.add_argument('-dcs', '--decay_steps', type=int, default=1e6)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.1)
    parser.add_argument('-ema', '--ema_rate', default="0.95, 0.9999")

    parser.add_argument('-slen', '--sequence_length', type=int, default=64)
    parser.add_argument('-nex', '--num_examples', type=int, default=8)
    parser.add_argument('-b', '--beta', type=float, default=3.0)

    parser.add_argument('-mdir', '--model_dir', type=str, required=True)
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-spm', '--spm_model', type=str, required=True)

    args = parser.parse_args()

    logfile_path = os.path.join(args.model_dir, "logfile.log")
    get_initialised_logger(logfile_path=logfile_path)

    tokenizer = SentencePieceTokenizer(args.spm_model)

    model = SimplexTransformerModel(
        num_classes=len(tokenizer),
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_prob=args.dropout_prob,
        layerdrop_prob=args.layerdrop_prob,
    )

    bayesian_flow = BayesianFlow(num_classes=len(tokenizer), beta=args.beta)

    dataset = TextDataset(path=args.data_path, tokenizer=tokenizer)

    conditional_starts = get_text("conditional_starts.txt")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        bayesian_flow=bayesian_flow,
        train_dataset=dataset,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ema_rate=args.ema_rate,
        model_dir=args.model_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval,
        sample_num_examples=args.num_examples,
        sample_conditioning=conditional_starts,
        sample_iterations=100,
        resume_checkpoint=True
    )

    trainer.sample()

    trainer.run_training()


if __name__ == "__main__":
    train()
