import copy
import os
import time
from logging import getLogger

import torch
from torch.utils.data import DataLoader

from data import Collate
from utils import count_parameters

logger = getLogger()


class Trainer:
    def __init__(
            self,
            model,
            tokenizer,
            bayesian_flow,
            train_dataset,
            batch_size,
            sequence_length,
            accumulation_steps,
            learning_rate,
            weight_decay,
            ema_rate,
            model_dir,
            log_interval,
            save_interval,
            sample_interval,
            sample_num_examples,
            sample_conditioning,
            sample_iterations,
            resume_checkpoint
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.bayesian_flow = bayesian_flow
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.accumulation_steps = accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ema_rate = ema_rate
        self.model_dir = model_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.sample_interval = sample_interval
        self.sample_num_examples = sample_num_examples
        self.sample_conditioning = sample_conditioning
        self.sample_iterations = sample_iterations
        self.resume_checkpoint = resume_checkpoint
        self.global_step = 0
        self.max_updates = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(self.device)

        self.ema_model = copy.deepcopy(model)

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98)
        )
        self.loss_elem_since_optim = None

        if self.resume_checkpoint:
            self.load()

        self.iters_since_log = 0
        self.prev_log_time = time.time()
        self.loss_iter = 0

    def load(self):
        checkpoint_path = os.path.join(self.model_dir, "checkpoint.pt")

        if os.path.exists(checkpoint_path):
            logger.info(f"Reloading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
        else:
            logger.info(f"Starting new checkpoint: {checkpoint_path}")
            checkpoint = {}

        if 'model_state_dict' in checkpoint:
            logger.info(f"Model state found in checkpoint")
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        num_params = count_parameters(self.model)
        logger.info(f"Total number of parameters: {num_params:,}")

        if 'ema_model_state_dict' in checkpoint:
            logger.info(f"EMA model state found in checkpoint")
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        else:
            logger.info(f"Initialising EMA model with model state")
            self.ema_model.load_state_dict(self.model.state_dict())

        if 'optimizer_state_dict' in checkpoint:
            logger.info(f"Optimizer state found in checkpoint")
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

        self.global_step = checkpoint.get('global_step', 0)
        logger.info(f"Number of completed training steps: {self.global_step:,}")

    def save(self):
        checkpoint_path = os.path.join(self.model_dir, "checkpoint.pt")
        logger.info(f"Saving checkpoint: {checkpoint_path}.")
        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)

    def log(self):
        time_now = time.time()
        duration = time_now - self.prev_log_time

        iter_rate = self.iters_since_log / duration

        logger.info(
            f"Global step: {self.global_step:,}, "
            f"Iteration loss: {self.loss_iter:,.4e}, "
            f"Learning rate: {self.optim.param_groups[0]['lr']:,.4e}, "
            f"Throughput: {iter_rate:,.4f} it/s"
        )

        self.iters_since_log = 0
        self.prev_log_time = time_now

    def update_ema_model(self):
        for weight, ema_weight in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_weight.mul_(self.ema_rate).add_(weight, alpha=1 - self.ema_rate)

    def sample(self):
        logger.info(f"Sampling started...")

        model = self.ema_model if hasattr(self, 'ema_model') else self.model
        tokenizer, bayesian_flow = self.tokenizer, self.bayesian_flow

        conditional_ids = tokenizer.encode(self.sample_conditioning)

        size = (self.sample_num_examples, self.sequence_length)
        ids = torch.zeros(size, dtype=torch.int64, device=self.device)
        conditional_mask = torch.zeros_like(ids, dtype=torch.bool)

        if conditional_ids is not None:
            for i, sublist in enumerate(conditional_ids):
                sublist_len = len(sublist)
                ids[i, :sublist_len] = torch.tensor(sublist, device=self.device)
                conditional_mask[i, :sublist_len] = True

        model.eval()
        probs = bayesian_flow.discrete_data_sample(
            model,
            size=size,
            num_steps=self.sample_iterations,
            device=self.device,
            conditional_mask=conditional_mask,
            conditional_ids=ids
        )
        model.train()

        output_ids = probs.argmax(-1).cpu().tolist()
        decoded = tokenizer.decode(output_ids)
        [logger.info(f"Sample {i}:\t{text}") for i, text in enumerate(decoded)]

    def optimise(self):
        # Determine the number of elements that contributed to the loss
        loss_elem = self.loss_elem_since_optim if self.loss_elem_since_optim is not None else self.accumulation_steps

        # Reset loss_elem_since_optim to 0 only if it was not None
        if self.loss_elem_since_optim is not None:
            self.loss_elem_since_optim = 0

        # Check for invalid number of elements in loss
        if loss_elem == 0:
            logger.error("Tried to optimize, but number of elements in loss was 0")
            return

        # Scale the gradients by number of elements, i.e. trainable indexes since last update
        if loss_elem > 1:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.div_(loss_elem)

        # Update learning rate and perform optimization
        self.optim.param_groups[0]['lr'] = self.learning_rate * min(self.global_step / 10000, 1.0)

        # Clip gradients and update model parameters
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        self.optim.zero_grad()

        # Increment the global step counter
        self.global_step += 1

    def run_training(self):
        model, tokenizer, bayesian_flow = self.model, self.tokenizer, self.bayesian_flow

        collate = Collate(
            crop_length=self.sequence_length,
            eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id,
            length_includes_pad=True
        )

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=False,
            collate_fn=collate
        )
        data_iter = iter(train_dataloader)

        logger.info(f"Training loop running...")

        self.loss_elem_since_optim = 0

        while True:
            for _ in range(self.accumulation_steps):
                try:
                    ids, length_mask, conditional_mask = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_dataloader)
                    ids, length_mask, conditional_mask = next(data_iter)

                ids = ids.to(self.device)
                length_mask = length_mask.to(self.device)
                conditional_mask = conditional_mask.to(self.device)

                loss = bayesian_flow.discrete_data_continuous_loss(
                    model=model,
                    target=ids,
                    reduction='none',
                    length_mask=length_mask,
                    conditional_mask=conditional_mask,
                    conditional_ids=ids
                )

                loss_mask = torch.logical_and(length_mask, ~conditional_mask)
                loss = (loss * loss_mask.float()).sum()

                # For optimisation
                loss.backward()
                self.loss_elem_since_optim += loss_mask.sum().item()

                # For logging
                self.loss_iter = (loss / loss_mask.sum()).item()
                self.iters_since_log += 1

            self.optimise()

            if self.global_step % self.log_interval == 0:
                self.log()

            if self.global_step % self.save_interval == 0:
                self.save()

            if self.global_step % self.sample_interval == 0:
                self.sample()

            if self.max_updates is not None and self.global_step >= self.max_updates:
                break
