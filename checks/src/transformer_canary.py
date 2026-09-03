#!/usr/bin/env python3

"""Run a small reproducible transformer training workload on one GPU."""

import argparse
import csv
import math
import os
import platform
import random
import socket
import statistics
import sys
import time
from pathlib import Path

# cuBLAS reads this before its first CUDA operation. ReFrame also sets it in
# the job environment, while this default keeps direct invocations consistent.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size, heads):
        super().__init__()
        if hidden_size % heads != 0:
            raise ValueError('hidden-size must be divisible by heads')

        self.heads = heads
        self.head_size = hidden_size // heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, values):
        batch, sequence, hidden = values.shape
        query, key, value = self.qkv(values).chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.view(
                batch, sequence, self.heads, self.head_size
            ).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch, sequence, hidden
        )
        return self.output(attended)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, heads, ffn_size):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.attention = CausalSelfAttention(hidden_size, heads)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size, bias=False),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_size, hidden_size, bias=False),
        )

    def forward(self, values):
        values = values + self.attention(self.attention_norm(values))
        return values + self.ffn(self.ffn_norm(values))


class TinyCausalTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        sequence_length,
        layers,
        hidden_size,
        heads,
        ffn_size,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Parameter(
            torch.empty(sequence_length, hidden_size)
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, heads, ffn_size)
            for _ in range(layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.apply(self._initialize)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, token_ids):
        sequence = token_ids.shape[1]
        values = (
            self.token_embedding(token_ids)
            + self.position_embedding[:sequence].unsqueeze(0)
        )
        for block in self.blocks:
            values = block(values)

        return self.lm_head(self.output_norm(values))


def positive_integer(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('value must be positive')

    return parsed


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--vocab-size', type=positive_integer, default=8192)
    parser.add_argument(
        '--sequence-length', type=positive_integer, default=512
    )
    parser.add_argument('--batch-size', type=positive_integer, default=8)
    parser.add_argument('--dataset-batches', type=positive_integer, default=8)
    parser.add_argument('--layers', type=positive_integer, default=8)
    parser.add_argument('--hidden-size', type=positive_integer, default=768)
    parser.add_argument('--heads', type=positive_integer, default=12)
    parser.add_argument('--ffn-size', type=positive_integer, default=3072)
    parser.add_argument('--warmup-steps', type=positive_integer, default=20)
    parser.add_argument('--measured-steps', type=positive_integer, default=100)
    parser.add_argument('--learning-rate', type=float, default=3.0e-4)
    parser.add_argument('--output', default='canary_samples.csv')
    arguments = parser.parse_args()
    if arguments.hidden_size % arguments.heads != 0:
        parser.error('--hidden-size must be divisible by --heads')
    if arguments.learning_rate <= 0.0:
        parser.error('--learning-rate must be positive')

    return arguments


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def coefficient_of_variation_percent(values):
    if len(values) == 1:
        return 0.0

    return 100.0 * statistics.stdev(values) / statistics.mean(values)


def make_dataset(arguments, device):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(arguments.seed + 1)
    sequences = torch.randint(
        low=0,
        high=arguments.vocab_size,
        size=(
            arguments.dataset_batches,
            arguments.batch_size,
            arguments.sequence_length + 1,
        ),
        generator=generator,
        dtype=torch.int64,
        device='cpu',
    )
    checksum = int(sequences.sum(dtype=torch.int64).item())
    sequences = sequences.to(device=device, non_blocking=False)
    return sequences[..., :-1], sequences[..., 1:], checksum


def evaluate_loss(model, inputs, targets):
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type='cuda', dtype=torch.bfloat16
    ):
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
    torch.cuda.synchronize()
    model.train()
    return loss.item()


def gradient_norm(model):
    squared_norm = torch.zeros((), device='cuda', dtype=torch.float32)
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared_norm.add_(
                parameter.grad.detach().float().square().sum()
            )

    return squared_norm.sqrt().item()


def train_step(model, optimizer, inputs, targets):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
    loss.backward()
    optimizer.step()
    return loss.detach()


def write_samples(
    path,
    metadata,
    elapsed_milliseconds,
    tokens_per_second,
    losses,
):
    output_path = Path(path)
    with output_path.open('w', newline='', encoding='utf-8') as output_file:
        for key, value in metadata.items():
            output_file.write(f'# {key}={value}\n')
        writer = csv.writer(output_file)
        writer.writerow([
            'step', 'milliseconds', 'tokens_per_second', 'loss'
        ])
        for step, (milliseconds, throughput, loss) in enumerate(zip(
            elapsed_milliseconds, tokens_per_second, losses
        )):
            writer.writerow([
                step,
                f'{milliseconds:.10f}',
                f'{throughput:.10f}',
                f'{loss:.10f}',
            ])


def main():
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch cannot access a CUDA GPU')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('the allocated GPU does not support BF16')

    random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    properties = torch.cuda.get_device_properties(device)

    inputs, targets, input_checksum = make_dataset(arguments, device)
    model = TinyCausalTransformer(
        vocab_size=arguments.vocab_size,
        sequence_length=arguments.sequence_length,
        layers=arguments.layers,
        hidden_size=arguments.hidden_size,
        heads=arguments.heads,
        ffn_size=arguments.ffn_size,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )

    initial_loss = evaluate_loss(model, inputs[0], targets[0])
    for step in range(arguments.warmup_steps):
        dataset_index = step % arguments.dataset_batches
        train_step(
            model,
            optimizer,
            inputs[dataset_index],
            targets[dataset_index],
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    elapsed_milliseconds = []
    tokens_per_second = []
    losses = []
    tokens_per_step = arguments.batch_size * arguments.sequence_length
    for step in range(arguments.measured_steps):
        dataset_index = (
            arguments.warmup_steps + step
        ) % arguments.dataset_batches
        started = time.perf_counter()
        loss = train_step(
            model,
            optimizer,
            inputs[dataset_index],
            targets[dataset_index],
        )
        torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - started
        elapsed_milliseconds.append(1000.0 * elapsed_seconds)
        tokens_per_second.append(tokens_per_step / elapsed_seconds)
        losses.append(loss.item())

    max_allocated_memory_gib = (
        torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
    )
    measured_gradient_norm = gradient_norm(model)
    final_loss = evaluate_loss(model, inputs[0], targets[0])
    parameter_checksum = (
        model.token_embedding.weight[:16, :16].detach().float().sum().item()
    )
    loss_decrease_percent = 100.0 * (
        initial_loss - final_loss
    ) / initial_loss

    numeric_values = [
        initial_loss,
        final_loss,
        measured_gradient_norm,
        parameter_checksum,
        max_allocated_memory_gib,
        *elapsed_milliseconds,
        *tokens_per_second,
        *losses,
    ]
    if not all(math.isfinite(value) for value in numeric_values):
        raise RuntimeError('the canary produced a non-finite value')
    if final_loss >= initial_loss:
        raise RuntimeError(
            f'loss did not decrease: initial={initial_loss}, '
            f'final={final_loss}'
        )
    if measured_gradient_norm <= 0.0:
        raise RuntimeError('the final gradient norm is not positive')

    metadata = {
        'schema_version': 1,
        'hostname': socket.gethostname(),
        'slurm_job_id': os.getenv('SLURM_JOB_ID', 'unset'),
        'slurm_cpus_on_node': os.getenv('SLURM_CPUS_ON_NODE', 'unset'),
        'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES', 'unset'),
        'python_executable': sys.executable,
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'torch_cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'gpu_name': properties.name,
        'gpu_compute_capability': (
            f'{properties.major}.{properties.minor}'
        ),
        'deterministic_algorithms': (
            torch.are_deterministic_algorithms_enabled()
        ),
        'cublas_workspace_config': os.environ.get(
            'CUBLAS_WORKSPACE_CONFIG', 'unset'
        ),
        'dtype': 'bfloat16',
        'seed': arguments.seed,
        'vocab_size': arguments.vocab_size,
        'sequence_length': arguments.sequence_length,
        'batch_size': arguments.batch_size,
        'dataset_batches': arguments.dataset_batches,
        'layers': arguments.layers,
        'hidden_size': arguments.hidden_size,
        'heads': arguments.heads,
        'ffn_size': arguments.ffn_size,
        'warmup_steps': arguments.warmup_steps,
        'measured_steps': arguments.measured_steps,
        'parameter_count': parameter_count,
        'input_checksum': input_checksum,
    }
    write_samples(
        arguments.output,
        metadata,
        elapsed_milliseconds,
        tokens_per_second,
        losses,
    )

    print('CANARY_SCHEMA_VERSION=1')
    print(f'HOSTNAME={metadata["hostname"]}')
    print(f'SLURM_JOB_ID={metadata["slurm_job_id"]}')
    print(f'PYTHON_EXECUTABLE={sys.executable}')
    print(f'PYTHON_VERSION={metadata["python_version"]}')
    print(f'TORCH_VERSION={torch.__version__}')
    print(f'TORCH_CUDA_VERSION={torch.version.cuda}')
    print(f'CUDNN_VERSION={torch.backends.cudnn.version()}')
    print('CUDA_AVAILABLE=True')
    print(f'CUDA_DEVICE_COUNT={torch.cuda.device_count()}')
    print('BF16_SUPPORTED=True')
    print(f'GPU_NAME={properties.name}')
    print(
        f'GPU_COMPUTE_CAPABILITY={properties.major}.{properties.minor}'
    )
    print(
        'DETERMINISTIC_ALGORITHMS='
        f'{torch.are_deterministic_algorithms_enabled()}'
    )
    print(
        'CUBLAS_WORKSPACE_CONFIG='
        f'{os.environ.get("CUBLAS_WORKSPACE_CONFIG", "unset")}'
    )
    print('DTYPE=bfloat16')
    print(f'PARAMETER_COUNT={parameter_count}')
    print(f'INPUT_CHECKSUM={input_checksum}')
    print(f'INITIAL_LOSS={initial_loss:.10f}')
    print(f'FINAL_LOSS={final_loss:.10f}')
    print(f'LOSS_DECREASE_PERCENT={loss_decrease_percent:.6f}')
    print(f'GRADIENT_NORM={measured_gradient_norm:.10f}')
    print(f'PARAMETER_CHECKSUM={parameter_checksum:.10f}')
    print(f'MAX_ALLOCATED_MEMORY_GIB={max_allocated_memory_gib:.6f}')
    print(
        f'STEP_MEDIAN_MS={statistics.median(elapsed_milliseconds):.6f}'
    )
    print(f'STEP_P95_MS={percentile(elapsed_milliseconds, 0.95):.6f}')
    print(
        'TOKENS_PER_SECOND_MEDIAN='
        f'{statistics.median(tokens_per_second):.6f}'
    )
    print(
        f'TOKENS_PER_SECOND_P05={percentile(tokens_per_second, 0.05):.6f}'
    )
    print(
        'STEP_CV_PERCENT='
        f'{coefficient_of_variation_percent(elapsed_milliseconds):.6f}'
    )
    print('CANARY_SANITY_PASS')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'ERROR={error}', file=sys.stderr)
        raise
