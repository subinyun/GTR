#!/usr/bin/env python3
"""Train LoRA on completion-style prompts without chat templates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path
from typing import Any


def require_package(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise ModuleNotFoundError(f"Missing package '{name}'.")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--valid-data-path", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--save-every-steps", type=int, default=0, help="Save LoRA adapter every N optimizer steps.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-chat-template", action="store_true")
    p.add_argument("--system-message", default="You are a legal assistant.")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    require_package("torch")
    require_package("transformers")
    require_package("peft")

    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    rows = load_jsonl(args.data_path)
    random.seed(args.seed)
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_prompt(prompt: str) -> str:
        if not args.use_chat_template:
            return prompt
        messages = [
            {"role": "system", "content": args.system_message},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float16
        if torch.cuda.is_available()
        else torch.float32,
    }
    if args.device == "auto":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    if args.device != "auto":
        model.to(args.device)
    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    def save_adapter_checkpoint(name: str) -> None:
        ckpt_dir = args.output_dir / "step_checkpoints" / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        (ckpt_dir / "training_args.json").write_text(
            json.dumps(
                {
                    "model_name_or_path": args.model_name_or_path,
                    "data_path": str(args.data_path),
                    "valid_data_path": str(args.valid_data_path) if args.valid_data_path else None,
                    "epochs": args.epochs,
                    "max_steps": args.max_steps,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "max_length": args.max_length,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "save_every_steps": args.save_every_steps,
                    "seed": args.seed,
                    "use_chat_template": args.use_chat_template,
                    "system_message": args.system_message if args.use_chat_template else None,
                    "global_step": global_step,
                    "loss_masks_prompt_tokens": True,
                    "uses_chat_template": False,
                    "is_step_checkpoint": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved step checkpoint to {ckpt_dir}", flush=True)

    class CompletionDataset(Dataset):
        def __init__(self, data: list[dict[str, Any]]) -> None:
            self.data = data

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            row = self.data[idx]
            prompt = format_prompt(str(row["prompt"]))
            completion = str(row["completion"]) + (tokenizer.eos_token or "")
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            if not completion_ids:
                completion_ids = [tokenizer.eos_token_id]
            if len(completion_ids) >= args.max_length:
                completion_ids = completion_ids[: args.max_length]
                prompt_ids = []
            else:
                prompt_budget = args.max_length - len(completion_ids)
                # Keep the end of the prompt where candidates, output format, and Answer appear.
                prompt_ids = prompt_ids[-prompt_budget:]
            input_list = prompt_ids + completion_ids
            attention_list = [1] * len(input_list)
            label_list = [-100] * len(prompt_ids) + completion_ids[:]
            pad_len = args.max_length - len(input_list)
            if pad_len > 0:
                input_list.extend([tokenizer.pad_token_id] * pad_len)
                attention_list.extend([0] * pad_len)
                label_list.extend([-100] * pad_len)
            input_ids = torch.tensor(input_list, dtype=torch.long)
            attention_mask = torch.tensor(attention_list, dtype=torch.long)
            labels = torch.tensor(label_list, dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(CompletionDataset(rows), batch_size=args.batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        losses: list[float] = []
        pending_update = False
        for step, batch in enumerate(loader, 1):
            target_device = next(model.parameters()).device
            batch = {key: value.to(target_device) for key, value in batch.items()}
            out = model(**batch)
            if not torch.isfinite(out.loss.detach()):
                print(f"warning: non-finite loss at epoch={epoch + 1} step={step}; skipping batch", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue
            loss = out.loss / max(args.gradient_accumulation_steps, 1)
            loss.backward()
            pending_update = True
            losses.append(float(out.loss.detach().cpu()))
            if step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending_update = False
                global_step += 1
                if args.log_every > 0 and global_step % args.log_every == 0:
                    recent = losses[-args.log_every :] if len(losses) >= args.log_every else losses
                    print(f"epoch={epoch + 1} global_step={global_step} recent_loss={sum(recent) / max(len(recent), 1):.4f}", flush=True)
                if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                    save_adapter_checkpoint(f"step_{global_step:04d}")
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        if pending_update:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                save_adapter_checkpoint(f"step_{global_step:04d}")
        print(f"epoch={epoch + 1} mean_loss={sum(losses) / max(len(losses), 1):.4f}")
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "training_args.json").write_text(
        json.dumps(
            {
                "model_name_or_path": args.model_name_or_path,
                "data_path": str(args.data_path),
                "valid_data_path": str(args.valid_data_path) if args.valid_data_path else None,
                "epochs": args.epochs,
                "max_steps": args.max_steps,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_length": args.max_length,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "save_every_steps": args.save_every_steps,
                "seed": args.seed,
                "use_chat_template": args.use_chat_template,
                "system_message": args.system_message if args.use_chat_template else None,
                "global_step": global_step,
                "loss_masks_prompt_tokens": True,
                "uses_chat_template": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
