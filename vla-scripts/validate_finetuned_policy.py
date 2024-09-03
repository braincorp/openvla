"""Make sure the fine-tuned policy performs as expected."""
import json
import os

import torch
from peft import prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig, AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset

ATTENTION_IMPLEMENTATION_NAME = "flash_attention_2"
DEFAULT_DEVICE_NAME = "cuda:0"
DEFAULT_CHECKPOINT_NAME = "openvla-7b-finetune-pick-mustard"
DEFAULT_DATASET_NAME = "episodes_pick_mustard_rlds"


def get_checkpoints_dir_path() -> str:
    """Get the path to the checkpoints directory."""
    return os.path.expanduser('~/brawn_artifacts/checkpoints')


def get_datasets_dir_path() -> str:
    """Get the path to the datasets directory."""
    return os.path.expanduser('~/brawn_artifacts/datasets/widowx_250s/episodes_pick_mustard')


def _evaluate_model_on_dataset(
        vla: AutoModelForVision2Seq,
        action_tokenizer: ActionTokenizer,
        dataloader: DataLoader,
        device_name: str
):
    """Helper to evaluate the model on the dataset."""
    accuracies = []
    losses = []
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device_name)
        attention_mask = batch["attention_mask"].to(device_name)
        pixel_values = batch["pixel_values"].to(torch.bfloat16).to(device_name)
        labels = batch["labels"].to(device_name)

        output: CausalLMOutputWithPast = vla(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = output.loss
        losses.append(loss.item())

        action_logits = output.logits[:, vla.vision_backbone.featurizer.patch_embed.num_patches: -1]
        action_preds = action_logits.argmax(dim=2)
        action_gt = batch["labels"][:, 1:].to(action_preds.device)
        mask = action_gt > action_tokenizer.action_token_begin_idx

        correct_preds = (action_preds == action_gt) & mask
        action_accuracy = correct_preds.sum().float() / mask.sum().float()
        accuracies.append(action_accuracy.item())
        print(f"Batch {batch_idx} Action Accuracy: {action_accuracy.item()}, Loss: {loss.item()}")

    print(f'Mean accuracy: {sum(accuracies) / len(accuracies)}')
    print(f'Mean loss: {sum(losses) / len(losses)}')


def validate_finetuned_policy(
        checkpoint_name: str = DEFAULT_CHECKPOINT_NAME,
        dataset_name: str = DEFAULT_DATASET_NAME,
        device_name: str = DEFAULT_DEVICE_NAME
):
    """Test that the fine-tuned policy performs as expected."""
    checkpoint_path = os.path.join(get_checkpoints_dir_path(), checkpoint_name)
    dataset_dir_path = get_datasets_dir_path()

    # Load model
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        attn_implementation=ATTENTION_IMPLEMENTATION_NAME,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        ),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    norm_stats_json = os.path.join(checkpoint_path, "dataset_statistics.json")
    with open(norm_stats_json, "r") as f:
        norm_stats = json.load(f)

    if len(norm_stats) != 1:
        raise ValueError(f"Expected one dataset key, got {len(norm_stats)}")

    vla.norm_stats = norm_stats  # hacky

    # Load dataset
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder
    )
    vla_dataset = RLDSDataset(
        data_root_dir=dataset_dir_path,
        data_mix=dataset_name,
        batch_transform=batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=100,
        image_aug=False,
        train=False
    )

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=1,
        sampler=None,
        collate_fn=collator,
        num_workers=0
    )
    _evaluate_model_on_dataset(
        vla=vla,
        action_tokenizer=action_tokenizer,
        dataloader=dataloader,
        device_name=device_name
    )


def validate_finetuned_policy_training_way(
        checkpoint_name: str = DEFAULT_CHECKPOINT_NAME,
        dataset_name: str = DEFAULT_DATASET_NAME,
        device_name: str = DEFAULT_DEVICE_NAME,
        use_quantization: bool = True
):
    """Build the model the same way as finetune.py"""
    checkpoint_path = os.path.join(get_checkpoints_dir_path(), checkpoint_name)
    dataset_dir_path = get_datasets_dir_path()

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if use_quantization:
        pass  # vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_name)

    norm_stats_json = os.path.join(checkpoint_path, "dataset_statistics.json")
    with open(norm_stats_json, "r") as f:
        norm_stats = json.load(f)

    if len(norm_stats) != 1:
        raise ValueError(f"Expected one dataset key, got {len(norm_stats)}")

    vla.norm_stats = norm_stats  # hacky

    # Load dataset
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder
    )
    vla_dataset = RLDSDataset(
        data_root_dir=dataset_dir_path,
        data_mix=dataset_name,
        batch_transform=batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=100,
        image_aug=False,
        train=False
    )

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=1,
        sampler=None,
        collate_fn=collator,
        num_workers=0
    )
    _evaluate_model_on_dataset(
        vla=vla,
        action_tokenizer=action_tokenizer,
        dataloader=dataloader,
        device_name=device_name
    )


if __name__ == '__main__':
    validate_finetuned_policy_training_way(use_quantization=False)
