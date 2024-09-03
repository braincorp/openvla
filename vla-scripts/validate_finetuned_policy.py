"""Make sure the fine-tuned policy performs as expected."""
import json
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset

ATTENTION_IMPLEMENTATION_NAME = "flash_attention_2"
DEFAULT_DEVICE_NAME = "cuda:0"


def get_checkpoints_dir_path() -> str:
    """Get the path to the checkpoints directory."""
    return os.path.expanduser('~/brawn_artifacts/checkpoints')


def get_datasets_dir_path() -> str:
    """Get the path to the datasets directory."""
    return os.path.expanduser('~/brawn_artifacts/widowx_250s/datasets')


def validate_finetuned_policy():
    """Test that the fine-tuned policy performs as expected."""
    checkpoint_name = "openvla-7b-finetune-pick-mustard"
    checkpoint_path = os.path.join(get_checkpoints_dir_path(), checkpoint_name)
    dataset_name = "episodes_pick_mustard_rlds"
    dataset_dir_path = get_datasets_dir_path()
    device_name = DEFAULT_DEVICE_NAME

    # Load model
    processor = AutoProcessor.from_pretrained(checkpoint_name, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint_name,
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
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=0,
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

    import pdb;
    pdb.set_trace()
    accuracies = []
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

        action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches: -1]
        action_preds = action_logits.argmax(dim=2)
        action_gt = batch["labels"][:, 1:].to(action_preds.device)
        mask = action_gt > action_tokenizer.action_token_begin_idx

        correct_preds = (action_preds == action_gt) & mask
        action_accuracy = correct_preds.sum().float() / mask.sum().float()
        accuracies.append(action_accuracy.item())
        print(f"Batch {batch_idx} Action Accuracy: {action_accuracy.item()}")


if __name__ == '__main__':
    test_finetuned_policy()
