"""Script to make sure the fine-tuned policy performs as expected."""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

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


def test_finetuned_policy(
        checkpoint_name: str = DEFAULT_CHECKPOINT_NAME,
        dataset_name: str = DEFAULT_DATASET_NAME,
        device_name: str = DEFAULT_DEVICE_NAME
) -> None:
    """Make sure the fine-tuned policy performs as expected."""
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
        image_aug=True,
        train=True
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

    # Evaluate the model on the dataset
    accuracies = []
    accuracies_policy = []
    losses = []
    max_batches = 100
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device_name)
        attention_mask = batch["attention_mask"].to(device_name)
        pixel_values = batch["pixel_values"].to(torch.bfloat16).to(device_name)
        labels = batch["labels"].to(device_name)

        # Generate tokens the way it's done during training
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

        # Generate tokens the way it's done in the policy
        image = np.round(np.interp(
            pixel_values[0][3:].to(torch.float32).cpu().numpy().transpose(1, 2, 0),
            (-1, 1),
            (0, 255)
        )).astype(np.uint8)
        instruction = processor.decode(input_ids[0][10:-13], skip_special_tokens=True)
        image_pil = Image.fromarray(image)
        network_inputs = processor(
            f"In: What action should the robot take to {instruction.lower()}?\nOut: ",  # note space at end!
            image_pil
        ).to(device_name, dtype=torch.bfloat16)
        unnorm_key = 'episodes_pick_mustard_rlds'

        # Validate inputs are the same
        input_ids_match = bool((input_ids[0, :-8] == network_inputs['input_ids'][0]).all().cpu().numpy())
        image_recovered = np.round(np.interp(
            network_inputs['pixel_values'][0][3:].to(torch.float32).cpu().numpy().transpose(1, 2, 0),
            (-1, 1),
            (0, 255)
        )).astype(np.uint8)
        image_match = bool((image == image_recovered).all())

        if not input_ids_match:
            raise RuntimeError('input ids mismatch!')

        if not image_match:
            raise RuntimeError('image mismatch!')

        generated_ids = vla.generate(
            pixel_values=network_inputs['pixel_values'],
            input_ids=network_inputs['input_ids'],
            attention_mask=network_inputs['attention_mask'],
            max_new_tokens=7
        )

        action_tokens_training = action_preds.to('cpu').numpy()[0][mask.to('cpu').numpy()[0]]
        action_tokens_policy = generated_ids[0, -vla.get_action_dim(unnorm_key):].cpu().numpy()
        action_tokens_gt = action_gt.to('cpu').numpy()[0][mask.to('cpu').numpy()[0]]

        accuracy_policy = (action_tokens_gt == action_tokens_policy).sum() / len(action_tokens_training)
        accuracies_policy.append(accuracy_policy)

        print(
            f"Batch {batch_idx} Loss: {loss.item()}, Action accuracy: {action_accuracy.item()}, "
            f"Action accuracy (policy): {accuracy_policy}"
        )
        if action_tokens_training[0] == action_tokens_gt[0] and (action_tokens_policy[0] != action_tokens_gt[0]):
            raise RuntimeError(f'first policy token does not match!')

        if batch_idx == max_batches:
            break

    print(f'Mean loss: {sum(losses) / len(losses)}')
    print(f'Mean accuracy: {sum(accuracies) / len(accuracies)}')
    print(f'Mean accuracy (policy): {sum(accuracies_policy) / len(accuracies_policy)}')


if __name__ == '__main__':
    test_finetuned_policy()
