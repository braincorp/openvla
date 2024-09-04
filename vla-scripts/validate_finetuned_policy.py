"""Make sure the fine-tuned policy performs as expected."""
import json
import os

import torch
from PIL import Image
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
import numpy as np

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


def patch_predict_action_tokens(vla: AutoModelForVision2Seq) -> None:
    @torch.inference_mode()
    def predict_action_tokens(
            image: Image,
            instruction: str,
            unnorm_key: Optional[str] = None,
            **kwargs: str
    ) -> np.ndarray:
        """
        Monkey patch for OpenVLA class.
        Core function for VLA inference; maps input image and task instruction to action tokens.

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
            was trained only on a single dataset, and retrieves those statistics.
        @return action tokens
        """
        image_transform, tokenizer = vla.vision_backbone.image_transform, vla.llm_backbone.tokenizer

        # Build VLA Prompt
        prompt_builder = vla.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(vla.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            # Note: We need to add this special empty token ('') after the colon (':') token in "ASSISTANT:"
            #       in order for the predictions to match the training configuration and be accurate.
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(vla.device)), dim = 1
            )
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(vla.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(vla.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = vla.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=vla.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super(PrismaticVLM, self).generate(
                input_ids=input_ids,  # Shape: [1, seq]
                pixel_values=pixel_values,  # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=vla.get_action_dim(unnorm_key),
                **kwargs
            )
            # fmt: on
        return generated_ids

    vla.predict_action_tokens = predict_action_tokens
    return vla


def _evaluate_model_on_dataset(
        vla: AutoModelForVision2Seq,
        action_tokenizer: ActionTokenizer,
        dataloader: DataLoader,
        device_name: str,
        processor: AutoProcessor
):
    """Helper to evaluate the model on the dataset."""
    action_norm_stats = vla.get_action_stats('episodes_pick_mustard_rlds')
    accuracies = []
    accuracies_policy = []
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

        # Generate tokens the way it's done in the policy
        image = np.interp(
            batch["pixel_values"][0][3:].numpy().transpose(1, 2, 0),
            (-1, 1),
            (0, 255)
        ).astype(np.uint8)
        instruction = processor.decode(input_ids[0][10:-13], skip_special_tokens=True)
        image_pil = Image.fromarray(image)
        network_inputs = processor(
            f"In: What action should the robot take to {instruction.lower()}?\nOut:",
            image_pil
        ).to(device_name, dtype=torch.bfloat16)
        unnorm_key = 'episodes_pick_mustard_rlds'
        generated_ids = vla.generate(**network_inputs, max_new_tokens=vla.get_action_dim(unnorm_key))

        action_tokens_training = action_preds.to('cpu').numpy()[0][mask.to('cpu').numpy()[0]]
        action_tokens_policy = generated_ids[0, -vla.get_action_dim(unnorm_key) :].cpu().numpy()
        action_tokens_gt = action_gt.to('cpu').numpy()[0][mask.to('cpu').numpy()[0]]

        accuracy_policy = (action_tokens_gt == action_tokens_policy).sum() / len(action_tokens_training)
        accuracies_policy.append(accuracy_policy)
        print(f"Batch {batch_idx} Loss: {loss.item()}, Action Accuracy: {action_accuracy.item()}, Action Accuracy (policy): {accuracy_policy}")
        import pdb; pdb.set_trace()

        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        action_preds_token_ids = action_preds[0, -vla.get_action_dim('episodes_pick_mustard_rlds'):]
        action_preds_normalized = action_tokenizer.decode_token_ids_to_actions(action_preds_token_ids.cpu().numpy())
        action_mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_preds_unnorm = np.where(
            action_mask,
            0.5 * (action_preds_normalized + 1) * (action_high - action_low) + action_low,
            action_preds_normalized,
        )

        image = np.interp(
            batch["pixel_values"][0][3:].numpy().transpose(1, 2, 0),
            (-1, 1),
            (0, 255)
        ).astype(np.uint8)
        import cv2
        cv2.imshow('image', image[:, :, ::-1])
        cv2.waitKey(0)

        instruction = processor.decode(input_ids[0][10:-13], skip_special_tokens=True)
        print(f'instruction: {instruction}')
        image_pil = Image.fromarray(image)
        network_inputs = processor(f"In: What action should the robot take to {instruction.lower()}?\nOut:", image_pil).to(device_name, dtype=torch.bfloat16)
        import pdb; pdb.set_trace()

        action_policy = vla.predict_action(**network_inputs, unnorm_key='episodes_pick_mustard_rlds')  # from modeling_prismatic.py
        print(f'action preds: {action_preds_unnorm}')
        print(f'action policy: {action_policy}')
        # TODO: Figure out why these disagree... see recent terminal

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
    _evaluate_model_on_dataset(
        vla=vla,
        action_tokenizer=action_tokenizer,
        dataloader=dataloader,
        device_name=device_name,
        processor=processor
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
        device_name=device_name,
        processor=processor
    )


if __name__ == '__main__':
    validate_finetuned_policy()
