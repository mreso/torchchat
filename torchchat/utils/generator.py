# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import base64
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from PIL import Image
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from torchchat.cli.builder import (
    _initialize_tokenizer,
    BuilderArgs,
    TokenizerArgs,
)
from torchchat.model import Model, ModelType
from torchchat.utils.build_utils import device_sync

# torchtune model definition dependencies
from torchtune.data import Message, padded_collate_tiled_images_and_mask
from torchtune.models.llama3_2_vision._model_builders import llama3_2_vision_transform
from torchtune.training import set_default_dtype


class _ChatFormatter(ABC):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def encode_dialog_prompt(self, dialog) -> List[int]:
        raise NotImplementedError()


class Llama3ChatFormatter(_ChatFormatter):
    """Format a chat prompt using special tokens to demarcate roles and messages.

    Refer to the LLaMA3 documentation for more details https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3

    """

    def encode_header(self, role) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(role, bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_message(self, message) -> List[int]:
        tokens = self.encode_header(message["role"])
        if isinstance(message["content"], str):
            tokens.extend(
                self.tokenizer.encode(message["content"], bos=False, eos=False)
            )
        elif isinstance(message["content"], list):
            for content in message["content"]:
                if content["type"] == "text":
                    tokens.extend(
                        self.tokenizer.encode(content["text"], bos=False, eos=False)
                    )

        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        return tokens

    def encode_dialog_prompt(self, dialog) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
        for message in dialog:
            tokens.extend(self.encode_message(message))
        # Add the start of an assistant message for the model to complete.
        tokens.extend(self.encode_header("assistant"))  # Pass role directly as a string
        return tokens


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"


class Llama2ChatFormatter(_ChatFormatter):
    def encode_dialog_prompt(self, dialog) -> List[int]:
        tokens = self.tokenizer.encode(f"{B_INST} ")
        first_message = True  # Bool to handle placing the B_INST token. Behavior is weird - the system prompt should have the B_INST, but not the first user message. All following user messages *should* have it. Also, if there is no system prompt, then the user message should have it.
        for message in dialog:
            if isinstance(message["content"], list):
                content = message["content"][0]["text"]
            else:
                content = message["content"]
            content = content.strip()
            if message["role"] == "system":
                encoded = self.tokenizer.encode(f"{B_SYS}\n{content}\n{E_SYS}")
                first_message = False
            elif message["role"] == "user":
                encoded = [self.tokenizer.bos_id()] + self.tokenizer.encode(
                    f"{B_INST if first_message else ''} {content} {E_INST} "
                )
                first_message = True
            elif message["role"] == "assistant":
                encoded = self.tokenizer.encode(f"{content}\n\n") + [
                    self.tokenizer.eos_id()
                ]
            tokens += encoded
        return tokens


@dataclass
class GeneratorArgs:
    prompt: Optional[str] = (
        None  # When passed into the Generator, this will be used as the system prompt
    )
    encoded_prompt: Optional[torch.Tensor] = None
    image_prompts: Optional[Sequence[Union[str, PathLike, bytes]]] = (
        None  # string or Path to an image file or the raw base64 bytes of an image
    )
    chat_mode: bool = False
    gui_mode: bool = False
    num_samples: int = 1
    max_new_tokens: int = 200
    top_k: int = 200
    temperature: float = 0.0  # deterministic argmax if 0.0
    compile: bool = False
    compile_prefill: bool = False
    speculate_k: int = 5
    sequential_prefill: bool = False
    max_autotune: bool = False
    # (Misnomer) See Issue: https://github.com/pytorch/torchchat/issues/1273
    is_torchtune_model: bool = False

    def __post_init__(self):
        if self.compile_prefill and self.sequential_prefill:
            raise RuntimeError("prefill compilation requires parallel prefill")

    def validate_build(
        self, builder_args: BuilderArgs, model_description: str = "model"
    ):
        reason = ""
        model_type = ""
        if not self.sequential_prefill:
            reason = "parallel prefill"
        if self.compile_prefill:
            reason = "model compilation for prefill"
        if self.compile:
            reason = "model compilation"
        if builder_args.aoti_package_path:
            model_type = "PT2"
        if builder_args.dso_path:
            model_type = "DSO"
        if builder_args.pte_path:
            model_type = "PTE"
        if model_type and reason:
            raise RuntimeError(
                f"cannot perform {reason} because a {model_type} {model_description} is used"
            )

    @classmethod
    def from_args(cls, args):
        dso_path = getattr(args, "dso_path", None)
        pte_path = getattr(args, "pte_path", None)
        aoti_package_path = getattr(args, "aoti_package_path", None)
        sequential_prefill = (
            args.sequential_prefill or bool(aoti_package_path) or bool(pte_path) or bool(dso_path)
        )

        # Validate that all image prompts exist before expensive model load
        if image_prompts := getattr(args, "image_prompts", None):
            non_existent_image_prompts = [
                image_prompt
                for image_prompt in image_prompts
                if (not os.path.exists(image_prompt))
            ]
            if non_existent_image_prompts:
                raise RuntimeError(
                    f"Image prompt {non_existent_image_prompts} does not exist"
                )

        return cls(
            prompt=getattr(args, "prompt", ""),
            encoded_prompt=None,
            image_prompts=image_prompts,
            chat_mode=args.chat,
            gui_mode=args.gui,
            num_samples=getattr(args, "num_samples", 1),
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            temperature=args.temperature,
            compile=args.compile,
            compile_prefill=args.compile_prefill,
            speculate_k=args.speculate_k,
            sequential_prefill=sequential_prefill,
            max_autotune=args.max_autotune,
            is_torchtune_model=args.model and args.model.endswith("tune"),
        )


class Generator(object):
    """
    Base class for generators that can be used to generate text samples based on a pre-trained Transformer model and tokenizer.
    """

    def __init__(
        self,
        builder_args: BuilderArgs,
        tokenizer_args: TokenizerArgs,
        generator_args: GeneratorArgs,
    ):
        self.builder_args = builder_args
        self.tokenizer_args = tokenizer_args
        self.generate_args = generator_args

        self.dtype = builder_args.precision

        self.tokenizer = _initialize_tokenizer(self.tokenizer_args)

        self.model : Optional[Model] = None
        self.draft_model : Optional[Model] = None
        self.profile : Optional[Path] = None
        self.quantize: bool = False
        self.draft_quantize: bool = False
        self.is_speculative: bool = False

        # Right now the assumption is only llama3 uses tiktokenizer and it
        # must use tiktokenizer.
        # Piggy backing off of this flag then for now to identify llama3
        # without prompting user.
        self.is_llama3_model = self.tokenizer_args.is_tiktoken
        if self.is_llama3_model:
            self.chat_formatter = Llama3ChatFormatter(self.tokenizer)
            if generator_args.chat_mode:
                logging.debug(
                    "Llama3 model detected in chat mode. Using updated sentence schemas"
                )
        else:
            self.chat_formatter = Llama2ChatFormatter(self.tokenizer)

    
    def chat(
        self,
        generator_args: GeneratorArgs,
    ):
        if generator_args.chat_mode:
            print("Starting Interactive Chat")

        model_size = self.model_size

        self.system_prompt = None
        # Set up our max_seq_length

        max_seq_length = self.max_seq_length

        encoded, batch = self._gen_model_input(
            generator_args.prompt,
            generator_args.image_prompts,
            generator_args.max_new_tokens,
            max_seq_length,
        )

        model_type = self.model.config.model_type if self.model else None
        if generator_args.chat_mode:
            print(
                f"Entering Chat Mode. Will continue chatting back and forth with the language model until the models max context length of {max_seq_length} tokens is hit or until the user says /bye"
            )
            get_system_prompt = input(
                "Do you want to enter a system prompt? Enter y for yes and anything else for no. \n"
            )
            if get_system_prompt == "y" or get_system_prompt == "Y":
                self.system_prompt = input("What is your system prompt? \n")

        # `is_torchtune_model` is a misnomer since it doesn't capture all
        # torchtune models (i.e. Flamingo)
        # See Issue: https://github.com/pytorch/torchchat/issues/1273
        elif (
            not generator_args.is_torchtune_model
            and model_type != ModelType.Flamingo
        ):
            text_transformer_args = self.model.text_transformer_args if self.model else None
            max_seq_length = min(
                encoded.size(0) + generator_args.max_new_tokens,
                (
                    text_transformer_args.block_size
                    if text_transformer_args is not None
                    else 2048
                ),
                max_seq_length,
            )

        if self.draft_model is not None:
            max_seq_length += self.speculative_builder_args.speculate_k + 1

        aggregate_metrics = {
            "tokens_per_sec": [],
            "first_token_per_sec": [],
            "next_tokens_per_sec": [],
            "accept_counts": [],
        }
        start_pos = 0

        # arbitrarily large number as chat mode goes until max_seq length
        # or user exits
        num_samples = (
            generator_args.num_samples if not generator_args.chat_mode else 100000
        )
        for i in range(num_samples):
            device_sync(device=self.builder_args.device)
            if generator_args.chat_mode:
                prompt = input("User: ")
                if prompt == "/bye":
                    print("Exiting Chat.\n")
                    break
                if not self.is_llama3_model:
                    if self.system_prompt:
                        prompt = f"{B_INST} {B_SYS}\n{self.system_prompt.strip()}\n{E_SYS}\n\n{prompt.strip()} {E_INST}"
                        self.system_prompt = (
                            None  # can only provide system prompt on first interaction
                        )
                    else:
                        prompt = f"{B_INST} {prompt.strip()} {E_INST}"
                    encoded = self.encode_tokens(
                        prompt, bos=True, device=self.builder_args.device
                    )
                else:
                    if self.system_prompt:
                        encoded = self.chat_formatter.encode_dialog_prompt(
                            [
                                {"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": prompt},
                            ]
                        )
                        self.system_prompt = None
                    elif i == 0:
                        encoded = self.chat_formatter.encode_dialog_prompt(
                            [{"role": "user", "content": prompt}]
                        )
                    else:
                        encoded = self.chat_formatter.encode_message(
                            {"role": "user", "content": prompt}
                        )
                        encoded.extend(self.chat_formatter.encode_header("assistant"))
                    encoded = torch.tensor(
                        encoded, dtype=torch.int, device=self.builder_args.device
                    )
                if encoded.size(0) + start_pos > max_seq_length:
                    print(
                        "This prompt would take us past the max_seq_length. Ending Conversation."
                    )
                    break

                print("Model: ", end="")

                buffer = []

                def callback(x, *, done_generating=False):
                    return self._callback(
                        x,
                        buffer=buffer,
                        done_generating=done_generating,
                    )

            else:
                assert not generator_args.chat_mode

                buffer = [generator_args.prompt]

                def callback(x, *, done_generating=False):
                    return self._callback(
                        x,
                        buffer=buffer,
                        done_generating=done_generating,
                    )

            if self.profile:
                torch._inductor.config.profiler_mark_wrapper_call = True
                torch._inductor.config.cpp.enable_kernel_profile = True
            if (i != generator_args.num_samples - 1 or not self.profile) or (
                self.builder_args.distributed and self.rank != 0
            ):
                import contextlib

                prof = contextlib.nullcontext()
            else:
                torch.profiler._utils._init_for_cuda_graphs()
                prof = torch.profiler.profile()
            t0 = time.perf_counter()
            num_tokens_generated = 0
            with prof:
                generator_func = self.generate(
                    encoded,
                    generator_args.max_new_tokens,
                    speculate_k=generator_args.speculate_k,
                    chat_mode=generator_args.chat_mode,
                    batch=batch,
                    callback=callback,
                    temperature=generator_args.temperature,
                    top_k=generator_args.top_k,
                    sequential_prefill=generator_args.sequential_prefill,
                    start_pos=start_pos,
                    max_seq_length=max_seq_length,
                )
                for token_tensor, metrics in generator_func:
                    if token_tensor is not None:
                        start_pos += token_tensor.size(0)
                        num_tokens_generated += token_tensor.size(0)
                    if metrics is not None:
                        aggregate_metrics.update(metrics)
                    yield token_tensor, metrics
            jit_compile = (i == 0) and (
                generator_args.compile or generator_args.compile_prefill
            )
            compilation_time = time.perf_counter() - t0
            device_sync(device=self.builder_args.device)
            t = time.perf_counter() - t0
            if hasattr(prof, "export_chrome_trace"):
                if self.builder_args.device == "cpu":
                    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                else:
                    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
                if self.builder_args.distributed:
                    prof.export_chrome_trace(f"{self.profile}_rank_{self.rank}.json")
                else:
                    prof.export_chrome_trace(f"{self.profile}.json")

            if start_pos >= max_seq_length:
                print(
                    f"[Max Sequence Length {max_seq_length} Reached. Ending Conversation.]"
                )
                print("---------------------------------------------------")

            tokens_sec = (num_tokens_generated + 1) / t
            first_token_sec = 1 / aggregate_metrics.get("time_to_first_token", 0)
            next_tokens_sec = num_tokens_generated / (
                t - aggregate_metrics.get("time_to_first_token", 0)
            )

            if jit_compile:
                print(
                    f"just-in-time compilation time (incl run time): {compilation_time:.2} seconds"
                )
            aggregate_metrics["tokens_per_sec"].append(tokens_sec)
            aggregate_metrics["first_token_per_sec"].append(first_token_sec)
            aggregate_metrics["next_tokens_per_sec"].append(next_tokens_sec)

            logging.info(
                f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\
                \nGenerated {num_tokens_generated} tokens \
                \nTime for inference {i + 1}: {t:.04f} sec total \
                \nTime to first token: {aggregate_metrics.get('time_to_first_token', 0):.04f} sec \
with {'sequential' if generator_args.sequential_prefill else 'parallel'} prefill.\
                \n\n      Total throughput: {tokens_sec:.04f} tokens/sec, {1 / tokens_sec:.04f} s/token \
                \nFirst token throughput: {first_token_sec:.04f} tokens/sec, {1 / first_token_sec:.04f} s/token \
                \n Next token throughput: {next_tokens_sec:.04f} tokens/sec, {1 / next_tokens_sec:.04f} s/token \
                    "
            )
            logging.info(
                f"\nBandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s"
            )
            if i == 0:
                logging.info(
                    f"*** This first iteration will include cold start effects for dynamic import, hardware caches{', JIT compilation' if jit_compile else ''}. ***"
                )
            print("\n========================================\n")
            if start_pos >= max_seq_length:
                if generator_args.chat_mode:
                    break

            if not generator_args.chat_mode:
                start_pos = 0

        if self.is_speculative:
            counts_aggregated = [
                sum(i) for i in zip(*aggregate_metrics["accept_counts"])
            ]
            acceptance_probs = [i / sum(counts_aggregated) for i in counts_aggregated]
            print(f"Acceptance probs: {acceptance_probs}")
            print(
                f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated)}"
            )

        print(
            f"\n      Average tokens/sec (total): {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f} \
                \nAverage tokens/sec (first token): {torch.mean(torch.tensor(aggregate_metrics['first_token_per_sec'])).item():.2f} \
                \nAverage tokens/sec (next tokens): {torch.mean(torch.tensor(aggregate_metrics['next_tokens_per_sec'])).item():.2f} \n\
                "
        )
        if torch.cuda.is_available():
            print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

    @abstractmethod
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        *,
        chat_mode: bool,
        batch: Optional[
            Dict[str, Any]
        ] = None,  # List of Image prompt tensors for multimodal models
        start_pos: int = 0,
        speculate_k: Optional[int] = 8,
        sequential_prefill=True,
        callback=lambda x: x,
        max_seq_length: int,
        seed: Optional[int] = None,
        **sampling_kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def is_text_only(self) -> bool:
        """
        Returns True if the model is text-only, False otherwise.
        """
        raise NotImplementedError()

    @property
    def max_seq_length(self) -> int:
        """
        Returns the maximum sequence length supported by the model.
        """
        raise NotImplementedError()

    @property
    def model_size(self) -> int:
        """
        Returns the size of the model.
        """
        raise NotImplementedError()

    def _gen_model_input(
        self,
        prompt: Union[str | List[Any]],
        image_prompts: Optional[List[str | Image.Image]] = None,
        max_new_tokens: Optional[int] = None,
        max_seq_len: Optional[int] = 2048,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Convert prompt and image prompts into consumable model input args.

        When prompt is a list, the anticipated format is OpenAI API Inspired:
            [ ..., {"role": message["role"], "content": message["content"]}, ...]

        Args:
            prompt (Union[str, List[Any]]): Prompt or list of dialog.
            image_prompts (Optional[List[str | Image.Image]]): List of image prompts. Used only with Llama 3.2 11B.
            max_new_tokens (Optional[int]): Maximum number of new tokens to generate. Used only with Llama 3.2 11B.

        Returns:
            Tuple[torch.Tensor, Optional[Dict[str, Any]]]: Encoded prompt and batch config for multimodal models.
        """

        # Text-Only model
        if self.is_text_only():
            # Single String prompt
            if isinstance(prompt, str):
                encoded = self.encode_tokens(
                    prompt, bos=True, device=self.builder_args.device
                )
            # List of dialog
            else:
                tokens = self.chat_formatter.encode_dialog_prompt(prompt)
                encoded = torch.tensor(
                    tokens, dtype=torch.int, device=self.builder_args.device
                )

            logging.debug(encoded)
            return encoded, None

        # Llama 3.2 11B
        assert (
            image_prompts is None or len(image_prompts) == 1
        ), "At most one image is supported at the moment"

        if image_prompts and isinstance(image_prompts[0], str):
            images = [Image.open(image_prompts[0])]
        else:
            images = None

        assert (
            max_new_tokens is not None
        ), "max_new_tokens must be specified for Flamingo models"

        # Wrap string prompts into a list
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        image_found = False
        messages = []
        for message in prompt:
            if isinstance(message["content"], str):
                if not image_found and image_prompts:
                    messages.append(
                        Message(
                            role=message["role"],
                            content=[
                                {"type": "image", "content": images[0]},
                                {"type": "text", "content": message["content"]},
                            ],
                        )
                    )
                    image_found = True
                else:
                    messages.append(Message(**message))

            elif isinstance(message["content"], list):
                images = None
                for content_dict in message["content"]:
                    if content_dict["type"] == "text":
                        prompt_arg = content_dict["text"]
                    elif content_dict["type"] == "image_url":
                        assert (
                            images is None
                        ), "At most one image is supported at the moment"

                        base64_decoded = base64.b64decode(
                            content_dict["image_url"].split(";base64,")[1]
                        )
                        images = [Image.open(BytesIO(base64_decoded))]
                        image_found = True

                is_multimodal = images is not None
                content = [{"type": "text", "content": prompt_arg}]

                if is_multimodal:
                    content = [{"type": "image", "content": images[0]}] + content

                messages.append(
                    Message(
                        role=message["role"],
                        content=content,
                    )
                )

        messages.append(
            Message(
                role="assistant",
                content="",
            )
        )

        transform = llama3_2_vision_transform(str(self.tokenizer_args.tokenizer_path))

        device = torch.device(device=self.builder_args.device)

        with device, set_default_dtype(self.dtype):
            data = transform({"messages": messages}, inference=True)

            if image_found:
                batch = padded_collate_tiled_images_and_mask(
                    [data], pad_direction="left", pad_max_images=1
                )
                encoded = batch.pop("tokens").to(device).view(-1)
                seq_len = encoded.size(0)
                batch["encoder_mask"] = batch["encoder_mask"][:, :seq_len]
                batch["encoder_input"]["images"] = batch["encoder_input"]["images"].to(
                    self.dtype
                )

            else:
                encoded = torch.tensor(data["tokens"], device=device).view(-1)
                seq_len = encoded.size(0)
                batch = {}

            total_response_length = seq_len + max_new_tokens
            batch["causal_mask"] = torch.nn.functional.pad(
                torch.tril(
                    torch.ones(
                        size=(total_response_length, total_response_length),
                        dtype=torch.bool,
                    )
                ),
                (
                    0,
                    max_seq_len - total_response_length,
                    0,
                    max_seq_len - total_response_length,
                ),
                value=0,
            )

        logging.debug(encoded)
        return encoded, batch

    def _callback(self, x, *, buffer, done_generating):
        # TODO: Refactor this callback to only include basic functionality & remove print statements
        period_id = self.tokenizer.encode(".")[0]
        buffer.append(
            self.tokenizer.decode([period_id] + x.tolist())[1:]
        )  # I think this results in the first output token being dropped from the display which is wrong.
        if x.item() == self.tokenizer.eos_id():
            done_generating = True
        if (
            self.is_llama3_model
            and x.item() == self.tokenizer.special_tokens["<|eot_id|>"]
        ):
            done_generating = True
            buffer = buffer[:-1]  # drop the eot_id from the output buffer
        if len(buffer) == 4 or done_generating:
            print("".join(buffer), end="", flush=True)
            buffer.clear()
        # print(, end='', flush=True)

    def encode_tokens(self, string, bos=True, device="cpu"):
        tokens = self.tokenizer.encode(string)
        if bos:
            tokens = [self.tokenizer.bos_id()] + tokens
        logging.debug(f"Size after encode_tokens: {len(tokens)}")
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def shutdown(self):
        """
        This method can be used to release resources.
        """
        pass
