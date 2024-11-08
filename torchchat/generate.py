# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import textwrap
import time

from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from typing_extensions import override

import torch
import torch._dynamo.config
import torch._inductor.config

from torchtune.generation import sample as tune_sample

from torchchat.cli.builder import (
    _initialize_model,
    BuilderArgs,
    TokenizerArgs,
)
from torchchat.distributed.generate import DistributedGenerator
from torchchat.model import Model, ModelType
from torchchat.utils.build_utils import set_precision
from torchchat.utils.device_info import get_device_info
from torchchat.utils.generator import Generator, GeneratorArgs, E_INST, B_INST, E_SYS, B_SYS


class LocalGenerator(Generator):
    """
    Generates text samples based on a pre-trained Transformer model and tokenizer.
    Args:
        builder_args: Defines the model configuration
        speculative_builder_args: Defines the speculative model configuration for speculative decode
        tokenizer_args: Defines the tokenizer configuration for both the model and speculative model
        generator_args: Controls the generation parameters
        profile: A Path to a directory where the profiling results will be stored, if enabled.
        quantize: If True, quantize the model. Please refer to docs/quantization.md for details.
        draft_quantize: If True, quantize the draft model.
    """

    def __init__(
        self,
        builder_args: BuilderArgs,
        speculative_builder_args: BuilderArgs,
        tokenizer_args: TokenizerArgs,
        generator_args: GeneratorArgs,
        profile: Optional[Path],
        quantize: bool,
        draft_quantize: bool,
    ):
        super().__init__(builder_args, tokenizer_args, generator_args)
        torch._inductor.config.coordinate_descent_tuning = (
            builder_args.device != "cpu"
        )
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

        self.builder_args = builder_args
        self.speculative_builder_args = speculative_builder_args
        self.profile = profile
        self.quantize = quantize
        self.draft_quantize = draft_quantize
        self.is_torchtune_model = generator_args.is_torchtune_model

        self.rank: Optional[int] = None

        print(
            f"Using device={self.builder_args.device} {get_device_info(self.builder_args.device)}"
        )
        set_precision(self.builder_args.precision)

        self.is_speculative = self.speculative_builder_args.checkpoint_path is not None

        if generator_args.chat_mode and not self.builder_args.is_chat_model:
            # fmt: off
            print(textwrap.dedent(
                """
                *******************************************************
                This model is not known to support the chat function
                and may produce nonsensical or false output.
                *******************************************************
                """
            ))
            # fmt: on
        self.system_prompt = generator_args.prompt

        self.builder_args.setup_caches = False
        self.model = _initialize_model(self.builder_args, self.quantize, self.tokenizer)

        if self.is_speculative:
            self.draft_model = _initialize_model(
                self.speculative_builder_args,
                (
                    self.quantize
                    if self.draft_quantize == "quantize"
                    else self.draft_quantize
                ),
                self.tokenizer,
            )
        else:
            self.draft_model = None

        # torchtune model does not contain essential info for validation
        # TODO: refactor model config to be more generic
        if not self.is_torchtune_model:
            self.tokenizer_args.validate_model(self.model)
        self.tokenizer_args.validate_model(self.draft_model, "draft model")
        generator_args.validate_build(self.builder_args)
        generator_args.validate_build(self.speculative_builder_args, "draft model")

        if generator_args.compile:
            self.apply_compile_to_models(generator_args)

    def multinomial_sample_one_no_sync(
        self,
        probs_sort,
    ):  # Does multinomial sampling without a cuda synchronization
        q = torch.empty_like(probs_sort).exponential_(1)
        return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

    def logits_to_probs(
        self, logits, temperature: float = 1.0, top_k: Optional[int] = None
    ):
        logits = logits / max(
            temperature, 1e-5 if logits.dtype != torch.float16 else 1e-3
        )

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, -float("Inf"), logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    def sample(
        self,
        logits,
        need_probs: bool,
        temperature: float = 0,
        top_k: Optional[int] = None,
    ):
        if temperature == 0 and not need_probs:
            _, idx_next = torch.topk(logits[0, -1], k=1, dim=-1)
            return (idx_next, None)
        probs = self.logits_to_probs(logits[0, -1], temperature, top_k)
        idx_next = self.multinomial_sample_one_no_sync(probs)
        return idx_next, probs

    def prefill(
        self,
        model: Model,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        batch: Optional[Dict[str, Any]] = None,  # Inputs for multimodal models
        *,
        sequential_prefill=True,
        **sampling_kwargs,
    ) -> torch.Tensor:
        # logging.debug(f"x: {x}, input_pos: {input_pos}")
        width = x.size(1)
        assert input_pos.size(0) == width

        if self.model.config.model_type == ModelType.Flamingo:
            assert batch is not None, "Flamingo requires batch"

            # TODO: Verify sequential prefill works with multimodal models
            is_multimodal = True
            if "encoder_input" in batch:
                encoder_input = batch["encoder_input"]
                encoder_mask = batch["encoder_mask"]
                is_multimodal = True
            else:
                encoder_input = None
                encoder_mask = None
                is_multimodal = False

            seq_len = x.size(1)
            mask = batch["causal_mask"][None, :seq_len]
            input_pos = input_pos.view(1, -1)
            logits = model(
                tokens=x,
                mask=mask,
                encoder_input=encoder_input,
                input_pos=input_pos,
                encoder_mask=encoder_mask,
            )[:, -1]

            if is_multimodal:
                batch["encoder_mask"] = batch["encoder_mask"][:, -1:]

            return tune_sample(logits, temperature=0, top_k=500)
        elif sequential_prefill:
            for i in range(width):
                x_sliced, ip_sliced = x[:, i].view(-1, 1), input_pos[i].view(-1)
                # logging.debug(f"<sliced> x: {x_sliced}, input_pos: {ip_sliced}")
                logits = model(x_sliced, ip_sliced)  # (x[:, i], input_pos[i])da
        else:
            # input_pos: [B, S]
            logits = model(x, input_pos)
            # print(f"logits {logits.shape}")

        # print(f"x: {x},\n  input_pos: {input_pos}\n")
        return self.sample(logits, need_probs=False, **sampling_kwargs)[0]

    def decode_one_token(
        self,
        model: Model,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        need_probs: bool,
        batch: Optional[Dict[str, Any]] = None,  # Inputs for multimodal models
        **sampling_kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # input_pos: [B, 1]
        assert input_pos.shape[-1] == 1
        x = x.view(1, -1)
        if model.config.model_type == ModelType.Flamingo:
            assert batch is not None, "Flamingo requires batch"
            mask = batch["causal_mask"][None, input_pos.item(), None, :]
            encoder_mask = batch["encoder_mask"] if "encoder_mask" in batch else None
            logits = model(
                x, encoder_mask=encoder_mask, mask=mask, input_pos=input_pos
            )[:, -1:]
        else:
            logits = model(x, input_pos)
        # print(f"x: {x},\n  input_pos: {input_pos}\n")
        return self.sample(logits, need_probs=need_probs, **sampling_kwargs)

    """
    Decode the next n tokens.

    Yields a tuple of (token, prob) for each token.
    """

    def decode_n_tokens(
        self,
        model: Model,
        cur_token: torch.Tensor,
        input_pos: torch.Tensor,
        num_new_tokens: int,
        need_probs: bool,
        batch=Optional[Dict[str, Any]],  # Inputs for multimodal models
        callback=lambda _: _,
        eos_token_id: int = 2,
        eot_id: Optional[int] = None,
        **sampling_kwargs,
    ):
        new_tokens, new_probs = [], []
        encountered_eos = False
        for _i in range(
            num_new_tokens - 1
        ):  # -1 to save space to run an EoS if dont generate it naturally
            # Actually better for Inductor to codegen attention here
            with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):

                out_token = cur_token.clone()
                next_token, next_prob = self.decode_one_token(
                    model,
                    out_token,
                    input_pos,
                    batch=batch,
                    need_probs=need_probs,
                    **sampling_kwargs,
                )
                input_pos += 1
                new_tokens.append(next_token.clone())
                callback(new_tokens[-1], done_generating=_i == num_new_tokens - 2)
                if need_probs or next_prob is None:
                    yield out_token, None
                else:
                    new_probs.append(next_prob.clone())
                    yield out_token, next_prob.clone()
                cur_token = next_token

                # encountered eos
                if next_token.item() == eos_token_id or (
                    eot_id is not None and next_token.item() == eot_id
                ):
                    encountered_eos = True
                    final_token, next_prob = self.decode_one_token(
                        model,
                        cur_token,
                        input_pos,
                        need_probs,
                        batch=batch,
                        **sampling_kwargs,
                    )
                    input_pos += 1
                    break

        if not encountered_eos:
            eos_token = torch.tensor(
                [eos_token_id if eot_id is None else eot_id],
                dtype=cur_token.dtype,
                device=cur_token.device,
            )
            new_tokens.append(eos_token.clone())
            eos_token, next_prob = self.decode_one_token(
                model,
                eos_token.view(1, -1),
                input_pos,
                need_probs,
                batch=batch,
                **sampling_kwargs,
            )
            input_pos += 1
            yield eos_token.clone(), (
                next_prob.clone() if next_prob is not None else None
            )

    def model_forward(self, model, x, input_pos):
        return model(x, input_pos)

    def speculative_decode(
        self,
        model: Model,
        draft_model: Model,
        cur_token: torch.Tensor,
        input_pos: int,
        speculate_k: int,
        batch: Optional[Dict[str, Any]] = None,  # Inputs for multimodal models
        **sampling_kwargs,
    ) -> torch.Tensor:
        # draft model inference sequentially
        device = cur_token.device
        orig_input_pos = torch.tensor(
            [input_pos], dtype=torch.int64, device=cur_token.device
        )
        draft_tokens, draft_probs = self.decode_n_tokens(
            draft_model,
            cur_token,
            orig_input_pos.clone(),
            speculate_k,
            batch=batch,
            need_probs=True,
            **sampling_kwargs,
        )

        draft_tokens = torch.cat(draft_tokens)
        # parallel inference on target model using draft tokens
        target_logits = self.model_forward(
            model,
            torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
            torch.arange(
                input_pos, input_pos + speculate_k + 1, device=cur_token.device
            ),
        )
        target_probs = self.logits_to_probs(target_logits[0], **sampling_kwargs)
        draft_probs = torch.stack(draft_probs)
        # q: target prob, p: draft prob
        # q >= p: always accept draft token
        # q < p: q/p prob to accept draft token
        p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
        q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
        accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k] / p)
        rejected_locations = (
            torch.rand_like(accept_draft_prob) > accept_draft_prob
        ).nonzero()

        if rejected_locations.shape[0] == 0:  # All draft tokens have been accepted
            accept_length = speculate_k + 1
            last_token = self.multinomial_sample_one_no_sync(target_probs[-1])
            # fill last token into draft model
            self.model_forward(
                draft_model,
                draft_tokens[-1].view(1, -1),
                orig_input_pos + speculate_k,
            )
            return torch.cat([draft_tokens, last_token])
        else:
            accept_length = rejected_locations[0].item()
            p = draft_probs[accept_length]
            q = target_probs[accept_length]
            new = q - p
            new = torch.where(new > 0, new, 0.0)
            new = new / new.sum()
            next_token = self.multinomial_sample_one_no_sync(new)
            return torch.cat([draft_tokens[:accept_length], next_token])

    @override
    @torch.no_grad()
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
        """
        Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
        """
        if seed:
            torch.manual_seed(seed)

        is_speculative = self.draft_model is not None
        device = prompt.device

        if len(prompt.shape) > 1:
            prompt = prompt.squeeze(0)
        prompt_length = prompt.size(0)
        max_new_tokens = min(max_new_tokens, max_seq_length - start_pos - prompt_length)
        # set up caches only if first inference
        if start_pos == 0:
            self.model = self.model.to(device=device)
            with torch.device(device):
                if (
                    self.is_torchtune_model
                    or self.model.config.model_type == ModelType.Flamingo
                ):
                    # 6404 is one-gpu affordable max_seq_length for single image input
                    self.model.setup_caches(
                        batch_size=1,
                        dtype=self.dtype,
                        encoder_max_seq_len=6404,
                        decoder_max_seq_len=max_seq_length,
                    )
                else:
                    self.model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)
                if is_speculative and self.draft_model is not self.model:
                    self.draft_model.setup_caches(
                        max_batch_size=1,
                        max_seq_length=max_seq_length,
                    )
            if self.model.config.model_type == ModelType.Flamingo:
                self.model.reset_caches()

        input_pos = torch.arange(
            start_pos, prompt_length + start_pos, device=device, dtype=torch.int
        )

        prefill_t0 = time.perf_counter()
        next_token = self.prefill(
            self.model,
            prompt.view(1, -1),
            input_pos,
            batch=batch,
            sequential_prefill=sequential_prefill,
            **sampling_kwargs,
        )
        if is_speculative:
            self.prefill(
                self.draft_model,
                prompt.view(1, -1),
                input_pos,
                sequential_prefill=sequential_prefill,
                **sampling_kwargs,
            )

        time_to_first_token = time.perf_counter() - prefill_t0
        yield None, {"time_to_first_token": time_to_first_token}
        # max_new_tokens <= 2 means we are effectively not calling decode_n_tokens().
        callback(next_token.clone().view(-1), done_generating=max_new_tokens <= 2)

        input_pos = torch.tensor(
            [start_pos + prompt_length], device=device, dtype=torch.int
        )
        accept_counts = [0] * (
            speculate_k + 1
        )  # creates array of [0, 0, 0, ...] that is speculate_k + 1 long

        if is_speculative:
            input_pos = (
                input_pos.item()
            )  # for speculative decoding easier to keep on host
            while input_pos < max_new_tokens - 1:
                cur_token = next_token.view(())

                next_tokens = self.speculative_decode(
                    self.model,
                    self.draft_model,
                    cur_token,
                    input_pos,
                    speculate_k,
                    batch=batch,
                    **sampling_kwargs,
                )

                accept_counts[len(next_tokens) - 1] += 1
                num_added = min(max_new_tokens - input_pos - 1, len(next_tokens))
                for token in next_tokens[:num_added,]:
                    callback(token)
                    yield token, None
                input_pos = input_pos + num_added
                next_token = next_tokens[-1]
        else:
            generated_tokens = []
            for generated_token, _ in self.decode_n_tokens(
                self.model,
                next_token,
                input_pos,
                max_new_tokens - 1,
                batch=batch,
                callback=callback,
                need_probs=False,
                eos_token_id=self.tokenizer.eos_id() if self.tokenizer else 2,
                eot_id=(
                    self.tokenizer.special_tokens["<|eot_id|>"]
                    if self.is_llama3_model
                    else None
                ),
                **sampling_kwargs,
            ):
                generated_tokens.append(generated_token.view(-1))
                yield generated_token, None

        generate_stats = {
            "accept_counts": accept_counts,
        }
        yield None, generate_stats
    
    @override
    def is_text_only(self) -> bool:
        return self.model.config.model_type != ModelType.Flamingo

    @override
    @property
    def max_seq_length(self) -> int:
        """
        Returns the maximum sequence length supported by the model.
        """
        # This is a hack to get around the fact that different models have different ways to record their max_seq_length and might be wrong
        # TODO: unify the max_seq_length config representation.
        text_transformer_args = self.model.text_transformer_args
        max_seq_length = (
            text_transformer_args.max_seq_length if text_transformer_args else 2048
        )
        return max_seq_length
    
    @override
    @property
    def model_size(self) -> int:
        """
        Returns the size of the model 
        """
        return sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(self.model.parameters(), self.model.buffers())
            ]
        )

    def apply_compile_to_models(self, generator_args: GeneratorArgs):
        if (
            self.is_speculative and self.builder_args.use_distributed
        ):  # and ("cuda" in builder_args.device):
            torch._inductor.config.triton.cudagraph_trees = (
                False  # Bug with cudagraph trees in this case
            )

        if self.builder_args.device == "cpu":
            if generator_args.max_autotune:
                kwargs = {"mode": "max-autotune"}
            else:
                kwargs = {}
        else:
            kwargs = {"mode": "reduce-overhead"}

        if self.is_speculative:
            self.model_forward = torch.compile(
                self.model_forward, fullgraph=True, **kwargs
            )

        if self.model.config.model_type == ModelType.Flamingo:
            # Based on https://github.com/pytorch/torchtune/blob/57ab583c84c4a9dcacac23aeabc81f2a679670fe/torchtune/training/_compile.py#L42-L52
            from torchtune.modules import (
                TransformerCrossAttentionLayer,
                TransformerSelfAttentionLayer,
            )

            decoder = self.model.model.decoder
            for m in reversed(list(decoder.modules())):
                if isinstance(m, TransformerSelfAttentionLayer) or isinstance(
                    m, TransformerCrossAttentionLayer
                ):
                    m.compile()
        else:
            self.decode_one_token = torch.compile(
                self.decode_one_token, fullgraph=True, **kwargs
            )

        if generator_args.compile_prefill:
            self.prefill = torch.compile(
                self.prefill, fullgraph=True, dynamic=True, **kwargs
            )

def create_generator(
    args: Namespace,
) -> Generator:
    """
    Factory function to create a Generator object.
    """

    builder_args = BuilderArgs.from_args(args)
    speculative_builder_args = BuilderArgs.from_speculative_args(args)
    tokenizer_args = TokenizerArgs.from_args(args)
    generator_args = GeneratorArgs.from_args(args)

    if builder_args.distributed:
        return DistributedGenerator(
            args.model,
            builder_args,
            tokenizer_args,
            generator_args,
            args.profile,
            args.quantize,
            args.draft_quantize,
            )
    else:
        return LocalGenerator(
            builder_args,
            speculative_builder_args,
            tokenizer_args,
            generator_args,
            args.profile,
            args.quantize,
            args.draft_quantize,
            )


def main(args):
    gen = create_generator(args)

    generator_args = GeneratorArgs.from_args(args)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for _ in gen.chat(generator_args):
        pass
    
    gen.shutdown()
