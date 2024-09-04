# Created LORA+ optimizer as per https://arxiv.org/abs/2402.12354
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils import is_sagemaker_mp_enabled, logging

from dataclasses import dataclass, field
from functools import reduce
from typing import Callable, Dict, List, Optional, Tuple, Union

from peft.tuners import lora
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import (EvalPrediction, PreTrainedModel,
                                  PreTrainedTokenizerBase, TrainerCallback)

import torch
import torch.nn as nn
from torch.utils.data import Dataset

logger = logging.get_logger(__name__)

LORA = "lora"
LORA_PARENT_INDEX = 2
LORA_NONE_INDEX = 1

def create_optimizer(
    opt_model,
    base_optimizer,
    lr_ratio,
    optimizer_kwargs,
):
    """
    Creates an optimizer for the given model, applying LoRA-specific learning rates to different parameter groups.
    
    Args:
        opt_model: The model for which the optimizer is being created.
        optimizer_cls: The class of the optimizer to be used (e.g., torch.optim.Adam).
        optimizer_kwargs: A dictionary of keyword arguments for the optimizer's initialization.
        lr_ratio: The learning rate ratio to be applied to LoRA parameters.
    
    Returns:
        An instance of the optimizer class configured into groups with custom learning rates.
    """
    
    assert lr_ratio is not None, "lr_ratio must be provided."

    lr = optimizer_kwargs["lr"]

    decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    parameters = {
        "groupA": {},
        "groupB": {},
        "groupB_no_decay": {},
        "embedding": {},
    }

    for name, param in opt_model.named_parameters():
        if not param.requires_grad:
            continue

        # Retrieve a module based on the model.
        p_index = LORA_PARENT_INDEX if LORA in name else LORA_NONE_INDEX
        module_names = name.split(sep=".")[:-p_index]
        module = reduce(getattr, module_names, opt_model)
        
        if isinstance(module, lora.Embedding):
            parameters["embedding"][name] = param
        elif "lora_B" in name or param.ndim == 1:
            if name in decay_parameters:
                parameters["groupB"][name] = param
            else:
                parameters["groupB_no_decay"][name] = param
        else:
            parameters["groupA"][name] = param

    assigned_parameters = ""
    for group in parameters:
        assigned_parameters += f"{group}\n {list(parameters[group].keys())}\n\n"
    logger.debug(assigned_parameters)

    weight_decay = 0 # weight decay is always fine to be 0 because of LORA matrix A already starting with zeros

    optimizer_grouped_parameters = [
        {
            "params": list(parameters["groupA"].values()),
            "lr": lr,
            "weight_decay": weight_decay,
        },
        {
            "params": list(parameters["embedding"].values()),
            "lr": None, # We do not want embeddings.
            "weight_decay": weight_decay,
        },
        {
            "params": list(parameters["groupB"].values()),
            "lr": lr_ratio * lr,
            "weight_decay": weight_decay,
        },
        {
            "params": list(parameters["groupB_no_decay"].values()),
            "lr": lr_ratio * lr,
            "weight_decay": 0.0,
        },
    ]

    optimizer = base_optimizer(optimizer_grouped_parameters, **optimizer_kwargs)

    return optimizer

optimizer = create_optimizer(
    opt_model=model,
    base_optimizer=torch.optim.AdamW, # 8 bits adam not supported since there is no need for it
    lr_ratio = 16.0, # As per LORA+ paper set to 16 for balanced dataset and a large model
    optimizer_kwargs = {'betas': (0.9, 0.99), 'lr': 1e-4, 'eps': 1e-6, 'weight_decay': 0.0},
)
