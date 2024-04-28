import numpy as np
import torch
import torch.nn as nn
from torch import optim
import ipdb

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from einops import rearrange

from ltsm.models.embed import DataEmbedding, DataEmbedding_wo_time

from transformers.modeling_utils import PreTrainedModel
from .config import LTSMConfig

class Chronos(PreTrainedModel):

    config_class = LTSMConfig

    # To load the LTSM model from pretrained weight, Run:
    # LTSM.from_pretrained("/home/sl237/ltsm/ltsm_hf/output/ltsm_debug")

    def __init__(self, configs, device=torch.device("cpu")):
        super().__init__(configs)
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len + configs.prompt_len - self.patch_size) // self.stride + 1
        self.d_type = torch.bfloat16
        self.pred_len = configs.pred_len

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        if configs.is_gpt:
            # if configs.local_pretrain != "None":
            #     print("------------------Load pretrain from {}-----------------\n".format(configs.local_pretrain))
            #     self.gpt2 = GPT2Model.from_pretrained(configs.local_pretrain, output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('gpt2-medium',output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------\n")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))

        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)



    def forward(self, x, iters=None):
        # ipdb.set_trace()
        x = x.unsqueeze(-1)

        x = x.int()
        outputs = self.gpt2(input_ids = x).last_hidden_state
        outputs = outputs[:, -self.pred_len:, :]

        return outputs
