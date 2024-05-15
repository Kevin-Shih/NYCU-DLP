import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        codebook_mapping, codebook_indices, _ = self.vqgan.encode(x)
        codebook_indices = codebook_indices.view(codebook_mapping.shape[0], -1)
        return codebook_mapping, codebook_indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda gamma: 1 - gamma
        elif mode == "cosine":
            return lambda gamma: np.cos(gamma * np.pi / 2)
        elif mode == "square":
            return lambda gamma: 1 - gamma ** 2
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        #z_indices: ground truth
        #logits:    transformer predict the probability of tokens
        _, z_indices = self.encode_to_z(x)
        device = z_indices.device
        ratio = math.floor(self.gamma(np.random.uniform()) * z_indices.shape[1])
        mask_index = torch.rand(z_indices.shape, device= device).topk(ratio, dim= 1).indices
        mask = torch.zeros(z_indices.shape, dtype= torch.bool, device= device)
        mask.scatter_(dim= 1, index= mask_index, value= True)
        masked_indices = mask * z_indices + (~mask) * torch.full_like(z_indices, self.mask_token_id)
        logits = self.transformer(masked_indices)
        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding
    @torch.no_grad()
    def inpainting(self):
        raise Exception('TODO3 step1-1!')
        logits = self.transformer(None)
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = None

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = None

        ratio=None 
        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = None  # gumbel noise
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank 
        #define how much the iteration remain predicted tokens by mask scheduling
        #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens
        mask_bc=None
        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
