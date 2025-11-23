# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    num_block=0
    predicted_length=gen_length

   # for num_block in range(num_blocks):
    while num_block < num_blocks:
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if factor is None:
                x0_initial,confidence_initial, x0, transfer_index, confidence = get_transfer_index_initial(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]

            if i==0:              
                unmask_length=prompt.shape[1] + num_block * block_length
                response_tokens_x0 = x0_initial[0, unmask_length:]
                response_confidence=confidence_initial[0, unmask_length:]
                
                special_tokens_126081 = (response_tokens_x0 == 126081)
                special_tokens_126081[response_confidence < 0.9] = False
                special_tokens_126081[:min(128,gen_length)]= False
                
                special_tokens_17188 = (response_tokens_x0 == 17188)
                special_tokens_17188[response_confidence < 0.9] = False
                special_tokens = special_tokens_126081 | special_tokens_17188
              
                if torch.any(special_tokens):
                    first_special_token_position = torch.nonzero(special_tokens, as_tuple=True)[0][0].item()
                    gen_length =  ((first_special_token_position  + block_length) // block_length) * block_length
                    total_length=min(x.shape[1],gen_length+unmask_length)
                    x = x[:, :total_length]  
                    num_blocks = (total_length-prompt.shape[1])// block_length
                    predicted_length=total_length -  prompt.shape[1]   

            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break

        num_block += 1

    return x, nfe,predicted_length



@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    num_block=0
    predicted_length=gen_length
            
    #for num_block in range(num_blocks):
    while num_block < num_blocks:
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0_initial,confidence_initial, x0, transfer_index, confidence = get_transfer_index_initial(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]
        
        unmask_length=prompt.shape[1] + num_block * block_length
        response_tokens_x0 = x0_initial[0, unmask_length:]
        response_confidence=confidence_initial[0, unmask_length:]

        special_tokens_126081 = (response_tokens_x0 == 126081)
        special_tokens_126081[response_confidence < 0.95] = False
        special_tokens_126081[:min(128,gen_length)]= False

        special_tokens_17188 = (response_tokens_x0 == 17188)
        special_tokens_17188[response_confidence < 0.95] = False
        special_tokens = special_tokens_126081 | special_tokens_17188               
        if torch.any(special_tokens):
            first_special_token_position = torch.nonzero(special_tokens, as_tuple=True)[0][0].item()
            gen_length =  ((first_special_token_position  + block_length) // block_length) * block_length
            total_length=min(x.shape[1],gen_length+unmask_length)
            x = x[:, :total_length]  
            num_blocks = (total_length-prompt.shape[1])// block_length
            predicted_length=total_length -  prompt.shape[1]  

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            i += 1

        num_block += 1



    return x, nfe,predicted_length


@ torch.no_grad()
def generate_with_dual_cache(model, task, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0  
    num_block=0
    predicted_length=gen_length
    predict_total=[]
    
    while num_block < num_blocks:

        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # cache init and update
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0_initial,confidence_initial, x0, transfer_index, confidence = get_transfer_index_initial(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        unmask_length=prompt.shape[1] + num_block * block_length
        response_tokens_x0 = x0_initial[0, unmask_length:]
        response_confidence=confidence_initial[0, unmask_length:]

        if task == "gsm8k":
            special_tokens_126081 = (response_tokens_x0 == 126081)
            special_tokens_126081[response_confidence < 0.93] = False
            special_tokens_126081[:min(128,gen_length)]= False

            special_tokens_17188 = (response_tokens_x0 == 17188)
            special_tokens_17188[response_confidence < 0.93] = False
            special_tokens = special_tokens_126081 | special_tokens_17188
        elif task == "minerva_math":
            special_tokens126081 = (response_tokens_x0 == 126081)   
            special_tokens126081[response_confidence < 0.8] = False
            #special_tokens126081[:min(128,gen_length)]= False
                    
            special_tokens38649 = (response_tokens_x0 == 38649)    
            special_tokens38649[response_confidence < 0.8] = False
            special_tokens = special_tokens126081|special_tokens38649  
        elif task == "bbh":
            special_tokens126081 = (response_tokens_x0 == 126081)   
            special_tokens126081[response_confidence < 0.80] = False
            #special_tokens126081[:min(128,gen_length)]= False
                    
            special_tokens48 = (response_tokens_x0 == 48)    
            special_tokens48[response_confidence < 0.80] = False
            
            special_tokens_double_198 = torch.zeros_like(special_tokens126081, dtype=torch.bool)
            if len(response_tokens_x0) > 1:
                is_198 = (response_tokens_x0 == 198)
                is_198[response_confidence< 0.80] = False
                if len(response_tokens_x0) > 1:
                    next_is_198 = torch.zeros_like(is_198, dtype=torch.bool)
                    next_one_confidence= torch.zeros_like(is_198, dtype=torch.bool)
                    next_is_198[:-1] = is_198[1:]
                    next_one_confidence[:-1] = response_confidence[1:]
                    next_is_198[next_one_confidence < 0.80] = False
                    special_tokens_double_198[:-1] = is_198[:-1] & next_is_198[:-1]
                    if len(response_tokens_x0) > 1:
                        special_tokens_double_198[-1] = is_198[-1] & is_198[-2]  
            special_tokens = special_tokens126081|special_tokens48| special_tokens_double_198 
        elif task == "humaneval":
            special_tokens_126081 = (response_tokens_x0 == 126081)
            special_tokens_126081[response_confidence < 0.85] = False
            #special_tokens_126081[:min(128,gen_length)]= False

            special_tokens_198 = torch.zeros_like(special_tokens_126081, dtype=torch.bool)
            target_tokens = torch.tensor([2439, 1413, 2, 7442, 371, 3384], device=response_tokens_x0.device)
            if len(response_tokens_x0) > 1:
                is_198 = (response_tokens_x0[:-1] == 198)
                is_198[response_confidence[:-1]< 0.85] = False
                next_token_in_targets = torch.any(response_tokens_x0[1:].unsqueeze(1) == target_tokens, dim=1)
                next_token_in_targets[response_confidence[1:]< 0.85] = False
                special_tokens_198[:-1] = is_198 & next_token_in_targets
            special_tokens = special_tokens_126081 | special_tokens_198
        elif task == "mbpp":
            special_tokens126081 = (response_tokens_x0 == 126081)   
            special_tokens126081[response_confidence < 0.9] = False
            special_tokens126081[:min(128,gen_length)]= False
                    
            special_tokens103193 = (response_tokens_x0 == 103193)    
            special_tokens103193[response_confidence < 0.9] = False
            special_tokens = special_tokens126081|special_tokens103193

        if torch.any(special_tokens):
            first_special_token_position = torch.nonzero(special_tokens, as_tuple=True)[0][0].item()
            gen_length =  ((first_special_token_position  + block_length) // block_length) * block_length
            total_length=min(x.shape[1],gen_length+unmask_length)
            x = x[:, :total_length]  
            num_blocks = (total_length-prompt.shape[1])// block_length
            predicted_length=total_length -  prompt.shape[1]  
                         
            new_past_key_values = []
            for i in range(len(past_key_values)):
                new_past_key_values.append(())
                for j in range(len(past_key_values[i])):
                    new_past_key_values[i] += (past_key_values[i][j][:, :, :x.shape[1]],)
        
            past_key_values = new_past_key_values
        predict_total.append(predicted_length)    
        nfe += 1

        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1
        while True:
            if (x[0, current_block_start:current_block_end] == mask_id).sum() == 0:
                
                if task == "gsm8k":
                    special_tokens = (x[0, current_block_start:current_block_end]== 126081) | (x[0, current_block_start:current_block_end]== 17188)
                elif task == "minerva_math":
                    special_tokens = (x[0, current_block_start:current_block_end] == 126081) | (x[0, current_block_start:current_block_end] == 38649)
                elif task == "bbh":
                    special_tokens_126081 = (x[0, current_block_start:current_block_end]== 126081)
                    special_tokens_48 = (x[0, current_block_start:current_block_end] == 48)
                    special_tokens_double_198 = torch.zeros_like(special_tokens_126081, dtype=torch.bool)
                    is_198 = (x[0, current_block_start:current_block_end] == 198)
                    next_is_198 = torch.zeros_like(is_198, dtype=torch.bool)
                    next_is_198[:-1] = is_198[1:]
                    special_tokens_double_198[:-1] = is_198[:-1] & next_is_198[:-1]
                    special_tokens_double_198[-1] = is_198[-1] & is_198[-2]
                    special_tokens = special_tokens_126081 | special_tokens_48 | special_tokens_double_198
                elif task == "humaneval":
                    special_tokens_126081 = (x[0, current_block_start:current_block_end]== 126081)
                    special_tokens_198 = torch.zeros_like(special_tokens_126081, dtype=torch.bool)
                    target_tokens = torch.tensor([2439, 1413, 2, 7442, 371, 3384], device=response_tokens_x0.device)
                    is_198 = (x[0, current_block_start:current_block_end][:-1] == 198)
                    next_token_in_targets = torch.any(x[0, current_block_start:current_block_end][1:].unsqueeze(1) == target_tokens, dim=1)
                    special_tokens_198[:-1] = is_198 & next_token_in_targets
                    special_tokens = special_tokens_126081 | special_tokens_198
                elif task == "mbpp":
                    special_tokens = (x[0, current_block_start:current_block_end]== 126081) | (x[0, current_block_start:current_block_end]== 103193)
                
                if torch.any(special_tokens):
                    first_special_token_position = torch.nonzero(special_tokens, as_tuple=True)[0][0].item()
                    decode_length=first_special_token_position+num_block*block_length+1
                    x[:, current_block_end: ] = 126081
                    decode_length_include_eos=x.shape[1]-prompt.shape[1]
                    average_predict_length = sum(predict_total) / len(predict_total)
                    return x, nfe, average_predict_length, decode_length
                break
            nfe += 1
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            # cache position is the position between current_block_start and current_block_end
            logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values, use_cache=True, replace_position=replace_position).logits

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:current_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:current_block_end], None, factor)
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            i += 1
        num_block += 1

    decode_length=decode_length_include_eos=x.shape[1]-prompt.shape[1]
    average_predict_length = sum(predict_total) / len(predict_total)
    return x, nfe, average_predict_length, decode_length



def get_transfer_index_initial(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0_initial = torch.argmax(logits_with_noise, dim=-1)
    x0 = x0_initial.clone()

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0_initial,x0_p, x0, transfer_index, confidence

def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index

def main():
    device = 'cuda'

    model = LLaDAModelLM.from_pretrained('/data/linyewei/models/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/9275bf8f5a5687507189baf4657e91c51b2be338', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('/data/linyewei/models/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/9275bf8f5a5687507189baf4657e91c51b2be338', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    

    out = generate_with_dual_cache(model, input_ids, steps=1024, gen_length=1024, block_length=32, temperature=0., remasking='low_confidence',threshold=0.9)
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])
    print("nfe:",out[1])
    print("average_predicted_length",out[2])
    print("decode_length",out[3])

if __name__ == '__main__':
    main()
