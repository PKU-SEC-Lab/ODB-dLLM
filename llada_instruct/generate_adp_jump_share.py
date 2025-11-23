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
from model.modeling_llada_spec_v2 import LLaDAModelLM
import json
import time

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
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None, token_step_file=None, token_step_file_2 = None, verbose=False, tokenizer =None):
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

    first_126081_x0_step = -1
    first_126081_x0_position = -1
    first_17188_x0_step = -1
    first_17188_x0_position = -1
    
    first_126081_unmask_step = -1
    first_126081_unmask_position = -1
    first_17188_unmask_step = -1
    first_17188_unmask_position = -1

    nfe = 0
    
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0

        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if factor is None:
                x0, transfer_index, confidence = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index, confidence = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index] 
            

            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
        


    return x, nfe, first_126081_x0_step, first_126081_x0_position, first_17188_x0_step, first_17188_x0_position, \
           first_126081_unmask_step, first_126081_unmask_position, first_17188_unmask_step, first_17188_unmask_position


@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None, token_step_file=None, verbose=False, tokenizer =None):
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
            
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index, confidence = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index, confidence = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

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
                x0, transfer_index, confidence = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index, confidence = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            i += 1


    return x, nfe


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
        output = model(x, use_cache=True, current_block_start = current_block_start)
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0_initial,confidence_initial, x0, transfer_index, confidence = get_transfer_index_initial(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index, confidence = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
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
                    new_past_key_values[i] += (past_key_values[i][j][:, :, :x.shape[1]+96],)
            
            past_key_values = new_past_key_values
        predict_total.append(predicted_length)

        i = 1
        nfe += 1

        position_ids = torch.arange(current_block_start, current_block_end, device=model.device )
        position_ids = position_ids.repeat(4)
        position_ids_original =position_ids.clone()

        current_block = x[:, current_block_start:current_block_end].clone()
        block_mask_index = (current_block == mask_id)
        masked_positions = block_mask_index[0]

        if confidence is not None:
            current_confidence = confidence[0, current_block_start:current_block_end]
            
            masked_confidences = torch.where(masked_positions, current_confidence, torch.tensor(-float('inf'), device=current_confidence.device))
            
            _, top2_indices = torch.topk(masked_confidences, min(2, masked_positions.sum().item()), dim=0)
            
            block1 = current_block.clone()
            block2 = current_block.clone()
            block3 = current_block.clone()
            
            if len(top2_indices) >= 1:
                highest_conf_idx = top2_indices[0]
                block1[0, highest_conf_idx] = x0[0, current_block_start + highest_conf_idx]
                
                block3[0, highest_conf_idx] = x0[0, current_block_start + highest_conf_idx]
            
            if len(top2_indices) >= 2:
                second_highest_conf_idx = top2_indices[1]
                block2[0, second_highest_conf_idx] = x0[0, current_block_start + second_highest_conf_idx]
                
                block3[0, second_highest_conf_idx] = x0[0, current_block_start + second_highest_conf_idx]
            
            additional_blocks = torch.cat([block1, block2, block3], dim=1)
        else:
            additional_blocks = current_block.repeat(1, 3)

        x = torch.cat([x[:, :current_block_end], additional_blocks, x[:, current_block_end:]], dim=1)

        spec_end_1 = current_block_end + block_length
        spec_end_2 = current_block_end + 2 * block_length
        spec_end_3 = current_block_end + 3 * block_length
        spec_end_4 = current_block_end + 4 * block_length
        spec_end_5 = current_block_end + 5 * block_length
        spec_end_6 = current_block_end + 6 * block_length
        spec_end_7 = current_block_end + 7 * block_length

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:spec_end_3] = 1


        attention_mask = torch.zeros(1, 1, 128, x.shape[1], device=model.device)
        attention_mask[:, :, :block_length, current_block_end:spec_end_3] = float('-inf')
        attention_mask[:, :, block_length:block_length * 2, current_block_start:current_block_end] = float('-inf')
        attention_mask[:, :, block_length:block_length * 2, spec_end_1:spec_end_3] = float('-inf')
        attention_mask[:, :, block_length * 2:block_length * 3, current_block_start:spec_end_1] = float('-inf')
        attention_mask[:, :, block_length * 2:block_length * 3, spec_end_2:spec_end_3] = float('-inf')
        attention_mask[:, :, block_length * 3:, current_block_start:spec_end_2] = float('-inf')
        attention_mask = attention_mask.to(torch.bfloat16)

        # not really 129, only represent initial state
        decode_length = 129
        decode_length_threshold = 128
        restore_indices = None
        predict_length = None
        new_spec_end_1 = 0
        new_spec_end_2 = 0
        new_spec_end_3 = 0
        new_spec_end_4 = 0
        new_spec_end_5 = 0
        new_spec_end_6 = 0
        new_spec_end_7 = 0
        share_end = 0
        block_share_end = 0
        later_tokens = x[:, spec_end_3:].clone()
        seven_candidate = 0
        seven_threshold = 0
        candidate_restore = None

        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                x = torch.cat([x[:, :current_block_end], later_tokens], dim=1)

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
                    average_predict_length = sum(predict_total) / len(predict_total)
                    return x, nfe, average_predict_length, decode_length
                break
            
            nfe += 1
            
            # not really 129, only represent initial state
            # not really 127, only a mark that there are still many mask tokens
            if decode_length == 129 or decode_length == 127:
                logits = model(x[:, current_block_start:current_block_start+128], attention_mask=attention_mask, 
                           position_ids=position_ids, current_block_start = current_block_start,
                           past_key_values=past_key_values, use_cache=True, 
                           replace_position=replace_position, decode_length=decode_length).logits
            else:
                logits = model(x[:, current_block_start:current_block_start+decode_length], attention_mask=attention_mask, 
                            position_ids=position_ids, current_block_start = current_block_start,
                            past_key_values=past_key_values, use_cache=True, 
                            replace_position=replace_position, decode_length=decode_length).logits
            
            # not really 129, only represent initial state
            # not really 127, only a mark that there are still many mask tokens
            if decode_length == 129 or decode_length == 127:
                mask_index = torch.zeros(x.shape[0], 4, block_length, dtype=torch.bool, device=x.device)
                mask_index[:, 0] = (x[:, current_block_start:current_block_end] == mask_id)
                mask_index[:, 1] = (x[:, current_block_end:spec_end_1] == mask_id)
                mask_index[:, 2] = (x[:, spec_end_1:spec_end_2] == mask_id)
                mask_index[:, 3] = (x[:, spec_end_2:spec_end_3] == mask_id)
            else:
                x = torch.cat([x[:, :current_block_start], candidate_restore, x[:, new_spec_end_7:]], dim=1)
                mask_index = torch.zeros(x.shape[0], 8, block_length, dtype=torch.bool, device=x.device)
                mask_index[:, 0] = (x[:, current_block_start:current_block_end] == mask_id)
                mask_index[:, 1] = (x[:, current_block_end:spec_end_1] == mask_id)
                mask_index[:, 2] = (x[:, spec_end_1:spec_end_2] == mask_id)
                mask_index[:, 3] = (x[:, spec_end_2:spec_end_3] == mask_id)
                mask_index[:, 4] = (x[:, spec_end_3:spec_end_4] == mask_id)
                mask_index[:, 5] = (x[:, spec_end_4:spec_end_5] == mask_id)
                mask_index[:, 6] = (x[:, spec_end_5:spec_end_6] == mask_id)
                mask_index[:, 7] = (x[:, spec_end_6:spec_end_7] == mask_id)

            if factor is None:
                # not really 129, only represent initial state
                # not really 127, only a mark that there are still many mask tokens
                if decode_length == 129 or decode_length == 127:
                    x0, transfer_index, confidence = get_transfer_index_3(logits, temperature, remasking, mask_index, 
                                                    x[:, current_block_start:spec_end_3], num_transfer_tokens[:, i] if threshold is None else None, threshold)
                else:
                    # if seven_candidate == 1:
                    x0, transfer_index, confidence = get_transfer_index_5(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:spec_end_7], num_transfer_tokens[:, i] if threshold is None else None, threshold, block_share_end, predict_length, restore_indices)
            # else:
            #     x0, transfer_index, confidence = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
            #                                     x[:, current_block_start:current_block_end], None, factor)
           
            if seven_candidate == 1:
                accept_1 = False
                accept_2 = False
                accept_12 = False
                accept_3 = False
                accept_123 = False
                accept_4 = False
                accept_1234 = False
                if len(top4_indices) >= 1:
                    highest_conf_idx_1 = top4_indices[0]
                    if transfer_index[0, 0, highest_conf_idx_1]:
                        if x0[0, 0, highest_conf_idx_1] == x0[0, 1, highest_conf_idx_1]:
                            accept_1 = True
                    if len(top4_indices) >= 2:
                        second_highest_conf_idx_1 = top4_indices[1]
                        if transfer_index[0, 0, second_highest_conf_idx_1]:
                            if x0[0, 0, second_highest_conf_idx_1] == x0[0, 2, second_highest_conf_idx_1]:
                                accept_2 = True
                                if transfer_index[0, 2, highest_conf_idx_1]:
                                    if x0[0, 2, highest_conf_idx_1] == x0[0, 1, highest_conf_idx_1]:
                                        accept_12 = True
                        if accept_1 == True:
                            if transfer_index[0, 1, second_highest_conf_idx_1]:
                                if x0[0, 1, second_highest_conf_idx_1] == x0[0, 2, second_highest_conf_idx_1]:
                                    accept_12 = True
                        if len(top4_indices) >= 3:
                            third_highest_conf_idx_1 = top4_indices[2]
                            if transfer_index[0, 0, third_highest_conf_idx_1]:
                                if x0[0, 0, third_highest_conf_idx_1] == x0[0, 4, third_highest_conf_idx_1]:
                                    accept_3 = True
                                    if accept_12 == True:
                                        accept_123 = True
                                    elif transfer_index[0, 4, highest_conf_idx_1] and transfer_index[0, 4, second_highest_conf_idx_1]:
                                        if x0[0, 4, highest_conf_idx_1] == x0[0, 1, highest_conf_idx_1] and x0[0, 4, second_highest_conf_idx_1] == x0[0, 2, second_highest_conf_idx_1]:
                                            accept_123 = True
                            if accept_12 == True:
                                if transfer_index[0, 3, third_highest_conf_idx_1]:
                                    if x0[0, 3, third_highest_conf_idx_1] == x0[0, 4, third_highest_conf_idx_1]:
                                        accept_123 = True
                                if transfer_index[0, 1, third_highest_conf_idx_1]:
                                    if x0[0, 1, third_highest_conf_idx_1] == x0[0, 4, third_highest_conf_idx_1]:
                                        accept_123 = True
                                if transfer_index[0, 2, third_highest_conf_idx_1]:
                                    if x0[0, 2, third_highest_conf_idx_1] == x0[0, 4, third_highest_conf_idx_1]:
                                        accept_123 = True
                        if len(top4_indices) == 4:
                            fourth_highest_conf_idx_1 = top4_indices[3]
                            if transfer_index[0, 0, fourth_highest_conf_idx_1]:
                                if x0[0, 0, fourth_highest_conf_idx_1] == x0[0, 6, fourth_highest_conf_idx_1]:
                                    accept_4 = True
                                    if accept_123 == True:
                                        accept_1234 = True
                                    if transfer_index[0, 6, highest_conf_idx_1] and transfer_index[0, 6, second_highest_conf_idx_1] and transfer_index[0, 6, third_highest_conf_idx_1]:
                                        if x0[0, 6, highest_conf_idx_1] == x0[0, 1, highest_conf_idx_1] and x0[0, 6, second_highest_conf_idx_1] == x0[0, 2, second_highest_conf_idx_1] and x0[0, 6, third_highest_conf_idx_1] == x0[0, 4, third_highest_conf_idx_1]:
                                            accept_1234 = True
                                    if transfer_index[0, 6, third_highest_conf_idx_1] and accept_12:
                                        if x0[0, 6, third_highest_conf_idx_1] == x0[0, 4, highest_conf_idx_1]:
                                            accept_1234 = True 
                            if accept_123 == True:
                                if transfer_index[0, 5, fourth_highest_conf_idx_1]:
                                    if x0[0, 5, fourth_highest_conf_idx_1] == x0[0, 6, fourth_highest_conf_idx_1]:
                                        accept_1234 = True
                                if transfer_index[0, 4, fourth_highest_conf_idx_1]:
                                    if x0[0, 4, fourth_highest_conf_idx_1] == x0[0, 6, fourth_highest_conf_idx_1]:
                                        accept_1234 = True
                                if transfer_index[0, 3, fourth_highest_conf_idx_1]:
                                    if x0[0, 3, fourth_highest_conf_idx_1] == x0[0, 6, fourth_highest_conf_idx_1]:
                                        accept_1234 = True
                                if transfer_index[0, 2, fourth_highest_conf_idx_1]:
                                    if x0[0, 2, fourth_highest_conf_idx_1] == x0[0, 6, fourth_highest_conf_idx_1]:
                                        accept_1234 = True
                                if transfer_index[0, 1, fourth_highest_conf_idx_1]:
                                    if x0[0, 1, fourth_highest_conf_idx_1] == x0[0, 6, fourth_highest_conf_idx_1]:
                                        accept_1234 = True
                
                if accept_1234 or (accept_1 and accept_2 and accept_3 and accept_4):
                    choice_id = 7
                elif accept_123 or (accept_1 and accept_2 and accept_3):
                    choice_id = 5
                elif (accept_1 and accept_2) or accept_12:
                    choice_id = 3
                elif accept_1:
                    choice_id = 1
                elif accept_2:
                    choice_id = 2
                elif accept_3:
                    choice_id = 4
                elif accept_4:
                    choice_id = 6
                else:
                    choice_id = 0
                
                x[:, current_block_start:current_block_end] = x[:, current_block_start + choice_id * block_length:current_block_end + choice_id * block_length]
                x[:, current_block_start:current_block_end][transfer_index[:, choice_id]] = x0[:, choice_id][transfer_index[:, choice_id]]
                later_tokens = x[:, spec_end_7:].clone() 
            else:
                accept_1 = False
                accept_2 = False
                accept_12 = False
                accept_21 = False
                if len(top2_indices) >= 1:
                    highest_conf_idx_1 = top2_indices[0]
                    if transfer_index[0, 0, highest_conf_idx_1]:
                        if x0[0, 0, highest_conf_idx_1] == x0[0, 1, highest_conf_idx_1]:
                            accept_1 = True
                    if len(top2_indices) == 2:
                        second_highest_conf_idx_1 = top2_indices[1]
                        if transfer_index[0, 0, second_highest_conf_idx_1]:
                            if x0[0, 0, second_highest_conf_idx_1] == x0[0, 2, second_highest_conf_idx_1]:
                                accept_2 = True
                                if transfer_index[0, 2, highest_conf_idx_1]:
                                    if x0[0, 2, highest_conf_idx_1] == x0[0, 1, highest_conf_idx_1]:
                                        accept_21 = True
                        if accept_1 == True:
                            if transfer_index[0, 1, second_highest_conf_idx_1]:
                                if x0[0, 1, second_highest_conf_idx_1] == x0[0, 2, second_highest_conf_idx_1]:
                                    accept_12 = True
                
                if (accept_1 and accept_2) or accept_12 or accept_21:
                    choice_id = 3
                    seven_threshold = 92
                elif accept_1:
                    choice_id = 1
                    seven_threshold = 89
                elif accept_2:
                    choice_id = 2
                    seven_threshold = 89
                else:
                    choice_id = 0
                    seven_threshold = 86
                
                x[:, current_block_start:current_block_end] = x[:, current_block_start + choice_id * block_length:current_block_end + choice_id * block_length]
                x[:, current_block_start:current_block_end][transfer_index[:, choice_id]] = x0[:, choice_id][transfer_index[:, choice_id]]
            
            if decode_length_threshold <= seven_threshold or seven_candidate == 1:
                if (x[:, current_block_start:current_block_end] == mask_id).sum() != 0:
                    seven_candidate = 1
                    
                    current_block = x[:, current_block_start:current_block_end].clone()
                    block_mask_index = (current_block == mask_id)
                    masked_positions = block_mask_index[0]
                    
                    if x.shape[1] < spec_end_7:
                        n_missing = spec_end_7 - x.shape[1] + 1
                        zeros = torch.zeros(1, n_missing, dtype=x.dtype, device=x.device)
                        x = torch.cat([x, zeros], dim=1)
                    
                    if confidence is not None:
                        current_confidence = confidence[0, choice_id]
                        
                        masked_confidences = torch.where(masked_positions, current_confidence, torch.tensor(-float('inf'), device=current_confidence.device))
                        
                        _, top4_indices = torch.topk(masked_confidences, min(4, masked_positions.sum().item()), dim=0)
                        
                        x[:, current_block_end:spec_end_1] = current_block
                        x[:, spec_end_1:spec_end_2] = current_block
                        x[:, spec_end_2:spec_end_3] = current_block
                        x[:, spec_end_3:spec_end_4] = current_block
                        x[:, spec_end_4:spec_end_5] = current_block
                        x[:, spec_end_5:spec_end_6] = current_block
                        x[:, spec_end_6:spec_end_7] = current_block
                        
                        if len(top4_indices) >= 1:
                            highest_conf_idx_2 = top4_indices[0]
                            x[0, current_block_end + highest_conf_idx_2] = x0[0, choice_id, highest_conf_idx_2]
                            
                            x[0, spec_end_2 + highest_conf_idx_2] = x0[0, choice_id, highest_conf_idx_2]
                            x[0, spec_end_4 + highest_conf_idx_2] = x0[0, choice_id, highest_conf_idx_2]
                            x[0, spec_end_6 + highest_conf_idx_2] = x0[0, choice_id, highest_conf_idx_2]
                        
                        if len(top4_indices) >= 2:
                            second_highest_conf_idx_2 = top4_indices[1]
                            x[0, spec_end_1 + second_highest_conf_idx_2] = x0[0, choice_id, second_highest_conf_idx_2]
                            
                            x[0, spec_end_2 + second_highest_conf_idx_2] = x0[0, choice_id, second_highest_conf_idx_2]
                            x[0, spec_end_4 + second_highest_conf_idx_2] = x0[0, choice_id, second_highest_conf_idx_2]
                            x[0, spec_end_6 + second_highest_conf_idx_2] = x0[0, choice_id, second_highest_conf_idx_2]
                        
                        if len(top4_indices) >= 3:
                            third_highest_conf_idx_2 = top4_indices[2]
                            x[0, spec_end_3 + third_highest_conf_idx_2] = x0[0, choice_id, third_highest_conf_idx_2]
                            
                            x[0, spec_end_4 + third_highest_conf_idx_2] = x0[0, choice_id, third_highest_conf_idx_2]
                            x[0, spec_end_6 + third_highest_conf_idx_2] = x0[0, choice_id, third_highest_conf_idx_2]
                        
                        if len(top4_indices) >= 4:
                            fourth_highest_conf_idx_2 = top4_indices[3]
                            x[0, spec_end_5 + fourth_highest_conf_idx_2] = x0[0, choice_id, fourth_highest_conf_idx_2]
                            
                            x[0, spec_end_6 + fourth_highest_conf_idx_2] = x0[0, choice_id, fourth_highest_conf_idx_2]
            
                    else:
                        x[:, current_block_end:spec_end_1] = current_block
                        x[:, spec_end_1:spec_end_2] = current_block
                        x[:, spec_end_2:spec_end_3] = current_block
                        x[:, spec_end_3:spec_end_4] = current_block
                        x[:, spec_end_4:spec_end_5] = current_block
                        x[:, spec_end_5:spec_end_6] = current_block
                        x[:, spec_end_6:spec_end_7] = current_block
                    
                    candidate_restore = x[:, current_block_start:spec_end_7].clone()
                    
                    current_block_tokens = x[:, current_block_start:current_block_end].clone()
                    current_position_ids = position_ids_original[:32].clone()
                    block_tokens_1 = x[:, current_block_end:spec_end_1].clone()
                    block_tokens_2 = x[:, spec_end_1:spec_end_2].clone()
                    block_tokens_3 = x[:, spec_end_2:spec_end_3].clone()
                    block_tokens_4 = x[:, spec_end_3:spec_end_4].clone()
                    block_tokens_5 = x[:, spec_end_4:spec_end_5].clone()
                    block_tokens_6 = x[:, spec_end_5:spec_end_6].clone()
                    block_tokens_7 = x[:, spec_end_6:spec_end_7].clone()

                    is_mask = (current_block_tokens == mask_id)

                    non_mask_tokens = current_block_tokens[:, ~is_mask[0]]
                    mask_tokens = current_block_tokens[:, is_mask[0]]
                    mask_block_1 = block_tokens_1[:, is_mask[0]]
                    mask_block_2 = block_tokens_2[:, is_mask[0]]
                    mask_block_3 = block_tokens_3[:, is_mask[0]]
                    mask_block_4 = block_tokens_4[:, is_mask[0]]
                    mask_block_5 = block_tokens_5[:, is_mask[0]]
                    mask_block_6 = block_tokens_6[:, is_mask[0]]
                    mask_block_7 = block_tokens_7[:, is_mask[0]]

                    non_mask_positions = current_position_ids[~is_mask[0]]
                    mask_positions = current_position_ids[is_mask[0]]
                    predict_length = mask_positions.shape[0]
                    decode_length = block_length + predict_length * 7
                    
                    share_end = current_block_end - predict_length
                    block_share_end = 32 - predict_length
                    new_spec_end_1 = current_block_end + predict_length
                    new_spec_end_2 = new_spec_end_1 + predict_length
                    new_spec_end_3 = new_spec_end_2 + predict_length
                    new_spec_end_4 = new_spec_end_3 + predict_length
                    new_spec_end_5 = new_spec_end_4 + predict_length
                    new_spec_end_6 = new_spec_end_5 + predict_length
                    new_spec_end_7 = new_spec_end_6 + predict_length

                    reordered_tokens = torch.cat([non_mask_tokens, mask_tokens, mask_block_1, mask_block_2, mask_block_3, mask_block_4, mask_block_5, mask_block_6, mask_block_7], dim=1)
                    reordered_positions = torch.cat([non_mask_positions, mask_positions], dim=0)

                    restore_indices = torch.argsort(reordered_positions - current_block_start)

                    x = torch.cat([x[:, :current_block_start], reordered_tokens, later_tokens], dim=1)
                    additional_position_ids = mask_positions.repeat(7)
                    position_ids = torch.cat([ reordered_positions, additional_position_ids], dim=0)
                    
                    inf_value = torch.finfo(torch.bfloat16).min
                    attention_mask = torch.zeros(1, 1, decode_length, x.shape[1], device=model.device, dtype=torch.bfloat16)
                    attention_mask[:, :, :block_length, current_block_end:new_spec_end_7] = inf_value
                    attention_mask[:, :, block_length:block_length + predict_length, share_end:current_block_end] = inf_value
                    attention_mask[:, :, block_length:block_length + predict_length, new_spec_end_1:new_spec_end_7] = inf_value
                    attention_mask[:, :, block_length + predict_length:block_length + predict_length * 2, share_end:new_spec_end_1] = inf_value
                    attention_mask[:, :, block_length + predict_length:block_length + predict_length * 2, new_spec_end_2:new_spec_end_7] = inf_value
                    attention_mask[:, :, block_length + predict_length * 2:block_length + predict_length * 3, share_end:new_spec_end_2] = inf_value
                    attention_mask[:, :, block_length + predict_length * 2:block_length + predict_length * 3, new_spec_end_3:new_spec_end_7] = inf_value
                    attention_mask[:, :, block_length + predict_length * 3:block_length + predict_length * 4, share_end:new_spec_end_3] = inf_value
                    attention_mask[:, :, block_length + predict_length * 3:block_length + predict_length * 4, new_spec_end_4:new_spec_end_7] = inf_value
                    attention_mask[:, :, block_length + predict_length * 4:block_length + predict_length * 5, share_end:new_spec_end_4] = inf_value
                    attention_mask[:, :, block_length + predict_length * 4:block_length + predict_length * 5, new_spec_end_5:new_spec_end_7] = inf_value
                    attention_mask[:, :, block_length + predict_length * 5:block_length + predict_length * 6, share_end:new_spec_end_5] = inf_value
                    attention_mask[:, :, block_length + predict_length * 5:block_length + predict_length * 6, new_spec_end_6:new_spec_end_7] = inf_value
                    attention_mask[:, :, block_length + predict_length * 6:, share_end:new_spec_end_6] = inf_value

            else: 
                if (x[:, current_block_start:current_block_end] == mask_id).sum() != 0:
                    current_block = x[:, current_block_start:current_block_end].clone()
                    block_mask_index = (current_block == mask_id)
                    masked_positions = block_mask_index[0]
                    
                    mask_num = masked_positions.sum().item()
                    decode_length_threshold = 32 + mask_num * 3
                    # not really 127, only a mark that there are still many mask tokens
                    decode_length = 127

                    if confidence is not None:
                        current_confidence = confidence[0, choice_id]
                        
                        masked_confidences = torch.where(masked_positions, current_confidence, torch.tensor(-float('inf'), device=current_confidence.device))
                        
                        _, top2_indices = torch.topk(masked_confidences, min(2, mask_num), dim=0)
                        
                        x[:, current_block_end:spec_end_1] = current_block
                        x[:, spec_end_1:spec_end_2] = current_block
                        x[:, spec_end_2:spec_end_3] = current_block
                        
                        if len(top2_indices) >= 1:
                            highest_conf_idx_2 = top2_indices[0]
                            x[0, current_block_end + highest_conf_idx_2] = x0[0, choice_id, highest_conf_idx_2]
                            
                            x[0, spec_end_2 + highest_conf_idx_2] = x0[0, choice_id, highest_conf_idx_2]
                        
                        if len(top2_indices) >= 2:
                            second_highest_conf_idx_2 = top2_indices[1]
                            x[0, spec_end_1 + second_highest_conf_idx_2] = x0[0, choice_id, second_highest_conf_idx_2]
                            
                            x[0, spec_end_2 + second_highest_conf_idx_2] = x0[0, choice_id, second_highest_conf_idx_2]
            
                    else:
                        x[:, current_block_end:spec_end_1] = current_block
                        x[:, spec_end_1:spec_end_2] = current_block
                        x[:, spec_end_2:spec_end_3] = current_block
      


            i += 1
            
        num_block += 1

    decode_length = x.shape[1]-prompt.shape[1]
    average_predict_length = sum(predict_total) / len(predict_total)
    return x, nfe, average_predict_length, decode_length


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
    return x0, transfer_index, confidence

def get_transfer_index_2(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, L
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf) #shape: b, L

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device) #shape: b, L
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True) #shape: b, 1
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j]) #shape: k
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index, confidence

def get_transfer_index_3(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_reshaped = logits.view(1, 4, 32, -1)
    x_reshaped = x.view(1, 4, 32)

    logits_with_noise = add_gumbel_noise(logits_reshaped, temperature=temperature) #shape: b, 4, L, v
    x0 = torch.argmax(logits_with_noise, dim=-1) #shape: b, 4, L 

    if remasking == 'low_confidence':
        p = F.softmax(logits_reshaped.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(-1) #shape: b, 4, L
    elif remasking == 'random':
        x0_p = torch.rand((1, 4, 32), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x_reshaped) #shape: b, 4, L
    confidence = torch.where(mask_index, x0_p, -np.inf) #shape: b, 4, L

    max_conf_indices = torch.argmax(confidence, dim=2, keepdim=True)
    confidence.scatter_(2, max_conf_indices,1.0)
    confidence = torch.where(mask_index, confidence, -np.inf) #shape: b, L
    confidence = confidence.to(device=x0_p.device, dtype=x0_p.dtype)

    transfer_index = torch.ones_like(x0, dtype=torch.bool, device=x0.device) #shape: b, L
    transfer_index = confidence >= threshold

    return x0, transfer_index, confidence

def get_transfer_index_4(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None, 
                         block_share_end=None, predict_length=None, restore_indices=None):
    x_reshaped = x.view(1, 4, 32)

    logits_with_noise = add_gumbel_noise(logits, temperature=temperature) #shape: b, L+3L', v
    x0 = torch.argmax(logits_with_noise, dim=-1) #shape: b, L+3L' 

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, L+3L'
        
        x_reorderedp = x0_p[:, :32]
        block1_reorderedp = torch.cat([x0_p[:, :block_share_end], x0_p[:, 32:32+predict_length]], dim=1)
        block2_reorderedp = torch.cat([x0_p[:, :block_share_end], x0_p[:, 32+predict_length:32+predict_length*2]], dim=1)
        block3_reorderedp = torch.cat([x0_p[:, :block_share_end], x0_p[:, 32+predict_length*2:]], dim=1)

        x_restorep = x_reorderedp[:, restore_indices]
        block1_restorep = block1_reorderedp[:, restore_indices]
        block2_restorep = block2_reorderedp[:, restore_indices]
        block3_restorep = block3_reorderedp[:, restore_indices]
        
        x0_p = torch.cat([x_restorep, block1_restorep, block2_restorep, block3_restorep], dim=1)
        x0_p = x0_p.view(1, 4, 32)

    elif remasking == 'random':
        x0_p = torch.rand((1, 4, 32), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x_reordered0 = x0[:, :32]
    block1_reordered0 = torch.cat([x0[:, :block_share_end], x0[:, 32:32+predict_length]], dim=1)
    block2_reordered0 = torch.cat([x0[:, :block_share_end], x0[:, 32+predict_length:32+predict_length*2]], dim=1)
    block3_reordered0 = torch.cat([x0[:, :block_share_end], x0[:, 32+predict_length*2:]], dim=1)

    x_restore0 = x_reordered0[:, restore_indices]
    block1_restore0 = block1_reordered0[:, restore_indices]
    block2_restore0 = block2_reordered0[:, restore_indices]
    block3_restore0 = block3_reordered0[:, restore_indices]
    
    x0 = torch.cat([x_restore0, block1_restore0, block2_restore0, block3_restore0], dim=1)
    x0 = x0.view(1, 4, 32)
    
    x0 = torch.where(mask_index, x0, x_reshaped) #shape: b, 4, L
    confidence = torch.where(mask_index, x0_p, -np.inf) #shape: b, 4, L

    max_conf_indices = torch.argmax(confidence, dim=2, keepdim=True)
    confidence.scatter_(2, max_conf_indices,1.0)
    confidence = torch.where(mask_index, confidence, -np.inf) #shape: b, L
    confidence = confidence.to(device=x0_p.device, dtype=x0_p.dtype)

    transfer_index = torch.ones_like(x0, dtype=torch.bool, device=x0.device) #shape: b, L
    transfer_index = confidence >= threshold

    return x0, transfer_index, confidence

def get_transfer_index_5(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None, 
                         block_share_end=None, predict_length=None, restore_indices=None):
    x_reshaped = x.view(1, 8, 32)

    logits_with_noise = add_gumbel_noise(logits, temperature=temperature) #shape: b, L+7L', v
    x0 = torch.argmax(logits_with_noise, dim=-1) #shape: b, L+7L' 

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, L+7L'

        x_reorderedp = x0_p[:, :32]
        block1_reorderedp = torch.cat([x0_p[:, :block_share_end], x0_p[:, 32:32+predict_length]], dim=1)
        block2_reorderedp = torch.cat([x0_p[:, :block_share_end], x0_p[:, 32+predict_length:32+predict_length*2]], dim=1)
        block3_reorderedp = torch.cat([x0_p[:, :block_share_end], x0_p[:, 32+predict_length*2:32+predict_length*3]], dim=1)
        block4_reorderedp = torch.cat([x0_p[:, :block_share_end], x0_p[:, 32+predict_length*3:32+predict_length*4]], dim=1)
        block5_reorderedp = torch.cat([x0_p[:, :block_share_end], x0_p[:, 32+predict_length*4:32+predict_length*5]], dim=1)
        block6_reorderedp = torch.cat([x0_p[:, :block_share_end], x0_p[:, 32+predict_length*5:32+predict_length*6]], dim=1)
        block7_reorderedp = torch.cat([x0_p[:, :block_share_end], x0_p[:, 32+predict_length*6:]], dim=1)

        x_restorep = x_reorderedp[:, restore_indices]
        block1_restorep = block1_reorderedp[:, restore_indices]
        block2_restorep = block2_reorderedp[:, restore_indices]
        block3_restorep = block3_reorderedp[:, restore_indices]
        block4_restorep = block4_reorderedp[:, restore_indices]
        block5_restorep = block5_reorderedp[:, restore_indices]
        block6_restorep = block6_reorderedp[:, restore_indices]
        block7_restorep = block7_reorderedp[:, restore_indices]
        
        x0_p = torch.cat([x_restorep, block1_restorep, block2_restorep, block3_restorep, block4_restorep, block5_restorep, block6_restorep, block7_restorep], dim=1)
        x0_p = x0_p.view(1, 8, 32)

    elif remasking == 'random':
        x0_p = torch.rand((1, 8, 32), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x_reordered0 = x0[:, :32]
    block1_reordered0 = torch.cat([x0[:, :block_share_end], x0[:, 32:32+predict_length]], dim=1)
    block2_reordered0 = torch.cat([x0[:, :block_share_end], x0[:, 32+predict_length:32+predict_length*2]], dim=1)
    block3_reordered0 = torch.cat([x0[:, :block_share_end], x0[:, 32+predict_length*2:32+predict_length*3]], dim=1)
    block4_reordered0 = torch.cat([x0[:, :block_share_end], x0[:, 32+predict_length*3:32+predict_length*4]], dim=1)
    block5_reordered0 = torch.cat([x0[:, :block_share_end], x0[:, 32+predict_length*4:32+predict_length*5]], dim=1)
    block6_reordered0 = torch.cat([x0[:, :block_share_end], x0[:, 32+predict_length*5:32+predict_length*6]], dim=1)
    block7_reordered0 = torch.cat([x0[:, :block_share_end], x0[:, 32+predict_length*6:]], dim=1)

    x_restore0 = x_reordered0[:, restore_indices]
    block1_restore0 = block1_reordered0[:, restore_indices]
    block2_restore0 = block2_reordered0[:, restore_indices]
    block3_restore0 = block3_reordered0[:, restore_indices]
    block4_restore0 = block4_reordered0[:, restore_indices]
    block5_restore0 = block5_reordered0[:, restore_indices]
    block6_restore0 = block6_reordered0[:, restore_indices]
    block7_restore0 = block7_reordered0[:, restore_indices]
    
    x0 = torch.cat([x_restore0, block1_restore0, block2_restore0, block3_restore0, block4_restore0, block5_restore0, block6_restore0, block7_restore0], dim=1)
    x0 = x0.view(1, 8, 32)
    
    x0 = torch.where(mask_index, x0, x_reshaped) #shape: b, 8, L
    confidence = torch.where(mask_index, x0_p, -np.inf) #shape: b, 8, L

    max_conf_indices = torch.argmax(confidence, dim=2, keepdim=True)
    confidence.scatter_(2, max_conf_indices,1.0)
    confidence = torch.where(mask_index, confidence, -np.inf) #shape: b, L
    confidence = confidence.to(device=x0_p.device, dtype=x0_p.dtype)

    transfer_index = torch.ones_like(x0, dtype=torch.bool, device=x0.device) #shape: b, L
    transfer_index = confidence >= threshold

    return x0, transfer_index, confidence

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

    return x0, transfer_index, confidence

def main():
    device = 'cuda'

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-1.5', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-1.5', trust_remote_code=True)

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
