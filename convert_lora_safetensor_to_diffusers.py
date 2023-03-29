import argparse

import torch
from safetensors.torch import load_file, save_file


LORA_PREFIX_UNET = 'lora_unet'


def convert_name_to_bin(name):
    
    new_name = name.replace(LORA_PREFIX_UNET+'_', '')
    new_name = new_name.replace('.weight', '')
    
    parts = new_name.split('.')
    
    if 'out' in parts[0]:
        parts[0] = "_".join(parts[0].split('_')[:-1])
    parts[1] = parts[1].replace('_', '.')
    
    sub_parts = parts[0].split('_')

    new_sub_parts = ""
    for i in range(len(sub_parts)):
        if sub_parts[i] in ['block', 'blocks', 'attentions'] or sub_parts[i].isnumeric() or 'attn' in sub_parts[i]:
            if 'attn' in sub_parts[i]:
                new_sub_parts += sub_parts[i] + ".processor."
            else:
                new_sub_parts += sub_parts[i] + "."
        else:
            new_sub_parts += sub_parts[i] + "_"
    
    new_sub_parts += parts[1]
    
    new_name =  new_sub_parts + '.weight'
    
    return new_name


parser = argparse.ArgumentParser()

parser.add_argument(
    "--lora_path", default=None, type=str, required=True, help="Path to the LoRA in safetensors format."
)
parser.add_argument(
    "--dump_path", default=None, type=str, required=True, help="Where you want the Bin file to be put"
)

args = parser.parse_args()

bin_state_dict = {}
safetensors_state_dict = load_file(args.lora_path)

for key_safetensors in safetensors_state_dict:
    if 'text' in key_safetensors:
        continue
    if 'unet' not in key_safetensors:
        continue
    if 'transformer_blocks' not in key_safetensors:
        continue
    if 'ff_net' in key_safetensors or 'alpha' in key_safetensors:
        continue
    key_bin = convert_name_to_bin(key_safetensors)
    bin_state_dict[key_bin] = safetensors_state_dict[key_safetensors]

torch.save(bin_state_dict, args.dump_path + "\pytorch_lora_weights.bin")
    

