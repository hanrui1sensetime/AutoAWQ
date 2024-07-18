from transformers import AutoTokenizer, TextStreamer, GenerationConfig, AutoModelForCausalLM
import torch
import glob
import os.path
from datasets import load_dataset
import json
from tqdm import tqdm
model_path = '/root/workspace/externel_data/pulse_v14_20b_gpt4_hf/base'
# quant_path = "/root/workspace/external_data/13bv9.1_autoawq_w4_gemv_calib_512x512_custom_flash_attn_32_new_seed_fix/"
# quant_path = "/root/workspace/external_data/13bv9.1_autoawq_w4_gemv_calib_64x16384_custom_flash_attn_32"
# quant_path = "/root/workspace/external_data/13bv9.1_autoawq_w4_gemv_calib_512x512_custom_flash_attn_32_new_seed"
# quant_path = "/root/workspace/external_data//"
quant_path = '/root/workspace/externel_data/pulse_v14_20b_gpt4_hf/quant_gemm'
#'/root/workspace/external_data/pulse_v11_123b_gpt4_hf/quant'
# data_path = '/root/workspace/external_data/dayi_data'
# data_path = 'dayi_data_qa_debug'
# predict_output_path = "/workspace/AutoAWQ/predict_13bv9.1_autoawq_w4_gemv_calib_512x512_custom_flash_attn_32_new_seed_fix_test"
# predict_output_path = "/workspace/AutoAWQ/predict_pulse_v11_123b_autoawq_w4_gemv_calib_512x512_custom_migrate_flash_attn_result"
# predict_13bv9.1_quant_gemv_calib_16384_50_custom/'

# Load model
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_quantized(quant_path,
                                          fuse_layers=True,
                                          )
# assert 1 == 2


tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# print(tokenizer.decode([56728, 56728, 56728, 56728, 56728, 56728, 56728, 56728, 56728, 56728, 56728]))
'''pyt
print(tokenizer.decode([  333,  8619,   266,   449,  5064,   352, 53522,   657,   262, 60368,
         60837,   328,   395,  3622,  4287,  1762,   446,   523,  1236,  1063,
         17786, 16308,   684,   262, 60731, 61325, 68636,   281, 22061,   569,
          3690,   416,  1114,   569,  3369,   756, 38248,  5049, 43836,   334,
           262,   638,  1807,   285,  2640,   285,  1836,   364,  5564,  2554,
           334,   262,   638,  1311,   285,  2800,   285,  2934,   402,   324,
         11740,   402,  2770,   777,  1745, 13175,  2189,   435,   395, 21989,
           751, 56342,   596,  7186,  2342,  4736,   281,  1592,   777,  1317,
          1130,   410, 11607,   586,  3480,  6052, 18943,   862,   442,   765,
          1614, 10779,   684,   410,  1341,   328,   810,  1148,  5899,   454,
         19792,   657,  8520,   281, 92548, 92546, 92550,   586,  3480,  6052,
           499, 92548, 92547, 70447, 61297, 92548, 92546, 68734,   299,  1236,
          1063, 60353, 68252, 60620, 60731, 61325, 68636, 69233, 77278, 68790,
         70218, 60355, 74010, 68417, 85625, 60353, 68403, 68347, 60353, 73165,
         70695, 60353, 68274, 68983, 60455, 60355, 92550,   586,  3480,  6052,
           499, 92548]))
'''
'''
model = AutoModelForCausalLM.from_pretrained(model_path, 
                                             # torch_dtype=torch.bfloat16
                                             ).to('cuda:0')
'''
input_ids = tokenizer(f"<s><|iim_start|>You are 大医, a large language model of PULSE architecture trained by 商汤科技. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2024-06-03\n\n# Environment\n\nYou can run Python code in a restricted Jupyter Notebook environment. You can only use the '\''get_user_input'\'' function to get text entered by the user, all other functions and operators are disabled.<|im_end|><|aim_start|><|code_start|>get_user_input()<|im_end|><|fim_start|>你是谁<|im_end|><|aim_start|>", add_special_tokens=False).input_ids
start_pos = len(input_ids)
print(f'start_pos: {start_pos}')
tokens = torch.tensor([input_ids]).cuda()
generation_config = GenerationConfig(max_length=16384,
                                     max_new_tokens=2048,
                                     num_beams=1,
                                     do_sample=True,
                                     temperature=0.001,
                                     top_p=0.1,
                                     top_k=1,
                                     repetition_penalty=1.0,
                                     pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=[tokenizer.convert_tokens_to_ids("<|im_end|>")],
                                     suppress_tokens=[tokenizer.eos_token_id])
try:
    for layer in model.model.model.layers:
        layer.self_attn.start_pos = 0
except Exception:
    for layer in model.model.layers:
        layer.self_attn.start_pos = 0
        print(layer.self_attn.q_proj.weight.dtype)
# tokens[0, 78] = 495
# tokens[0, 82] = 259
generation_output = model.generate(
                        tokens,
                        generation_config,
                        # streamer=streamer,
                    )
print(generation_output)
print(tokenizer.decode(generation_output[0, 113:]))
'''
print(tokenizer.decode([61176]))
print(tokenizer.decode([299, 1236, 1063])) # PULSE
print(tokenizer.decode([68734,   299,  1236,
          1063, 60353, 68252, 60620, 60731, 61325, 68636, 69233, 77278, 68790,
         70218, 60355, 74010, 68417, 85625, 60353, 68403, 68347, 60353, 73165,
         70695, 60353, 68274, 68983, 60455, 60355, 92550,   586,  3480,  6052,
           499, 92548]))
'''