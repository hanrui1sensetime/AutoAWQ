from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# model_path = '/root/workspace/external_data/tigerbot-13b-base_v9_gpt4_hf'
# model_path = '/root/workspace/externel_data/pulse_v13_1_20b_gpt4_hf/base'
# quant_path = '/root/workspace/externel_data/pulse_v13_1_20b_gpt4_hf/quant_gemm'
model_path = '/root/workspace/externel_data/pulse_v14_20b_gpt4_hf/base'
quant_path = '/root/workspace/externel_data/pulse_v14_20b_gpt4_hf/quant_gemm'
assert os.path.exists(quant_path)
#13bv9.1_autoawq_w4_gemv_calib_64x16384_custom_flash_attn_32'
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path,
                                           use_flash_attention_2=True,
                                           # device_map='balanced'
                                        )

# model = model.to('cuda:0')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config,
               calib_data='custom'
              )

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
