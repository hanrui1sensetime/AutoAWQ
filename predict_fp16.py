from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer, GenerationConfig, AutoModelForCausalLM
import torch
import glob
import os.path
from datasets import load_dataset
import json
from tqdm import tqdm
model_path = '/root/workspace/externel_data/pulse_v12_70b_gpt4_bf16_hf/base'
# quant_path = "/root/workspace/external_data/13bv9.1_autoawq_w4_gemv_calib_512x512_custom_flash_attn_32_new_seed_fix/"
# quant_path = "/root/workspace/external_data/13bv9.1_autoawq_w4_gemv_calib_64x16384_custom_flash_attn_32"
# quant_path = "/root/workspace/external_data/13bv9.1_autoawq_w4_gemv_calib_512x512_custom_flash_attn_32_new_seed"
# quant_path = "/root/workspace/external_data//"
# quant_path = '/root/workspace/externel_data/pulse_v12_70b_gpt4_bf16_hf/quant'
data_path = '/root/workspace/externel_data/MedBench/DrugCA'
# data_path = 'dayi_data_qa_debug'
# predict_output_path = "/workspace/AutoAWQ/predict_13bv9.1_autoawq_w4_gemv_calib_512x512_custom_flash_attn_32_new_seed_fix_test"
predict_output_path = "/workspace/AutoAWQ/predict_pulse_v12_70b_gpt4_bf16_hf_result"
# predict_13bv9.1_quant_gemv_calib_16384_50_custom/'

# Load model

model_debug = AutoAWQForCausalLM.from_pretrained(model_path,
                                                 use_flash_attention_2=True,
                                                 # device_map='balanced'
                                                 )
model = model_debug.to('cuda:0')
#assert 1 == 2

# model = AutoModelForCausalLM.from_pretrained('/root/workspace/external_data/tigerbot-13b-base_v9_gpt4_hf').to('cuda:0')

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# streamer = TextStreamer(tokenizer, skip_special_tokens=True)

# Convert prompt to tokens
prompt_template = "You are 大医, a large language model of PULSE architecture trained by 商汤科技. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2024-06-03\n\n# Environment\n\nYou can run Python code in a restricted Jupyter Notebook environment. You can only use the '\''get_user_input'\'' function to get text entered by the user, all other functions and operators are disabled."

answer_start_ids = [tokenizer.convert_tokens_to_ids("<|aim_start|>")]
end_token = '<|im_end|>'
eos_token_ids=[
    tokenizer.convert_tokens_to_ids("<|im_end|>"),
    ]
suppress_token_ids=[
    tokenizer.eos_token_id,
    ]
max_retry = 5
for test_file_path in sorted(glob.glob(os.path.join(data_path, "**/*.jsonl"), recursive=True)):
    if test_file_path.endswith("提交结果示例.jsonl"):
        continue
    predict_file_path = test_file_path.replace(data_path, predict_output_path)
    print(f"run eval on {test_file_path}")
    print(f"save eval on {predict_file_path}")

    if os.path.exists(predict_file_path) == True:
        print(f"{predict_file_path} is finish, continue")
        continue

    test_dataset = load_dataset(
        "json",
        data_files=test_file_path,
        split="train",
    )
    predict_output = []
    for data in tqdm(test_dataset):
        retry = 0
        question = data['question']
        input_ids = tokenizer(f"<|iim_start|>{prompt_template}<|im_end|>").input_ids
        input_ids += tokenizer(f"<|aim_start|><|code_start|>get_user_input()<|im_end|><|fim_start|>{question}<|im_end|>", add_special_tokens=False).input_ids
        input_ids += answer_start_ids
        start_pos = len(input_ids)
        # already fill <s> automatically.
        print(f'debugging start_pos: {start_pos}')
        tokens = torch.tensor([input_ids]).cuda()
        generation_config = GenerationConfig(max_length=16384,
                                             max_new_tokens=2048,
                                             num_beams=1,
                                             do_sample=False,
                                             temperature=0,
                                             top_p=0.1,
                                             top_k=1,
                                             repetition_penalty=1.0,
                                             pad_token_id=tokenizer.pad_token_id,
                                             eos_token_id=eos_token_ids,
                                             suppress_tokens=suppress_token_ids)
        try:
            for layer in model.model.model.layers:
                layer.self_attn.start_pos = 0
        except Exception:
            for layer in model.model.layers:
                layer.self_attn.start_pos = 0

        while retry < max_retry:
            try:
                generation_output = model.generate(
                                    tokens,
                                    generation_config,
                                    # streamer=streamer,
                                    )
                break
            except Exception as e:
                print(e)
                retry += 1
                print('retry')

        predict_output += generation_output[:, start_pos:]

    os.makedirs(os.path.dirname(predict_file_path), exist_ok=True)

    with open(predict_file_path, "w", encoding="utf8") as f:
        for test_dataset_item, predict_output_item in zip(test_dataset, predict_output):
            test_dataset_item['answer'] = tokenizer.decode(predict_output_item).strip().split("<|code_start|>")[0]
            f.write(json.dumps(test_dataset_item, ensure_ascii=False) + "\n")
