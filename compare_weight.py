import torch

weight1_folder = '/root/workspace/external_data/13bv9.1_autoawq_w4_gemv_calib_512x512_custom_flash_attn_32/'
weight2_folder = '/root/workspace/external_data/13bv9.1_autoawq_w4_gemv_calib_512x512_custom_flash_attn_32_new_seed/'

model1 = torch.load(weight1_folder + 'pytorch_model.bin')
model2 = torch.load(weight2_folder + 'pytorch_model.bin')

for key in model1.keys():
    assert key in model2.keys()
    assert torch.allclose(model1[key], model2[key])
print('success')
