import torch
from diffusers import FluxPipeline
import os
# from huggingface_hub import login
# login(token="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")

pipe = FluxPipeline.from_pretrained("/ytech_m2v2_hdd/baijianhong/flux/ckpt_dev", torch_dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A logo with white background. Note that the logo clearly contains the following words, “SynCamMaster” in artistic font, and an additional blended object that accords with the title."

prompt = "A logo with white background. Note that the logo clearly contains the following words, “SynCamMaster” in artistic font, and an additional blended camera object."
prompt = "A logo with multiple cameras shooting. The logo should be in a white background."
prompt = "The logo was shot using four cameras, with the word 'SynCamMaster' next to it. They blend together organically and have an artistic feel. The logo should be in a white background."

prompt = "The logo was shot using four cameras, with the word 'SynCamMaster' in artistic font next to it. They blend together organically and have an artistic feel. The logo should be in a white background."
prompt = "The logo is composed of four cameras, with the word 'SynCamMaster' in artistic font next to it. They blend together organically and have an artistic feel. The logo is situated against a crisp white background."
prompt = "The logo is composed of four cameras, with the word 'SynCamMaster' in WordArt font next to it. They blend together organically and have an artistic feel. The logo should be in a white background."
save_dir = '/ytech_m2v2_hdd/baijianhong/flux/384-672-camera5_pe'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for seed in range(500):
    if seed == 0:
        seed = 9
    image = pipe(
        prompt,
        height=384,
        width=672,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(seed)
    ).images[0]
    image.save(f"{save_dir}/flux-dev_font_{seed}.png")