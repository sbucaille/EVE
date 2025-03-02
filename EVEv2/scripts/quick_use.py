import math
from io import BytesIO

import requests
import torch
from PIL import Image

from eve.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from eve.conversation import conv_templates, SeparatorStyle
from eve.mm_utils import KeywordsStoppingCriteria
from eve.model.builder import load_pretrained_model


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def process_images(images, processor, model_cfg=None):
    new_images = []
    for raw_image in images:
        width, height = raw_image.size
        scale_ratio = math.sqrt(width * height) / processor.max_size
        min_edge = processor.patch_stride * processor.dense_stride

        width = max(int(math.ceil(width / scale_ratio / min_edge)) * min_edge, min_edge)
        height = max(int(math.ceil(height / scale_ratio / min_edge)) * min_edge, min_edge)

        new_image = raw_image.resize((width, height))
        image = processor.preprocess(new_image, return_tensors='pt')['pixel_values'][0]
        new_images.append(image)

    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [
        tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

model_path = "/home/steven/models/evev2"
model_type = "qwen2"

image_path = "/home/steven/EVE/EVEv2/examples/mac.jpg"
query = "What is in this image ?"


max_new_tokens = 1024

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_type=model_type,
    device_map="auto",
    device="cuda"
)


qs = query
qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

conv = conv_templates[model_type].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

image = load_image(image_path)
image_tensor = process_images([image], image_processor, None)[0]
image_tensor = image_tensor.unsqueeze(0).half().cuda()

input_ids = tokenizer_image_token(
    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
input_ids = input_ids.to(device='cuda', non_blocking=True)

stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(
    keywords, tokenizer, input_ids)

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        # do_sample=args.do_sample,
        # top_p=args.top_p,
        # top_k=args.top_k,
        # num_beams=args.num_beams,
        # temperature=args.temperature,
        use_cache=True,
        max_new_tokens=max_new_tokens,
        stopping_criteria=[stopping_criteria])

input_token_len = input_ids.shape[1]
n_diff_input_output = (
    input_ids != output_ids[:, :input_token_len]).sum().item()
if n_diff_input_output > 1:
    print(
        f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
outputs = tokenizer.batch_decode(
    output_ids[:, input_token_len:], skip_special_tokens=True)[0]
outputs = outputs.strip()
if outputs.endswith(stop_str):
    outputs = outputs[:-len(stop_str)]
outputs = outputs.strip()
print(outputs)