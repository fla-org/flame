from transformers import AutoModelForCausalLM, AutoTokenizer

import fla

name = "fla-hub/gla-1.3B-100B"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).cuda().bfloat16()
model.eval()
input_prompt = "Power goes with permanence. Impermanence is impotence. And rotation is castration."
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.cuda()
outputs = model.generate(input_ids, max_length=64, use_cache=True)
tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
