from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "ckpts/7b_pretrain/hf", trust_remote_code=True
).cuda()

tokenizer = AutoTokenizer.from_pretrained(
    "ckpts/7b_pretrain/hf", trust_remote_code=True
)

input_ids = tokenizer("I am ", return_tensors="pt").input_ids.cuda()
output = model.generate(input_ids, max_length=30)
print(tokenizer.decode(output[0]))
