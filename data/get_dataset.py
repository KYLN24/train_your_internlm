from datasets import load_dataset

dataset = load_dataset("monology/pile-uncopyrighted")

dataset["train"].to_json("./data/train/pile.train.jsonl")
dataset["test"].to_json("./data/test/pile.test.jsonl")
dataset["validation"].to_json("./data/val/pile.validation.jsonl")
