from datasets import load_dataset

dataset = load_dataset("ola13/small-the_pile")
dataset = dataset["train"].train_test_split(test_size=0.01, seed=42)

dataset["train"].to_json("data/train/small-the_pile.train.jsonl")
dataset["test"].to_json("data/val/small-the_pile.val.jsonl")
