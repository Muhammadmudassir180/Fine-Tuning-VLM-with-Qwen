import torch
from PIL import Image
from unsloth import FastVisionModel
import json
from transformers import AutoProcessor
from tqdm import tqdm


# MODEL_PATH = r"C:\Users\Hey-BUDD\Desktop\Deep Learining Project\Datasets\RadFig-VQA\Qwen_25_VL_Finetuned_RadFig"     
MODEL_PATH="unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"
TEST_FILE = r"path to\your\test_file.jsonl"  # Update with your test file path



model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_PATH,
    load_in_4bit = True,   
    max_seq_length=4096, # must match training
    device_map = "auto",
    dtype=torch.bfloat16,
    use_gradient_checkpointing="unsloth" 
)

model.eval()
import re

def extract_option(text):
    match = re.search(r"\b([A-F])\.", text)
    return match.group(1) if match else None


# processor = AutoProcessor.from_pretrained(MODEL_PATH)
samples = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        user_content = data["messages"][0]["content"]
        gt_text = data["messages"][1]["content"][0]["text"]

        question = user_content[0]["text"]
        image_path = user_content[1]["image"]

        samples.append({
            "question": question,
            "image": image_path,
            "ground_truth": gt_text
        })

print(f"Loaded {len(samples)} test samples")
correct = 0
total = 0
samples = samples[:200]  # Limit to first 50 samples for quicker evaluation
for s in tqdm(samples, desc="Evaluating VQA samples"):
# for s in samples:
    image = Image.open(s["image"]).convert("RGB")
    # image.show()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": s["question"]}
            ]
        }
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to("cuda")

    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,       # IMPORTANT
            temperature=0.0,
            use_cache=True,
        )

    # 🔥 REMOVE PROMPT TOKENS
    gen_ids = outputs[0][prompt_len:]

    pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    gt_text = s["ground_truth"].strip()

    pred_option = extract_option(pred_text)
    gt_option = extract_option(gt_text)
    # print("Question:", s["question"])
    # print("Assistant Output:", pred_text)
    # print("Ground Truth:", gt_text)
    # print("Predicted Option:", pred_option)
    # print("GT Option:", gt_option)

    if pred_option is not None and pred_option == gt_option:
        correct += 1
        print("✅ Correct\n")
    else:
        print("❌ Incorrect\n")

    total += 1
    tqdm.write(f"Running Acc: {correct/total:.3f}")


accuracy = correct / total if total > 0 else 0
print(f"\nFinal Accuracy: {accuracy:.4f}")
