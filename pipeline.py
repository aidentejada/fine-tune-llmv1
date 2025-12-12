import json
import time
import re
import google.generativeai as genai
from tqdm import tqdm
import os

# configuration
INPUT_FILE = "input.jsonl"
OUTPUT_FILE = "output.jsonl"
CHECKPOINT_FILE = "checkpoint.json"

# batch size: 50 items per batch
BATCH_SIZE = 50
API_KEY = "your_api_key_here"

# setup
genai.configure(api_key=API_KEY)
MODEL_NAME = 'gemini-2.5-flash-lite'

model = genai.GenerativeModel(MODEL_NAME, 
                              system_instruction="""
You are a dataset sanitization expert. Rewrite messages to be safe while keeping the original tone and style.

CRITICAL RULES:
1. Output MUST be a JSON array with EXACTLY the same number of items as input
2. Each input message gets exactly ONE output message
3. Never skip, merge, or duplicate items
4. Remove explicit sexual content, severe profanity, and PII
5. Keep slang, emotion, and conversational style

Example:
Input: ["yo what's good", "lmaooo that's wild"]
Output: ["hey what's up", "haha that's crazy"]
""")

def clean_json_text(text):
    """removes markdown code blocks if present."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def load_checkpoint():
    """load progress from checkpoint file."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"completed_batches": [], "cleaned_texts": []}

def save_checkpoint(completed_batches, cleaned_texts):
    """save progress to checkpoint file."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({
            "completed_batches": completed_batches,
            "cleaned_texts": cleaned_texts
        }, f)

def process_batch_with_retry(batch, batch_idx):
    """process a batch with 2 retries, then skip if it keeps failing."""
    retries = 0
    max_retries = 2
    
    while retries < max_retries:
        try:
            # split into numbered items to force 1:1 mapping
            numbered_batch = [f"[{i}] {item}" for i, item in enumerate(batch)]
            
            prompt = f"""Rewrite exactly {len(batch)} messages. Output EXACTLY {len(batch)} strings.

Input:
{json.dumps(numbered_batch, ensure_ascii=False)}"""
            
            response = model.generate_content(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.0,
                    "max_output_tokens": 10000
                }
            )
            
            clean_response = clean_json_text(response.text)
            batch_results = json.loads(clean_response)
            
            # validation
            if not isinstance(batch_results, list):
                raise ValueError(f"Output is not a list")
            
            # accept if we got at least 90% of items
            if len(batch_results) >= len(batch) * 0.9:
                # truncate or pad to exact size
                if len(batch_results) > len(batch):
                    batch_results = batch_results[:len(batch)]
                else:
                    while len(batch_results) < len(batch):
                        batch_results.append(batch[len(batch_results)])
                
                if len(batch_results) != len(batch):
                    print(f"    auto-corrected to {len(batch_results)} items")
                
                return batch_results
            else:
                raise ValueError(f"Too few items: {len(batch_results)}/{len(batch)}")
            
        except Exception as e:
            retries += 1
            error_str = str(e)
            
            # check if it's a rate limit error
            if "429" in error_str or "quota" in error_str.lower():
                print(f"\n    rate limit hit on batch {batch_idx}. waiting 65 seconds...")
                time.sleep(65)  # wait a full minute to reset the window
                # don't count this as a retry
                retries -= 1
                continue
            
            if retries < max_retries:
                print(f"\n    batch {batch_idx} attempt {retries}/{max_retries}: {error_str[:80]}")
                time.sleep(10)
    
    # failed after 2 retries - just skip this batch
    print(f"    skipping batch {batch_idx} ({len(batch)} items) - keeping originals")
    return batch  # return originals unchanged

def process_dataset():
    print(f"loading {INPUT_FILE}...")
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"file not found: {INPUT_FILE}")
        return

    # load checkpoint
    checkpoint = load_checkpoint()
    completed_batch_indices = set(checkpoint["completed_batches"])
    cleaned_texts = checkpoint["cleaned_texts"]
    
    if completed_batch_indices:
        print(f"resuming from checkpoint: {len(completed_batch_indices)} batches already done")

    # find all items needing generalization
    dirty_indices = []
    dirty_texts = []

    print("scanning for <TO_GENERALIZE> tags...")
    for i, line in enumerate(lines):
        if "<TO_GENERALIZE>" in line:
            match = re.search(r"<TO_GENERALIZE>(.*?)</TO_GENERALIZE>", line, re.DOTALL)
            if match:
                dirty_indices.append(i)
                dirty_texts.append(match.group(1))

    total_dirty = len(dirty_indices)
    print(f"found {total_dirty} items to process")

    # create batches
    batches = [dirty_texts[i:i + BATCH_SIZE] for i in range(0, total_dirty, BATCH_SIZE)]
    total_batches = len(batches)
    
    print(f"processing {total_batches} batches (size={BATCH_SIZE})")
    
    # process batches
    for batch_idx, batch in enumerate(tqdm(batches, desc="batches")):
        # skip already completed batches
        if batch_idx in completed_batch_indices:
            continue
        
        # process batch
        batch_results = process_batch_with_retry(batch, batch_idx)
        cleaned_texts.extend(batch_results)
        
        # save checkpoint after each batch
        completed_batch_indices.add(batch_idx)
        save_checkpoint(list(completed_batch_indices), cleaned_texts)
        
        # rate limiting: 15 rpm limit means 1 request every 4+ seconds
        # using 5 seconds to be safe (12 requests/min)
        if batch_idx < total_batches - 1:
            time.sleep(5)

    # validate we got everything
    loss_percentage = (1 - len(cleaned_texts) / total_dirty) * 100 if total_dirty > 0 else 0
    
    if len(cleaned_texts) != total_dirty:
        print(f"processed {len(cleaned_texts)}/{total_dirty} items ({loss_percentage:.1f}% loss)")
        if loss_percentage > 5:
            print("   this is more than 5% loss - consider checking for issues")
        else:
            print("   acceptable loss for large dataset")
        
        # if somehow short, pad with originals
        while len(cleaned_texts) < total_dirty:
            idx = len(cleaned_texts)
            cleaned_texts.append(dirty_texts[idx])

    # reconstruct file
    print("rebuilding file...")
    for i, line_idx in enumerate(dirty_indices):
        original_line = lines[line_idx]
        safe_text = cleaned_texts[i]
        
        # ensure safe_text is a string
        if not isinstance(safe_text, str):
            if isinstance(safe_text, list):
                safe_text = " ".join(str(x) for x in safe_text)
            else:
                safe_text = str(safe_text)
        
        # replace the tag
        new_line = re.sub(
            r"<TO_GENERALIZE>.*?</TO_GENERALIZE>",
            lambda m: safe_text,
            original_line,
            flags=re.DOTALL
        )
        lines[line_idx] = new_line

    # write output
    print(f"writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    # clean up checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    
    print(f"done! processed {total_dirty} items")
    print(f"output: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_dataset()