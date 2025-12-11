import os
import json
import time
from tqdm import tqdm
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

from short_term_memory import ShortTermMemory
from mid_term_memory import MidTermMemory
from long_term_memory import LongTermMemory
from dynamic_update import DynamicUpdate
from retrieval_and_answer import RetrievalAndAnswer
from utils import OpenAIClient

from main_loco_parse import update_user_profile_from_top_segment, generate_system_response_with_meta

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "")

client = OpenAIClient(api_key=API_KEY, base_url=BASE_URL)

lock = Lock()

def process_conversation(sessions: list[list[dict[str, str | bool]]], dates: list[str]) -> list[dict[str, str]]:
    processed = []
    for session, date in zip(sessions, dates):
        # 2023/04/10 (Mon) 17:50 to "%Y-%m-%d %H:%M:%S"
        formatted_date = time.strftime("%Y-%m-%d %H:%M:%S", time.strptime(date, "%Y/%m/%d (%a) %H:%M"))
        
        for message in session:
            if message["role"] == "user":
                processed.append({
                    "user_input": message["content"],
                    "agent_response": "",
                    "timestamp": formatted_date
                })
            elif processed:
                processed[-1]["agent_response"] = message["content"]
            else:
                processed.append({
                    "user_input": "",
                    "agent_response": message["content"],
                    "timestamp": formatted_date
                })
    return processed

def process_sample(sample: dict, idx: int, output_file: str):
    with lock:
        with open(output_file, 'r') as f:
            existing_results = {json.loads(line)["question_id"] for line in f}
        if sample["question_id"] in existing_results:
            print(f"Sample {sample['question_id']} already processed. Skipping.")
            return
    
    short_mem = ShortTermMemory(max_capacity=1, file_path=f"mem_tmp_longmemeval_final/{idx}_short_term.json")
    mid_mem = MidTermMemory(max_capacity=2000, file_path=f"mem_tmp_longmemeval_final/{idx}_mid_term.json")
    long_mem = LongTermMemory(file_path=f"mem_tmp_longmemeval_final/{idx}_long_term.json")
    dynamic_updater = DynamicUpdate(short_mem, mid_mem, long_mem, topic_similarity_threshold=0.6, client=client)
    retrieval_system = RetrievalAndAnswer(short_mem, mid_mem, long_mem, dynamic_updater=dynamic_updater, queue_capacity=10)
    
    processed_dialogs = process_conversation(sample["haystack_sessions"], sample["haystack_dates"])
    
    for dialog in processed_dialogs:
        short_mem.add_qa_pair(dialog)
        if short_mem.is_full():
            dynamic_updater.bulk_evict_and_update_mid_term()
        update_user_profile_from_top_segment(mid_mem, long_mem, sample["question_id"], client)
    
    retrieval_result = retrieval_system.retrieve(
        sample["question"],
        segment_threshold=0.1,
        page_threshold=0.1,
        knowledge_threshold=0.1,
        client=client
    )
    
    system_answer, _, _ = generate_system_response_with_meta(
        sample["question"],
        short_mem,
        long_mem,
        retrieval_result["retrieval_queue"],
        retrieval_result["long_term_knowledge"],
        client,
        sample["question_id"],
        "user",
        "assistant",
        {}
    )

    with lock:
        with open(output_file, 'a') as f:
            f.write(json.dumps({
                "question_id": sample["question_id"],
                "hypothesis": system_answer,
            }, ensure_ascii=False) + "\n")

def main(data_path: str):
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} entries from {data_path}")
    
    output_file = os.path.basename(data_path).replace('.json', '_results.jsonl')
    output_file = os.path.join("LongMemEval", output_file)
    os.makedirs("LongMemEval", exist_ok=True)

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for idx, sample in enumerate(data):
            futures.append(executor.submit(process_sample, sample, idx, output_file))
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing sample: {e}")

if __name__ == "__main__":
    main("../data/longmemeval_oracle.json")
    