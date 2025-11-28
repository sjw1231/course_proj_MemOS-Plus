


import json
import os
from llmlingua import PromptCompressor
from tqdm import tqdm

def compress(prompt,rate=0.5):
    compressor = PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2=True
    )

    compress_res = compressor.compress_prompt_llmlingua2(
        prompt,
        rate=rate,
        force_tokens=['\n', '.', '!', '?', ','],
        chunk_end_tokens=['.', '\n'],
        return_word_label=True,
        drop_consecutive=True
    )

    return compress_res


def process_conversation(conversation_data):
    """
    Process conversation data from locomo10 format into memory system format.
    Handles both text-only and image-containing messages.
    """
    processed = []
    speaker_a = conversation_data["speaker_a"]
    speaker_b = conversation_data["speaker_b"]
    
    # Find all session keys
    session_keys = [key for key in conversation_data.keys() if key.startswith("session_") and not key.endswith("_date_time")]
    
    for session_key in session_keys:
        timestamp_key = f"{session_key}_date_time"
        timestamp = conversation_data.get(timestamp_key, "")
        
        for dialog in conversation_data[session_key]:
            speaker = dialog["speaker"]
            text = dialog["text"]
            
            # Handle image content if present
            if "blip_caption" in dialog and dialog["blip_caption"]:
                text = f"{text} (image description: {dialog['blip_caption']})"
            
            # Alternate between speakers as user and assistant
            if speaker == speaker_a:
                processed.append({
                    "user_input": text,
                    "agent_response": "",
                    "timestamp": timestamp
                })
            else:
                if processed:
                    processed[-1]["agent_response"] = text
                else:
                    processed.append({
                        "user_input": "",
                        "agent_response": text,
                        "timestamp": timestamp
                    })
    
    return processed



def main():
    
    rt_dir="plots"
    save_path=f"{rt_dir}/compressed_locomo.json"
    locomo_path="eval/locomo10.json"
    dataset=json.load(open(locomo_path, 'r', encoding='utf-8'))
    total_samples = len(dataset)

    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            existing_samples = json.load(f)
        cur_idx=len(existing_samples)    
    else:
        cur_idx=0
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)
        
    dataset=dataset[cur_idx:]
    compressed_dataset=[]
    for idx, sample in enumerate(dataset):
        sample_id = sample.get("sample_id", "unknown_sample")
        conversation_data = sample["conversation"]

        processed_dialogs = process_conversation(conversation_data)
        compressed_dialogs=[]

        num_token_original=0
        num_token_compressed=0
        num_char_original=0
        num_char_compressed=0
        for dialog in tqdm(processed_dialogs):
            user_input = dialog["user_input"]
            agent_response = dialog["agent_response"]
            time_stamp = dialog["timestamp"]
            compress_res_user = compress(user_input, rate=0.4)
            compress_res_agent = compress(agent_response, rate=0.4)

            cur_n_token_ori=compress_res_user["origin_tokens"]+compress_res_agent['origin_tokens']
            cur_n_token_comp=compress_res_user['compressed_tokens']+compress_res_agent['compressed_tokens']
            num_token_original+=cur_n_token_ori
            num_token_compressed+=cur_n_token_comp
            num_char_original+=len(user_input)+len(agent_response)
            num_char_compressed+=len(compress_res_user['compressed_prompt'])+len(compress_res_agent['compressed_prompt'])
            print("num_token_original_cur_dialog: ",cur_n_token_ori)
            print("num_token_compressed_cur_dialog: ",cur_n_token_comp)
            
            compressed_dialogs.append({
                "user_input_compressed": compress_res_user['compressed_prompt'],
                "agent_response_compressed": compress_res_agent['compressed_prompt'],
                "u_token_num_0": compress_res_user['origin_tokens'],
                "u_token_num_1": compress_res_user['compressed_tokens'],
                "a_token_num_0": compress_res_agent['origin_tokens'],
                "a_token_num_1": compress_res_agent['compressed_tokens'],
                "timestamp": time_stamp,
            })
        
        compressed_dataset.append({
            "sample_id": sample_id,
            "compressed_dialogs": compressed_dialogs,
            "num_token_original":num_token_original,
            "num_token_compressed":num_token_compressed,
            "num_char_original":num_char_original,
            "num_char_compressed":num_char_compressed,
        })
        
        with open(save_path, 'r') as f:
            existing_samples = json.load(f)
        existing_samples.append(compressed_dataset[-1])
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(existing_samples, f, ensure_ascii=False, indent=4)
    
    global_num_token_compressed=0
    global_num_token_original=0
    global_num_char_compressed=0
    global_num_char_original=0
    
    for x in compressed_dataset:
        global_num_token_compressed+=x['num_token_compressed']
        global_num_token_original+=x['num_token_original']
        global_num_char_compressed+=x['num_char_compressed']
        global_num_char_original+=x['num_char_original']
    print(f"Token Compression Rate: {global_num_token_compressed/global_num_token_original}")
    print(f"Character Compression Rate: {global_num_char_compressed/global_num_char_original}")
        
main()
        