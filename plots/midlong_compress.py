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


def compress_mid():
    dir="eval/mem_tmp_loco_final"
    summ_list=[]
    for filename in tqdm(os.listdir(dir)):
        if "mid" in filename:
            filepath = os.path.join(dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data=json.load(f)["sessions"]
                for k,v in data.items():
                    summ_list.append(v["summary"])
    all_sum="\n".join(summ_list)
    compress_res=compress(all_sum,rate=0.6)
    compressed_midterm_mem=compress_res['compressed_prompt']
    print(len(all_sum))
    print(len(compressed_midterm_mem))
    print(compress_res["origin_tokens"])
    print(compress_res['compressed_tokens'])
    print("midterm token ratio: ", compress_res['compressed_tokens'] / compress_res["origin_tokens"])
    print("midterm char ratio: ", len(compressed_midterm_mem) / len(all_sum))

def compress_long():
    dir="eval/mem_tmp_loco_final"
    user_klg_list=[]
    agent_klg_list=[]
    for filename in tqdm(os.listdir(dir)):
        if "long" in filename:
            filepath = os.path.join(dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data=json.load(f)
                user_klg_list.extend([x["knowledge"] for x in data["knowledge_base"]])
                agent_klg_list.extend([x["knowledge"] for x in data["assistant_knowledge"]])
                
    all_sum="\n".join(user_klg_list+agent_klg_list)
    compress_res=compress(all_sum,rate=0.6)
    compressed_midterm_mem=compress_res['compressed_prompt']
    print(compress_res["origin_tokens"])
    print(compress_res['compressed_tokens'])
    print(len(all_sum))
    print(len(compressed_midterm_mem))
    print("longterm token ratio: ", compress_res['compressed_tokens'] / compress_res["origin_tokens"])
    print("longterm char ratio: ", len(compressed_midterm_mem) / len(all_sum))
    
    
if __name__=="__main__":
    compress_mid()
    # compress_long()