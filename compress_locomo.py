from llmlingua import PromptCompressor

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True
)

original_prompt = """"""

results = compressor.compress_prompt_llmlingua2(
    original_prompt,
    rate=0.6,
    force_tokens=['\n', '.', '!', '?', ','],
    chunk_end_tokens=['.', '\n'],
    return_word_label=True,
    drop_consecutive=True
)

compressed_prompt = results['compressed_prompt']
compress_rate = results['rate']
