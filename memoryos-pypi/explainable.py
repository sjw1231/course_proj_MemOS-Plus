import os
import json
from memoryos import Memoryos

# --- Basic Configuration ---
USER_ID = "demo_user"
ASSISTANT_ID = "demo_assistant"
DEFAULT_OPENAI_API_KEY = ""
DEFAULT_OPENAI_BASE_URL = "https://openrouter.ai/api/v1"

API_KEY = os.environ.get("OPENAI_API_KEY", DEFAULT_OPENAI_API_KEY)
BASE_URL = os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
DATA_STORAGE_PATH = "data_memoryos_demo"
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COMPRESS_MODE = False


# ====== Explainable Memory Agent ======
EXPLAINABLE_SYSTEM_PROMPT = """You are an AI assistant with an explicit memory system.
You will receive:
- The user's new query.
- Recent short-term dialogue history.
- Retrieved mid-term memories (historical dialogue pages).
- Retrieved long-term user and assistant knowledge.

Your goals:
1. Give the best possible answer to the user's query.
2. EXPLAIN which memories you used and why.

Return your output as a JSON object with two fields:
- "answer": the final reply to the user (natural language).
- "explanation": a short explanation (2-6 bullet points) describing:
    - which memories (short-term / mid-term / user knowledge / assistant knowledge) you relied on,
    - and how they influenced your answer.

Make sure the JSON is valid and does NOT contain trailing commas.
"""

EXPLAINABLE_USER_PROMPT_TEMPLATE = """[User Query]
{query}

[Short-Term Dialogue History]
{history_text}

[Retrieved Mid-Term Memories]
{retrieval_text}

[User Background (Profile + Knowledge)]
{background_context}

[Assistant Knowledge]
{assistant_knowledge_text}

You are answering as the user's {relationship}.
"""


def get_response_with_explanations(
    memo: Memoryos,
    query: str,
    relationship_with_user: str = "friend",
    user_conversation_meta_data=None,
):
    """
    使用已有的 Memoryos 实例，返回:
      - answer: 给用户的回答
      - explanation: 对“用了哪些记忆、为什么”的解释
      - debug_info: 原始检索结果（方便你在作业里展示）
    """

    # 1. 检索上下文（直接用 Memoryos 内部的 retriever）
    retrieval_results = memo.retriever.retrieve_context(
        user_query=query,
        user_id=memo.user_id,
    )
    retrieved_pages = retrieval_results["retrieved_pages"]
    retrieved_user_knowledge = retrieval_results["retrieved_user_knowledge"]
    retrieved_assistant_knowledge = retrieval_results["retrieved_assistant_knowledge"]

    # 2. 短期历史（STM）
    short_term_history = memo.short_term_memory.get_all()
    history_text = "\n".join([
        f"User: {qa.get('user_input', '')}\nAssistant: {qa.get('agent_response', '')} (Time: {qa.get('timestamp', '')})"
        for qa in short_term_history
    ])

    # 3. 中期记忆（MTM）格式化
    retrieval_text = "\n".join([
        f"[Historical Memory]\nUser: {page.get('user_input', '')}\nAssistant: {page.get('agent_response', '')}\nTime: {page.get('timestamp', '')}\nConversation chain overview: {page.get('meta_info','N/A')}"
        for page in retrieved_pages
    ])

    # 4. 用户画像 + 用户知识（LTM）
    user_profile_text = memo.user_long_term_memory.get_raw_user_profile(memo.user_id)
    if not user_profile_text or user_profile_text.lower() == "none":
        user_profile_text = "No detailed profile available yet."

    user_knowledge_background = ""
    if retrieved_user_knowledge:
        user_knowledge_background = "\n[Relevant User Knowledge Entries]\n"
        for kn_entry in retrieved_user_knowledge:
            user_knowledge_background += f"- {kn_entry['knowledge']} (Recorded: {kn_entry['timestamp']})\n"

    background_context = f"[User Profile]\n{user_profile_text}\n{user_knowledge_background}"

    # 5. 助手知识（assistant LTM）
    assistant_knowledge_text = "[Assistant Knowledge Base]\n"
    if retrieved_assistant_knowledge:
        for ak_entry in retrieved_assistant_knowledge:
            assistant_knowledge_text += f"- {ak_entry['knowledge']} (Recorded: {ak_entry['timestamp']})\n"
    else:
        assistant_knowledge_text += "- No relevant assistant knowledge found for this query.\n"


    user_prompt_text = EXPLAINABLE_USER_PROMPT_TEMPLATE.format(
        query=query,
        history_text=history_text,
        retrieval_text=retrieval_text,
        background_context=background_context,
        assistant_knowledge_text=assistant_knowledge_text,
        relationship=relationship_with_user,
    )

    messages = [
        {"role": "system", "content": EXPLAINABLE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt_text},
    ]

    # 7. 调 LLM
    print("Explainable agent: Calling LLM for answer + explanation...")
    raw_content = memo.client.chat_completion(
        model=memo.llm_model,
        messages=messages,
        temperature=0.4,
        max_tokens=1500,
    )

    # 8. 解析 JSON
    try:
        parsed = json.loads(raw_content)
        answer = parsed.get("answer", raw_content)
        explanation = parsed.get("explanation", "")
    except Exception:
        answer = raw_content
        explanation = "Failed to parse JSON explanation; model returned non-JSON text."

    # 9. 把真正回答写回记忆（解释可以不存）
    from memoryos import get_timestamp  # 如果你有暴露；否则用 utils.get_timestamp
    try:
        memo.add_memory(user_input=query, agent_response=answer, timestamp=get_timestamp())
    except Exception:
        # 保底：不带 timestamp 也可以
        memo.add_memory(user_input=query, agent_response=answer)

    debug_info = {
        "retrieved_pages": retrieved_pages,
        "retrieved_user_knowledge": retrieved_user_knowledge,
        "retrieved_assistant_knowledge": retrieved_assistant_knowledge,
    }

    return answer, explanation, debug_info


def init_memoryos():
    print("Initializing MemoryOS (explainable demo)...")
    memo = Memoryos(
        user_id=USER_ID,
        openai_api_key=API_KEY,
        openai_base_url=BASE_URL,
        data_storage_path=DATA_STORAGE_PATH,
        llm_model=LLM_MODEL,
        assistant_id=ASSISTANT_ID,
        short_term_capacity=7,
        mid_term_heat_threshold=1000,  
        retrieval_queue_capacity=10,
        long_term_knowledge_capacity=100,
        mid_term_similarity_threshold=0.6,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        compress_mode=COMPRESS_MODE,
    )
    print("MemoryOS initialized!\n")
    return memo


def explainable_demo():
    memo = init_memoryos()

    print("Adding some memories (same as test.py)...")
    memo.add_memory(
        user_input="Hi! I'm Tom, I work as a data scientist in San Francisco.",
        agent_response="Hello Tom! Nice to meet you. Data science is such an exciting field. What kind of data do you work with?"
    )
    memo.add_memory(
        user_input="I love hiking on weekends, especially in the mountains.",
        agent_response="That sounds wonderful! Do you have a favorite trail or mountain you like to visit?"
    )
    memo.add_memory(
        user_input="Recently, I've been reading a lot about artificial intelligence.",
        agent_response="AI is a fascinating topic! Are you interested in any specific area of AI?"
    )
    memo.add_memory(
        user_input="My favorite food is sushi, especially salmon nigiri.",
        agent_response="Sushi is delicious! Have you ever tried making it at home?"
    )
    memo.add_memory(
        user_input="I have a golden retriever named Max.",
        agent_response="Max must be adorable! How old is he?"
    )
    memo.add_memory(
        user_input="I traveled to Japan last year and visited Tokyo and Kyoto.",
        agent_response="That must have been an amazing experience! What did you enjoy most about Japan?"
    )
    memo.add_memory(
        user_input="I'm currently learning how to play the guitar.",
        agent_response="That's awesome! What songs are you practicing right now?"
    )
    memo.add_memory(
        user_input="I usually start my day with a cup of black coffee.",
        agent_response="Coffee is a great way to kickstart the day! Do you prefer it hot or iced?"
    )
    memo.add_memory(
        user_input="My favorite movie genre is science fiction.",
        agent_response="Sci-fi movies can be so imaginative! Do you have a favorite film?"
    )
    memo.add_memory(
        user_input="I enjoy painting landscapes in my free time.",
        agent_response="Painting is such a creative hobby! Do you use oils, acrylics, or watercolors?"
    )

    test_query = "What do you remember about my job?"
    print(f"\nUser: {test_query}")

    answer, explanation, debug_info = get_response_with_explanations(
        memo,
        query=test_query,
        relationship_with_user="friend",
    )

    print(f"\nAssistant: {answer}")
    print("\n[Memory Explanation]")
    print(explanation)


if __name__ == "__main__":
    explainable_demo()
