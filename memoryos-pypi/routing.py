import os
import json
import shutil

from memoryos import Memoryos

# --- Basic Configuration (与原 test.py 风格保持一致) ---
USER_ID = "routing_demo_user"
ASSISTANT_ID = "routing_demo_assistant"

DEFAULT_OPENAI_API_KEY = ""
DEFAULT_OPENAI_BASE_URL = "https://openrouter.ai/api/v1"

API_KEY = os.environ.get("OPENAI_API_KEY", DEFAULT_OPENAI_API_KEY)
BASE_URL = os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)

# 两个不同的数据目录，避免相互污染
DATA_PATH_ROUTING_ON = "data_memoryos_routing_on"
DATA_PATH_ROUTING_OFF = "data_memoryos_routing_off"

LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COMPRESS_MODE = False  # 为了方便观察内容，这里关掉压缩


# ===== 工具函数 =====

def reset_data_dir(path: str):
    """删除旧数据目录，保证每次测试都是干净的。"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def build_memoryos_instance(data_path: str, routing_enabled: bool) -> Memoryos:
    """
    构造 Memoryos 实例。
    - routing_enabled = True  : 使用当前 routing 逻辑（rule / LLM）
    - routing_enabled = False : 通过 monkey-patch 把 classifier 固定为 'personal'，
                                等价于“全部写入 LTM”，模拟旧版行为。
    """
    if routing_enabled:
        print(f"\n[INIT] Building Memoryos with routing ENABLED (knowledge_classifier_mode='rule')\n")
        memo = Memoryos(
            user_id=USER_ID,
            openai_api_key=API_KEY,
            openai_base_url=BASE_URL,
            data_storage_path=data_path,
            llm_model=LLM_MODEL,
            assistant_id=ASSISTANT_ID,
            short_term_capacity=7,
            mid_term_heat_threshold=0.0,      # 低一点，让 profile/knowledge 更新更容易触发
            retrieval_queue_capacity=10,
            long_term_knowledge_capacity=200,
            mid_term_similarity_threshold=0.6,
            embedding_model_name=EMBEDDING_MODEL_NAME,
            compress_mode=COMPRESS_MODE,
            knowledge_classifier_mode="rule",  # 你在 memoryos.py 里加过这个参数
        )
        return memo
    else:
        print(f"\n[INIT] Building Memoryos with routing DISABLED (monkey-patched classifier → 'personal')\n")

        # 备份原有分类器
        original_classifier = getattr(Memoryos, "_classify_knowledge_line", None)

        # 定义一个永远返回 "personal" 的分类器，等价于“全都写进 LTM”
        def always_personal(self, line: str) -> str:
            return "personal"

        Memoryos._classify_knowledge_line = always_personal

        memo = Memoryos(
            user_id=USER_ID,
            openai_api_key=API_KEY,
            openai_base_url=BASE_URL,
            data_storage_path=data_path,
            llm_model=LLM_MODEL,
            assistant_id=ASSISTANT_ID,
            short_term_capacity=7,
            mid_term_heat_threshold=0.0,
            retrieval_queue_capacity=10,
            long_term_knowledge_capacity=200,
            mid_term_similarity_threshold=0.6,
            embedding_model_name=EMBEDDING_MODEL_NAME,
            compress_mode=COMPRESS_MODE,
            knowledge_classifier_mode="rule",  # 实际上已经被我们 patch 掉了
        )

        # 用完恢复，避免影响其它代码
        if original_classifier is not None:
            Memoryos._classify_knowledge_line = original_classifier

        return memo


def load_user_ltm_json(data_path: str) -> dict:
    """直接读 long_term_user.json，方便对比 LTM 内容。"""
    ltm_file = os.path.join(
        os.path.abspath(data_path),
        "users",
        USER_ID,
        "long_term_user.json",
    )
    if not os.path.exists(ltm_file):
        print(f"[WARN] LTM file not found: {ltm_file}")
        return {}
    with open(ltm_file, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"[WARN] Failed to parse JSON from {ltm_file}")
            return {}


def print_ltm_summary(ltm_data: dict, headline: str):
    """把 LTM 的结构清晰打印出来，重点看 user_profiles 和 knowledge_base。"""
    print("\n" + "=" * 80)
    print(headline)
    print("=" * 80)

    raw_user_profiles = ltm_data.get("user_profiles", None)
    knowledge_base = ltm_data.get("knowledge_base", [])
    assistant_knowledge = ltm_data.get("assistant_knowledge", [])

    # ---- 规范化 user_profiles 成 list[snapshot] ----
    snapshots = []
    if isinstance(raw_user_profiles, list):
        snapshots = raw_user_profiles
    elif isinstance(raw_user_profiles, dict):
        # 情况 A：这个 dict 本身就是一个 profile 对象
        if any(k in raw_user_profiles for k in ["profile", "content", "data", "summary", "text"]):
            snapshots = [raw_user_profiles]
        else:
            # 情况 B：可能是 {id1: {...}, id2: {...}}
            snapshots = list(raw_user_profiles.values())
    else:
        snapshots = []

    print("\n[User Profiles] ({} snapshots)".format(len(snapshots)))
    if not snapshots:
        print("  (none)")
    else:
        latest = snapshots[-1]
        if isinstance(latest, dict):
            text = (
                latest.get("profile")
                or latest.get("content")
                or latest.get("data")
                or latest.get("summary")
                or latest.get("text")
                or str(latest)
            )
        else:
            text = str(latest)
        print("  [LATEST PROFILE]")
        print("  " + text.replace("\n", "\n  "))

    # ---- 用户知识条目（我们 routing 控制的其实就是这里）----
    print("\n[Knowledge Base Entries] ({} entries)".format(len(knowledge_base)))
    if not knowledge_base:
        print("  (none)")
    else:
        for i, item in enumerate(knowledge_base, start=1):
            if isinstance(item, dict):
                k_type = item.get("type", "unknown")
                k_text = item.get("knowledge", str(item))
                ts = item.get("timestamp", "")
                print(f"  {i:02d}. [{k_type}] {k_text}")
                if ts:
                    print(f"       (timestamp: {ts})")
            else:
                print(f"  {i:02d}. {item}")

    # ---- 助手知识（顺便看看）----
    print("\n[Assistant Knowledge Entries] ({} entries)".format(len(assistant_knowledge)))
    if not assistant_knowledge:
        print("  (none)")
    else:
        for i, item in enumerate(assistant_knowledge, start=1):
            if isinstance(item, dict):
                k_text = item.get("knowledge", str(item))
            else:
                k_text = str(item)
            print(f"  {i:02d}. {k_text}")

    print("\n[Raw JSON keys in LTM]:", list(ltm_data.keys()))
    print("=" * 80 + "\n")


def print_stm_summary(memo: Memoryos, headline: str):
    """打印当前 STM 中的 QA 条数和前几条内容。"""
    stm_all = memo.short_term_memory.get_all()
    print("\n" + "-" * 80)
    print(headline)
    print("-" * 80)
    print(f"[STM] Number of QA pairs in short-term memory: {len(stm_all)}")

    for i, qa in enumerate(stm_all[:5], start=1):
        u = qa.get("user_input", "")[:100].replace("\n", " ")
        a = qa.get("agent_response", "")[:100].replace("\n", " ")
        print(f"  STM[{i:02d}] User: {u}")
        print(f"           Assistant: {a}")
    print("-" * 80 + "\n")


# ===== 场景对话（虚构的 NLP 研究者，与你本人无关） =====

SCENARIO_QA = [
    # 明确个人身份 / 背景（虚构的 NLP 研究者）
    (
        "Hi! I'm Alex, a 2nd-year PhD student working on natural language processing.",
        "Nice to meet you Alex! A 2nd-year PhD in NLP sounds like a lot of interesting research."
    ),
    (
        "I currently live in a small city near the university campus, but I grew up in a different country.",
        "Got it, you live near your university now and you grew up in another country."
    ),
    # 明确长期目标 / 兴趣
    (
        "My long-term goal is to build language models that are more reliable and controllable in real-world applications.",
        "That's a great long-term goal. Reliability and controllability are crucial for real-world language models."
    ),
    (
        "Outside of research, I really enjoy hiking and board games, but I hate early morning meetings.",
        "Noted, you enjoy hiking and board games, and you prefer to avoid early morning meetings."
    ),
    # 明确技术偏好
    (
        "I prefer using PyTorch and the HuggingFace ecosystem instead of TensorFlow for most of my NLP experiments.",
        "Okay, I'll keep in mind that you prefer PyTorch and HuggingFace over TensorFlow for NLP work."
    ),
    # 接下来是知识性 / 定义类内容
    (
        "Reinforcement learning from human feedback is a training paradigm where a model is optimized using a reward model derived from human preference data.",
        "Yes, RLHF is indeed about optimizing a model with a reward model trained on human preference data."
    ),
    (
        "Self-consistency decoding is a technique where multiple reasoning paths are sampled and the most consistent answer across paths is selected.",
        "Correct, self-consistency decoding samples multiple reasoning paths and chooses the most consistent answer."
    ),
    (
        "The Transformer architecture was introduced in 2017 by Vaswani et al. in the paper 'Attention is All You Need'.",
        "Right, the Transformer architecture comes from the 2017 paper 'Attention is All You Need'."
    ),
    (
        "The BLEU score is a metric defined as a modified n-gram precision measure with a brevity penalty, commonly used for machine translation evaluation.",
        "Exactly, BLEU is a modified n-gram precision with a brevity penalty, widely used for MT evaluation."
    ),
    # 再加一点混合信息（个人偏好 + 研究方向）
    (
        "Even though I collaborate on a lot of speech and multimodal projects, my favorite part is still analyzing error cases in large language models.",
        "So you work on speech and multimodal projects, but your favorite part remains analyzing LLM error cases."
    ),
]


# ===== Debug: 直接测试 routing 对混合句子的分类效果 =====

DEBUG_KNOWLEDGE_LINES = [
    # 明显 personal
    "My name is Alex and I am a 2nd-year PhD student in NLP.",
    "I hate waking up before 8 a.m.",
    # 明显 factual / 定义
    "Reinforcement learning from human feedback is a training paradigm where a model is optimized using a reward model derived from human preference data.",
    "Self-consistency decoding is a technique where multiple reasoning paths are sampled and the most consistent answer is selected.",
    "The Transformer architecture was introduced in 2017 by Vaswani et al. in the paper 'Attention is All You Need'.",
    "The BLEU score is a metric defined as a modified n-gram precision with a brevity penalty, commonly used for machine translation evaluation.",
]


def debug_route_and_add(memo: Memoryos, line: str):
    """
    使用 Memoryos._classify_knowledge_line 对单句做分类，
    并根据结果决定是否写入 user LTM，同时打印 debug 信息。
    """
    try:
        mem_type = memo._classify_knowledge_line(line)
    except AttributeError:
        # 如果没这个方法，直接全部当 personal（不应该发生，因为你已经加过）
        mem_type = "personal"

    print(f"[DEBUG ROUTING] type={mem_type:<9} | line={line}")

    if mem_type == "personal":
        memo.user_long_term_memory.add_user_knowledge(line)
    else:
        print("                -> skipped for user LTM (non-personal)\n")


def run_debug_routing(memo: Memoryos):
    print("\n[DEBUG] Injecting mixed knowledge lines through classifier and LTM...\n")
    for l in DEBUG_KNOWLEDGE_LINES:
        debug_route_and_add(memo, l)


# ===== 主场景：跑一遍 MemoryOS，然后再跑 debug routing，最后对比 LTM =====

def run_scenario(routing_enabled: bool, data_path: str):
    """
    跑一遍场景：
    1. 初始化 Memoryos
    2. 依次 add_memory（正常对话产生记忆）
    3. 强制触发 mid-term → long-term 分析
    4. 额外 debug：手动用 routing 注入一批混合 personal / factual 句子
    5. 打印 STM 概况 + LTM JSON 概况
    """
    reset_data_dir(data_path)
    memo = build_memoryos_instance(data_path=data_path, routing_enabled=routing_enabled)

    print("Adding scenario QA pairs...\n")
    for i, (u, a) in enumerate(SCENARIO_QA, start=1):
        print(f"--- QA #{i} ---")
        print(f"User:      {u}")
        print(f"Assistant: {a}\n")
        memo.add_memory(user_input=u, agent_response=a)

    # 为了确保 LTM 写入逻辑跑一遍，主动触发分析
    print("\n[FORCE] Trigger mid-term analysis → profile/knowledge update")
    memo.force_mid_term_analysis()

    # Debug: 直接测试 routing 对混合知识句子的分类 & 写入
    run_debug_routing(memo)

    # 打印 STM 概况
    title_stm = "🔍 STM STATUS (with routing ENABLED)" if routing_enabled else "🔍 STM STATUS (with routing DISABLED)"
    print_stm_summary(memo, title_stm)

    # 读取并打印 LTM JSON 概况
    ltm_data = load_user_ltm_json(data_path)
    title_ltm = "🌱 USER LONG-TERM MEMORY (routing ENABLED)" if routing_enabled else "🌳 USER LONG-TERM MEMORY (routing DISABLED — everything kept)"
    print_ltm_summary(ltm_data, title_ltm)


def main():
    print("=" * 80)
    print("MEMORY ROUTING TEST (with debug injection)")
    print("=" * 80)
    print("This script will run the same conversation twice:\n"
          "  1) With memory routing ENABLED (rule-based / LLM classifier).\n"
          "  2) With memory routing DISABLED (all knowledge lines treated as personal).\n"
          "Then it injects a set of mixed personal/factual sentences directly through the\n"
          "routing pipeline and compares what ends up in long-term memory (LTM).\n")

    # 1) Routing 开启
    print("\n\n==================== 🧠 RUN 1: ROUTING ENABLED ====================\n")
    run_scenario(routing_enabled=True, data_path=DATA_PATH_ROUTING_ON)

    # 2) Routing 关闭（模拟旧行为）
    print("\n\n==================== 📦 RUN 2: ROUTING DISABLED (NO FILTER) ====================\n")
    run_scenario(routing_enabled=False, data_path=DATA_PATH_ROUTING_OFF)

    print("\n\n✅ Done. Please compare:")
    print("   - The [DEBUG ROUTING] logs between RUN 1 and RUN 2.")
    print("   - The 'Knowledge Base Entries' sections in the two USER LONG-TERM MEMORY blocks.\n")
    print("   With routing ENABLED, factual/definition lines should be skipped and not stored")
    print("   as user knowledge, while with routing DISABLED, everything is kept.\n")


if __name__ == "__main__":
    main()
