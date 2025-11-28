import json
import numpy as np
from collections import defaultdict
import faiss
import heapq
from datetime import datetime

try:
    from .utils import (
        get_timestamp, generate_id, get_embedding, normalize_vector, 
        compute_time_decay, ensure_directory_exists, OpenAIClient
    )
except ImportError:
    from utils import (
        get_timestamp, generate_id, get_embedding, normalize_vector, 
        compute_time_decay, ensure_directory_exists, OpenAIClient
    )

# Heat computation constants (can be tuned or made configurable)
HEAT_ALPHA = 1.0
HEAT_BETA = 1.0
HEAT_GAMMA = 1
RECENCY_TAU_HOURS = 24 # For R_recency calculation in compute_segment_heat

def compute_segment_heat(session, alpha=HEAT_ALPHA, beta=HEAT_BETA, gamma=HEAT_GAMMA, tau_hours=RECENCY_TAU_HOURS):
    N_visit = session.get("N_visit", 0)
    L_interaction = session.get("L_interaction", 0)
    
    # Calculate recency based on last_visit_time
    R_recency = 1.0 # Default if no last_visit_time
    if session.get("last_visit_time"):
        R_recency = compute_time_decay(session["last_visit_time"], get_timestamp(), tau_hours)
    
    session["R_recency"] = R_recency # Update session's recency factor
    return alpha * N_visit + beta * L_interaction + gamma * R_recency

class MidTermMemory:
    def __init__(self, file_path: str, client: OpenAIClient, max_capacity=2000, embedding_model_name: str = "all-MiniLM-L6-v2", embedding_model_kwargs: dict = None):
        self.file_path = file_path
        ensure_directory_exists(self.file_path)
        self.client = client
        self.max_capacity = max_capacity
        self.sessions = {} # {session_id: session_object}
        self.access_frequency = defaultdict(int) # {session_id: access_count_for_lfu}
        self.heap = []  # Min-heap storing (-H_segment, session_id) for hottest segments

        self.embedding_model_name = embedding_model_name
        self.embedding_model_kwargs = embedding_model_kwargs if embedding_model_kwargs is not None else {}
        self.load()
        
        # added
        self.graph_layer = GraphMemoryLayer(similarity_threshold=0.8)
        self.load()


    def get_page_by_id(self, page_id):
        for session in self.sessions.values():
            for page in session.get("details", []):
                if page.get("page_id") == page_id:
                    return page
        return None

    def update_page_connections(self, prev_page_id, next_page_id):
        if prev_page_id:
            prev_page = self.get_page_by_id(prev_page_id)
            if prev_page:
                prev_page["next_page"] = next_page_id
        if next_page_id:
            next_page = self.get_page_by_id(next_page_id)
            if next_page:
                next_page["pre_page"] = prev_page_id
        # self.save() # Avoid saving on every minor update; save at higher level operations

    def evict_lfu(self):
        if not self.access_frequency or not self.sessions:
            return
        
        lfu_sid = min(self.access_frequency, key=self.access_frequency.get)
        print(f"MidTermMemory: LFU eviction. Session {lfu_sid} has lowest access frequency.")
        
        if lfu_sid not in self.sessions:
            del self.access_frequency[lfu_sid] # Clean up access frequency if session already gone
            self.rebuild_heap()
            return
        
        session_to_delete = self.sessions.pop(lfu_sid) # Remove from sessions
        del self.access_frequency[lfu_sid] # Remove from LFU tracking

        # [Graph Mod] 同步从图中移除节点，防止产生悬空边
        self.graph_layer.remove_node(lfu_sid)
        
        # Clean up page connections if this session's pages were linked
        for page in session_to_delete.get("details", []):
            prev_page_id = page.get("pre_page")
            next_page_id = page.get("next_page")
            # If a page from this session was linked to an external page, nullify the external link
            if prev_page_id and not self.get_page_by_id(prev_page_id): # Check if prev page is still in memory
                 # This case should ideally not happen if connections are within sessions or handled carefully
                 pass 
            if next_page_id and not self.get_page_by_id(next_page_id):
                 pass
            # More robustly, one might need to search all other sessions if inter-session linking was allowed
            # For now, assuming internal consistency or that MemoryOS class manages higher-level links

        self.rebuild_heap()
        self.save()
        print(f"MidTermMemory: Evicted session {lfu_sid}.")

    def add_session(self, summary, details, summary_keywords=None):
        session_id = generate_id("session")
        summary_vec = get_embedding(
            summary, 
            model_name=self.embedding_model_name, 
            **self.embedding_model_kwargs
        )
        summary_vec = normalize_vector(summary_vec).tolist()
        summary_keywords = summary_keywords if summary_keywords is not None else []
        
        processed_details = []
        for page_data in details:
            page_id = page_data.get("page_id", generate_id("page"))
            
            # 检查是否已有embedding，避免重复计算
            if "page_embedding" in page_data and page_data["page_embedding"]:
                print(f"MidTermMemory: Reusing existing embedding for page {page_id}")
                inp_vec = page_data["page_embedding"]
                # 确保embedding是normalized的
                if isinstance(inp_vec, list):
                    inp_vec_np = np.array(inp_vec, dtype=np.float32)
                    if np.linalg.norm(inp_vec_np) > 1.1 or np.linalg.norm(inp_vec_np) < 0.9:  # 检查是否需要重新normalize
                        inp_vec = normalize_vector(inp_vec_np).tolist()
            else:
                print(f"MidTermMemory: Computing new embedding for page {page_id}")
                full_text = f"User: {page_data.get('user_input','')} Assistant: {page_data.get('agent_response','')}"
                inp_vec = get_embedding(
                    full_text,
                    model_name=self.embedding_model_name,
                    **self.embedding_model_kwargs
                )
                inp_vec = normalize_vector(inp_vec).tolist()
            
            # 使用已有keywords或设置为空（由multi-summary提供）
            if "page_keywords" in page_data and page_data["page_keywords"]:
                print(f"MidTermMemory: Using existing keywords for page {page_id}")
                page_keywords = page_data["page_keywords"]
            else:
                print(f"MidTermMemory: Setting empty keywords for page {page_id} (will be filled by multi-summary)")
                page_keywords = []
            
            processed_page = {
                **page_data, # Carry over existing fields like user_input, agent_response, timestamp
                "page_id": page_id,
                "page_embedding": inp_vec,
                "page_keywords": page_keywords,
                "preloaded": page_data.get("preloaded", False), # Preserve if passed
                "analyzed": page_data.get("analyzed", False),   # Preserve if passed
                # pre_page, next_page, meta_info are handled by DynamicUpdater
            }
            processed_details.append(processed_page)
        
        current_ts = get_timestamp()
        session_obj = {
            "id": session_id,
            "summary": summary,
            "summary_keywords": summary_keywords,
            "summary_embedding": summary_vec,
            "details": processed_details,
            "L_interaction": len(processed_details),
            "R_recency": 1.0, # Initial recency
            "N_visit": 0,
            "H_segment": 0.0, # Initial heat, will be computed
            "timestamp": current_ts, # Creation timestamp
            "last_visit_time": current_ts, # Also initial last_visit_time for recency calc
            "access_count_lfu": 0 # For LFU eviction policy
        }
        session_obj["H_segment"] = compute_segment_heat(session_obj)
        self.sessions[session_id] = session_obj
        self.access_frequency[session_id] = 0 # Initialize for LFU
        heapq.heappush(self.heap, (-session_obj["H_segment"], session_id)) # Use negative heat for max-heap behavior
        
        # [Graph Mod] 将新 Session 作为节点加入图，并计算相似度建立连接
        # 注意：graph_layer.add_node 内部会计算 cosine similarity 并连边
        self.graph_layer.add_node(session_id, np.array(summary_vec))
        
        print(f"MidTermMemory: Added new session {session_id}. Initial heat: {session_obj['H_segment']:.2f}.")
        if len(self.sessions) > self.max_capacity:
            self.evict_lfu()
        self.save()
        return session_id
    
    
    def rebuild_heap(self):
        self.heap = []
        for sid, session_data in self.sessions.items():
            # Ensure H_segment is up-to-date before rebuilding heap if necessary
            # session_data["H_segment"] = compute_segment_heat(session_data)
            heapq.heappush(self.heap, (-session_data["H_segment"], sid))
        # heapq.heapify(self.heap) # Not needed if pushing one by one
        # No save here, it's an internal operation often followed by other ops that save

    def insert_pages_into_session(self, summary_for_new_pages, keywords_for_new_pages, pages_to_insert, 
                                  similarity_threshold=0.6, keyword_similarity_alpha=1.0):
        if not self.sessions: # If no existing sessions, just add as a new one
            print("MidTermMemory: No existing sessions. Adding new session directly.")
            return self.add_session(summary_for_new_pages, pages_to_insert, keywords_for_new_pages)

        new_summary_vec = get_embedding(
            summary_for_new_pages,
            model_name=self.embedding_model_name,
            **self.embedding_model_kwargs
        )
        new_summary_vec = normalize_vector(new_summary_vec)
        
        best_sid = None
        best_overall_score = -1

        for sid, existing_session in self.sessions.items():
            existing_summary_vec = np.array(existing_session["summary_embedding"], dtype=np.float32)
            semantic_sim = float(np.dot(existing_summary_vec, new_summary_vec))
            
            # Keyword similarity (Jaccard index based)
            existing_keywords = set(existing_session.get("summary_keywords", []))
            new_keywords_set = set(keywords_for_new_pages)
            s_topic_keywords = 0
            if existing_keywords and new_keywords_set:
                intersection = len(existing_keywords.intersection(new_keywords_set))
                union = len(existing_keywords.union(new_keywords_set))
                if union > 0:
                    s_topic_keywords = intersection / union 
            
            overall_score = semantic_sim + keyword_similarity_alpha * s_topic_keywords
            
            if overall_score > best_overall_score:
                best_overall_score = overall_score
                best_sid = sid
        
        if best_sid and best_overall_score >= similarity_threshold:
            print(f"MidTermMemory: Merging pages into session {best_sid}. Score: {best_overall_score:.2f} (Threshold: {similarity_threshold})")
            target_session = self.sessions[best_sid]
            
            processed_new_pages = []
            for page_data in pages_to_insert:
                page_id = page_data.get("page_id", generate_id("page")) # Use existing or generate new ID
                
                # 检查是否已有embedding，避免重复计算
                if "page_embedding" in page_data and page_data["page_embedding"]:
                    print(f"MidTermMemory: Reusing existing embedding for page {page_id}")
                    inp_vec = page_data["page_embedding"]
                    # 确保embedding是normalized的
                    if isinstance(inp_vec, list):
                        inp_vec_np = np.array(inp_vec, dtype=np.float32)
                        if np.linalg.norm(inp_vec_np) > 1.1 or np.linalg.norm(inp_vec_np) < 0.9:  # 检查是否需要重新normalize
                            inp_vec = normalize_vector(inp_vec_np).tolist()
                else:
                    print(f"MidTermMemory: Computing new embedding for page {page_id}")
                    full_text = f"User: {page_data.get('user_input','')} Assistant: {page_data.get('agent_response','')}"
                    inp_vec = get_embedding(
                        full_text,
                        model_name=self.embedding_model_name,
                        **self.embedding_model_kwargs
                    )
                    inp_vec = normalize_vector(inp_vec).tolist()
                
                # 使用已有keywords或继承session的keywords
                if "page_keywords" in page_data and page_data["page_keywords"]:
                    print(f"MidTermMemory: Using existing keywords for page {page_id}")
                    page_keywords_current = page_data["page_keywords"]
                else:
                    print(f"MidTermMemory: Using session keywords for page {page_id}")
                    page_keywords_current = keywords_for_new_pages

                processed_page = {
                    **page_data, # Carry over existing fields
                    "page_id": page_id,
                    "page_embedding": inp_vec,
                    "page_keywords": page_keywords_current,
                    # analyzed, preloaded flags should be part of page_data if set
                }
                target_session["details"].append(processed_page)
                processed_new_pages.append(processed_page)

            target_session["L_interaction"] += len(pages_to_insert)
            target_session["last_visit_time"] = get_timestamp() # Update last visit time on modification
            target_session["H_segment"] = compute_segment_heat(target_session)
            self.rebuild_heap() # Rebuild heap as heat has changed
            self.save()
            return best_sid
        else:
            print(f"MidTermMemory: No suitable session to merge (best score {best_overall_score:.2f} < threshold {similarity_threshold}). Creating new session.")
            return self.add_session(summary_for_new_pages, pages_to_insert, keywords_for_new_pages)
    
    
    def search_sessions(self, query_text, segment_similarity_threshold=0.1, page_similarity_threshold=0.1, 
                          top_k_sessions=5, keyword_alpha=1.0, recency_tau_search=3600):
        if not self.sessions:
            return []

        query_vec = get_embedding(
            query_text,
            model_name=self.embedding_model_name,
            **self.embedding_model_kwargs
        )
        query_vec = normalize_vector(query_vec)
        query_keywords = set()  # Keywords extraction removed, relying on semantic similarity

        candidate_sessions = []
        session_ids = list(self.sessions.keys())
        if not session_ids: return []

        summary_embeddings_list = [self.sessions[s]["summary_embedding"] for s in session_ids]
        summary_embeddings_np = np.array(summary_embeddings_list, dtype=np.float32)

        dim = summary_embeddings_np.shape[1]
        index = faiss.IndexFlatIP(dim) # Inner product for similarity
        index.add(summary_embeddings_np)
        
        query_arr_np = np.array([query_vec], dtype=np.float32)
        distances, indices = index.search(query_arr_np, min(top_k_sessions, len(session_ids)))

        results = []
        current_time_str = get_timestamp()

        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            
            session_id = session_ids[idx]
            session = self.sessions[session_id]
            semantic_sim_score = float(distances[0][i]) # This is the dot product

            # Keyword similarity for session summary
            session_keywords = set(session.get("summary_keywords", []))
            s_topic_keywords = 0
            if query_keywords and session_keywords:
                intersection = len(query_keywords.intersection(session_keywords))
                union = len(query_keywords.union(session_keywords))
                if union > 0: s_topic_keywords = intersection / union
            
            # Time decay for session recency in search scoring
            # time_decay_factor = compute_time_decay(session["timestamp"], current_time_str, tau_hours=recency_tau_search)
            
            # Combined score for session relevance
            session_relevance_score =  (semantic_sim_score + keyword_alpha * s_topic_keywords)

            if session_relevance_score >= segment_similarity_threshold:
                matched_pages_in_session = []
                for page in session.get("details", []):
                    page_embedding = np.array(page["page_embedding"], dtype=np.float32)
                    # page_keywords = set(page.get("page_keywords", []))
                    
                    page_sim_score = float(np.dot(page_embedding, query_vec))
                    # Can also add keyword sim for pages if needed, but keeping it simpler for now

                    if page_sim_score >= page_similarity_threshold:
                        matched_pages_in_session.append({"page_data": page, "score": page_sim_score})
                
                if matched_pages_in_session:
                    # Update session access stats
                    session["N_visit"] += 1
                    session["last_visit_time"] = current_time_str
                    session["access_count_lfu"] = session.get("access_count_lfu", 0) + 1
                    self.access_frequency[session_id] = session["access_count_lfu"]
                    session["H_segment"] = compute_segment_heat(session)
                    self.rebuild_heap() # Heat changed
                    
                    results.append({
                        "session_id": session_id,
                        "session_summary": session["summary"],
                        "session_relevance_score": session_relevance_score,
                        "matched_pages": sorted(matched_pages_in_session, key=lambda x: x["score"], reverse=True) # Sort pages by score
                    })
                    
        # # [新增] Graph RAG 增强：获取这些 Top Session 的邻居
        # matched_session_ids = [s["session_id"] for s in results]
        # neighbor_sessions = self.get_direct_neighbors(matched_session_ids)
        
        # # [新增] 合并上下文 (去重)
        # seen_ids = set([s["session_id"] for s in results])
        # final_context_sessions = results # 这里是 search_sessions 返回的特定格式
        # for ns in neighbor_sessions:
        #     if ns['id'] not in seen_ids:
        #         matched_pages_in_session = []
        #         for page in ns.get("details", []):
        #             page_embedding = np.array(page["page_embedding"], dtype=np.float32)
        #             # page_keywords = set(page.get("page_keywords", []))
                    
        #             page_sim_score = float(np.dot(page_embedding, query_vec))
        #             # Can also add keyword sim for pages if needed, but keeping it simpler for now

        #             if page_sim_score >= page_similarity_threshold:
        #                 matched_pages_in_session.append({"page_data": page, "score": page_sim_score})
                
        #         # 将 neighbor session 转换为 search_results 相同的格式以便统一处理
        #         final_context_sessions.append({
        #             "session_id": ns['id'],
        #             "session_summary": ns['summary'],
        #             "session_relevance_score": 0.5, # 邻居赋予一个默认权重，或使用边的 weight
        #             "matched_pages": matched_pages_in_session # 邻居通常意味着整个 session 都相关，或者需要进一步过滤
        #         })
        #         seen_ids.add(ns['id'])
        # results = final_context_sessions
        
        self.save() # Save changes from access updates
        # Sort final results by session_relevance_score
        return sorted(results, key=lambda x: x["session_relevance_score"], reverse=True)

    def get_direct_neighbors(self, session_ids):
        """
        [Graph Mod] 获取指定 sessions 的直接邻居 session 对象
        """
        # 1. 从图层获取邻居 ID
        neighbor_ids = self.graph_layer.get_neighbor_ids(session_ids)
        
        # 2. 根据 ID 从 self.sessions 中检索完整对象
        neighbor_sessions = []
        for nid in neighbor_ids:
            if nid in self.sessions:
                neighbor_sessions.append(self.sessions[nid])
            else:
                # 如果图里有ID但sessions里没有，说明数据不一致（通常由evict导致），需要清理
                self.graph_layer.remove_node(nid)
        
        return neighbor_sessions
    
    def save(self):
        # Make a copy for saving to avoid modifying heap during iteration if it happens
        # Though current heap is list of tuples, so direct modification risk is low
        # sessions_to_save = {sid: data for sid, data in self.sessions.items()}
        
        # 从 graph_layer 获取 [[u, v, w], ...] 格式的列表
        edges_to_save = self.graph_layer.get_edges_serializable()
        
        data_to_save = {
            "sessions": self.sessions,
            "access_frequency": dict(self.access_frequency), # Convert defaultdict to dict for JSON
            "graph_edges": edges_to_save ,
            # Heap is derived, no need to save typically, but can if desired for faster load
            # "heap_snapshot": self.heap 
        }
                
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Error saving MidTermMemory to {self.file_path}: {e}")

    def load(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.sessions = data.get("sessions", {})
                self.access_frequency = defaultdict(int, data.get("access_frequency", {}))
                self.rebuild_heap() # Rebuild heap from loaded sessions
                self.graph_layer = GraphMemoryLayer(similarity_threshold=self.graph_layer.similarity_threshold)
                
                for sid, session_data in self.sessions.items():
                    emb = session_data.get("summary_embedding")
                    if emb:
                        # 使用 restore_node_no_sim 避免 O(N^2) 的相似度计算
                        self.graph_layer.restore_node_no_sim(sid, emb)
                # 重建图：恢复边
                saved_edges = data.get("graph_edges", [])
                if saved_edges:
                    self.graph_layer.load_edges_from_list(saved_edges)
                
                self.rebuild_heap()
            
            print(f"MidTermMemory: Loaded from {self.file_path}. Sessions: {len(self.sessions)}. Graph Edges: {len(saved_edges)}.")
            print(f"MidTermMemory: Loaded from {self.file_path}. Sessions: {len(self.sessions)}.")
        except FileNotFoundError:
            print(f"MidTermMemory: No history file found at {self.file_path}. Initializing new memory.")
        except json.JSONDecodeError:
            print(f"MidTermMemory: Error decoding JSON from {self.file_path}. Initializing new memory.")
        except Exception as e:
            print(f"MidTermMemory: An unexpected error occurred during load from {self.file_path}: {e}. Initializing new memory.") 
    
            
            
            
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class GraphMemoryLayer:
    def __init__(self, similarity_threshold=0.6):
        self.graph = nx.Graph()
        self.similarity_threshold = similarity_threshold

    def add_node(self, session_id, embedding):
        """
        添加新节点，并计算与现有节点的相似度进行连边 (用于新Session进入时)
        """
        # 1. 格式化 Embedding
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
        elif len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)
            
        # 2. 添加节点
        self.graph.add_node(session_id, embedding=embedding)
        
        # 3. 寻找现有节点
        existing_nodes = [n for n in self.graph.nodes() if n != session_id]
        if not existing_nodes:
            return

        # 4. 计算相似度
        existing_embeddings = np.vstack([self.graph.nodes[n]['embedding'] for n in existing_nodes])
        sim_scores = cosine_similarity(embedding, existing_embeddings)[0]
        
        # 5. 连边
        for idx, score in enumerate(sim_scores):
            if score > self.similarity_threshold:
                target_node = existing_nodes[idx]
                # weight 必须转为 float，否则 numpy float 可能导致 json 序列化报错
                self.graph.add_edge(session_id, target_node, weight=float(score))

    def remove_node(self, session_id):
        if self.graph.has_node(session_id):
            self.graph.remove_node(session_id)

    def get_neighbor_ids(self, session_ids):
        neighbors = set()
        for sid in session_ids:
            if self.graph.has_node(sid):
                neighbors.update(self.graph.neighbors(sid))
        # 排除自己
        for sid in session_ids:
            if sid in neighbors:
                neighbors.remove(sid)
        return list(neighbors)
    
    def get_edges_serializable(self):
        """
        [关键] 返回用于 JSON 存储的列表格式: [[u, v, weight], ...]
        """
        edges_data = []
        for u, v, data in self.graph.edges(data=True):
            w = data.get('weight', 0.0)
            edges_data.append([u, v, w])
        return edges_data

    def load_edges_from_list(self, edges_list):
        """
        从 JSON 列表恢复边
        """
        if not edges_list:
            return
        # networkx add_weighted_edges_from 接受 (u, v, w) 元组列表
        self.graph.add_weighted_edges_from(edges_list)

    def restore_node_no_sim(self, session_id, embedding):
        """
        [关键] Load 专用：仅恢复节点，不计算相似度（避免 Load 时 O(N^2) 计算）
        """
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
        self.graph.add_node(session_id, embedding=embedding)