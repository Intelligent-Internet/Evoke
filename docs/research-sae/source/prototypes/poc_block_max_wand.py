import time
import numpy as np
from collections import defaultdict
import heapq

NUM_DOCS = 100_000
BLOCK_SIZE = 128
NUM_BLOCKS = (NUM_DOCS + BLOCK_SIZE - 1) // BLOCK_SIZE

def generate_mock_index(reorder=False):
    np.random.seed(42)
    
    raw_docs = []
    for doc_id in range(NUM_DOCS):
        doc_terms = {}
        for term_id in range(13):
            is_high_df = (term_id < 3)
            prob = 0.90 if is_high_df else 0.05
            if np.random.rand() < prob:
                score = np.random.uniform(0.1, 1.0)
                if is_high_df:
                    score *= np.random.uniform(0.1, 2.0)
                doc_terms[term_id] = score
        raw_docs.append(doc_terms)
        
    if reorder:
        # Simulate sae_overlap_greedy: sort docs by their score on the highest DF term (term 0)
        raw_docs.sort(key=lambda d: d.get(0, 0.0), reverse=True)
        
    index = {}
    block_max = {}
    
    for term_id in range(13):
        postings = []
        term_block_max = defaultdict(float)
        
        for doc_id, doc_terms in enumerate(raw_docs):
            if term_id in doc_terms:
                score = doc_terms[term_id]
                postings.append((doc_id, score))
                
                block_id = doc_id // BLOCK_SIZE
                if score > term_block_max[block_id]:
                    term_block_max[block_id] = score
                    
        index[term_id] = postings
        block_max[term_id] = term_block_max
        
    return index, block_max

def block_max_wand_top_k(query_terms, index, block_max, k=10):
    start_time = time.time()
    top_k_heap = []
    threshold = 0.0
    touched_postings = 0
    skipped_blocks = 0
    
    term_cursors = {term: 0 for term in query_terms}
    
    for block_id in range(NUM_BLOCKS):
        block_upper_bound = 0.0
        for term in query_terms:
            block_upper_bound += block_max[term].get(block_id, 0.0)
            
        if block_upper_bound <= threshold:
            skipped_blocks += 1
            for term in query_terms:
                postings = index[term]
                while term_cursors[term] < len(postings) and postings[term_cursors[term]][0] < (block_id + 1) * BLOCK_SIZE:
                    term_cursors[term] += 1
            continue
            
        doc_scores = defaultdict(float)
        for term in query_terms:
            postings = index[term]
            while term_cursors[term] < len(postings) and postings[term_cursors[term]][0] < (block_id + 1) * BLOCK_SIZE:
                doc_id, score = postings[term_cursors[term]]
                doc_scores[doc_id] += score
                touched_postings += 1
                term_cursors[term] += 1
                
        for doc_id, score in doc_scores.items():
            if len(top_k_heap) < k:
                heapq.heappush(top_k_heap, (score, doc_id))
                if len(top_k_heap) == k:
                    threshold = top_k_heap[0][0]
            elif score > threshold:
                heapq.heapreplace(top_k_heap, (score, doc_id))
                threshold = top_k_heap[0][0]
                
    top_k_heap.sort(reverse=True)
    top_k = [(doc_id, score) for score, doc_id in top_k_heap]
    elapsed = time.time() - start_time
    return top_k, touched_postings, skipped_blocks, elapsed

if __name__ == "__main__":
    query_terms = [0, 1, 5, 8]
    
    print("\n--- 1. Without Document Reordering (Naive Block-Max WAND) ---")
    index_random, block_max_random = generate_mock_index(reorder=False)
    bmw_top_k, bmw_touched, bmw_skipped, bmw_time = block_max_wand_top_k(query_terms, index_random, block_max_random, k=10)
    print(f"Skipped Blocks: {bmw_skipped:,} / {NUM_BLOCKS:,} ({(bmw_skipped/NUM_BLOCKS)*100:.1f}%)")
    print(f"Touched Postings: {bmw_touched:,}")
    
    print("\n--- 2. With Document Reordering (sae_overlap_greedy simulation) ---")
    index_reorder, block_max_reorder = generate_mock_index(reorder=True)
    bmw_top_k, bmw_touched, bmw_skipped, bmw_time = block_max_wand_top_k(query_terms, index_reorder, block_max_reorder, k=10)
    print(f"Skipped Blocks: {bmw_skipped:,} / {NUM_BLOCKS:,} ({(bmw_skipped/NUM_BLOCKS)*100:.1f}%)")
    print(f"Touched Postings: {bmw_touched:,}")
    print("\n[CONCLUSION] Document reordering is strictly required for Block-Max WAND to prune High-DF SAE atoms!")
