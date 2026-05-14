import faiss
import numpy as np

class FAISSSearchEngine:
    def __init__(self, dimension=128, metric='cosine'):
        self.dimension = dimension
        
        if metric == 'cosine':
            self.index = faiss.IndexFlatIP(dimension) 
        elif metric == 'l2':
            self.index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError("Chỉ hỗ trợ metric 'cosine' hoặc 'l2'")
            

    def build_database(self, database_vectors):
        # Đảm bảo vector ở dạng float32
        vectors_np = np.array(database_vectors).astype('float32')
        self.index.reset()
        self.index.add(vectors_np)

    def search_exemplar(self, query_vector, top_k):
        query_np = np.array(query_vector).astype('float32')
        
        if len(query_np.shape) == 1:
            query_np = query_np.reshape(1, -1)
            
        scores, indices = self.index.search(query_np, top_k)
        
        return scores[0], indices[0]