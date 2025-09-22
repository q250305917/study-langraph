"""
重排序器模块

提供多种重排序策略，用于优化检索结果的相关性和多样性。
支持CrossEncoder、分数调整、多样性优化等重排序方法。
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# LangChain导入
from langchain.schema import Document

# 可选依赖
try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """重排序器基类"""
    
    def __init__(self, top_k: int = 10):
        """
        初始化重排序器
        
        Args:
            top_k: 重排序后返回的文档数量
        """
        self.top_k = top_k
        
        # 性能统计
        self.stats = {
            "total_reranks": 0,
            "total_documents_processed": 0,
            "total_time": 0.0
        }
    
    @abstractmethod
    def rerank(self,
               query: str,
               documents: List[Document],
               scores: Optional[List[float]] = None) -> Tuple[List[Document], List[float]]:
        """
        重排序文档
        
        Args:
            query: 查询文本
            documents: 文档列表
            scores: 原始分数列表
            
        Returns:
            (重排序后的文档列表, 新分数列表)
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        if stats["total_reranks"] > 0:
            stats["avg_time_per_rerank"] = stats["total_time"] / stats["total_reranks"]
            stats["avg_docs_per_rerank"] = stats["total_documents_processed"] / stats["total_reranks"]
        
        return stats


class CrossEncoderReranker(BaseReranker):
    """CrossEncoder重排序器
    
    使用预训练的CrossEncoder模型对查询-文档对进行重新评分。
    """
    
    def __init__(self,
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 top_k: int = 10,
                 batch_size: int = 32):
        """
        初始化CrossEncoder重排序器
        
        Args:
            model_name: CrossEncoder模型名称
            top_k: 返回文档数量
            batch_size: 批处理大小
        """
        if not HAS_CROSS_ENCODER:
            raise ImportError("需要安装sentence-transformers: pip install sentence-transformers")
        
        super().__init__(top_k)
        self.model_name = model_name
        self.batch_size = batch_size
        
        # 加载模型
        try:
            self.model = CrossEncoder(model_name)
            logger.info(f"加载CrossEncoder模型: {model_name}")
        except Exception as e:
            logger.error(f"加载CrossEncoder模型失败: {e}")
            raise
    
    def rerank(self,
               query: str,
               documents: List[Document],
               scores: Optional[List[float]] = None) -> Tuple[List[Document], List[float]]:
        """使用CrossEncoder重排序"""
        import time
        start_time = time.time()
        
        try:
            if not documents:
                return [], []
            
            # 准备查询-文档对
            query_doc_pairs = []
            for doc in documents:
                # 截断文档内容以避免模型输入限制
                content = doc.page_content[:512]  # 限制为512字符
                query_doc_pairs.append([query, content])
            
            # 批量计算相关性分数
            all_scores = []
            for i in range(0, len(query_doc_pairs), self.batch_size):
                batch = query_doc_pairs[i:i + self.batch_size]
                batch_scores = self.model.predict(batch)
                all_scores.extend(batch_scores.tolist())
            
            # 创建(文档, 分数)对并排序
            doc_score_pairs = list(zip(documents, all_scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # 取top-k
            top_pairs = doc_score_pairs[:self.top_k]
            reranked_docs = [doc for doc, score in top_pairs]
            reranked_scores = [score for doc, score in top_pairs]
            
            # 更新统计
            processing_time = time.time() - start_time
            self.stats["total_reranks"] += 1
            self.stats["total_documents_processed"] += len(documents)
            self.stats["total_time"] += processing_time
            
            logger.debug(f"CrossEncoder重排序完成: {len(documents)} -> {len(reranked_docs)}")
            
            return reranked_docs, reranked_scores
        
        except Exception as e:
            logger.error(f"CrossEncoder重排序失败: {e}")
            # 回退到原始排序
            return documents[:self.top_k], scores[:self.top_k] if scores else []


class ScoreReranker(BaseReranker):
    """基于分数的重排序器
    
    使用多种分数调整策略对文档进行重排序。
    """
    
    def __init__(self,
                 top_k: int = 10,
                 boost_recent: bool = True,
                 boost_length: bool = False,
                 diversity_factor: float = 0.1):
        """
        初始化分数重排序器
        
        Args:
            top_k: 返回文档数量
            boost_recent: 是否提升最近文档的分数
            boost_length: 是否考虑文档长度
            diversity_factor: 多样性因子
        """
        super().__init__(top_k)
        self.boost_recent = boost_recent
        self.boost_length = boost_length
        self.diversity_factor = diversity_factor
    
    def rerank(self,
               query: str,
               documents: List[Document],
               scores: Optional[List[float]] = None) -> Tuple[List[Document], List[float]]:
        """基于分数调整的重排序"""
        import time
        start_time = time.time()
        
        try:
            if not documents:
                return [], []
            
            # 如果没有原始分数，使用均匀分数
            if scores is None:
                scores = [1.0] * len(documents)
            
            # 计算调整后的分数
            adjusted_scores = []
            
            for i, (doc, score) in enumerate(zip(documents, scores)):
                adjusted_score = score
                
                # 时间提升
                if self.boost_recent:
                    adjusted_score *= self._get_time_boost(doc)
                
                # 长度提升
                if self.boost_length:
                    adjusted_score *= self._get_length_boost(doc)
                
                # 多样性调整
                if self.diversity_factor > 0:
                    diversity_penalty = self._calculate_diversity_penalty(
                        doc, documents[:i], self.diversity_factor
                    )
                    adjusted_score *= (1.0 - diversity_penalty)
                
                adjusted_scores.append(adjusted_score)
            
            # 排序
            doc_score_pairs = list(zip(documents, adjusted_scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # 取top-k
            top_pairs = doc_score_pairs[:self.top_k]
            reranked_docs = [doc for doc, score in top_pairs]
            reranked_scores = [score for doc, score in top_pairs]
            
            # 更新统计
            processing_time = time.time() - start_time
            self.stats["total_reranks"] += 1
            self.stats["total_documents_processed"] += len(documents)
            self.stats["total_time"] += processing_time
            
            return reranked_docs, reranked_scores
        
        except Exception as e:
            logger.error(f"分数重排序失败: {e}")
            return documents[:self.top_k], scores[:self.top_k] if scores else []
    
    def _get_time_boost(self, doc: Document) -> float:
        """计算时间提升因子"""
        try:
            from datetime import datetime, timedelta
            
            # 尝试从元数据获取时间信息
            created_time = doc.metadata.get('created_time')
            modified_time = doc.metadata.get('modified_time')
            
            if created_time:
                if isinstance(created_time, str):
                    created_time = datetime.fromisoformat(created_time)
                elif isinstance(created_time, (int, float)):
                    created_time = datetime.fromtimestamp(created_time)
                
                # 计算时间差（天数）
                days_old = (datetime.now() - created_time).days
                
                # 时间衰减：最近的文档获得更高分数
                # 使用指数衰减，半衰期为30天
                time_boost = math.exp(-days_old / 30.0)
                return max(0.5, time_boost)  # 最低0.5倍
            
            return 1.0  # 无时间信息时不调整
        
        except Exception:
            return 1.0
    
    def _get_length_boost(self, doc: Document) -> float:
        """计算长度提升因子"""
        try:
            content_length = len(doc.page_content)
            
            # 中等长度的文档获得最高分数
            # 太短可能信息不足，太长可能不够聚焦
            optimal_length = 1000
            
            if content_length < optimal_length:
                # 短文档的惩罚
                length_boost = content_length / optimal_length
            else:
                # 长文档的轻微惩罚
                length_boost = optimal_length / content_length
                length_boost = max(0.7, length_boost)  # 最低0.7倍
            
            return length_boost
        
        except Exception:
            return 1.0
    
    def _calculate_diversity_penalty(self,
                                   current_doc: Document,
                                   previous_docs: List[Document],
                                   diversity_factor: float) -> float:
        """计算多样性惩罚"""
        if not previous_docs or not HAS_SKLEARN:
            return 0.0
        
        try:
            # 使用TF-IDF计算文档相似度
            all_texts = [current_doc.page_content] + [doc.page_content for doc in previous_docs]
            
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # 计算当前文档与已选文档的最大相似度
            current_vector = tfidf_matrix[0:1]
            other_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(current_vector, other_vectors)[0]
            max_similarity = np.max(similarities) if len(similarities) > 0 else 0.0
            
            # 相似度越高，惩罚越大
            penalty = max_similarity * diversity_factor
            
            return min(penalty, 0.5)  # 最大惩罚50%
        
        except Exception:
            return 0.0


class DiversityReranker(BaseReranker):
    """多样性重排序器
    
    使用最大边际相关性(MMR)算法确保结果的多样性。
    """
    
    def __init__(self,
                 top_k: int = 10,
                 lambda_mult: float = 0.5,
                 similarity_threshold: float = 0.8):
        """
        初始化多样性重排序器
        
        Args:
            top_k: 返回文档数量
            lambda_mult: 相关性与多样性的平衡参数
            similarity_threshold: 相似度阈值
        """
        super().__init__(top_k)
        self.lambda_mult = lambda_mult
        self.similarity_threshold = similarity_threshold
        
        if not HAS_SKLEARN:
            logger.warning("sklearn未安装，多样性重排序可能受限")
    
    def rerank(self,
               query: str,
               documents: List[Document],
               scores: Optional[List[float]] = None) -> Tuple[List[Document], List[float]]:
        """使用MMR算法重排序"""
        import time
        start_time = time.time()
        
        try:
            if not documents or not HAS_SKLEARN:
                return documents[:self.top_k], scores[:self.top_k] if scores else []
            
            # 如果没有原始分数，使用均匀分数
            if scores is None:
                scores = [1.0] * len(documents)
            
            # 计算文档向量
            doc_vectors = self._compute_document_vectors(documents)
            query_vector = self._compute_query_vector(query, doc_vectors.shape[1])
            
            # MMR算法
            selected_indices = []
            remaining_indices = list(range(len(documents)))
            
            while len(selected_indices) < self.top_k and remaining_indices:
                best_score = -float('inf')
                best_idx = None
                
                for idx in remaining_indices:
                    # 相关性分数（原始分数 + 查询相似度）
                    relevance_score = scores[idx]
                    if query_vector is not None:
                        query_sim = cosine_similarity(
                            query_vector.reshape(1, -1),
                            doc_vectors[idx:idx+1]
                        )[0][0]
                        relevance_score = 0.5 * relevance_score + 0.5 * query_sim
                    
                    # 多样性分数（与已选文档的最大相似度）
                    diversity_score = 0.0
                    if selected_indices:
                        similarities = cosine_similarity(
                            doc_vectors[idx:idx+1],
                            doc_vectors[selected_indices]
                        )[0]
                        diversity_score = np.max(similarities)
                    
                    # MMR分数
                    mmr_score = (
                        self.lambda_mult * relevance_score - 
                        (1 - self.lambda_mult) * diversity_score
                    )
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx
                
                if best_idx is not None:
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)
                else:
                    break
            
            # 构建结果
            reranked_docs = [documents[i] for i in selected_indices]
            reranked_scores = [scores[i] for i in selected_indices]
            
            # 更新统计
            processing_time = time.time() - start_time
            self.stats["total_reranks"] += 1
            self.stats["total_documents_processed"] += len(documents)
            self.stats["total_time"] += processing_time
            
            logger.debug(f"多样性重排序完成: {len(documents)} -> {len(reranked_docs)}")
            
            return reranked_docs, reranked_scores
        
        except Exception as e:
            logger.error(f"多样性重排序失败: {e}")
            return documents[:self.top_k], scores[:self.top_k] if scores else []
    
    def _compute_document_vectors(self, documents: List[Document]) -> np.ndarray:
        """计算文档向量"""
        try:
            texts = [doc.page_content for doc in documents]
            
            # 使用TF-IDF向量化
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            doc_vectors = vectorizer.fit_transform(texts).toarray()
            self.vectorizer = vectorizer  # 保存以便查询向量化
            
            return doc_vectors
        
        except Exception as e:
            logger.error(f"计算文档向量失败: {e}")
            # 返回随机向量作为备用
            return np.random.rand(len(documents), 100)
    
    def _compute_query_vector(self, query: str, vector_dim: int) -> Optional[np.ndarray]:
        """计算查询向量"""
        try:
            if hasattr(self, 'vectorizer'):
                query_vector = self.vectorizer.transform([query]).toarray()[0]
                return query_vector
            else:
                return None
        
        except Exception:
            return None


class CombinedReranker(BaseReranker):
    """组合重排序器
    
    结合多个重排序器的结果。
    """
    
    def __init__(self,
                 rerankers: List[BaseReranker],
                 weights: Optional[List[float]] = None,
                 top_k: int = 10):
        """
        初始化组合重排序器
        
        Args:
            rerankers: 重排序器列表
            weights: 权重列表
            top_k: 返回文档数量
        """
        super().__init__(top_k)
        self.rerankers = rerankers
        
        # 权重归一化
        if weights is None:
            weights = [1.0] * len(rerankers)
        
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]
        
        logger.info(f"初始化组合重排序器，包含 {len(rerankers)} 个子重排序器")
    
    def rerank(self,
               query: str,
               documents: List[Document],
               scores: Optional[List[float]] = None) -> Tuple[List[Document], List[float]]:
        """组合多个重排序器的结果"""
        import time
        start_time = time.time()
        
        try:
            if not documents:
                return [], []
            
            # 收集所有重排序器的结果
            all_results = []
            for reranker in self.rerankers:
                try:
                    reranked_docs, reranked_scores = reranker.rerank(query, documents, scores)
                    all_results.append((reranked_docs, reranked_scores))
                except Exception as e:
                    logger.warning(f"重排序器失败: {e}")
                    continue
            
            if not all_results:
                return documents[:self.top_k], scores[:self.top_k] if scores else []
            
            # 使用倒数排名融合
            doc_scores = {}
            
            for i, (reranked_docs, reranked_scores) in enumerate(all_results):
                weight = self.weights[i] if i < len(self.weights) else 1.0
                
                for rank, doc in enumerate(reranked_docs):
                    doc_id = self._get_doc_id(doc)
                    
                    # 倒数排名分数
                    rrf_score = weight / (rank + 60)
                    
                    if doc_id in doc_scores:
                        doc_scores[doc_id]['score'] += rrf_score
                    else:
                        doc_scores[doc_id] = {
                            'document': doc,
                            'score': rrf_score
                        }
            
            # 排序并取top-k
            sorted_docs = sorted(
                doc_scores.values(),
                key=lambda x: x['score'],
                reverse=True
            )[:self.top_k]
            
            final_documents = [item['document'] for item in sorted_docs]
            final_scores = [item['score'] for item in sorted_docs]
            
            # 更新统计
            processing_time = time.time() - start_time
            self.stats["total_reranks"] += 1
            self.stats["total_documents_processed"] += len(documents)
            self.stats["total_time"] += processing_time
            
            return final_documents, final_scores
        
        except Exception as e:
            logger.error(f"组合重排序失败: {e}")
            return documents[:self.top_k], scores[:self.top_k] if scores else []
    
    def _get_doc_id(self, doc: Document) -> str:
        """获取文档ID"""
        import hashlib
        return hashlib.md5(doc.page_content.encode()).hexdigest()


# 便捷函数
def create_reranker(reranker_type: str,
                   **kwargs) -> BaseReranker:
    """
    创建重排序器的工厂函数
    
    Args:
        reranker_type: 重排序器类型
        **kwargs: 额外参数
        
    Returns:
        重排序器实例
    """
    reranker_map = {
        "cross_encoder": CrossEncoderReranker,
        "score": ScoreReranker,
        "diversity": DiversityReranker
    }
    
    if reranker_type not in reranker_map:
        raise ValueError(f"不支持的重排序器类型: {reranker_type}")
    
    reranker_class = reranker_map[reranker_type]
    return reranker_class(**kwargs)