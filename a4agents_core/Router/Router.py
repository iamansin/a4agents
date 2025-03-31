from typing import Dict, Any, List, Union, Callable, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass
import torch
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import lru_cache

class EdgeType(Enum):
    DIRECT = "direct"
    CONDITIONAL = "conditional"

@dataclass
class Edge:
    from_node: str
    to_node: Union[str, List[str]]
    edge_type: EdgeType
    condition: Optional[str] = None
    
class Router:
    def __init__(
        self,
        from_node: str,
        to_node: Optional[str] = None,
        direct_nodes: Optional[List[str]] = None,
        conditional_nodes: Optional[Dict[str, str]] = None,
        use_embeddings: bool = False,
        embedding_model: Optional[Any] = None,
        router_type: Optional[EdgeType] = EdgeType.DIRECT,
    ):
        """
        Initialize the Router with source node and destination nodes configuration.
        
        Args:
            from_node (str): Source node name
            to_nodes (Union[str, List[Union[Tuple[Dict[str, str], List[str]], str]]]): 
                - If str: Single destination node (direct routing)
                - If List: List of either strings (direct routing) or tuples (conditional routing)
                    where tuple is (conditions_dict, target_nodes)
            use_embeddings (bool): Whether to use embeddings for similarity-based routing
            embedding_model: Pre-initialized embedding model (optional)
            device (str): Device to run embeddings on
            cache_size (int): Size of the LRU cache for embeddings
        """
        self.logger = logging.getLogger(__name__)
        self.from_node = from_node
        self.use_embeddings = use_embeddings
        # Initialize edges
        self.router_type = router_type
        self.edges: List[Edge] = []
        self._initialize_edges(to_node, direct_nodes, conditional_nodes)
        
        # Initialize embedding related attributes if needed
        if use_embeddings:
            self.embedding_model = embedding_model 
            self._node_embeddings = {}
            # self._compute_node_embeddings()
        
    def _initialize_edges(
        self,
        to_node: Optional[str],
        direct_nodes: Optional[List[str]],
        conditional_nodes: Optional[Dict[str, str]]
    ) -> None:
        """Initialize edges based on the provided configuration."""
        if to_node:
            self.edges.append(Edge(
                from_node=self.from_node,
                to_node=to_node,
                edge_type=EdgeType.DIRECT
            ))
            self.router_type = EdgeType.DIRECT
            return

        if direct_nodes:
            for node in direct_nodes:
                self.edges.append(Edge(
                    from_node=self.from_node,
                    to_node=node,
                    edge_type=EdgeType.DIRECT
                ))
            self.router_type = EdgeType.DIRECT
            return 
        
        # Handle conditional nodes
        if conditional_nodes:
            for condition, target_node in conditional_nodes.items():
                self.edges.append(Edge(
                    from_node=self.from_node,
                    to_node=target_node,
                    edge_type=EdgeType.CONDITIONAL,
                    condition= condition
                ))
            self.router_type = EdgeType.CONDITIONAL
            return 

    def get_routing_function(self) -> Callable:
        """Return a routing function for use with LangGraph."""
        if self.router_type == EdgeType.DIRECT:
            raise ValueError("Direct routing is not supported in this context.")
        
        conditional_nodes = {edge.condition: edge.to_node for edge in self.edges} # {"web":"websearch","email": "email-hanlder", "type": "type-checker"}
                
        def routing_function(state, _condition_space = conditional_nodes) -> Union[str, List[str]]:
            current_node_result = state.get(self.from_node).get("_next")
                    
            if current_node_result is None:
                raise ValueError(f"No result found for node {self.from_node}")
            
            _route_to = []
            try:
                for val in current_node_result:# ["web", "email"]:
                    _route_to.append(_condition_space[val])
                    
            except KeyError as e:
                raise ValueError(f"Condition {val} not found in routing conditions. {_condition_space}") from e
            
            if _route_to:
                return _route_to
            
            raise ValueError(f"No next node for routing found.")
            
        return routing_function
    


    # def __del__(self):
    #     """Cleanup resources."""
    #     self.thread_pool.shutdown(wait=False)
    
    # @lru_cache(maxsize=1000)
    # def _compute_embedding(self, text: str) -> np.ndarray:
    #     """Compute and cache embeddings for a given text."""
    #     with torch.no_grad():
    #         embedding = self.embedding_model.encode(
    #             text,
    #             convert_to_tensor=True,
    #             device=self.device
    #         )
    #         return embedding.cpu().numpy()

    # def _compute_node_embeddings(self) -> None:
    #     """Pre-compute embeddings for all node names."""
    #     if not self.use_embeddings:
    #         return
            
    #     unique_nodes = set()
    #     for edge in self.edges:
    #         if isinstance(edge.to_nodes, list):
    #             unique_nodes.update(edge.to_nodes)
    #         else:
    #             unique_nodes.add(edge.to_nodes)
                
    #     for node in unique_nodes:
    #         self._node_embeddings[node] = self._compute_embedding(node)

    # def _compute_similarity(
    #     self,
    #     text_embedding: np.ndarray,
    #     node_embeddings: Dict[str, np.ndarray]
    # ) -> Dict[str, float]:
    #     """Compute cosine similarity between text and node embeddings."""
    #     similarities = {}
    #     text_embedding_norm = np.linalg.norm(text_embedding)
        
    #     for node, node_embedding in node_embeddings.items():
    #         node_embedding_norm = np.linalg.norm(node_embedding)
    #         similarity = np.dot(text_embedding, node_embedding) / (
    #             text_embedding_norm * node_embedding_norm
    #         )
    #         similarities[node] = float(similarity)
            
    #     return similarities

