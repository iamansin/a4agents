import ray
import httpx
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional
from ray.util.collective import allreduce
from sentence_transformers import SentenceTransformer

class Router:
    def __init__(self, model_registry: Dict[str, Any], use_ray: bool = True):
        """
        Router class implementing MCP with efficient communication.
        
        :param model_registry: Dictionary mapping model names to their respective endpoints or Ray actors.
        :param use_ray: Whether to use Ray for distributed processing.
        """
        self.model_registry = model_registry  # Maps model names to endpoints/actors
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Embedding model
        self.use_ray = use_ray
        if use_ray:
            ray.init(ignore_reinit_error=True)
        
    async def route_request(self, query: str, transport: str = "auto", top_k: int = 1) -> Any:
        """
        Routes a request based on the query semantics and transport type.
        
        :param query: The input query string.
        :param transport: The communication mode ("http", "ray", "local", "auto").
        :param top_k: Number of top relevant models to consider.
        :return: Response from the routed model(s).
        """
        best_models = self._find_best_models(query, top_k)
        
        if not best_models:
            raise ValueError("No suitable models found for the given query.")
        
        if transport == "auto":
            transport = self._choose_best_transport(best_models)
        
        responses = []
        for model_name in best_models:
            if transport == "http":
                responses.append(await self._http_request(self.model_registry[model_name], query))
            elif transport == "ray":
                responses.append(await self._ray_request(self.model_registry[model_name], query))
            else:  # Default to local function calls
                responses.append(self.model_registry[model_name](query))
        
        return responses if len(responses) > 1 else responses[0]
    
    def _find_best_models(self, query: str, top_k: int) -> List[str]:
        """Finds the best model(s) based on embedding similarity."""
        query_embedding = self.embedding_model.encode(query)
        model_scores = {}
        for model_name in self.model_registry:
            model_embedding = self.embedding_model.encode(model_name)
            similarity = np.dot(query_embedding, model_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(model_embedding))
            model_scores[model_name] = similarity
        
        sorted_models = sorted(model_scores, key=model_scores.get, reverse=True)
        return sorted_models[:top_k]
    
    def _choose_best_transport(self, models: List[str]) -> str:
        """Chooses the best transport mode dynamically."""
        # Here, we could implement a more sophisticated decision-making process
        # For now, prefer Ray if available, otherwise HTTP
        return "ray" if self.use_ray else "http"
    
    async def _http_request(self, endpoint: str, query: str) -> Any:
        """Sends an HTTP request asynchronously."""
        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint, json={"query": query})
            return response.json()
    
    async def _ray_request(self, actor: ray.actor.ActorHandle, query: str) -> Any:
        """Sends a request to a Ray actor using Ray collective communication."""
        result = await actor.handle_query.remote(query)
        return result
    
    def distributed_reduce(self, tensor: np.ndarray) -> np.ndarray:
        """Performs an all-reduce operation across Ray actors for distributed consensus."""
        if not self.use_ray:
            raise RuntimeError("Ray must be enabled for distributed reduction.")
        allreduce(tensor)
        return tensor  # The updated reduced tensor