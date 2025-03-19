import os
import asyncio
import inspect
from typing import Dict, Any, Union, Optional, Callable, Type, TypeVar, cast
from typing_extensions import Protocol, runtime_checkable
from langgraph.graph import StateGraph
import logging
from pathlib import Path
from functools import partial
from tqdm.asyncio import tqdm_asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DynamicGraphCompiler")

# Type definitions
T = TypeVar('T')
State = TypeVar('State')

@runtime_checkable
class Router(Protocol):
    """Protocol for Router objects that can provide routing functions."""
    def get_routing_function(self) -> Callable:
        """Return a routing function compatible with langgraph conditional edges."""
        ...

@runtime_checkable
class SystemNode(Protocol):
    """Protocol for SystemNode objects."""
    async def run(self, state: Any) -> Any:
        """Run the node with the given state."""
        ...

class DynamicGraphCompiler:
    """
    A class for dynamically compiling and building a graph of nodes, tools, and agents.
    
    This compiler takes nodes, router mappings, tools, and agents and builds a langgraph
    StateGraph by resolving dependencies, initializing external tools/agents, and setting
    up the appropriate edges between nodes.
    """
    
    def __init__(
        self, 
        state_type: Type[State],
        nodes: Dict[str, SystemNode], 
        router_mappings: Dict[str, Union[str, Router, Dict[str, str]]], 
        tools: Dict[str, Any], 
        agents: Dict[str, Any], 
        registry_instance: Any,
        entry_point: str = "start"
    ):
        """
        Initialize the DynamicGraphCompiler.
        
        Args:
            state_type: The state type used for the StateGraph
            nodes: Dictionary of node names to SystemNode instances
            router_mappings: Dictionary mapping source nodes to destination nodes or routers
            tools: Dictionary of available tools
            agents: Dictionary of available agents
            registry_instance: An instance of the Registry class for managing tools and agents
            entry_point: The name of the entry point node (default: "start")
        """
        self._state_type = state_type
        self._nodes = nodes
        self._router_mappings = router_mappings
        self._tools = tools
        self._agents = agents
        self._registry = registry_instance
        self._graph = StateGraph(state_type)
        self._entry_point = entry_point
        self._resolved_nodes: Dict[str, Callable] = {}
        self._visited_nodes: set = set()
        self.compiled_graph = False
        
    @classmethod
    async def create(cls, 
        state_type: Type[State],
        nodes: Dict[str, SystemNode], 
        router_mappings: Dict[str, Union[str, Router, Dict[str, str]]], 
        tools: Dict[str, Any], 
        agents: Dict[str, Any], 
        registry_instance: Any,
        entry_point: str = "start"
    ):
        """
        Asynchronously create and initialize a DynamicGraphCompiler instance.
        
        Returns:
            A fully initialized DynamicGraphCompiler
        """
        instance = cls(state_type, nodes, router_mappings, tools, agents, registry_instance, entry_point)
        try:
            await instance._compile_graph()
        except Exception as e:
            raise e
        
        if instance.compiled_graph:
            return instance.compiled_graph
        
        return "ERROR OBJECT"
    
    async def _compile_graph(self) -> None:
        """
        Compile the graph by loading references and building the graph structure.
        """
        try:
            logger.info("Starting graph compilation")
            await self._load_references()
            self._build_graph_from_entry_point()
            logger.info("Graph compilation completed successfully")
        except Exception as e:
            logger.error(f"Graph compilation failed: {str(e)}")
            raise RuntimeError(f"Failed to compile graph: {str(e)}") from e
    
    async def _load_references(self) -> None:
        """
        Load all node references, resolving any tools or agents from the registry concurrently.
        This async implementation significantly improves loading time by running operations in parallel.
        """
        logger.info("Loading node references asynchronously")
        
        # Create node functions (this is typically fast so we do it sequentially)
        for node_name, node in self._nodes.items():
            self._resolved_nodes[node_name] = self._create_node_function(node_name, node)
        
        # Gather tasks for loading tools and agents concurrently
        tasks = []
        
        # Create tasks for loading tools
        for tool_name, ETool_instance in self._tools.items():
            tasks.append(self._load_tool_async(tool_name))
        
        # Create tasks for loading agents
        for agent_name, EAgent_instance in self._agents.items():
            tasks.append(self._load_agent_async(agent_name))
        
        # Run all loading tasks concurrently
        if tasks:
            logger.info(f"Concurrently loading {len(tasks)} tools and agents")
            await tqdm_asyncio.gather(*tasks)
        
        logger.info(f"Successfully loaded {len(self._resolved_nodes)} total nodes")

    async def _load_tool_async(self, tool_name: str) -> None:
        """
        Asynchronously load a tool from the registry and create its function wrapper.
        
        Args:
            tool_name: Name of the tool to load
        """
        try:
            logger.info(f"Loading tool: {tool_name}")
            tool_obj = await self._registry.load_tool(tool_name)
            self._resolved_nodes[tool_name] = self._create_tool_function(tool_name, tool_obj)
            logger.debug(f"Successfully loaded tool: {tool_name}")
        except ValueError as e:
            logger.warning(f"Failed to load tool {tool_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error loading tool {tool_name}: {str(e)}")

    async def _load_agent_async(self, agent_name: str) -> None:
        """
        Asynchronously load an agent from the registry and create its function wrapper.
        
        Args:
            agent_name: Name of the agent to load
        """
        try:
            logger.info(f"Loading agent: {agent_name}")
            # Note: If load_agent is a blocking operation, wrap it with run_in_executor
            if inspect.iscoroutinefunction(self._registry.load_agent):
                agent_obj = await self._registry.load_agent(agent_name)
            else:
                # Run blocking registry operations in a thread pool to avoid blocking the event loop
                agent_obj = await asyncio.to_thread(self._registry.load_agent, agent_name)
                
            self._resolved_nodes[agent_name] = self._create_agent_function(agent_name, agent_obj)
            logger.debug(f"Successfully loaded agent: {agent_name}")
        except ValueError as e:
            logger.warning(f"Failed to load agent {agent_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error loading agent {agent_name}: {str(e)}")
    
    def _create_node_function(self, node_name: str, node: SystemNode) -> Callable:
        """
        Create a function wrapper for a system node.
        
        Args:
            node_name: Name of the node
            node: The node instance
            
        Returns:
            A callable function that can be added to the graph
        """
        async def node_wrapper(state: Any) -> Any:
            logger.debug(f"Executing node: {node_name}")
            try:
                return await node.run(state)
            except Exception as e:
                logger.error(f"Error in node {node_name}: {str(e)}")
                raise RuntimeError(f"Node execution failed: {node_name}") from e
                
        # Set the function name for better debugging
        node_wrapper.__name__ = f"_{node_name}_"
        return node_wrapper
    
    def _create_tool_function(self, tool_name: str, tool_obj: Any) -> Callable:
        """
        Create a function wrapper for a tool.
        
        Args:
            tool_name: Name of the tool
            tool_obj: The tool object loaded from registry
            
        Returns:
            A callable function that can be added to the graph
        """
        # For local tools
        if hasattr(tool_obj, "run"):
            async def tool_wrapper(state: Any) -> Any:
                logger.debug(f"Executing local tool: {tool_name}")
                try:
                    if inspect.iscoroutinefunction(tool_obj.run):
                        return await tool_obj.run(state)
                    else:
                        return tool_obj.run(state)
                except Exception as e:
                    logger.error(f"Error in tool {tool_name}: {str(e)}")
                    raise RuntimeError(f"Tool execution failed: {tool_name}") from e
        # For remote tools
        else:
            async def tool_wrapper(state: Any) -> Any:
                logger.debug(f"Executing remote tool: {tool_name}")
                try:
                    endpoint = tool_obj.get("endpoint")
                    api_key = tool_obj.get("api_key")
                    
                    if not endpoint:
                        raise ValueError(f"No endpoint specified for remote tool: {tool_name}")
                    
                    # Here you would implement the logic to call the remote endpoint
                    # For example using aiohttp or httpx
                    # This is a placeholder for the actual implementation
                    return await self._call_remote_endpoint(endpoint, api_key, state)
                except Exception as e:
                    logger.error(f"Error in remote tool {tool_name}: {str(e)}")
                    raise RuntimeError(f"Remote tool execution failed: {tool_name}") from e
                
        # Set the function name for better debugging
        if 'tool_wrapper' in locals():
            tool_wrapper.__name__ = f"_{tool_name}_"
            return tool_wrapper
        else:
            raise ValueError(f"Could not create wrapper for tool: {tool_name}")
    
    def _create_agent_function(self, agent_name: str, agent_obj: Any) -> Callable:
        """
        Create a function wrapper for an agent.
        
        Args:
            agent_name: Name of the agent
            agent_obj: The agent object loaded from registry
            
        Returns:
            A callable function that can be added to the graph
        """
        # For local agents
        if hasattr(agent_obj, "run"):
            async def agent_wrapper(state: Any) -> Any:
                logger.debug(f"Executing local agent: {agent_name}")
                try:
                    if inspect.iscoroutinefunction(agent_obj.run):
                        return await agent_obj.run(state)
                    else:
                        return agent_obj.run(state)
                except Exception as e:
                    logger.error(f"Error in agent {agent_name}: {str(e)}")
                    raise RuntimeError(f"Agent execution failed: {agent_name}") from e
        # For remote agents
        else:
            async def agent_wrapper(state: Any) -> Any:
                logger.debug(f"Executing remote agent: {agent_name}")
                try:
                    endpoint = agent_obj.get("endpoint")
                    
                    if not endpoint:
                        raise ValueError(f"No endpoint specified for remote agent: {agent_name}")
                    
                    # Here you would implement the logic to call the remote endpoint
                    return await self._call_remote_endpoint(endpoint, None, state)
                except Exception as e:
                    logger.error(f"Error in remote agent {agent_name}: {str(e)}")
                    raise RuntimeError(f"Remote agent execution failed: {agent_name}") from e
                
        # Set the function name for better debugging
        if 'agent_wrapper' in locals():
            agent_wrapper.__name__ = f"{agent_name}_wrapper"
            return agent_wrapper
        else:
            raise ValueError(f"Could not create wrapper for agent: {agent_name}")
    
    async def _call_remote_endpoint(self, endpoint: str, api_key: Optional[str], state: Any) -> Any:
        """
        Call a remote endpoint with the given state.
        
        Args:
            endpoint: The endpoint URL
            api_key: Optional API key for authentication
            state: The state to pass to the endpoint
            
        Returns:
            The response from the endpoint
        """
        # Placeholder for actual remote endpoint call implementation
        # In a real implementation, you would use aiohttp or httpx to make the request
        logger.info(f"Calling remote endpoint: {endpoint}")
        # This is where you'd make the actual HTTP request
        return state  # Placeholder return value
    
    def _build_graph_from_entry_point(self) -> None:
        """
        Build the graph starting from the entry point and following the router mappings.
        """
        if self._entry_point not in self._resolved_nodes:
            raise ValueError(f"Entry point node '{self._entry_point}' not found in resolved nodes")
        
        # Start building the graph from the entry point
        self._process_node(self._entry_point)
    
    def _process_node(self, node_name: str) -> None:
        """
        Process a node by adding it to the graph and following its routing mappings.
        
        Args:
            node_name: The name of the node to process
        """
        # Check if we've already visited this node
        if node_name in self._visited_nodes:
            return
        
        # Mark the node as visited
        self._visited_nodes.add(node_name)
        
        # Add the node to the graph
        if node_name in self._resolved_nodes:
            self._graph.add_node(node_name, self._resolved_nodes[node_name])
            logger.info(f"Added node to graph: {node_name}")
        else:
            logger.warning(f"Node {node_name} not found in resolved nodes, skipping")
            return
        
        # Check if this node has routing mappings
        if node_name not in self._router_mappings:
            logger.info(f"No routing mappings found for node: {node_name}")
            return
        
        # Process the routing mappings
        self._process_routing(node_name, self._router_mappings[node_name])
    
    def _process_routing(self, source_node: str, routing: Union[str, Router, Dict[str, str]]) -> None:
        """
        Process routing for a source node.
        
        Args:
            source_node: The source node name
            routing: The routing configuration (string, Router object, or dict)
        """
        # Case 1: Simple string mapping to another node
        if isinstance(routing, str):
            target_node = routing
            self._process_node(target_node)
            self._graph.add_edge(source_node, target_node)
            logger.info(f"Added edge: {source_node} -> {target_node}")
            
        # Case 2: Router object for conditional routing
        elif isinstance(routing, Router):
            router_func = routing.get_routing_function()
            # Find all possible targets in the router mappings
            # This might require additional logic depending on your Router implementation
            # For now, we'll assume the router can work with any node in our resolved nodes
            for target_node in self._resolved_nodes.keys():
                if target_node != source_node:
                    self._process_node(target_node)
            
            # Add conditional edge
            self._graph.add_conditional_edges(
                source_node,
                router_func,
                # Pass all possible targets
                self._resolved_nodes.keys()
            )
            logger.info(f"Added conditional edges from: {source_node}")
            
        # Case 3: Dictionary mapping conditions to target nodes
        elif isinstance(routing, dict):
            for condition, target_node in routing.items():
                if isinstance(target_node, str):
                    self._process_node(target_node)
                    
            # Create a simple routing function based on the dictionary
            def condition_router(state: Any) -> str:
                for condition_key, target_node in routing.items():
                    # For simplicity, we'll check if the condition key exists in the state
                    # In a real implementation, you might want more sophisticated condition checking
                    if condition_key in state:
                        return target_node
                # Default to the first target if no condition matches
                return next(iter(routing.values()))
            
            # Add conditional edge
            self._graph.add_conditional_edges(
                source_node,
                condition_router,
                list(routing.values())
            )
            logger.info(f"Added conditional edges from: {source_node} with {len(routing)} conditions")
        else:
            logger.warning(f"Unknown routing type for node {source_node}: {type(routing)}")
    
    def get_graph(self) -> StateGraph:
        """
        Get the compiled graph.
        
        Returns:
            The compiled StateGraph
        """
        return self._graph
    
    def compile_with_entry_point(self, entry_point: str) -> StateGraph:
        """
        Recompile the graph with a different entry point.
        
        Args:
            entry_point: The new entry point node name
            
        Returns:
            The newly compiled StateGraph
        """
        self._entry_point = entry_point
        self._visited_nodes = set()
        self._graph = StateGraph(self._state_type)
        self._compile_graph()
        return self._graph