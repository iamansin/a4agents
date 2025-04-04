from typing_extensions import (
    Literal,
    Callable, 
    Optional,
    Dict,
    Any, 
    Tuple, 
    List, 
    Union,
    TypedDict)
import time
import logging
import base64
import ray 
from dataclasses import dataclass, field
import functools
from enum import Enum, auto
from pydantic import BaseModel
from urllib.parse import urlparse
import warnings
import aiohttp
import asyncio
#self Imports
from a4agents_core.Utils._Singleton import Singleton
from Registry.BaseRegistry import Registry, ETYPES, ToolValidationError, ToolCreationError
from Router.Router import Router, EdgeType
from Registry.Handlers import AgentHandler, ToolHandler
from GraphCompiler import DynamicGraphCompiler

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors in the registry."""
    pass

class SelfLoopCondition(Exception):
    """Custom exception for self-loop conditions in the routing graph."""
    pass

class RouteValidationError(Exception):
    """Custom exception for route validation errors in the routing graph."""
    pass

class NodeCreationError(Exception):
    """Custom exeception for Node Creation errors in the System class"""

class NodeType(Enum):
    NODE = auto()
    TOOL = auto()
    AGENT = auto()
    
@dataclass(slots=True)
class ETool:
    tool_type : Literal[ETYPES.LOCAL, ETYPES.REMOTE]
    func : Optional[Callable] = None
    repo_url : Optional[str] = None
    endpoint : Optional[str] = None
    api_key : Optional[str] = None

@dataclass(slots=True)
class EAgent:
    agent_type : Literal[ETYPES.LOCAL, ETYPES.REMOTE]
    func : Optional[Callable] = None
    repo_url : Optional[str] = None
    endpoint : Optional[str] = None
    api_key : Optional[str] = None
    
@dataclass(frozen=True, slots=True)
class SystemNode:
    """Encapsulates a node in the LangGraph System with enhanced performance using slots."""
    name: str
    node_type: NodeType
    function: Optional[Callable] = None
    handler: Optional[Any] = None  # ToolHandler or AgentHandler
    scale: bool = False
    resources: Dict[str, Any] = field(default_factory=dict)
    task: Optional[Any] = field(default=None, init=False, compare=False) # To store the remote function

    def __post_init__(self):
        """Validate the node configuration and set up Ray task if scaling is enabled."""
        # Validate function or handler based on node type
        if self.node_type == NodeType.NODE and not callable(self.function):
            raise ValueError(f"Node '{self.name}' requires a callable function")
            
        if self.node_type in (NodeType.TOOL, NodeType.AGENT) and self.handler is None:
            raise ValueError(f"{self.node_type.name.capitalize()} '{self.name}' requires a handler")

        logger.info(f"SystemNode '{self.name}' initialized as {self.node_type.name}")

    # Here get_callable() method returns a wrapper which encapsulates the 
    # original logic for execution of the particular node type.
    # We are creating a wrapper here so as to provide an additional functionality,
    # this wrapper takes state as an argument and returns the state after node execution.
    # The state is updated based on the node result rather then keyword this decreases KeyNotFound error.
    # as well as provides flexibility to us.`
    def get_callable(self) -> Callable:
        """
        Returns a callable function suitable for use in LangGraph.
        
        The returned function will:
        1. Have the same name as the node for better error reporting
        2. Execute the appropriate logic based on node type
        3. Handle state management for LangGraph compatibility
        
        Returns:
            Callable: A function that can be registered with LangGraph
        """
        if self.node_type == NodeType.NODE:
            # For regular nodes, return the function directly or wrapped for Ray
            if self.scale and self.task is not None:
                # Create a wrapper for Ray remote execution
                @functools.wraps(self.function)
                async def ray_wrapper(state):
                    try:
                        # Execute remotely and get result
                        result = ray.get(self.task.remote(state))
                        return result
                    except Exception as e:
                        logger.error(f"Error executing node '{self.name}' remotely: {str(e)}")
                        raise RuntimeError(f"Failed to execute node '{self.name}': {str(e)}") from e
                
                # Ensure the wrapper has the same name as the node
                ray_wrapper.__name__ = self.name
                return ray_wrapper
            else:
                # For non-Ray execution, return the function with proper name
                @functools.wraps(self.function)
                async def function_wrapper(state):
                    try:
                        return self.function(state)
                    except Exception as e:
                        logger.error(f"Error executing node '{self.name}': {str(e)}")
                        raise RuntimeError(f"Failed to execute node '{self.name}': {str(e)}") from e
                    
                function_wrapper.__name__ = self.name
                return function_wrapper

                
        elif self.node_type in (NodeType.TOOL, NodeType.AGENT):
            # For tools and agents, create a wrapper that uses the handler
            @functools.wraps(self.handler.execute)
            async def handler_wrapper(state):
                try:
                    # Execute the handler and update state with the result
                    result = self.handler.execute(state)
                    
                    # Create a new state to avoid mutation issues
                    new_state = state.copy() if hasattr(state, 'copy') else dict(state)
                    new_state[self.name] = result
                    return new_state
                except Exception as e:
                    logger.error(f"Error executing {self.node_type.name.lower()} '{self.name}': {str(e)}")
                    raise RuntimeError(f"Failed to execute {self.node_type.name.lower()} '{self.name}': {str(e)}") from e
            
            # Ensure the wrapper has the same name as the node
            handler_wrapper.__name__ = self.name
            return handler_wrapper
        
        # This should never happen due to validation in __post_init__
        raise ValueError(f"Unsupported node type: {self.node_type}")


@Singleton(strict=True,thread_safe=True,debug=False)
@dataclass(slots=True)
class System:
    """A robust AI agent system integrated with LangGraph."""
    
    _nodes :Dict[str , SystemNode]= {}  # {"node_name" : SystemNode() , ....}
    _routers :Dict[str, List[Router]] = {} # {"from_node" : [Router() ,....]}
    _routing_dict :Dict[str, Dict[str,str]]= {} # {"from_node" : {"to_node": "Direct"/"Conditional"}}
    _lazy_tools : Dict[str,ETool] = {}
    _lazy_agents : Dict[str,EAgent] = {}
    _resource_distribution_map = {}
    _registry = Registry()
    _entry_node = None
    _end_node = None
    _embedding_model =  None # Optional embedding model for routing

    def __init__(self, name: str, 
                 state_schema :dict, 
                 embedding_model =None, 
                 use_model :bool = False, 
                 **kwargs):
        
        self.name :str = name
        self.State :Union[BaseModel,Dict] = state_schema
        self._system_resources = {**kwargs}
        # self._system_actor= Actor_Initializer(*kwargs)
        if embedding_model and use_model:
            self._embedding_model = embedding_model
        
    @classmethod
    def add_node(cls, 
             name: Optional[str] = None, 
             func: Optional[Callable] = None, 
             resources: Optional[Dict[str, Any]] = None, 
             scale :bool =False,):
        """
        Registers a function as a SystemNode in the LangGraph, supporting both decorator and direct function call.

        Usage:
        - As a decorator: `@system.node(name="my_node")`
        - As a function: `system.node(my_function, name="my_node")`

        :param func: Function to register (optional for decorator usage).
        :param name: Unique name for the node.
        :param resources: Resource allocation for Ray DAG (e.g., {"num_cpus": 2, "num_gpus": 1}).
        :param is_tool: Flag to indicate if the node represents a tool.
        :return: SystemNode instance (when used directly) or decorator (when used as `@system.node`).
        """

        def register_node(func: Callable):
            """Inner function to handle function registration logic."""
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Node name must be a non-empty string.")
            if not callable(func):
                raise TypeError(f"Expected a callable function, got {type(func).__name__}")
            if name in cls._nodes:
                raise  NodeCreationError(f"Node name '{name}' already exists in system '{cls.name}'.")

            try:
                node = SystemNode(name= name,
                                  function = func,
                                  node_type= NodeType.NODE,
                                  scale= scale,
                                  resources=resources,
                                  )
                
                cls._nodes[name] = node
                if scale:
                    cls._resource_distribution_map[name] = resources
                logger.info(f"Node '{name}' added to System '{cls.name}'.")
                return node
            except Exception as e:
                logger.error(f"Error creating node '{name}': {str(e)}")
                raise NodeCreationError(f"Error creating node '{name}': {str(e)}")

        # If `func` is provided, it's being used directly: `system.node(my_function, name="my_node")`
        if func is not None:
            return register_node(func)

        # If `func` is None, return a decorator: `@system.node(name="my_node")`
        def decorator(func: Callable):
            return register_node(func)

        return decorator
        
    @classmethod
    def add_route(cls, from_node: str, 
                  to_node : Optional[str] =None,
                  to_multiple_nodes: Optional[List[str]] = None,
                  use_model_routing :bool = False,
                  allow_self_loop: bool = False):
        
        if not any(to_node, to_multiple_nodes):
            raise RouteValidationError("Must Provide either single node name or a List containing multiple node names.")
        if to_node and to_multiple_nodes:
            raise RouteValidationError("Provide either single node name or a List containing multiple node names.")
        if from_node not in cls._nodes:
            raise ValueError(f"From node '{from_node}' not registered in the system.")
        
        _node_list = [to_node] if to_node else to_multiple_nodes
        for node in _node_list: # here we are directly raising the error.
            if not isinstance(node, str) or not node.strip():
                raise TypeError(f"Node name '{node}' must be a non-empty string.")
            if node not in cls._nodes:
                raise ValueError(f"No Node with name {node} found in the System!")
            if node == from_node and not allow_self_loop:
                raise SelfLoopCondition(f"Node '{from_node}' is routing to itself. May cause infinite loop.")
            if node in cls._routing_dict[from_node]: #_self.routing_dict : {from_node: {node: "Direct"/"Conditional"}}
                raise RouteValidationError(f" {cls._routing_dict[from_node].get(node)} Route from '{from_node}' to '{node}' already exists.")
        try:
            _router = Router(from_node=from_node, 
                             direct_nodes=_node_list,
                             use_embeddings=use_model_routing,
                             embedding_model= cls._embedding_model if use_model_routing else None, 
                             router_type= EdgeType.DIRECT,
                             )# Assuming this is defined somewhere in the class
            if cls._routers.get(from_node,None) is not None:
                cls._routers[from_node].append(_router)
            else:
                cls._routers[from_node] = [_router]
                
        except Exception as e: # here we are catching the error from lower level and then raising the error.
            raise RouteValidationError(f"Error creating router from '{from_node}' to '{to_multiple_nodes}': {str(e)}") from e
    
    @classmethod
    def add_conditional_route(cls, 
                              from_node: str, 
                              to_node : Optional[Dict[str,str]] = None,
                              to_multiple_nodes: Optional[List[Dict[str, str]]] = None, 
                              use_model_routing :bool = False
                              ,allow_self_loop: bool = False):
        """
        Adds a conditional route from one node to multiple nodes based on a condition function.
        :param from_node: Name of the source node.
        :param to_nodes: Dictionary where keys are condition names and values are target node names.
        :param condition_func: Function that determines the routing condition.
        """
        if not any(to_node, to_multiple_nodes):
            raise RouteValidationError("Must Provide either single node name or a List containing multiple node names.")
        if to_node and to_multiple_nodes:
            raise RouteValidationError("Provide either single node name or a List containing multiple node names.")
        if from_node not in cls._nodes:
            raise ValueError(f"From node '{from_node}' not registered in the system.")
        
        _node_dict : Dict = to_node if to_node else to_multiple_nodes
        for condition, node in _node_dict.items():
            if not isinstance(condition, str) or not condition.strip():
                raise TypeError(f"Condition name '{condition}' must be a non-empty string.")
            if not isinstance(node, str) or not node.strip():
                raise TypeError(f"Node name '{node}' must be a non-empty string.")
            if node not in cls._nodes:
                raise ValueError(f"No Node with name {node} found in the System!")
            if node == from_node and not allow_self_loop:
                raise SelfLoopCondition(f"Node '{from_node}' is routing to itself. May cause infinite loop.")
            if node in cls._routing_dict[from_node]:
                raise RouteValidationError(f"Route from '{from_node}' to '{node}' already exists.")

        try:
            _router = Router(from_node=from_node, 
                             conditional_nodes=_node_dict, 
                             use_embeddings= use_model_routing,
                             embedding_model= cls._embedding_model if use_model_routing else None,
                             router_type= EdgeType.CONDITIONAL,
                            )
            if cls._routers.get(from_node,None) is not None:
                cls._routers[from_node].append(_router)
            else:
                cls._routers[from_node] = [_router]
        except Exception as e:
            raise RouteValidationError(f"Error creating router from '{from_node}' to '{_node_dict}': {str(e)}") from e
    
    @classmethod
    def add_tool(cls,
            tool_name :str, 
            tool_func :Optional[Callable] = None,
            resources: Optional[Dict[str, Any]] = None):
        """This method is used to add tools into the system."""
        
        def register_tool(tool_func:Union[Callable]):
            
            if tool_name is None or not isinstance(tool_name, str) or not tool_name.strip():
                raise TypeError("Tool name must be a non-empty string.")
            
            if tool_name in cls._lazy_tools:
                raise ToolValidationError(f"Tool with name {tool_name} already exists in the Registry.")
                
            if not isinstance(tool_func,Callable):
                raise ValueError("The tool_func should be a type of Callable function.")
            
            #manage Resources also (ray update)
            cls._lazy_tools[tool_name] = ETool(tool_type = ETYPES.CUSTOM,
                                           func = tool_func)
        
        if tool_func is not None:
            return register_tool(tool_func)
        
        def decorator(tool_func :Callable):
            return register_tool(tool_func)
        
        return decorator
    
    #Here we have to add the logic for lazy loading.
    def add_etool(cls,
                    tool_name: str, 
                    endpoint: Optional[str] = None, 
                    api_key: Optional[str] = None,
                    repo_url: Optional[str] = None):
        
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValidationError("Tool name must be a non-empty string")
        if tool_name in cls._tools:
            raise ToolValidationError(f"Tool with name {tool_name} already exists in the Registry.")
        if not any(endpoint,repo_url): 
            raise ToolValidationError("Tool must have either an endpoint, repository URL")
        if endpoint and repo_url:
            raise ToolValidationError("Tool cannot have both an endpoint and repository URL")
        
        tool_type : ETYPES = ETYPES.REMOTE if endpoint else ETYPES.LOCAL

        cls._lazy_tools[tool_name] = ETool(tool_type = tool_type,
                                           repo_url =repo_url,
                                           endpoint = endpoint,
                                           api_key = api_key)
    
    @classmethod
    def add_multiple_nodes(cls, node_dict: Dict[str, Tuple[Callable, Optional[Dict[str, Any]]]]):
        """
        Adds multiple functions as SystemNodes from a dictionary.

        :param func_dict: Dictionary where keys are node names and values are functions.
        :param resources: Optional dictionary specifying resource allocation.
        :raises TypeError: If func_dict is not a dictionary or contains invalid keys/values.
        :raises ValueError: If any node name is already registered.
        """
        if not isinstance(node_dict, dict):
            raise TypeError(f"Expected a dictionary mapping, but got {type(func_dict).__name__}")

        for name, args in node_dict.items():
            try:
                cls.add_node(name=name, func=args[0], resources= args[1] if len(args)>1 else None)
            except Exception as e:
                logger.error(f"Error adding node '{name}': {str(e)}")
                raise
    
    @classmethod
    def add_multiple_tools(cls, tool_dict : Dict[str, Tuple[Callable, Optional[Dict[str, Any]]]]) -> None:
        
        if not isinstance(tool_dict, dict):
            raise TypeError(f"Expected a dictionary mapping, but got {type(func_dict).__name__}")

        for name, args in tool_dict.items():
            try:
                cls.add_tool(name=name, func=args[0], resources= args[1] if len(args)>1 else None)
            except Exception as e:
                logger.error(f"Error adding node '{name}': {str(e)}")
                raise

    def add_multiple_etools(cls, etool_dict :Dict[str,Dict[str,str]]) -> None:
        
        if not isinstance(etool_dict, dict):
            raise TypeError("Expected a dictionary with {name: function} mapping, but got {type(func_dict).__name__}")

        # for name, args in etool_dict.items():
        #     try:
        #         cls.add_etool(name=name,
        #                       endpoint=args.get(""))
        #     except Exception as e:
        #         logger.error(f"Error adding node '{name}': {str(e)}")
        #         raise

    async def register4agent(self,
                            endpoint: Optional[str] = None, 
                            api_key: Optional[str] = None,
                            repo_url: Optional[str] = None):
        

            
        try:
            await self._registry.add_agent_package(name, agent_type, endpoint, api_key, repo_url)
            
        except Exception as e :
            raise e
        
    def set_entry_node(self, node_name: str):
        """Sets the entry point node for the LangGraph graph."""
        if not isinstance(node_name,str) or not node_name.strip():
            raise TypeError("The entry point name should be a non-empty string.")
        if self._entry_node:
            raise ValueError(f"Node :{self._entry_node} is already set as entry point.")
        if node_name not in self._nodes:
            raise ValueError(f"Entry point node '{node_name}' not registered in the system.")
        self._entry_node = node_name
        
    def set_end_node(self, node_name: str):
        """Sets a node as the end node of the LangGraph workflow."""
        if not isinstance(node_name,str) or not node_name.strip():
            raise TypeError("The entry point name should be a non-empty string.")
        if self._entry_node:
            raise ValueError(f"Node :{self._entry_node} is already set as entry point.")
        if node_name not in self._nodes:
            raise ValueError(f"End node '{node_name}' not registered in the system.")
        self._end_point = node_name 

    async def _load_tools(
        self,
        _tools_list: Dict[str, ETool],
        logger: Optional[logging.Logger] = None
    ) -> Dict[str, ETool]:
        """
        Asynchronously loads all tools concurrently and returns dictionary of failed tools.
        
        Args:
            _tools_list (Dict[str, ETool]): Dictionary mapping tool names to their configurations
            logger (Optional[logging.Logger]): Logger instance for error tracking
            
        Returns:
            Dict[str, ETool]: Dictionary of tools that failed to load
        """
        _failed_tools: Dict[str, ETool] = {}
        
        async def _process_single_tool(tool_name: str, etool: ETool) -> Tuple[str, Any]:
            """Inner function to process a single tool and return its result."""
            try:
                tool_handler = await self._registry.add_tool(
                    tool_type=etool.tool_type,
                    func=etool.func,
                    endpoint=etool.endpoint,
                    api_key=etool.api_key,
                    repo_url=etool.repo_url
                )
                return tool_name, tool_handler
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to load tool {tool_name}: {str(e)}", exc_info=True)
                return tool_name, e

        # Start all tool loading tasks concurrently
        tasks: List[asyncio.Task] = [
            asyncio.create_task(_process_single_tool(tool_name, etool))
            for tool_name, etool in _tools_list.items()
        ]
        
        if logger:
            logger.info(f"Started loading {len(tasks)} tools concurrently")

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for (tool_name, result) in results:
            if isinstance(result, Exception):
                _failed_tools[tool_name] = _tools_list[tool_name]
                warnings.warn(
                    f"Tool {tool_name} failed to load: {str(result)}",
                    RuntimeWarning
                )
                continue
                
            if not result or not isinstance(result, ToolHandler):
                _failed_tools[tool_name] = _tools_list[tool_name]
                warnings.warn(
                    f"Invalid tool handler for {tool_name}. Got: {type(result)}",
                    RuntimeWarning
                )
                continue
                
            # Store successfully created tool
            self._nodes[tool_name] = SystemNode(
                name=tool_name,
                node_type=NodeType.TOOL,
                handler=result
            )

        return _failed_tools

    async def _load_lazy_tools(
        self,
        retry: bool = False,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Loads all lazy-initialized tools concurrently with optional retry.
        
        Args:
            retry (bool): Whether to retry loading failed tools
            logger (Optional[logging.Logger]): Logger instance for error tracking
            
        Raises:
            ToolCreationError: If tools fail to load after retry
            RuntimeError: If tools cannot be loaded even after retry
        """
        start_time = time.perf_counter
        tools_list = self._lazy_tools.copy()
        
        if logger:
            logger.info(f"Starting concurrent load of {len(tools_list)} tools")
        
        try:
            # First attempt - load all tools concurrently
            failed_tools = await self._load_tools(tools_list, logger)
            
            if failed_tools:
                failed_count = len(failed_tools)
                if logger:
                    logger.warning(f"{failed_count} tools failed to load")
                
                # Retry failed tools if requested
                if retry:
                    if logger:
                        logger.info(f"Retrying {failed_count} failed tools concurrently")
                    
                    _failed_tools = await self._load_tools(failed_tools, logger)
                    
                    if _failed_tools:
                        failed_names = ", ".join(_failed_tools.keys())
                        raise RuntimeError(
                            f"Tools failed to load after retry: {failed_names}"
                        )
                
        except Exception as e:
            if logger:
                logger.error(f"Critical error in tool loading: {str(e)}", exc_info=True)
            raise ToolCreationError(f"Tool loading failed: {str(e)}") from e
            
        finally:
            duration = time.perf_counter - start_time
            if logger:
                logger.info(
                    f"Tool loading completed in {duration:.2f}s. "
                    f"Success: {len(tools_list) - len(failed_tools)}, "
                    f"Failed: {len(failed_tools)}"
                )
        
    async def _load_lazy_agents():
        pass
    @classmethod
    def compile(cls,**kwargs):
        """Compiles the LangGraph graph to finalize the workflow definition."""
        if cls._lazy_tools:
            try:
                cls._load_lazy_tools()
            except Exception as e:
                raise e
            
        if cls._lazy_agents:
            try: 
                cls._load_lazy_agents()
            except Exception as e:
                raise e
            
        Dynamic_Graph : CompiledGraphObj = DynamicGraphCompiler(cls._nodes, 
                                                                cls._registry)._compile_graph

    def execute(self, start_node: str, data: Any, config: Optional[dict] = None): # Added LangGraph config
        """
        Executes the LangGraph workflow starting from a specified node.

        Parameters:
        - start_node (str): Name of the starting node (entry point needs to be set separately).
        - data (Any): Input data for execution.
        - config (Optional[dict]): Optional configuration for LangGraph execution.

        Returns:
        - The output of the LangGraph workflow execution.
        """
        start_time = time.time()

        if not hasattr(self._graph, "entry_point"): # Check if entry point is set
            raise ValueError("No entry point defined. Call `set_entry_point()` first.")

        try:
            result = self._DAGExecutor.execute(input=data, config=config) # Pass config to executor
            return result, (time.time() - start_time)
        except Exception as e:
            raise RuntimeError(f"Error executing LangGraph workflow: {e}")

    def draw_graph(self) -> None:
        """Display the mermaid diagram using mermaid.ink with error handling."""

        def generate_mermaid() -> str:
            """Generate the mermaid.js diagram string with proper error handling."""
            if not self._nodes:
                raise RuntimeError("Cannot generate a Mermaid diagram without any nodes.")

            graph = self._dag
            mermaid_string = "graph TD\n"

            shape_map = {
                "rectangle": '[\"{label}\"]',
                "diamond": '{{\"{label}\"}}',
                "circle": '(\"{label}\")'
            }

            for vertex in graph.vs: # Iterate over igraph vertices
                node_name = vertex['name'] # Get node name from vertex attribute
                shape = vertex.attributes().get("shape", "circle")  # Get shape, default to circle if not found
                shape_str_mermaid = shape_map[shape].format(label=node_name)
                mermaid_string += f"    {node_name}{shape_str_mermaid}\n"

            # Add edges with labels and conditional styles
            for edge in graph.es: # Iterate over igraph edges
                from_node_name = edge.source_vertex['name'] # Get source vertex name
                to_node_name = edge.target_vertex['name'] # Get target vertex name
                label = edge['type'] # Get edge 'type' attribute
                edge_style = "-->" if label == "Direct" else "-.->"
                mermaid_string += f"    {from_node_name} {edge_style}{to_node_name}\n"

            return mermaid_string.strip()
        try:
            mermaid_string = generate_mermaid()
            base64_string = base64.urlsafe_b64encode(mermaid_string.encode("utf8")).decode("ascii")
            mermaid_url = "https://mermaid.ink/img/" + base64_string
            display(Image(url=mermaid_url))
        except Exception as e:
            raise e
            print(f"Error while generating Mermaid diagram: {e}")

