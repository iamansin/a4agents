from typing import Callable, Optional, Dict, Any, Tuple, List, Union
import time
import logging
import base64
import ray 

#self Imports
from ETools.ETools import ETool
from Registry import BaseRegistry

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors in the registry."""
    pass

class Main_Actor_Class:
    func_store : Dict

    async def add_function(self, func_name, func_ref):
        self.func_store[func_name] = func_ref
        


class Actor_Initializer:
    def __init__(self,**kwargs):
        self._Actor = ray.remote.options(**kwargs)(Main_Actor_Class)

class SystemNode:
    """Encapsulates a function as a node in the LangGraph System."""

    def __init__(self, name: str, func: Callable, is_tool: bool = False, scale :bool = False, **kwargs):
        """
        Initializes the SystemNode.

        :param func: Function to be used in the node. Should ideally be a LangChain Runnable.
        :param name: Unique name for the node.
        :param resources: Resource allocation (currently conceptual for Ray integration later).
        :param is_tool: Flag to indicate if the node represents a tool.
        """
        self._name = name
        self._resources = {**kwargs}
        self._func = func # Store the original function directly, LangGraph will handle Runnable conversion
        self._is_tool = is_tool
        self._scale = scale
        if scale:
            self._task = ray.remote.options(**kwargs)(func)
        logger.info(f"SystemNode '{self.name}' initialized.")


class System:
    """A robust AI agent system integrated with LangGraph."""

    def __init__(self, name: str, state_schema :dict, **kwargs):
        self.name = name
        self.State = state_schema
        self._system_resources = {**kwargs}
        self._system_actor= Actor_Initializer(*kwargs)
        self._nodes = {}  # Stores agent nodes (SystemNode instances)
        self._router_mappings = {} #stores (Router Instances)
        self._resource_distribution_map = {}
        self._graph = None
        self._registry = BaseRegistry()
         
    def node(self, name: Optional[str] = None, 
             func: Optional[Callable] = None, 
             resources: Optional[Dict[str, Any]] = None, 
             is_tool: bool = False, 
             scale :bool =False):
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
            if not callable(func):
                raise TypeError(f"Expected a callable function, got {type(func).__name__}")

            if not isinstance(name, str) or not name.strip():
                raise ValueError("Node name must be a non-empty string.")

            if name in self._nodes:
                raise ValueError(f"Node name '{name}' already exists in system '{self.name}'.")

            if any(node.func == func for node in self._nodes.values()): # Comparing original function, not task
                raise ValueError(f"Function '{func.__name__}' is already registered under a different name.")

            try:
                node = SystemNode(name, func, is_tool, scale,resources)
                self._nodes[name] = node
                if scale:
                    self._resource_distribution_map[name] = {**resources}
                logger.info(f"Node '{name}' added to System '{self.name}'.")
                return node
            except Exception as e:
                logger.error(f"Error creating node '{name}': {str(e)}")
                raise

        # If `func` is provided, it's being used directly: `system.node(my_function, name="my_node")`
        if func is not None:
            return register_node(func)

        # If `func` is None, return a decorator: `@system.node(name="my_node")`
        def decorator(func: Callable):
            return register_node(func)

        return decorator
    
    def nodes_from_dict(self, func_dict: Dict[str, Tuple[Callable, Optional[Dict[str, Any]]]]):
        """
        Adds multiple functions as SystemNodes from a dictionary.

        :param func_dict: Dictionary where keys are node names and values are functions.
        :param resources: Optional dictionary specifying resource allocation.
        :raises TypeError: If func_dict is not a dictionary or contains invalid keys/values.
        :raises ValueError: If any node name is already registered.
        """
        if not isinstance(func_dict, dict):
            raise TypeError("Expected a dictionary with {name: function} mapping, but got {type(func_dict).__name__}")

        for name, args in func_dict.items():
            try:
                self.node(func=args[0], name=name, resources=args[1])
            except Exception as e:
                logger.error(f"Error adding node '{name}': {str(e)}")
                raise

    def set_entry_point(self, entry_point_node: str):
        """Sets the entry point node for the LangGraph graph."""
        if self._entry_node:
            raise ValueError(f"Node :{self._entry_node} is already set as entry point.")
        if entry_point_node not in self._nodes:
            raise ValueError(f"Entry point node '{entry_point_node}' not registered in the system.")
        self._graph.set_entry_point(self._nodes[entry_point_node].func) # Assuming SystemNode.func is LangChain Runnable
        self._entry_node = entry_point_node
        
    def add_router(self, from_node: str, direct_edges: List, conditional_edges :Dict[str,str]):
        
        if from_node not in self._nodes:
            raise ValueError(f"From node '{from_node}' not registered in the system.")
        if direct_edges:
            for node in direct_edges:
                if node not in self._nodes:
                    raise ValueError(f"No Node with name {node} found in the System!")
        if conditional_edges:
            edges = {}
            for condition_value, to_node_name in conditional_edges.items():
                if to_node_name not in self._nodes:
                    raise ValueError(f"No Node with name {node} found in the System!")
                edges[condition_value] = self._nodes[to_node_name].func # Map condition value to LangGraph Runnable

            self._graph.add_conditional_edges(
                self._nodes[from_node].func, condition_func, edges
            )
        logger.info(f"Added conditional edges from '{from_node}' with conditions '{conditional_edge_mapping}'.")
        
        pass
    
    def add_edge(self, from_node :str, to_node: str, ): # Renamed from add_route to align with LangGraph naming
        """Adds a direct edge in the LangGraph graph."""
        if from_node not in self._nodes:
            raise ValueError(f"From node '{from_node}' not registered in the system.")
        if to_node not in self._nodes:
            raise ValueError(f"To node '{to_node}' not registered in the system.")

        self._graph.add_edge(self._nodes[from_node].func, self._nodes[to_node].func) # Connect LangGraph Runnables
        logger.info(f"Added direct edge from '{from_node}' to '{to_node}'.")

    def tool(self, tool_name :str, tool_func :Union[Callable,ETool]):
        """This method is used to add tools into the system."""
        
        def register_tool(tool_func:Union[Callable,ETool]):
            if tool_name in self._nodes:
                raise ValueError(f"A Node with name {tool_name} already exists.")
            
            if self._registry.check_existing_tool(tool_name):
                raise ValueError(f"The Tool {tool_name} already exists in the Registry.")
            
            if not isinstance(tool_func,Callable) or not isinstance(tool_func,ETool):
                raise ValueError("The tool_func should be a type of Callable or an isntance of ETool.")
            
            if isinstance(tool_func,Callable):
                tool_func = ETool(tool_name, tool_func)
        
            self._tools[name] = tool_func
        
        
        if tool_func is not None:
            return register_tool(tool_func)
        
        def decorator(tool_func :Callable):
            return register_tool(tool_func)
        
        return decorator
        
    def set_workflow_end_node(self, node_name: str):
        """Sets a node as the end node of the LangGraph workflow."""
        if node_name not in self._nodes:
            raise ValueError(f"End node '{node_name}' not registered in the system.")
        self._graph.set_end_point(self._nodes[node_name].func) # Set LangGraph end point
        logger.info(f"Set node '{node_name}' as workflow end point.")

    def compile(self,**kwargs):
        """Compiles the LangGraph graph to finalize the workflow definition."""
        Dynamic_Graph = DynamicGraphCompiler(self._nodes, self._router_mappings)

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

    async def register4agent(self,name: str, 
                    endpoint: Optional[str] = None, 
                    api_key: Optional[str] = None,
                    repo_url: Optional[str] = None):
        
        # if name in self._registry.get_agent_names():
        #     raise ValueError(f"Agent '{name}' is already registered.")
        
        if not endpoint and not repo_url:
            raise ValidationError("Agent must have either an endpoint or repository URL")
        
        if endpoint and repo_url:
            raise ValidationError("Agent cannot have both an endpoint and repository URL")
        
        if endpoint:
            agent_type = "MCP"
            
        elif repo_url:
            agent_type = "LOCAL"
            
        try:
            await self._registry.add_agent_package(name, agent_type, endpoint, api_key, repo_url)
            
        except Exception as e :
            raise e
    
    async def register4tool(self,name: str, 
                    endpoint: Optional[str] = None, 
                    api_key: Optional[str] = None,
                    repo_url: Optional[str] = None):
        
        if not endpoint and not repo_url:
            raise ValidationError("Tool must have either an endpoint, repository URL")
        
        if endpoint and repo_url:
            raise ValidationError("Tool cannot have both an endpoint and repository URL")
        
        
        # if name in self._registry.get_tool_names():
        #     raise ValueError(f"Tool '{name}' is already registered.")

        if endpoint:
            tool_type = "MCP"
            
        elif repo_url:
            tool_type = "LOCAL"
            
        try:
            await self._registry.add_tool_package(name, tool_type, endpoint, api_key, repo_url)
        
        except Exception as e:
            raise e