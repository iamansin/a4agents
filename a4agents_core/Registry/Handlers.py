import asyncio
import logging
import json
import os
import httpx
from enum import Enum, auto
from typing import Dict, Any, Optional, Union, List, TypeVar, cast
from dataclasses import dataclass, asdict, field
from pathlib import Path
import time
from abc import abstractmethod, ABC

from Runner import PackageExecutor

T = TypeVar('T')
    
class ETYPES(Enum):
    CUSTOM = auto()
    LOCAL = auto()
    REMOTE = auto()
    
class ValidationError(Exception):
    """Custom exception for validation errors in the registry."""
    pass

class ExecutionError(Exception):
    """Custom exception for execution errors."""
    pass

class ToolHandlerError(Exception):
    pass

class BaseObjectHandler(ABC):
    
    def __post_init__(self) -> None:
        """Post-initialization method for validation."""
        pass
    
    @abstractmethod
    def execute(self, state: Dict[str, Any], method: Optional[str] = None) -> Dict[str, Any]:
        """Execute the handler with the given state."""
        pass

@dataclass(slots=True)
class ToolHandler(BaseObjectHandler):
    """
    Class to handle and store details about tools.
    
    Attributes:
        name (str): Unique identifier for the tool
        tool_type (str): Type of tool (langchain, mcp-server, local, etc.)
        endpoint (Optional[str]): API endpoint for remote tools
        api_key (Optional[str]): API key for authentication
        venv_path (Optional[str]): Path to the virtual environment
        entry_point (Optional[str]): Entry point script for the tool
        dir (Optional[str]): Directory containing the tool
    """
    name: str
    tool_type: str
    func : Optional[callable] = None
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    venv_path: Optional[str] = None
    entry_point: Optional[str] = None
    package_path: Optional[str] = None
    dir: Optional[str] = None
    _logger: logging.Logger = field(default_factory=lambda: logging.getLogger("ToolHandler"), repr=False, compare=False)
    _http_client: Optional[httpx.AsyncClient] = field(default=None, repr=False, compare=False)
    
    def __post_init__(self) -> None:
        """Validate the ToolHandler attributes after initialization."""
        
        if self.tool_type == ETYPES.REMOTE and not self.endpoint:
            raise ValidationError("MCP Remote tool must have an endpoint")
            
        if self.tool_type == ETYPES.LOCAL:
            if not self.venv_path:
                raise ValidationError("Local tool must have a virtual environment path")
            if not self.entry_point:
                raise ValidationError("Local tool must have an entry point")
            if not self.dir:
                raise ValidationError("Local tool must have a directory path")
            
            self._executor = PackageExecutor(path)
                
            # Convert string paths to Path objects for validation
            venv_path = Path(self.venv_path)
            entry_point_path = Path(os.path.join(self.dir, self.entry_point))
            
            if not venv_path.exists():
                raise ValidationError(f"Virtual environment path does not exist: {self.venv_path}")
            
            if not entry_point_path.exists() or not entry_point_path.is_file():
                raise ValidationError(f"Entry point does not exist or is not a file: {entry_point_path}")
    
    def set_executor(self, executor: Any) -> None:
        """Set the PackageExecutor instance for this handler."""
        self._executor = executor 
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create an HTTP client for remote execution."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=60.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            )
        return self._http_client
    
    async def execute(self, state: Dict[str, Any], method: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the tool with the given state.
        
        Args:
            state: The current state to pass to the tool
            method: Optional method name to execute
            
        Returns:
            Dict containing the execution results
            
        Raises:
            ExecutionError: If there's an error during execution
            ValidationError: If the handler is not properly configured
        """
        start_time = time.time()
        
        try:
            if self.tool_type == ETYPES.LOCAL.value:
                return await self._execute_local(state, method)
            elif self.tool_type == ETYPES.REMOTE.value:
                return await self._execute_remote(state, method)
            else:
                raise ValidationError(f"Unsupported tool type: {self.tool_type}")
        except Exception as e:
            execution_time = time.time() - start_time
            self._logger.error(f"Error executing tool '{self.name}' ({execution_time:.2f}s): {str(e)}")
            raise ExecutionError(f"Failed to execute tool '{self.name}': {str(e)}") from e
    
    async def _execute_local(self, state: Dict[str, Any], method: Optional[str] = None) -> Dict[str, Any]:
        """Execute a local tool using PackageExecutor."""
        if not self._executor:
            raise ValidationError(f"Tool '{self.name}' has no executor configured")
            
        if not self.venv_path or not self.entry_point or not self.dir:
            raise ValidationError(f"Tool '{self.name}' is missing required local execution parameters")
        
        try:
            # Convert state dict to args/kwargs format expected by PackageExecutor
            args: List[str] = []
            kwargs: Dict[str, Any] = {"input_data": json.dumps(state)}
            
            # Execute the package and parse the result
            result = await self._executor.execute_package(
                handler=self,
                method=method,
                args=args,
                kwargs=kwargs
            )
            
            # Parse stdout as JSON if possible (assuming the tool outputs JSON)
            try:
                output_data = json.loads(result.get('stdout', '{}'))
            except json.JSONDecodeError:
                # If not JSON, use the raw output
                output_data = {"output": result.get('stdout', '')}
                
            # Add execution metadata
            output_data['_metadata'] = {
                'execution_time': result.get('execution_time', 0),
                'peak_memory_mb': result.get('peak_memory_mb', 0),
                'return_code': result.get('return_code', -1)
            }
            
            # Log any errors even if the execution technically succeeded
            if result.get('stderr'):
                self._logger.warning(f"Tool '{self.name}' reported errors: {result.get('stderr')}")
                
            return output_data
            
        except Exception as e:
            self._logger.error(f"Error during local execution of tool '{self.name}': {str(e)}")
            raise ExecutionError(f"Local execution failed for tool '{self.name}': {str(e)}") from e
    
    async def _execute_remote(self, state: Dict[str, Any], method: Optional[str] = None) -> Dict[str, Any]:
        """Execute a remote tool via API endpoint."""
        if not self.endpoint:
            raise ValidationError(f"Tool '{self.name}' has no endpoint configured")
            
        client = await self._get_http_client()
        
        request_data = {
            "state": state
        }
        
        if method:
            request_data["method"] = method
            
        try:
            # Make the API request
            endpoint = f"{self.endpoint.rstrip('/')}/{method if method else ''}"
            response = await client.post(
                endpoint,
                json=request_data,
                timeout=60.0  # 60 second timeout
            )
            
            # Check for successful response
            response.raise_for_status()
            
            # Parse and return response data
            return response.json()
            
        except httpx.HTTPStatusError as e:
            # Handle HTTP errors (4xx, 5xx)
            error_detail = f"HTTP {e.response.status_code}"
            try:
                error_json = e.response.json()
                if 'error' in error_json:
                    error_detail = f"{error_detail}: {error_json['error']}"
            except json.JSONDecodeError:
                error_detail = f"{error_detail}: {e.response.text[:100]}"
                
            raise ExecutionError(f"Remote tool '{self.name}' returned an error: {error_detail}") from e
            
        except httpx.RequestError as e:
            # Handle network/connection errors
            raise ExecutionError(f"Failed to connect to remote tool '{self.name}': {str(e)}") from e
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the ToolHandler instance to a dictionary."""
        result = asdict(self)
        # Remove private fields
        for key in list(result.keys()):
            if key.startswith('_'):
                del result[key]
        return result

    async def close(self) -> None:
        """Clean up resources."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()


@dataclass(slots=True)
class AgentHandler(BaseObjectHandler):
    """
    Class to handle and store details about agents.
    
    Attributes:
        name (str): Unique identifier for the agent
        agent_type (str): Type of agent (local, remote, github)
        endpoint (Optional[str]): API endpoint for remote agents
        venv_path (Optional[str]): Path to the virtual environment
        entry_point (Optional[str]): Entry point script for the agent
        dir (Optional[str]): Directory containing the agent
    """
    name: str
    agent_type: str
    endpoint: Optional[str] = None
    venv_path: Optional[str] = None
    entry_point: Optional[str] = None   
    dir: Optional[str] = None
    api_key: Optional[str] = None
    _executor: Optional[Any] = field(default=None, repr=False, compare=False)
    _logger: logging.Logger = field(default_factory=lambda: logging.getLogger("AgentHandler"), repr=False, compare=False)
    _http_client: Optional[httpx.AsyncClient] = field(default=None, repr=False, compare=False)
    
    def __post_init__(self) -> None:
        """Validate the AgentHandler attributes after initialization."""
        if not self.name or not isinstance(self.name, str):
            raise ValidationError("Agent name must be a non-empty string")
        
        if not self.agent_type or not isinstance(self.agent_type, str):
            raise ValidationError("Agent type must be a non-empty string")
        
        if self.agent_type == ETYPES.REMOTE and not self.endpoint:
            raise ValidationError("MCP Remote Agent must have an endpoint")

        if self.agent_type == ETYPES.LOCAL.value:
            if not self.venv_path:
                raise ValidationError("Local agent must have a virtual environment path")
            if not self.entry_point:
                raise ValidationError("Local agent must have an entry point")
            if not self.dir:
                raise ValidationError("Local agent must have a directory path")
                
            # Convert string paths to Path objects for validation
            venv_path = Path(self.venv_path)
            entry_point_path = Path(os.path.join(self.dir, self.entry_point))
            
            if not venv_path.exists():
                raise ValidationError(f"Virtual environment path does not exist: {self.venv_path}")
            
            if not entry_point_path.exists() or not entry_point_path.is_file():
                raise ValidationError(f"Entry point does not exist or is not a file: {entry_point_path}")
    
    def set_executor(self, executor: Any) -> None:
        """Set the PackageExecutor instance for this handler."""
        self._executor = executor
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create an HTTP client for remote execution."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=60.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            )
        return self._http_client
    
    async def execute(self, state: Dict[str, Any], method: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the agent with the given state.
        
        Args:
            state: The current state to pass to the agent
            method: Optional method name to execute
            
        Returns:
            Dict containing the execution results
            
        Raises:
            ExecutionError: If there's an error during execution
            ValidationError: If the handler is not properly configured
        """
        start_time = time.time()
        
        try:
            if self.agent_type == ETYPES.MCP_LOCAL.value:
                return await self._execute_local(state, method)
            elif self.agent_type == ETYPES.MCP_REMOTE.value:
                return await self._execute_remote(state, method)
            else:
                raise ValidationError(f"Unsupported agent type: {self.agent_type}")
        except Exception as e:
            execution_time = time.time() - start_time
            self._logger.error(f"Error executing agent '{self.name}' ({execution_time:.2f}s): {str(e)}")
            raise ExecutionError(f"Failed to execute agent '{self.name}': {str(e)}") from e
    
    async def _execute_local(self, state: Dict[str, Any], method: Optional[str] = None) -> Dict[str, Any]:
        """Execute a local agent using PackageExecutor."""
        if not self._executor:
            raise ValidationError(f"Agent '{self.name}' has no executor configured")
            
        if not self.venv_path or not self.entry_point or not self.dir:
            raise ValidationError(f"Agent '{self.name}' is missing required local execution parameters")
        
        try:
            # Convert state dict to args/kwargs format expected by PackageExecutor
            args: List[str] = []
            kwargs: Dict[str, Any] = {"input_data": json.dumps(state)}
            
            # Execute the package and parse the result
            result = await self._executor.execute_package(
                handler=self,
                method=method,
                args=args,
                kwargs=kwargs
            )
            
            # Parse stdout as JSON if possible (assuming the agent outputs JSON)
            try:
                output_data = json.loads(result.get('stdout', '{}'))
            except json.JSONDecodeError:
                # If not JSON, use the raw output
                output_data = {"output": result.get('stdout', '')}
                
            # Add execution metadata
            output_data['_metadata'] = {
                'execution_time': result.get('execution_time', 0),
                'peak_memory_mb': result.get('peak_memory_mb', 0),
                'return_code': result.get('return_code', -1)
            }
            
            # Log any errors even if the execution technically succeeded
            if result.get('stderr'):
                self._logger.warning(f"Agent '{self.name}' reported errors: {result.get('stderr')}")
                
            return output_data
            
        except Exception as e:
            self._logger.error(f"Error during local execution of agent '{self.name}': {str(e)}")
            raise ExecutionError(f"Local execution failed for agent '{self.name}': {str(e)}") from e
    
    async def _execute_remote(self, state: Dict[str, Any], method: Optional[str] = None) -> Dict[str, Any]:
        """Execute a remote agent via API endpoint."""
        if not self.endpoint:
            raise ValidationError(f"Agent '{self.name}' has no endpoint configured")
            
        client = await self._get_http_client()
        
        request_data = {
            "state": state
        }
        
        if method:
            request_data["method"] = method
            
        try:
            # Make the API request
            endpoint = f"{self.endpoint.rstrip('/')}/{method if method else ''}"
            response = await client.post(
                endpoint,
                json=request_data,
                timeout=60.0  # 60 second timeout
            )
            
            # Check for successful response
            response.raise_for_status()
            
            # Parse and return response data
            return response.json()
            
        except httpx.HTTPStatusError as e:
            # Handle HTTP errors (4xx, 5xx)
            error_detail = f"HTTP {e.response.status_code}"
            try:
                error_json = e.response.json()
                if 'error' in error_json:
                    error_detail = f"{error_detail}: {error_json['error']}"
            except json.JSONDecodeError:
                error_detail = f"{error_detail}: {e.response.text[:100]}"
                
            raise ExecutionError(f"Remote agent '{self.name}' returned an error: {error_detail}") from e
            
        except httpx.RequestError as e:
            # Handle network/connection errors
            raise ExecutionError(f"Failed to connect to remote agent '{self.name}': {str(e)}") from e
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the AgentHandler instance to a dictionary."""
        result = asdict(self)
        # Remove private fields
        for key in list(result.keys()):
            if key.startswith('_'):
                del result[key]
        return result

    async def close(self) -> None:
        """Clean up resources."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()