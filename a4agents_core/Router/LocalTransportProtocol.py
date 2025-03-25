import json
import sys
import os
import subprocess
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
import asyncio
import threading
import queue
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict

# Import the MCP Transport base class
from mcp.transports.base import Transport

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StdioToolTransport")

@dataclass
class ToolCall:
    """Represents a function call to a tool."""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class ToolResponse:
    """Represents a response from a tool call."""
    call_id: str
    result: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ToolRegistry:
    """Registry for tools that can be called."""
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        
    def register(self, name: str, func: Callable) -> None:
        """Register a tool function."""
        self.tools[name] = func
        
    def get(self, name: str) -> Optional[Callable]:
        """Get a tool function by name."""
        return self.tools.get(name)
        
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())

class StdioToolTransport(Transport):
    """
    A high-performance transport layer for tool communication using stdio.
    Implements the MCP Transport interface for compatibility with MCP clients.
    Works with MCP tools, Langchain tools, and custom tools.
    """
    
    def __init__(
        self, 
        registry: Optional[ToolRegistry] = None,
        subprocess_command: Optional[List[str]] = None,
        buffer_size: int = 16384,
        timeout: float = 60.0,
        auto_reconnect: bool = True,
        max_retries: int = 3
    ):
        """
        Initialize the StdioToolTransport.
        
        Args:
            registry: Optional tool registry for local function calls
            subprocess_command: Optional command to spawn a subprocess for tool execution
            buffer_size: Size of the read buffer
            timeout: Timeout for operations in seconds
            auto_reconnect: Whether to automatically reconnect on failure
            max_retries: Maximum number of retries for operations
        """
        # Initialize the base Transport class
        super().__init__()
        
        self.registry = registry or ToolRegistry()
        self.subprocess_command = subprocess_command
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.auto_reconnect = auto_reconnect
        self.max_retries = max_retries
        
        # Communication channels
        self.process = None
        self.stdin = None
        self.stdout = None
        self.stderr = None
        
        # Async support
        self.response_queue = queue.Queue()
        self.pending_calls: Dict[str, asyncio.Future] = {}
        self.lock = threading.RLock()
        self.read_thread = None
        self.is_running = False
        
        # Initialize if subprocess command is provided
        if self.subprocess_command:
            self._initialize_subprocess()
        else:
            # Use system stdin/stdout if no subprocess
            self.stdin = sys.stdin.buffer
            self.stdout = sys.stdout.buffer
            
            # Start the reader thread if using system stdin/stdout
            self.is_running = True
            self.read_thread = threading.Thread(target=self._read_output_loop)
            self.read_thread.daemon = True
            self.read_thread.start()
    
    def _initialize_subprocess(self) -> None:
        """Initialize the subprocess communication channel."""
        try:
            self.process = subprocess.Popen(
                self.subprocess_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.buffer_size,
                universal_newlines=False
            )
            self.stdin = self.process.stdin
            self.stdout = self.process.stdout
            self.stderr = self.process.stderr
            
            # Start the reader thread
            self.is_running = True
            self.read_thread = threading.Thread(target=self._read_output_loop)
            self.read_thread.daemon = True
            self.read_thread.start()
            
            logger.info(f"Subprocess initialized: {' '.join(self.subprocess_command)}")
        except Exception as e:
            logger.error(f"Failed to initialize subprocess: {e}")
            raise
    
    def _read_output_loop(self) -> None:
        """Background thread to continuously read from stdout."""
        while self.is_running and self.stdout:
            try:
                line = self.stdout.readline()
                if not line:
                    if self.process and self.process.poll() is not None:
                        logger.warning("Subprocess terminated unexpectedly")
                        if self.auto_reconnect:
                            self._initialize_subprocess()
                    break
                
                try:
                    response = json.loads(line.decode('utf-8').strip())
                    self.response_queue.put(response)
                    
                    # Handle async responses
                    if 'call_id' in response:
                        with self.lock:
                            future = self.pending_calls.pop(response['call_id'], None)
                            if future and not future.done():
                                future.set_result(response)
                                
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSON response: {line}")
            except Exception as e:
                logger.error(f"Error in read loop: {e}")
                
        logger.info("Read loop terminated")
    
    def _write_request(self, data: Dict[str, Any]) -> None:
        """Write a request to stdin."""
        if not self.stdin:
            raise ValueError("Stdin is not available")
            
        try:
            json_str = json.dumps(data)
            self.stdin.write((json_str + '\n').encode('utf-8'))
            self.stdin.flush()
        except Exception as e:
            logger.error(f"Failed to write request: {e}")
            raise
    
    # Implementation of the Transport interface method
    def send(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a request and wait for a response.
        
        Args:
            request: The request to send, following MCP protocol format
            
        Returns:
            The response dictionary according to MCP protocol
        """
        # Extract tool call from the request
        # Assuming the MCP request contains a 'functions' field with 'name' and 'arguments'
        function_call = request.get('functions', [{}])[0]
        tool_name = function_call.get('name', '')
        arguments = function_call.get('arguments', {})
        
        if not tool_name:
            return {"error": "No tool name specified in request"}
        
        # Create a call ID if not present
        call_id = request.get('call_id', str(uuid.uuid4()))
        
        # First, check if we have a local implementation
        tool_func = self.registry.get(tool_name)
        if tool_func:
            try:
                result = tool_func(**arguments)
                return {
                    "call_id": call_id,
                    "result": result,
                    "error": None
                }
            except Exception as e:
                logger.error(f"Local tool execution failed: {e}")
                return {
                    "call_id": call_id,
                    "result": None,
                    "error": str(e)
                }
        
        # If not local, use the subprocess
        if not self.subprocess_command:
            return {
                "call_id": call_id,
                "result": None,
                "error": f"Tool {tool_name} not found in registry and no subprocess configured"
            }
        
        # Prepare the request for the subprocess
        tool_call = ToolCall(tool_name=tool_name, arguments=arguments, call_id=call_id)
        subprocess_request = asdict(tool_call)
        
        # Send request
        with self.lock:
            self._write_request(subprocess_request)
            
        # Wait for the response with timeout
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            try:
                response = self.response_queue.get(timeout=0.1)
                if response.get('call_id') == call_id:
                    # Transform to MCP response format if needed
                    return self._transform_to_mcp_response(response)
                else:
                    # Put back responses for other calls
                    self.response_queue.put(response)
            except queue.Empty:
                continue
        
        return {
            "call_id": call_id,
            "result": None,
            "error": f"Timeout waiting for response from {tool_name}"
        }
    
    def _transform_to_mcp_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a tool response to MCP format if needed."""
        # If already in MCP format, return as is
        if 'generations' in response:
            return response
            
        # Otherwise, transform to MCP format
        result = response.get('result')
        error = response.get('error')
        
        mcp_response = {
            "call_id": response.get("call_id", ""),
            "generations": [
                {
                    "text": str(result) if result is not None else "",
                    "finish_reason": "error" if error else "stop"
                }
            ]
        }
        
        if error:
            mcp_response["error"] = error
            
        return mcp_response
    
    # Implementation of the Transport interface method
    def close(self) -> None:
        """Close the transport connection and clean up resources."""
        self.is_running = False
        
        # Clean up subprocess if we created one
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                self.process.kill()
            
            self.process = None
            self.stdin = None
            self.stdout = None
            self.stderr = None
            
        # Only close the thread if we're not using system stdin/stdout
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=5)
            
        logger.info("Transport closed")
    
    # Additional utility methods for direct tool calling
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool synchronously.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
        
        Returns:
            The tool's response
        """
        request = {
            "functions": [
                {
                    "name": tool_name,
                    "arguments": arguments
                }
            ],
            "call_id": str(uuid.uuid4())
        }
        
        response = self.send(request)
        return ToolResponse(
            call_id=response.get("call_id", ""),
            result=response.get("generations", [{}])[0].get("text", "") if "generations" in response else None,
            error=response.get("error")
        )
    
    async def call_tool_async(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool asynchronously.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
        
        Returns:
            The tool's response
        """
        call_id = str(uuid.uuid4())
        request = {
            "functions": [
                {
                    "name": tool_name,
                    "arguments": arguments
                }
            ],
            "call_id": call_id
        }
        
        # For local tools, just run directly
        tool_func = self.registry.get(tool_name)
        if tool_func:
            try:
                loop = asyncio.get_event_loop()
                if asyncio.iscoroutinefunction(tool_func):
                    result = await tool_func(**arguments)
                else:
                    result = await loop.run_in_executor(
                        None, lambda: tool_func(**arguments)
                    )
                return ToolResponse(
                    call_id=call_id,
                    result=result
                )
            except Exception as e:
                logger.error(f"Local tool execution failed: {e}")
                return ToolResponse(
                    call_id=call_id,
                    result=None,
                    error=str(e)
                )
        
        # If not local, use the subprocess with async waiting
        if not self.subprocess_command:
            raise ValueError(f"Tool {tool_name} not found in registry and no subprocess configured")
        
        future = asyncio.Future()
        with self.lock:
            self.pending_calls[call_id] = future
            self._write_request({
                "tool_name": tool_name,
                "arguments": arguments,
                "call_id": call_id
            })
        
        try:
            response = await asyncio.wait_for(future, timeout=self.timeout)
            return ToolResponse(
                call_id=response.get("call_id", ""),
                result=response.get("result"),
                error=response.get("error")
            )
        except asyncio.TimeoutError:
            with self.lock:
                self.pending_calls.pop(call_id, None)
            raise TimeoutError(f"Timeout waiting for response from {tool_name}")
    
    def register_tool(self, name: str, func: Callable) -> None:
        """Register a tool function in the registry."""
        self.registry.register(name, func)
    
    def register_langchain_tool(self, tool) -> None:
        """Register a Langchain tool in the registry."""
        def wrapper(**kwargs):
            return tool.run(kwargs)
        
        self.registry.register(tool.name, wrapper)
    
    def register_mcp_tool(self, tool) -> None:
        """Register an MCP tool in the registry."""
        def wrapper(**kwargs):
            return tool.execute(kwargs)
        
        self.registry.register(tool.name, wrapper)