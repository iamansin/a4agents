from .BaseRegistry import AgentHandler, ToolHandler
import sys
import asyncio
import logging
import time
from enum import Enum
from typing import Optional, List, Union, Dict, Any, TypeVar, Protocol
from pathlib import Path
from dataclasses import dataclass
import psutil
from abc import ABC, abstractmethod

# Type definitions and protocols
T = TypeVar('T')

class CommunicationMode(Enum):
    """Enum defining the available communication modes."""
    REAL_TIME = "real_time"
    BATCH = "batch"

@dataclass
class ExecutionMetrics:
    """Data class for storing execution metrics."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0
    last_execution_time: float = 0.0
    peak_memory_usage: float = 0.0  # in MB

class StreamHandler(Protocol):
    """Protocol defining the interface for stream handlers."""
    async def handle_output(self, line: str, stream_type: str) -> None:
        ...

class BaseStreamHandler(ABC):
    """Abstract base class for stream handlers."""
    @abstractmethod
    async def handle_output(self, line: str, stream_type: str) -> None:
        pass

class RealTimeStreamHandler(BaseStreamHandler):
    """Handler for real-time output processing."""
    def __init__(self, logger: logging.Logger, buffer_size: int = 1000):
        self.logger = logger
        self.buffer_size = buffer_size
        self.stdout_buffer: List[str] = []
        self.stderr_buffer: List[str] = []

    async def handle_output(self, line: str, stream_type: str) -> None:
        """Handle output in real-time with buffering."""
        if stream_type == "stdout":
            if len(self.stdout_buffer) >= self.buffer_size:
                self.stdout_buffer.pop(0)
            self.stdout_buffer.append(line)
            self.logger.info(f"[Output] {line}")
        else:
            if len(self.stderr_buffer) >= self.buffer_size:
                self.stderr_buffer.pop(0)
            self.stderr_buffer.append(line)
            self.logger.warning(f"[Error] {line}")

class BatchStreamHandler(BaseStreamHandler):
    """Handler for batch output processing."""
    def __init__(self):
        self.stdout_buffer: List[str] = []
        self.stderr_buffer: List[str] = []

    async def handle_output(self, line: str, stream_type: str) -> None:
        """Collect output for batch processing."""
        if stream_type == "stdout":
            self.stdout_buffer.append(line)
        else:
            self.stderr_buffer.append(line)
            
            
class PackageExecutor:
    """
    Advanced package executor for running actors and tools in isolated environments.
    Designed for high-performance, low-overhead package execution.
    """

    def __init__(
        self,
        communication_mode: CommunicationMode = CommunicationMode.BATCH,
        logger: Optional[logging.Logger] = None,
        max_concurrent_executions: int = 10,
        timeout: float = 300.0,
        semaphore_timeout: float = 10.0,
        buffer_size: int = 1000,
        memory_limit_mb: float = 1024.0  # 1GB default memory limit
    ):
        """
        Initialize PackageExecutor with advanced configuration.
        
        Args:
            communication_mode: Mode of output communication (real-time or batch)
            logger: Custom logger (creates default if not provided)
            max_concurrent_executions: Maximum number of concurrent package executions
            timeout: Default execution timeout in seconds
            buffer_size: Size of output buffer for real-time mode
            memory_limit_mb: Maximum memory limit in MB
        """
        # Validate inputs
        if max_concurrent_executions < 1:
            raise ValueError("max_concurrent_executions must be at least 1")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if buffer_size < 1:
            raise ValueError("buffer_size must be at least 1")
        if memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive")

        # Setup core components
        self.logger = logger or self._setup_logger()
        self.semaphore = asyncio.Semaphore(max_concurrent_executions)
        self.default_timeout = timeout
        self.memory_limit = memory_limit_mb
        self.communication_mode = communication_mode
        self.buffer_size = buffer_size
        self.semaphore_timeout = semaphore_timeout
        
        # Initialize metrics
        self.metrics = ExecutionMetrics()
        
        # Create appropriate stream handler
        self.stream_handler = (
            RealTimeStreamHandler(self.logger, buffer_size)
            if communication_mode == CommunicationMode.REAL_TIME
            else BatchStreamHandler()
        )

    def _setup_logger(self) -> logging.Logger:
        """
        Create a comprehensive logger for package execution.
        
        Returns:
            Configured logging.Logger
        """
        logger = logging.getLogger('PackageExecutor')
        logger.setLevel(logging.INFO)
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    async def execute_package(
        self,
        handler: Union[AgentHandler, ToolHandler],
        method: Optional[str] = None,
        args: Optional[List[str]] = None,
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a package method with error handling and performance tracking.

        Args:
            handler: ActorHandler or ToolHandler object
            method: Optional method name to execute
            args: Positional arguments for the method
            kwargs: Keyword arguments for the method

        Returns:
            Execution result dictionary
        
        Raises:
            ValueError: If handler is missing required attributes
            RuntimeError: For execution failures, timeouts, or memory errors
        """
        # Input validation
        if not hasattr(handler, 'venv_path') or not hasattr(handler, 'entry_point'):
            raise ValueError("Invalid handler: Missing venv_path or entry_point")

        # Initialize defaults
        args = args or []
        kwargs = kwargs or {}
        start_time = time.time()
        
        # Prepare execution
        python_executable = self._get_venv_python(handler.venv_path)
        execution_context = {
            'handler': handler,
            'method': method,
            'args': args,
            'kwargs': kwargs
        }
        
        try:
            # Try to acquire semaphore with timeout
            async with asyncio.timeout(self.semaphore_timeout):
                acquired = await self.semaphore.acquire()
                if not acquired:
                    raise RuntimeError("Failed to acquire execution slot")
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            self._update_metrics(False, execution_time)
            raise RuntimeError(
                f"Timeout waiting for execution slot after {self.semaphore_timeout} seconds"
            )

        try:
            # Execute the package with timeout
            try:
                result = await asyncio.wait_for(
                    self._safe_package_execution(
                        python_executable,
                        handler.entry_point,
                        execution_context
                    ),
                    timeout=self.default_timeout
                )
                
                # Log successful execution
                self.logger.info(
                    f"Successfully executed {handler.name} "
                    f"(took {result['execution_time']:.2f}s, "
                    f"peak memory: {result['peak_memory_mb']:.2f}MB)"
                )
                
                # Update metrics for success
                execution_time = time.time() - start_time
                self._update_metrics(True, execution_time)
                
                return result
                
            except asyncio.TimeoutError:
                execution_time = time.time() - start_time
                self._update_metrics(False, execution_time)
                raise RuntimeError(
                    f"Process execution timed out after {self.default_timeout} seconds"
                )
                
            except MemoryError as me:
                execution_time = time.time() - start_time
                self._update_metrics(False, execution_time)
                raise RuntimeError(
                    f"Memory limit exceeded for {handler.name}: {me}"
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                self._update_metrics(False, execution_time)
                raise RuntimeError(f"Execution error: {str(e)}") from e
                
        finally:
            # Always release the semaphore
            self.semaphore.release()
            
    def _get_venv_python(self, venv_path: Path) -> str:
        """
        Get the Python executable from a virtual environment.
        
        Args:
            venv_path: Path to the virtual environment
        
        Returns:
            Path to the Python executable
        """
        # OS-specific python executable detection
        if sys.platform == 'win32':
            return str(venv_path / 'Scripts' / 'python.exe')
        else:
            return str(venv_path / 'bin' / 'python')

    def _build_command(
        self,
        python_executable: str,
        entry_point: str,
        execution_context: Dict[str, Any]
    ) -> List[str]:
        
        """Build command with proper escaping and validation."""
        cmd = [python_executable, entry_point]
        
        if execution_context.get('method'):
            cmd.extend(['--method', execution_context['method']])
        
        if execution_context.get('args'):
            cmd.extend(str(arg) for arg in execution_context['args'])
            
        if execution_context.get('kwargs'):
            for key, value in execution_context['kwargs'].items():
                cmd.extend([f"{key}", str(value)])
                
        return cmd

    async def _monitor_process_resources(
                                        self,
                                        process: asyncio.subprocess.Process) -> None:
        """Monitor process resources and enforce limits."""
        try:
            while True:
                # Get process information using psutil
                proc = psutil.Process(process.pid)
                memory_mb = proc.memory_info().rss / 1024 / 1024
                
                # Update peak memory usage
                self.metrics.peak_memory_usage = max(
                    self.metrics.peak_memory_usage,
                    memory_mb
                )
                
                # Check memory limit
                if memory_mb > self.memory_limit:
                    process.kill()
                    raise MemoryError(
                        f"Process exceeded memory limit of {self.memory_limit}MB"
                    )
                
                await asyncio.sleep(1)  # Check every second
                
                # Exit if process has finished
                if process.returncode is not None:
                    break
                    
        except psutil.NoSuchProcess:
            # Process has already terminated
            pass
        except Exception as e:
            self.logger.error(f"Error monitoring process resources: {e}")
            raise

    async def _safe_package_execution(
        self,
        python_executable: str,
        entry_point: str,
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute package with comprehensive error handling and configurable output mode.
        """
        start_time = time.time()
        
        try:
            # Construct command
            cmd = self._build_command(python_executable, entry_point, execution_context)
            
            # Execute subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=self.buffer_size  # Limit buffer size
            )
            
            # Start resource monitoring
            monitor_task = asyncio.create_task(
                self._monitor_process_resources(process)
            )
            
            # Handle output streams
            stdout_task = asyncio.create_task(
                self._handle_stream(process.stdout, "stdout")
            )
            stderr_task = asyncio.create_task(
                self._handle_stream(process.stderr, "stderr")
            )
            
            # Wait for completion
            try:
                await asyncio.gather(stdout_task, stderr_task)
                await process.wait()
            finally:
                monitor_task.cancel()
            
            # Check execution status
            if process.returncode != 0:
                raise RuntimeError(
                    f"Execution failed with return code {process.returncode}"
                )
            
            execution_time = time.time() - start_time
            
            # Return results
            return {
                'stdout': '\n'.join(self.stream_handler.stdout_buffer),
                'stderr': '\n'.join(self.stream_handler.stderr_buffer),
                'return_code': process.returncode,
                'execution_time': execution_time,
                'peak_memory_mb': self.metrics.peak_memory_usage
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_metrics(False, execution_time)
            self.logger.error(f"Execution error: {e}")
            raise

    async def _handle_stream(
        self,
        stream: asyncio.StreamReader,
        stream_type: str
    ) -> None:
        """Handle output stream based on communication mode."""
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                line_decoded = line.decode().rstrip()
                await self.stream_handler.handle_output(line_decoded, stream_type)
        except Exception as e:
            self.logger.error(f"Error handling {stream_type} stream: {e}")
            raise

    def _update_metrics(self, success: bool, execution_time: float) -> None:
        """Update execution metrics."""
        self.metrics.total_executions += 1
        self.metrics.last_execution_time = execution_time
        
        if success:
            self.metrics.successful_executions += 1
        else:
            self.metrics.failed_executions += 1
            
        # Update rolling average
        prev_avg = self.metrics.average_execution_time
        self.metrics.average_execution_time = (
            (prev_avg * (self.metrics.total_executions - 1) + execution_time)
            / self.metrics.total_executions
        )

    def get_metrics(self) -> ExecutionMetrics:
        """Get current execution metrics."""
        return self.metrics

# Example usage demonstrating integration

# async def main():
#     # Example of how to use the PackageExecutor
#     executor = PackageExecutor()
    
#     # Create a mock handler (in real scenario, this would be from your system)
#     mock_handler = MockActorHandler(
#         name='example_actor',
#         venv_path=Path('/path/to/venv'),
#         entry_point='/path/to/entry_script.py'
#     )
    
#     try:
#         # Execute a package method
#         result = await executor.execute_package(
#             handler=mock_handler,
#             method='process_data',
#             args=['arg1', 'arg2'],
#             kwargs={'key': 'value'}
#         )
        
#         print("Execution Result:", result)
        
#         # Get performance metrics
#         metrics = executor.get_execution_metrics()
#         print("Execution Metrics:", metrics)
        
#     except Exception as e:
#         print(f"Execution failed: {e}")

# if __name__ == '__main__':
#     asyncio.run(main())