import os
import sys
import asyncio
import shutil
import logging
from typing import Optional, List, Union, Dict
from pathlib import Path
import platform
import subprocess
# import pkg_resources
from packaging.utils import parse_wheel_filename
import venv

class WheelInstaller:
    def __init__(self, 
                 logger: Optional[logging.Logger] = None, 
                 base_venv_dir: str = os.path.join(os.getcwd(), 'venvs')):
        """
        Initialize WheelInstaller with cross-platform support.
        
        Args:
            logger: Optional custom logger. If not provided, creates a default logger.
            base_venv_dir: Base directory for creating isolated virtual environments.
        """
        self.base_venv_dir = base_venv_dir
        os.makedirs(self.base_venv_dir, exist_ok=True)
        
        self.logger = logger or self._setup_logger()
        
        # OS-specific path configurations
        self.path_configs: Dict[str, Dict[str, str]] = {
            'Windows': {
                'bin_dir': 'Scripts',
                'activation_script': 'activate.bat',
                'path_separator': ';'
            },
            'Darwin': {
                'bin_dir': 'bin',
                'activation_script': 'activate',
                'path_separator': ':'
            },
            'Linux': {
                'bin_dir': 'bin',
                'activation_script': 'activate',
                'path_separator': ':'
            }
        }

        # Detect operating system
        self.os_type = self._detect_os()
        
    def _setup_logger(self) -> logging.Logger:
        """
        Set up a default logger with informative formatting.
        
        Returns:
            Configured logging.Logger instance
        """
        logger = logging.getLogger('WheelInstaller')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _detect_os(self) -> str:
        """
        Detect the current operating system.
        
        Returns:
            String representing the OS type
        """
        os_name = platform.system()
        print(f"The os name is : {os_name}")
        if os_name not in self.path_configs:
            raise OSError(f"Unsupported operating system: {os_name}")
        return os_name

    def _get_os_specific_commands(self) -> Dict[str, List[str]]:
        """
        Generate OS-specific commands for package management and activation.
        
        Returns:
            Dictionary of OS-specific commands
        """
        return {
            'Windows': {
                'pip': ['pip.exe'],
                'python': ['python.exe'],
                'activate': ['.\\Scripts\\activate.bat']
            },
            'Darwin': {
                'pip': ['pip3'],
                'python': ['python3'],
                'activate': ['source', './bin/activate']
            },
            'Linux': {
                'pip': ['pip3'],
                'python': ['python3'],
                'activate': ['source', './bin/activate']
            }
        }

    async def install_wheel(self, wheel_path: Union[str, Path], force_reinstall: bool = False) -> Path:
        """
        Comprehensive wheel installation with cross-platform support.
        
        Args:
            wheel_path: Path to the .whl file
            force_reinstall: Force reinstallation even if package exists
        
        Returns:
            Path to the virtual environment where package was installed
        """
        # Validate wheel file
        if not await self.validate_wheel_file(wheel_path):
            raise ValueError("Invalid wheel file")
        
        path = Path(wheel_path)
        package_name = path.stem.split('-')[0]
        
        # Create isolated virtual environment
        self.logger.info("Now creating venv")
        venv_path = await self.create_venv(package_name)
        
        # Get OS-specific configurations
        os_config = self.path_configs[self.os_type]
        bin_dir = venv_path / os_config['bin_dir']
        
        # Determine pip executable in the virtual environment
        pip_executable = bin_dir / ('pip.exe' if self.os_type == 'Windows' else 'pip')
        
        # Prepare installation command with OS-specific considerations
        install_cmd = [
            str(pip_executable), 
            'install', 
            str(path),
            '--no-cache-dir',]
        
            # '--no-deps' if not force_reinstall else '--force-reinstall']
        
        # Install wheel
        self.logger.info("Now Installing the package......")
        await self.run_command(install_cmd)
        
        # Ensure PATH is always set
        self.logger.info("Now adding path and finding the Entry point...")
        tasks = [self.ensure_path(venv_path) ,self._find_package_entry_point(venv_path)]
        try :
            _ , executable_path = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            raise e 
        
        print(f"The executable file path is : {executable_path}")
        print(f"The venv path : {venv_path}")
        if not executable_path:
            raise ValueError("Not able to find any Executable in the virtual environment")
        
        self.logger.info(f"Successfully installed {package_name} in {venv_path}")
        
        return venv_path ,executable_path

    async def ensure_path(self, venv_path: Path) -> None:
        """
        Ensure the virtual environment's bin/Scripts directory is in PATH.
        Handles path modification for different operating systems.
        
        Args:
            venv_path: Path to the virtual environment
        """
        # Get OS-specific configurations
        os_config = self.path_configs[self.os_type]
        bin_dir = venv_path / os_config['bin_dir']
        
        # Convert to string and normalize path
        bin_dir_str = str(bin_dir.resolve())
        
        # Add to PATH if not already present
        current_path = os.environ.get('PATH', '')
        path_separator = os.pathsep
        
        if bin_dir_str not in current_path.split(path_separator):
            # Modify PATH
            os.environ['PATH'] = f"{bin_dir_str}{path_separator}{current_path}"
            self.logger.info(f"Added {bin_dir_str} to PATH")
        
        # Additional OS-specific PATH handling
        if self.os_type in ['Darwin', 'Linux']:
            # For Unix-like systems, also modify shell environment
            try:
                # Use source command to modify shell environment
                await self.run_command([
                    'source', 
                    str(bin_dir / os_config['activation_script'])
                ])
            except Exception as e:
                self.logger.warning(f"Could not source activation script: {e}")

    async def run_command(self, command: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """
        Cross-platform asynchronous command execution.
        
        Args:
            command: List of command and arguments
            cwd: Optional working directory
        
        Returns:
            subprocess.CompletedProcess object
        """
        try:
            # Windows requires special handling for shell commands
            if self.os_type == 'Windows':
                process = await asyncio.create_subprocess_shell(
                    ' '.join(command),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd
                )
            else:
                # Unix-like systems
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd
                )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                self.logger.error(f"Command failed: {' '.join(command)}")
                self.logger.error(f"STDERR: {stderr.decode().strip()}")
                raise subprocess.CalledProcessError(
                    process.returncode, 
                    command, 
                    stdout, 
                    stderr
                )
            
            return subprocess.CompletedProcess(
                args=command, 
                returncode=process.returncode, 
                stdout=stdout, 
                stderr=stderr
            )
        except Exception as e:
            self.logger.error(f"Error executing command {command}: {e}")
            raise
        
    async def validate_wheel_file(self, wheel_path: Union[str, Path]) -> bool:
        """
        Validate that the provided path is a valid .whl file.
        
        Args:
            wheel_path: Path to the .whl file
        
        Returns:
            Boolean indicating whether the file is a valid wheel
        """
        path = Path(wheel_path)
        
        if not path.exists():
            self.logger.error(f"Wheel file does not exist: {path}")
            return False
        
        if not path.suffix == '.whl':
            self.logger.error(f"Invalid file type. Must be a .whl file: {path}")
            return False
        
        try:
            parse_wheel_filename(path.name)
            return True
        except Exception as e:
            self.logger.error(f"Invalid wheel filename: {e}")
            return False

    async def create_venv(self, package_name: str) -> Path:
        """
        Create an isolated virtual environment for package installation.
        
        Args:
            package_name: Base name for the virtual environment
        
        Returns:
            Path to the created virtual environment
        """
        venv_path = Path(self.base_venv_dir) / f"{package_name}_venv"
        try:
            venv.create(venv_path, with_pip=True)
            return venv_path
        except Exception as e:
            self.logger.error(f"Failed to create virtual environment: {e}")
            raise

    async def cleanup_old_venvs(self, max_keep: int = 5, venv_name : Union[str ,Path] = None) -> None:
        """
        Clean up old virtual environments to prevent disk space accumulation.
        
        Args:
            max_keep: Maximum number of virtual environments to keep
            
        """
        
        if venv_name:
            full_path = os.path.join(self.base_venv_dir, venv_name)
            self.logger.info(f"Removing the virtual environment at : {full_path}")
            try:
                shutil.rmtree(full_path)
                self.logger.info(f"Removed virtual environment: {venv_name}")
                return True
            except Exception as e:
                raise e 
            
        try:
            venvs = sorted(
                [d for d in os.listdir(self.base_venv_dir) if d.endswith('_venv')],
                key=lambda x: os.path.getctime(os.path.join(self.base_venv_dir, x))
            )
            
            for old_venv in venvs[:-max_keep]:
                full_path = os.path.join(self.base_venv_dir, old_venv)
                shutil.rmtree(full_path)
                self.logger.info(f"Removed old virtual environment: {old_venv}")
        except Exception as e:
            self.logger.warning(f"Error during venv cleanup: {e}")

    async def _find_package_entry_point(self, venv_path: Path) -> Optional[str]:
        """
        Find the main entry point for the installed package with improved precision.
        
        Args:
            venv_path: Path to the virtual environment
        
        Returns:
            Path to the entry point executable or None
        """
        # Determine OS-specific bin directory
        bin_dir = venv_path / ('Scripts' if sys.platform == 'win32' else 'bin')
        
        # Excluded file patterns
        excluded_files = {
            'python.exe', 'pythonw.exe',  # Python executables
            *{f for f in os.listdir(bin_dir) if f.startswith('pip')},  # Any pip executable
            'activate', 'activate.bat'  # Activation scripts
        }
        
        # Find executable files, prioritizing .exe on Windows
        if sys.platform == 'win32':
            # On Windows, look for .exe files first, excluding system/pip executables
            executable_scripts = [
                str(script) for script in bin_dir.glob('*.exe')
                if script.name not in excluded_files
            ]
        else:
            # On Unix-like systems, find executable files
            executable_scripts = [
                str(script) for script in bin_dir.glob('*')
                if os.access(script, os.X_OK) and 
                script.name not in excluded_files
            ]
        
        # Return the first matching executable, if any
        return executable_scripts[0] if executable_scripts else None