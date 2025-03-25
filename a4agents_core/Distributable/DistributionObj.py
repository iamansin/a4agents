import os
import sys
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
import pkg_resources
from contextlib import contextmanager

# In the Setup_Script we need to change the entry_points for the function that has to be called right now "main".

SETUP_SCRIPT = """
from setuptools import setup, find_packages;

setup(
    name="{package_name}",
    version="{version}",
    packages=find_packages(exclude={exclude_files}),
    include_package_data=True,
    entry_points={{
        'console_scripts': [
            '{package_name}={package_name}.runobject:main',
        ],
    }},
    install_requires={requirements},
    python_requires='>=3.6',
)
"""

class DistributionError(Exception):
    """Custom exception for distribution-related errors."""
    pass

class DistributionObj:
    """
    A class to handle creation and packaging of agent containers for distribution
    in both Python and non-Python environments.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the DistributionObj class.
        
        Args:
            logger: Optional logger instance for logging. If None, creates a default logger.
        """
        self.logger = logger or self._setup_logger()
        self._temp_dirs: List[str] = []
    
    def _setup_logger(self) -> logging.Logger:
        """Set up and return a logger instance."""
        logger = logging.getLogger("DistributionObj")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    @contextmanager
    def _create_temp_dir(self):
        """
        Create a temporary directory and ensure it's cleaned up afterwards.
        
        Yields:
            str: Path to the temporary directory
        """
        import tempfile
        temp_dir = tempfile.mkdtemp()
        self._temp_dirs.append(temp_dir)
        try:
            yield temp_dir
        finally:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    self._temp_dirs.remove(temp_dir)
                except Exception as e:
                    self.logger.warning(f"Failed to remove temporary directory {temp_dir}: {e}")
    
    def __del__(self) -> None:
        """Clean up any remaining temporary directories when the object is destroyed."""
        for temp_dir in self._temp_dirs[:]:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    self._temp_dirs.remove(temp_dir)
                except Exception as e:
                    self.logger.warning(f"Failed to remove temporary directory {temp_dir}: {e}")
    
    def create_agent_container(self, 
                               agent_name: str, 
                               agent_config: Dict[str, any]) -> Dict[str, any]:
        """
        Creates an agent container object.
        
        Args:
            agent_name: Name of the agent
            agent_config: Configuration dictionary for the agent
            
        Returns:
            Dict[str, any]: The agent container object
            
        Raises:
            DistributionError: If agent creation fails
        """
        try:
            self.logger.info(f"Creating agent container for: {agent_name}")
            
            # Validate inputs
            if not isinstance(agent_name, str) or not agent_name.strip():
                raise ValueError("Agent name must be a non-empty string")
            
            if not isinstance(agent_config, dict) or not agent_config:
                raise ValueError("Agent config must be a non-empty dictionary")
            
            # Create a container object with metadata
            container = {
                "agent_name": agent_name,
                "config": agent_config,
                "created_at": self._get_timestamp(),
                "version": "1.0.0",  # Default version
                "metadata": {}
            }
            
            self.logger.info(f"Successfully created agent container for: {agent_name}")
            return container
            
        except Exception as e:
            error_msg = f"Failed to create agent container: {str(e)}"
            self.logger.error(error_msg)
            raise DistributionError(error_msg) from e
    
    def _get_timestamp(self) -> str:
        """Get the current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _verify_runobject_exists(self, directory_path: Union[str, Path]) -> Path:
        """
        Verify that runobject.py exists in the given directory.
        
        Args:
            directory_path: Path to the directory to check
            
        Returns:
            Path: Path to the runobject.py file
            
        Raises:
            DistributionError: If runobject.py doesn't exist
        """
        directory_path = Path(directory_path)
        runobject_path = directory_path / "runobject.py"
        
        if not runobject_path.exists():
            raise DistributionError(
                f"runobject.py not found in {directory_path}. "
                "This file is required as the main entry point."
            )
        
        return runobject_path
    
    def _generate_setup_py(self, 
                           directory_path: Path, 
                           package_name: str,
                           requirements_path: Optional[Path],
                           version :str,
                           exclude_files :List[str] = None) -> Path:
        """
        Generate a setup.py file for Python packaging.
        
        Args:
            directory_path: Path to the main directory
            package_name: Name of the package
            requirements_path: Path to requirements.txt file
            
        Returns:
            Path: Path to the generated setup.py file
        """
        requirements = []
        if requirements_path and requirements_path.exists():
            with open(requirements_path, 'r') as f:
                requirements = [line.strip() for line in f if line.strip() 
                               and not line.startswith('#')]
        
        setup_path = directory_path /"setup.py"
        
        with open(setup_path, 'w') as f:
            f.write(SETUP_SCRIPT.format(package_name = package_name, version = version, requirements = requirements, exclude_files = exclude_files))
        return setup_path
    
    def _check_package_installed(self, package_name: str) -> bool:
        """
        Check if a Python package is installed.
        
        Args:
            package_name: Name of the package to check
            
        Returns:
            bool: True if package is installed, False otherwise
        """
        try:
            pkg_resources.get_distribution(package_name)
            return True
        except pkg_resources.DistributionNotFound:
            return False
    
    def _install_required_packages(self, packages: List[str]) -> None:
        """
        Install required Python packages if they are not already installed.
        
        Args:
            packages: List of package names to install
            
        Raises:
            DistributionError: If package installation fails
        """
        packages_to_install = [pkg for pkg in packages if not self._check_package_installed(pkg)]
        
        if not packages_to_install:
            return
        
        self.logger.info(f"Installing required packages: {', '.join(packages_to_install)}")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", *packages_to_install
            ])
        except subprocess.CalledProcessError as e:
            raise DistributionError(f"Failed to install required packages: {e}")
    
    def pack4python(self, 
                    root_directory: Union[str, Path],
                    main_directory: Union[str, Path], 
                    requirements_path: Union[str, Path],
                    exclude_files : Optional[List[str]] = None,
                    version: str = "0.1.0",) -> str:
        """
        Package the agent for Python distribution as a wheel package.

        Args:
            root_directory: Path to the root directory (where setup.py is created)
            main_directory: Path to the main package directory (where __init__.py is created)
            requirements_path: Path to requirements.txt file (optional)
            version: Version of the package

        Returns:
            str: Path to the generated wheel file

        Raises:
            DistributionError: If packaging fails
        """
        try:
            # Convert to absolute paths and validate them
            root_directory = Path(root_directory).resolve()
            main_directory = Path(main_directory).resolve()

            if not root_directory.exists() or not root_directory.is_dir():
                raise DistributionError(f"Root directory does not exist: {root_directory}")

            if not main_directory.exists() or not main_directory.is_dir():
                raise DistributionError(f"Main directory does not exist: {main_directory}")

            self.logger.info(f"Packaging for Python distribution from root: {root_directory}")

            # Ensure __init__.py exists in the main package directory
            init_file = main_directory / "__init__.py"
            if not init_file.exists():
                with open(init_file, "w") as f:
                    pass  # Create an empty __init__.py file
                self.logger.info(f"Created __init__.py in: {main_directory}")

            # Ensure `runobject.py` exists in `main_directory`
            self._verify_runobject_exists(main_directory)

            # Validate requirements.txt path
            if requirements_path:
                requirements_path = Path(requirements_path).resolve()
                if not requirements_path.exists():
                    raise DistributionError(f"Requirements file not found: {requirements_path}")

            # Install required packaging tools
            self._install_required_packages(["wheel", "setuptools"])

            # Generate setup.py in `root_directory`
            package_name = main_directory.name
            self._generate_setup_py(root_directory, package_name, requirements_path, version, exclude_files)

            # Build wheel inside `root_directory`
            self.logger.info("Building wheel package...")
            original_dir = os.getcwd()

            try:
                os.chdir(root_directory)  # Change to root directory
                dist_dir = root_directory / "dist"  # Ensure dist is inside root
                dist_dir.mkdir(exist_ok=True)

                subprocess.check_call([
                    sys.executable, "-m", "pip", "wheel", 
                    "--no-deps", "--wheel-dir", str(dist_dir), "."
                ])

                # Find the generated wheel file
                wheel_files = list(dist_dir.glob("*.whl"))

                if not wheel_files:
                    raise DistributionError("Wheel file was not created")

                wheel_path = str(wheel_files[0].resolve())
                self.logger.info(f"Successfully created wheel package: {wheel_path}")
                return wheel_path

            finally:
                os.chdir(original_dir)

        except Exception as e:
            error_msg = f"Failed to package for Python distribution: {str(e)}"
            self.logger.error(error_msg)
            raise DistributionError(error_msg) from e
    
#     def pack4non_python(self, 
#                        main_directory: Union[str, Path], 
#                        requirements_path: Optional[Union[str, Path]] = None,
#                        target_platforms: Optional[List[str]] = None) -> str:
#         """
#         Package the agent for non-Python distribution using cx_Freeze.
        
#         Args:
#             main_directory: Path to the main directory containing the agent code
#             requirements_path: Path to requirements.txt file (optional)
#             target_platforms: List of target platforms (e.g., ['windows', 'linux'])
            
#         Returns:
#             str: Path to the directory containing the packaged executables
            
#         Raises:
#             DistributionError: If packaging fails
#         """
#         try:
#             main_directory = Path(main_directory).resolve()
#             self.logger.info(f"Packaging for non-Python distribution from directory: {main_directory}")
            
#             # Default target platform is the current platform
#             if not target_platforms:
#                 import platform
#                 system = platform.system().lower()
#                 if system == "darwin":
#                     system = "macos"
#                 target_platforms = [system]
            
#             # Validate directory exists
#             if not main_directory.exists() or not main_directory.is_dir():
#                 raise DistributionError(f"Main directory does not exist: {main_directory}")
            
#             # Check for runobject.py
#             runobject_path = self._verify_runobject_exists(main_directory)
            
#             # Install cx_Freeze
#             self._install_required_packages(["cx_Freeze"])
            
#             # Process requirements
#             requirements = []
#             if requirements_path:
#                 requirements_path = Path(requirements_path).resolve()
#                 if not requirements_path.exists():
#                     raise DistributionError(f"Requirements file not found: {requirements_path}")
#                 with open(requirements_path, 'r') as f:
#                     requirements = [line.strip() for line in f if line.strip() 
#                                    and not line.startswith('#')]
            
#             # Create cx_Freeze setup script
#             package_name = main_directory.name.replace("-", "_").lower()
            
#             with self._create_temp_dir() as temp_dir:
#                 setup_path = Path(temp_dir) / "cx_setup.py"
                
#                 with open(setup_path, 'w') as f:
#                     f.write(f"""
# import sys
# from cx_Freeze import setup, Executable

# build_exe_options = {{
#     "packages": [],
#     "excludes": [],
#     "include_files": []
# }}

# base = None
# if sys.platform == "win32":
#     base = "Console"

# setup(
#     name="{package_name}",
#     version="0.1.0",
#     description="{package_name} Executable",
#     options={{"build_exe": build_exe_options}},
#     executables=[Executable("{runobject_path}", base=base, target_name="{package_name}")]
# )
# """)
                
#                 # Install requirements
#                 if requirements:
#                     self.logger.info("Installing requirements...")
#                     try:
#                         subprocess.check_call([
#                             sys.executable, "-m", "pip", "install", *requirements
#                         ])
#                     except subprocess.CalledProcessError as e:
#                         raise DistributionError(f"Failed to install requirements: {e}")
                
#                 # Build executable
#                 self.logger.info(f"Building executable for platforms: {', '.join(target_platforms)}")
                
#                 build_dir = main_directory / "build"
#                 if build_dir.exists():
#                     shutil.rmtree(build_dir)
                
#                 try:
#                     subprocess.check_call([
#                         sys.executable, str(setup_path), "build"
#                     ])
                    
#                     # Move build directory to main_directory
#                     temp_build_dir = Path(temp_dir) / "build"
#                     if temp_build_dir.exists():
#                         shutil.copytree(temp_build_dir, build_dir)
                    
#                     self.logger.info(f"Successfully created executable package in: {build_dir}")
#                     return str(build_dir)
                    
#                 except subprocess.CalledProcessError as e:
#                     raise DistributionError(f"Failed to build executable: {e}")
                
#         except Exception as e:
#             error_msg = f"Failed to package for non-Python distribution: {str(e)}"
#             self.logger.error(error_msg)
#             raise DistributionError(error_msg) from e