from typing import Type, TypeVar, Dict, Any, Optional, Callable
from threading import Lock
from functools import wraps
from datetime import datetime, timezone
import logging

# Type variable for generic class types
T = TypeVar('T')

class SingletonError(Exception):
    """Custom exception for Singleton-related errors."""
    pass

class SingletonRegistry:
    """
    Thread-safe registry for managing Singleton instances and their metadata.
    """
    _instances: Dict[Type, Any] = {}
    _metadata: Dict[Type, Dict[str, Any]] = {}
    _locks: Dict[Type, Lock] = {}
    _registry_lock = Lock()

    @classmethod
    def get_instance(cls, class_type: Type[T]) -> Optional[T]:
        """Get the singleton instance of a class if it exists."""
        return cls._instances.get(class_type,None)

    @classmethod
    def register_instance(cls, class_type: Type[T], instance: T) -> None:
        """Register a new singleton instance with metadata."""
        with cls._registry_lock:
            if class_type not in cls._metadata:
                cls._metadata[class_type] = {
                    'created_at': datetime.now(timezone.utc),
                    'created_by': _get_current_user(),
                    'access_count': 0,
                    'last_accessed': None
                }
            cls._instances[class_type] = instance

    @classmethod
    def get_lock(cls, class_type: Type) -> Lock:
        """Get or create a lock for a specific class type."""
        with cls._registry_lock:
            if class_type not in cls._locks:
                cls._locks[class_type] = Lock()
            return cls._locks[class_type]

    @classmethod
    def update_access_metadata(cls, class_type: Type) -> None:
        """Update access metadata for a singleton instance."""
        if class_type in cls._metadata:
            cls._metadata[class_type]['access_count'] += 1
            cls._metadata[class_type]['last_accessed'] = datetime.now(timezone.utc)

    @classmethod
    def get_metadata(cls, class_type: Type) -> Dict[str, Any]:
        """Get metadata for a singleton instance."""
        return cls._metadata.get(class_type, {})

def _get_current_user() -> str:
    """Get the current user's identity."""
    try:
        import getpass
        return getpass.getuser()
    except Exception:
        return "unknown"

def Singleton(
    cls: Optional[Type[T]] = None,
    *,
    strict: bool = True,
    thread_safe: bool = True,
    debug: bool = False
) -> Callable[[Type[T]], Type[T]]:
    """
    A production-ready Singleton decorator that ensures only one instance of a class exists.

    Args:
        cls: The class to be decorated
        strict: If True, raises error on attempted recreation. If False, returns existing instance
        thread_safe: Enable/disable thread safety mechanisms
        debug: Enable/disable debug logging

    Returns:
        The decorated class

    Raises:
        SingletonError: When attempting to create multiple instances in strict mode
        TypeError: When decorator is misused
    """
    def _setup_logging() -> logging.Logger:
        """Setup logging for the Singleton decorator."""
        logger = logging.getLogger(f'Singleton.{cls.__ne__}')
        if debug and not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
        return logger

    def _singleton_wrapper(cls: Type[T]) -> Type[T]:
        # Validate class
        if not isinstance(cls, type):
            raise TypeError(f"Singleton decorator must be applied to a class, not {type(cls)}")

        # Setup logging
        # logger = _setup_logging()

        # Store original methods
        original_new = cls.__new__
        original_init = cls.__init__

        @wraps(cls.__new__)
        def __new__(cls_: Type[T], *args: Any, **kwargs: Any) -> T:
            # Get or create lock for this class
            lock = SingletonRegistry.get_lock(cls_) if thread_safe else None

            try:
                # Check if instance exists
                instance = SingletonRegistry.get_instance(cls_)
                if instance is not None:
                    if strict:
                        # Get metadata for error message
                        metadata = SingletonRegistry.get_metadata(cls_)
                        raise SingletonError(
                            f"Attempted to create new instance of {cls_.__name__} but one already exists.\n"
                            f"Existing instance created at {metadata.get('created_at')} "
                            f"by {metadata.get('created_by')}"
                        )
                    # logger.debug(f"Returning existing instance of {cls_.__name__}")
                    SingletonRegistry.update_access_metadata(cls_)
                    return instance

                # Create new instance with thread safety if enabled
                if thread_safe:
                    with lock:  # type: ignore
                        # Double-check pattern
                        if SingletonRegistry.get_instance(cls_) is None:
                            if hasattr(original_new, '__call__'):
                                instance = original_new(cls_)
                            else:
                                instance = super(cls_, cls_).__new__(cls_)
                            SingletonRegistry.register_instance(cls_, instance)
                else:
                    if hasattr(original_new, '__call__'):
                        instance = original_new(cls_)
                    else:
                        instance = super(cls_, cls_).__new__(cls_)
                    SingletonRegistry.register_instance(cls_, instance)

                # logger.debug(f"Created new instance of {cls_.__name__}")
                return instance

            except Exception as e:
                if not isinstance(e, SingletonError):
                    print(f"Error creating singleton instance: {str(e)}")
                raise

        @wraps(cls.__init__)
        def __init__(self: T, *args: Any, **kwargs: Any) -> None:
            if not hasattr(self, '_initialized'):
                original_init(self, *args, **kwargs)
                self._initialized = True
                # logger.debug(f"Initialized {cls.__name__} instance")

        # Replace class methods
        cls.__new__ = __new__  # type: ignore
        cls.__init__ = __init__  # type: ignore

        # Add utility methods to the class
        @classmethod
        def get_instance(cls_: Type[T]) -> Optional[T]:
            """Get the singleton instance if it exists."""
            return SingletonRegistry.get_instance(cls_)

        @classmethod
        def get_metadata(cls_: Type[T]) -> Dict[str, Any]:
            """Get metadata about the singleton instance."""
            return SingletonRegistry.get_metadata(cls_)

        cls.get_instance = get_instance
        cls.get_metadata = get_metadata

        return cls

    # Handle both @Singleton and @Singleton() syntax
    if cls is None:
        return _singleton_wrapper
    return _singleton_wrapper(cls)
