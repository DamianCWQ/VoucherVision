"""
Model Cache Manager for VoucherVision
Ensures models are loaded once and reused across all instances
"""
import threading
import torch
import gc

class ModelCache:
    """Singleton class to cache loaded models and prevent repeated loading"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelCache, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._models = {}
        self._model_locks = {}
        
    def get_model(self, model_key, loader_func, *args, **kwargs):
        """
        Get a cached model or load it if not cached.
        
        Args:
            model_key: Unique identifier for this model
            loader_func: Function to call to load the model if not cached
            *args, **kwargs: Arguments to pass to loader_func
            
        Returns:
            The cached or newly loaded model
        """
        # Create a lock for this model if it doesn't exist
        if model_key not in self._model_locks:
            with self._lock:
                if model_key not in self._model_locks:
                    self._model_locks[model_key] = threading.Lock()
        
        # Acquire the model-specific lock
        with self._model_locks[model_key]:
            if model_key not in self._models:
                print(f"Loading model: {model_key} (first time)")
                self._models[model_key] = loader_func(*args, **kwargs)
            else:
                print(f"Using cached model: {model_key}")
            
            return self._models[model_key]
    
    def clear_cache(self):
        """Clear all cached models and free GPU memory"""
        with self._lock:
            for model_key in list(self._models.keys()):
                del self._models[model_key]
            self._models.clear()
            self._model_locks.clear()
            
            # Free GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("Model cache cleared")
    
    def remove_model(self, model_key):
        """Remove a specific model from cache"""
        with self._lock:
            if model_key in self._models:
                del self._models[model_key]
                if model_key in self._model_locks:
                    del self._model_locks[model_key]
                
                # Free GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                print(f"Model removed from cache: {model_key}")
