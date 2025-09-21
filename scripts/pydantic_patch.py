"""
Patch for Pydantic to handle Starlette Request objects in Gradio 4.x
"""
import sys
import os
from typing import Any, Callable

# Set environment variables to modify Pydantic behavior
os.environ["PYDANTIC_WARN_ABOUT_MISSING_FIELD_INSTANCE_ATTRIBUTE"] = "0"
os.environ["PYDANTIC_SCHEMAGENERATION_ERROR"] = "0"

# Try to patch Pydantic v2 for Starlette Request
try:
    from pydantic.config import ConfigDict
    from pydantic import create_model
    from pydantic.json_schema import JsonSchemaGenerator
    from pydantic._internal._model_construction import ModelMetaclass
    
    # Create a model with arbitrary_types_allowed=True as global config
    ARBITRARY_TYPES_CONFIG = ConfigDict(arbitrary_types_allowed=True)
    
    # Try to patch JsonSchemaGenerator to handle Starlette Request
    original_generate_schema = JsonSchemaGenerator.generate_schema
    
    def patched_generate_schema(self, obj: Any) -> dict[str, Any]:
        if hasattr(obj, "__module__") and "starlette" in obj.__module__:
            # Return a simple schema for starlette objects
            return {"type": "object"}
        return original_generate_schema(self, obj)
    
    # Apply monkey patch
    JsonSchemaGenerator.generate_schema = patched_generate_schema
    
    print("Pydantic v2 patch for Starlette objects applied")
except (ImportError, AttributeError):
    try:
        # Fallback for older Pydantic versions
        from pydantic import BaseConfig
        BaseConfig.arbitrary_types_allowed = True
        print("Pydantic v1 patch applied (arbitrary_types_allowed=True)")
    except (ImportError, AttributeError):
        print("Could not patch Pydantic, but we'll continue anyway")

# Try to modify FastAPI/Starlette types handler
try:
    from fastapi import FastAPI
    from fastapi.applications import AppType
    from starlette.applications import Starlette
    
    # Patch the get_openapi method to ignore certain types
    if hasattr(FastAPI, "openapi"):
        original_openapi = FastAPI.openapi
        
        def patched_openapi(self):
            try:
                return original_openapi(self)
            except Exception as e:
                # If schema generation fails, return a minimal schema
                print(f"OpenAPI schema generation failed, returning minimal schema: {e}")
                return {"openapi": "3.0.2", "info": {"title": "API", "version": "0.1.0"}, "paths": {}}
        
        FastAPI.openapi = patched_openapi
        print("FastAPI openapi patch applied")
except (ImportError, AttributeError):
    pass 