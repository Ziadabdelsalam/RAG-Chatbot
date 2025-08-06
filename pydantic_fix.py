"""
Pydantic compatibility fixes for Gradio/FastAPI integration
"""
from typing import Any
import pydantic
from pydantic import ConfigDict

# Global Pydantic configuration
class GlobalConfig:
    arbitrary_types_allowed = True
    validate_assignment = True
    use_enum_values = True

# Monkey patch for compatibility
def patch_pydantic():
    """Apply patches for Pydantic compatibility"""
    original_model_config = getattr(pydantic.BaseModel, 'model_config', None)
    
    if original_model_config is None:
        pydantic.BaseModel.model_config = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True,
            use_enum_values=True
        )

# Apply the patch
patch_pydantic()