"""General utility functions for the LiteLLM library."""

import json
import logging
from pathlib import Path
from typing import Union

from pydantic import BaseModel

logger = logging.getLogger(__name__)

def save_model_response(model: Union[BaseModel, str], output_path: Union[str, Path]) -> Path:
    """
    Save a Pydantic model or markdown string to a file.

    Args:
        model: The Pydantic model instance or markdown string to save.
        output_path: Path where the file will be saved (string or Path object). Extension will be added
                   automatically if not provided (.json for Pydantic models, .md for strings).

    Returns:
        Path: The absolute path to the saved file.

    Raises:
        OSError: If there is an issue creating directories or writing the file.
        IOError: If there is an issue writing to the file.
        ValueError: If the model type is not supported.
    """
    path = Path(output_path).resolve()
    
    # Add appropriate extension if not present
    if not path.suffix:
        if isinstance(model, BaseModel):
            path = path.with_suffix('.json')
        elif isinstance(model, str):
            path = path.with_suffix('.md')
    
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving data to: {path}")
        
        with open(path, 'w', encoding='utf-8') as f:
            if isinstance(model, BaseModel):
                json.dump(model.model_dump(exclude_none=True), f, indent=2, default=str)
            elif isinstance(model, str):
                f.write(model)
            else:
                raise ValueError(f"Unsupported model type: {type(model)}. Expected Pydantic model or string.")
            
        logger.info(f"✓ Successfully saved to {path}")
        return path
    except OSError as e:
        logger.error(f"✗ Error saving to {path}: {e}")
        raise
