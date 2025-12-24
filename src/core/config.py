import yaml
from pathlib import Path
from typing import Any, Dict

def load_project_config(project_name: str) -> Dict[str, Any]:
    # 1. Load Global Settings
    with open("config/settings.yaml", "r") as f:
        base_config = yaml.safe_load(f)
    
    # 2. Find Project File
    project_path = Path("config/projects") / f"{project_name}.yaml"
    if not project_path.exists():
        # Try adding .yaml if missing
        project_path = Path("config/projects") / project_name
        if not project_path.exists():
            raise FileNotFoundError(f"Project configuration '{project_name}' not found in config/projects/")
    
    # 3. Load Project Settings
    with open(project_path, "r") as f:
        project_config = yaml.safe_load(f)
    
    # 4. Merge: Project settings override defaults
    # We inject project specific settings into the main config structure
    final_config = base_config.copy()
    final_config.update(project_config)
    
    # Ensure nested 'search' keys exist
    if 'search' not in final_config:
        final_config['search'] = {}
        
    # Apply defaults if project didn't specify them
    for key, value in final_config['defaults'].items():
        if key not in final_config['search']:
            final_config['search'][key] = value
            
    return final_config
