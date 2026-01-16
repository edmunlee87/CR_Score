"""
Configuration templates for CR_Score scorecards.

Provides ready-to-use templates for different complexity levels:
- Beginner: Simple, minimal configuration
- Intermediate: Full-featured with EDA and feature selection
- Advanced: Enterprise-grade with monitoring and observability
"""

from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent


def get_template_path(level: str, name: str) -> Path:
    """
    Get path to configuration template.
    
    Args:
        level: Template level ('beginner', 'intermediate', 'advanced')
        name: Template name
    
    Returns:
        Path to template file
    
    Example:
        >>> path = get_template_path('beginner', 'basic_scorecard.yml')
        >>> with open(path) as f:
        ...     config = yaml.safe_load(f)
    """
    template_path = TEMPLATE_DIR / level / name
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    
    return template_path


def list_templates(level: Optional[str] = None) -> Dict[str, List[str]]:
    """
    List available templates.
    
    Args:
        level: Filter by level (None = all)
    
    Returns:
        Dictionary of templates by level
    """
    import os
    from typing import Dict, List, Optional
    
    templates = {}
    
    levels = [level] if level else ['beginner', 'intermediate', 'advanced']
    
    for lvl in levels:
        level_dir = TEMPLATE_DIR / lvl
        if level_dir.exists():
            templates[lvl] = [
                f.name for f in level_dir.glob('*.yml')
            ]
    
    return templates


__all__ = [
    "TEMPLATE_DIR",
    "get_template_path",
    "list_templates",
]
