"""
Template Engine for A.R.T.H
Loads and renders Jinja2-based attack templates from YAML files
Supports prompt converters for encoding attacks
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from jinja2 import Template, Environment, BaseLoader

from .converters import CONVERTERS, convert_prompt


@dataclass
class AttackTemplate:
    """Represents a loaded attack template"""
    name: str
    description: str
    category: str  # Derived from folder structure
    subcategory: Optional[str]
    parameters: List[str]
    template: str
    source: Optional[str] = None
    authors: Optional[List[str]] = None
    file_path: str = ""
    converter: Optional[str] = None  # Optional converter to apply to prompt

    def render(self, **kwargs) -> str:
        """Render the template with provided parameters"""
        # Apply converter to prompt if specified
        if self.converter and 'prompt' in kwargs:
            kwargs['prompt'] = convert_prompt(kwargs['prompt'], self.converter)
            kwargs['original_prompt'] = kwargs.get('original_prompt', kwargs['prompt'])

        env = Environment(loader=BaseLoader())
        template = env.from_string(self.template)
        return template.render(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "subcategory": self.subcategory,
            "parameters": self.parameters,
            "source": self.source,
            "authors": self.authors,
            "file_path": self.file_path
        }


class TemplateEngine:
    """
    Loads and manages attack templates from YAML files
    """

    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = Path(templates_dir)
        self.templates: Dict[str, AttackTemplate] = {}
        self.categories: Dict[str, List[str]] = {}  # category -> template names

    def load_all_templates(self) -> int:
        """
        Load all templates from the templates directory
        Returns the number of templates loaded
        """
        count = 0

        # Walk through all subdirectories
        for yaml_file in self.templates_dir.rglob("*.yaml"):
            try:
                template = self._load_template_file(yaml_file)
                if template:
                    self.templates[template.name] = template

                    # Track by category
                    if template.category not in self.categories:
                        self.categories[template.category] = []
                    self.categories[template.category].append(template.name)

                    count += 1
            except Exception as e:
                print(f"Warning: Failed to load {yaml_file}: {e}")

        return count

    def _load_template_file(self, file_path: Path) -> Optional[AttackTemplate]:
        """Load a single template from a YAML file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not data:
            return None

        # Determine category from path
        rel_path = file_path.relative_to(self.templates_dir)
        parts = list(rel_path.parts)

        category = parts[0] if len(parts) > 1 else "uncategorized"
        subcategory = parts[1] if len(parts) > 2 else None

        # Handle PyRIT format
        name = data.get('name', file_path.stem)
        description = data.get('description', '')
        parameters = data.get('parameters', ['prompt'])
        template_value = data.get('value', '')
        source = data.get('source', None)
        authors = data.get('authors', None)
        converter = data.get('converter', None)  # Optional converter

        return AttackTemplate(
            name=name,
            description=description,
            category=category,
            subcategory=subcategory,
            parameters=parameters,
            template=template_value,
            source=source,
            authors=authors,
            file_path=str(file_path),
            converter=converter
        )

    def get_template(self, name: str) -> Optional[AttackTemplate]:
        """Get a template by name"""
        return self.templates.get(name)

    def get_templates_by_category(self, category: str) -> List[AttackTemplate]:
        """Get all templates in a category"""
        names = self.categories.get(category, [])
        return [self.templates[name] for name in names]

    def get_all_templates(self) -> List[AttackTemplate]:
        """Get all loaded templates"""
        return list(self.templates.values())

    def get_categories(self) -> List[str]:
        """Get all available categories"""
        return list(self.categories.keys())

    def render_template(self, name: str, **kwargs) -> Optional[str]:
        """Render a template by name with provided parameters"""
        template = self.get_template(name)
        if template:
            return template.render(**kwargs)
        return None

    def render_all_with_prompt(self, prompt: str) -> Dict[str, str]:
        """
        Render all templates with a given prompt
        Returns dict of template_name -> rendered_prompt
        """
        results = {}
        for name, template in self.templates.items():
            try:
                results[name] = template.render(prompt=prompt)
            except Exception as e:
                print(f"Warning: Failed to render {name}: {e}")
        return results

    def list_templates(self) -> None:
        """Print a summary of all loaded templates"""
        print(f"\n{'='*60}")
        print(f"Loaded {len(self.templates)} templates in {len(self.categories)} categories")
        print(f"{'='*60}\n")

        for category, names in sorted(self.categories.items()):
            print(f"\n[{category.upper()}] ({len(names)} templates)")
            print("-" * 40)
            for name in sorted(names):
                template = self.templates[name]
                desc = template.description[:50] + "..." if len(template.description) > 50 else template.description
                print(f"  - {name}: {desc}")


# CLI utility for listing templates
if __name__ == "__main__":
    import sys

    templates_dir = sys.argv[1] if len(sys.argv) > 1 else "templates"

    engine = TemplateEngine(templates_dir)
    count = engine.load_all_templates()

    print(f"Loaded {count} templates")
    engine.list_templates()
