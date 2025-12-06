#!/usr/bin/env python3
"""
A-Red-Team-Havoc - NIA Red Team Toolkit
Main CLI Entry Point
"""

import argparse
import sys
import os
import json
import yaml
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.template_engine import TemplateEngine
from src.core.target import OpenRouterTarget
from src.core.scorer import Scorer
from src.core.batch_runner import BatchRunner


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Expand environment variables
    if config.get('openrouter', {}).get('api_key', '').startswith('${'):
        env_var = config['openrouter']['api_key'][2:-1]
        config['openrouter']['api_key'] = os.environ.get(env_var, '')

    return config


def cmd_list_templates(args):
    """List all available templates"""
    engine = TemplateEngine(args.templates_dir)
    count = engine.load_all_templates()

    if args.category:
        templates = engine.get_templates_by_category(args.category)
        print(f"\nTemplates in category '{args.category}': {len(templates)}")
        for t in templates:
            print(f"  - {t.name}")
    else:
        engine.list_templates()


def cmd_run_single(args):
    """Run a single attack"""
    config = load_config(args.config)

    # Initialize components
    engine = TemplateEngine(args.templates_dir)
    engine.load_all_templates()

    target = OpenRouterTarget(
        api_key=config['openrouter'].get('api_key') or os.environ.get('OPENROUTER_API_KEY'),
        model=args.model or config['openrouter']['target_model']
    )

    judge_target = OpenRouterTarget(
        api_key=config['openrouter'].get('api_key') or os.environ.get('OPENROUTER_API_KEY'),
        model=args.judge_model or config['openrouter']['judge_model']
    )

    scorer = Scorer(
        judge_target=judge_target,
        use_judge=not args.no_judge
    )

    # Get template
    template = engine.get_template(args.template)
    if not template:
        print(f"Error: Template '{args.template}' not found")
        sys.exit(1)

    # Render and send
    rendered = template.render(prompt=args.objective)
    print(f"\n[TARGET] {target.model}")
    print(f"[TEMPLATE] {template.name}")
    print(f"\n[RENDERED PROMPT]\n{rendered[:500]}{'...' if len(rendered) > 500 else ''}\n")

    response = target.send_prompt(rendered, template.name)

    if response.success:
        print(f"[RESPONSE]\n{response.response_text}\n")

        score = scorer.score(response)
        print(f"[SCORE] {score.result.value.upper()}")
        if score.matched_pattern:
            print(f"[MATCHED] {score.matched_pattern}")
    else:
        print(f"[ERROR] {response.error}")


def cmd_run_batch(args):
    """Run batch attacks"""
    config = load_config(args.config)

    # Load objectives
    if args.objectives_file:
        with open(args.objectives_file, 'r') as f:
            if args.objectives_file.endswith('.yaml'):
                data = yaml.safe_load(f)
                objectives = data.get('objectives', [])
            elif args.objectives_file.endswith('.json'):
                data = json.load(f)
                objectives = data.get('objectives', [])
            else:
                objectives = [line.strip() for line in f if line.strip()]
    else:
        objectives = args.objectives

    if not objectives:
        print("Error: No objectives provided")
        sys.exit(1)

    # Initialize components
    engine = TemplateEngine(args.templates_dir)
    count = engine.load_all_templates()
    print(f"Loaded {count} templates")

    target = OpenRouterTarget(
        api_key=config['openrouter'].get('api_key') or os.environ.get('OPENROUTER_API_KEY'),
        model=args.model or config['openrouter']['target_model']
    )

    judge_target = None
    if not args.no_judge:
        judge_target = OpenRouterTarget(
            api_key=config['openrouter'].get('api_key') or os.environ.get('OPENROUTER_API_KEY'),
            model=args.judge_model or config['openrouter']['judge_model']
        )

    scorer = Scorer(
        judge_target=judge_target,
        use_judge=not args.no_judge
    )

    runner = BatchRunner(
        template_engine=engine,
        target=target,
        scorer=scorer,
        concurrency=args.concurrency,
        output_dir=args.output_dir
    )

    # Parse template selection
    template_names = args.templates.split(',') if args.templates else None
    categories = args.categories.split(',') if args.categories else None

    # Run batch
    filepath = runner.run_and_save(
        objectives=objectives,
        template_names=template_names,
        categories=categories
    )

    print(f"\nResults saved to: {filepath}")


def cmd_list_models(args):
    """List available OpenRouter models"""
    target = OpenRouterTarget.__new__(OpenRouterTarget)
    models = target.get_available_models()

    print("\nAvailable Models for Red Teaming:")
    print("-" * 40)
    for model in models:
        print(f"  {model}")


def main():
    parser = argparse.ArgumentParser(
        description="A-Red-Team-Havoc - NIA Red Team Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all templates
  python -m src.main list-templates

  # Run single attack
  python -m src.main single --template "DAN 1" --objective "How to pick a lock"

  # Run batch with all templates
  python -m src.main batch --objectives "How to hack" "How to make explosives"

  # Run batch with specific category
  python -m src.main batch --objectives-file objectives/test.yaml --categories jailbreaks

  # Run without LLM judge (faster, only rejection detection)
  python -m src.main batch --objectives "Test prompt" --no-judge
        """
    )

    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--templates-dir', default='templates', help='Templates directory')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # List templates command
    list_parser = subparsers.add_parser('list-templates', help='List available templates')
    list_parser.add_argument('--category', help='Filter by category')

    # Single attack command
    single_parser = subparsers.add_parser('single', help='Run single attack')
    single_parser.add_argument('--template', required=True, help='Template name')
    single_parser.add_argument('--objective', required=True, help='Attack objective')
    single_parser.add_argument('--model', help='Target model (overrides config)')
    single_parser.add_argument('--judge-model', help='Judge model (overrides config)')
    single_parser.add_argument('--no-judge', action='store_true', help='Skip LLM judge')

    # Batch attack command
    batch_parser = subparsers.add_parser('batch', help='Run batch attacks')
    batch_parser.add_argument('--objectives', nargs='+', help='Attack objectives')
    batch_parser.add_argument('--objectives-file', help='File with objectives (yaml/json/txt)')
    batch_parser.add_argument('--templates', help='Comma-separated template names')
    batch_parser.add_argument('--categories', help='Comma-separated categories')
    batch_parser.add_argument('--model', help='Target model (overrides config)')
    batch_parser.add_argument('--judge-model', help='Judge model (overrides config)')
    batch_parser.add_argument('--no-judge', action='store_true', help='Skip LLM judge')
    batch_parser.add_argument('--concurrency', type=int, default=5, help='Parallel requests')
    batch_parser.add_argument('--output-dir', default='results', help='Output directory')

    # List models command
    models_parser = subparsers.add_parser('list-models', help='List available models')

    args = parser.parse_args()

    if args.command == 'list-templates':
        cmd_list_templates(args)
    elif args.command == 'single':
        cmd_run_single(args)
    elif args.command == 'batch':
        cmd_run_batch(args)
    elif args.command == 'list-models':
        cmd_list_models(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
