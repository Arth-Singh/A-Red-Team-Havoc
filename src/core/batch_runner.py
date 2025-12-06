"""
Batch Runner for A-Red-Team-Havoc
Orchestrates batch attacks using templates against target models
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import uuid

from .template_engine import TemplateEngine, AttackTemplate
from .target import OpenRouterTarget, TargetResponse
from .scorer import Scorer, Score, ScoreResult


@dataclass
class AttackResult:
    """Complete result of a single attack attempt"""
    id: str
    objective: str
    template_name: str
    template_category: str
    rendered_prompt: str
    response: TargetResponse
    score: Score
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective,
            "template_name": self.template_name,
            "template_category": self.template_category,
            "rendered_prompt": self.rendered_prompt,
            "response": self.response.to_dict(),
            "score": self.score.to_dict(),
            "timestamp": self.timestamp
        }


@dataclass
class BatchResult:
    """Complete results of a batch attack run"""
    run_id: str
    target_model: str
    start_time: str
    end_time: str
    total_attacks: int
    objectives: List[str]
    templates_used: List[str]
    results: List[AttackResult] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "target_model": self.target_model,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_attacks": self.total_attacks,
            "objectives": self.objectives,
            "templates_used": self.templates_used,
            "results": [r.to_dict() for r in self.results],
            "statistics": self.statistics
        }

    def save(self, output_dir: str = "results") -> str:
        """Save results to JSON file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"batch_{self.run_id}_{self.start_time.replace(':', '-').replace('.', '-')}.json"
        filepath = output_path / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        return str(filepath)


class BatchRunner:
    """
    Orchestrates batch attacks:
    1. Loads templates
    2. Renders with objectives
    3. Sends to target
    4. Scores responses
    5. Aggregates results
    """

    def __init__(
        self,
        template_engine: TemplateEngine,
        target: OpenRouterTarget,
        scorer: Scorer,
        concurrency: int = 5,
        output_dir: str = "results"
    ):
        self.template_engine = template_engine
        self.target = target
        self.scorer = scorer
        self.concurrency = concurrency
        self.output_dir = output_dir

    async def run_single_attack_async(
        self,
        objective: str,
        template: AttackTemplate
    ) -> AttackResult:
        """Run a single attack with one objective and one template"""
        # Render template
        rendered_prompt = template.render(prompt=objective)

        # Send to target
        response = await self.target.send_prompt_async(
            prompt=rendered_prompt,
            template_name=template.name
        )

        # Score response
        score = await self.scorer.score_async(response)

        return AttackResult(
            id=str(uuid.uuid4())[:8],
            objective=objective,
            template_name=template.name,
            template_category=template.category,
            rendered_prompt=rendered_prompt,
            response=response,
            score=score,
            timestamp=datetime.now().isoformat()
        )

    async def run_batch_async(
        self,
        objectives: List[str],
        template_names: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None
    ) -> BatchResult:
        """
        Run batch attacks with multiple objectives and templates

        Args:
            objectives: List of attack objectives/prompts
            template_names: Specific templates to use (None = all)
            categories: Template categories to use (None = all)
            progress_callback: Optional callback(completed, total) for progress

        Returns:
            BatchResult with all attack results
        """
        run_id = str(uuid.uuid4())[:8]
        start_time = datetime.now().isoformat()

        # Get templates to use
        if template_names:
            templates = [
                self.template_engine.get_template(name)
                for name in template_names
                if self.template_engine.get_template(name)
            ]
        elif categories:
            templates = []
            for cat in categories:
                templates.extend(self.template_engine.get_templates_by_category(cat))
        else:
            templates = self.template_engine.get_all_templates()

        if not templates:
            raise ValueError("No templates found matching criteria")

        # Build attack matrix
        attacks = []
        for objective in objectives:
            for template in templates:
                attacks.append((objective, template))

        total_attacks = len(attacks)
        print(f"\n[HAVOC] Starting batch run {run_id}")
        print(f"[HAVOC] Objectives: {len(objectives)}, Templates: {len(templates)}")
        print(f"[HAVOC] Total attacks: {total_attacks}")
        print(f"[HAVOC] Target: {self.target.model}")
        print(f"[HAVOC] Concurrency: {self.concurrency}\n")

        # Execute with concurrency control
        semaphore = asyncio.Semaphore(self.concurrency)
        results: List[AttackResult] = []
        completed = 0

        async def run_with_semaphore(objective: str, template: AttackTemplate) -> AttackResult:
            nonlocal completed
            async with semaphore:
                result = await self.run_single_attack_async(objective, template)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_attacks)
                else:
                    # Default progress output
                    status = result.score.result.value.upper()
                    print(f"[{completed}/{total_attacks}] {template.name[:30]:30} -> {status}")
                return result

        tasks = [run_with_semaphore(obj, tmpl) for obj, tmpl in attacks]
        results = await asyncio.gather(*tasks)

        end_time = datetime.now().isoformat()

        # Calculate statistics
        scores = [r.score for r in results]
        statistics = self.scorer.get_statistics(scores)

        # Add per-template statistics
        template_stats = {}
        for template in templates:
            template_results = [r for r in results if r.template_name == template.name]
            template_scores = [r.score for r in template_results]
            if template_scores:
                template_stats[template.name] = {
                    "total": len(template_scores),
                    "harmful": sum(1 for s in template_scores if s.result == ScoreResult.HARMFUL),
                    "rejected": sum(1 for s in template_scores if s.result == ScoreResult.REJECTED),
                    "success_rate": sum(1 for s in template_scores if s.result == ScoreResult.HARMFUL) / len(template_scores)
                }

        statistics["per_template"] = template_stats

        # Add per-objective statistics
        objective_stats = {}
        for objective in objectives:
            obj_results = [r for r in results if r.objective == objective]
            obj_scores = [r.score for r in obj_results]
            if obj_scores:
                objective_stats[objective[:50]] = {
                    "total": len(obj_scores),
                    "harmful": sum(1 for s in obj_scores if s.result == ScoreResult.HARMFUL),
                    "rejected": sum(1 for s in obj_scores if s.result == ScoreResult.REJECTED),
                    "success_rate": sum(1 for s in obj_scores if s.result == ScoreResult.HARMFUL) / len(obj_scores)
                }

        statistics["per_objective"] = objective_stats

        batch_result = BatchResult(
            run_id=run_id,
            target_model=self.target.model,
            start_time=start_time,
            end_time=end_time,
            total_attacks=total_attacks,
            objectives=objectives,
            templates_used=[t.name for t in templates],
            results=results,
            statistics=statistics
        )

        # Print summary
        print(f"\n{'='*60}")
        print(f"[HAVOC] Batch Complete: {run_id}")
        print(f"{'='*60}")
        print(f"Total Attacks: {statistics['total']}")
        print(f"Rejected:      {statistics['rejected']} ({statistics['rejection_rate']*100:.1f}%)")
        print(f"Harmful:       {statistics['harmful']} ({statistics['attack_success_rate']*100:.1f}%)")
        print(f"Safe:          {statistics['safe']} ({statistics['safe_rate']*100:.1f}%)")
        print(f"{'='*60}\n")

        return batch_result

    def run_batch(
        self,
        objectives: List[str],
        template_names: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None
    ) -> BatchResult:
        """Synchronous wrapper for run_batch_async"""
        return asyncio.run(self.run_batch_async(
            objectives, template_names, categories, progress_callback
        ))

    def run_and_save(
        self,
        objectives: List[str],
        template_names: Optional[List[str]] = None,
        categories: Optional[List[str]] = None
    ) -> str:
        """Run batch and save results to file"""
        result = self.run_batch(objectives, template_names, categories)
        filepath = result.save(self.output_dir)
        print(f"[HAVOC] Results saved to: {filepath}")
        return filepath


# CLI Entry Point
if __name__ == "__main__":
    import sys
    import os

    # Quick test
    print("BatchRunner module loaded successfully")
    print("Use main.py for full CLI functionality")
