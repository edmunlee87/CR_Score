"""
CR_Score command-line interface.

Provides commands for config validation, run execution, comparison, and more.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import click

from cr_score import __version__
from cr_score.core.config import load_config
from cr_score.core.exceptions import ConfigValidationError, CR_ScoreException
from cr_score.core.logging import get_audit_logger
from cr_score.core.registry import RunRegistry


@click.group()
@click.version_option(version=__version__, prog_name="CR_Score")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """
    CR_Score: Enterprise Scorecard Development Platform

    Config-driven, Spark-native platform for credit scorecard development.

    Examples:

        \b
        # Validate configuration
        cr-score validate --config config.yml

        \b
        # Run scorecard development
        cr-score run --config config.yml

        \b
        # List recent runs
        cr-score list-runs --limit 10

        \b
        # Compare two runs
        cr-score compare --run-id-a run_123 --run-id-b run_456
    """
    ctx.ensure_object(dict)


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to configuration YAML file",
)
def validate(config: str) -> None:
    """
    Validate configuration file.

    Checks schema, required fields, and value constraints.
    """
    try:
        click.echo(f"Validating configuration: {config}")
        cfg = load_config(config)

        click.secho("Configuration is valid", fg="green", bold=True)
        click.echo(f"\nProject: {cfg.project.name}")
        click.echo(f"Owner: {cfg.project.owner}")
        click.echo(f"Engine: {cfg.execution.engine.value}")
        click.echo(f"Compression: {'enabled' if cfg.compression.enabled else 'disabled'}")

    except ConfigValidationError as e:
        click.secho(f"Configuration validation failed: {e}", fg="red", bold=True)
        sys.exit(1)
    except Exception as e:
        click.secho(f"Unexpected error: {e}", fg="red")
        sys.exit(1)


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to configuration YAML file",
)
@click.option(
    "--run-id",
    type=str,
    default=None,
    help="Custom run ID (auto-generated if not provided)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate configuration without executing run",
)
def run(config: str, run_id: Optional[str], dry_run: bool) -> None:
    """
    Execute scorecard development run.

    Runs the full pipeline: EDA → Binning → Modeling → Scaling → Reporting
    """
    try:
        # Load config
        click.echo(f"Loading configuration: {config}")
        cfg = load_config(config)

        if dry_run:
            click.secho("Dry run mode - configuration is valid", fg="yellow")
            click.echo(f"Would execute: {cfg.project.name}")
            return

        # TODO: Implement full pipeline execution
        click.secho("Full pipeline execution not yet implemented", fg="yellow")
        click.echo("Core infrastructure is ready. Pipeline modules pending.")

    except ConfigValidationError as e:
        click.secho(f"Configuration error: {e}", fg="red", bold=True)
        sys.exit(1)
    except CR_ScoreException as e:
        click.secho(f"CR_Score error: {e}", fg="red")
        sys.exit(1)
    except Exception as e:
        click.secho(f"Unexpected error: {e}", fg="red")
        sys.exit(1)


@cli.command()
@click.option(
    "--status",
    type=click.Choice(["created", "running", "completed", "failed", "cancelled"]),
    default=None,
    help="Filter by run status",
)
@click.option(
    "--user-id",
    type=str,
    default=None,
    help="Filter by user ID",
)
@click.option(
    "--limit",
    type=int,
    default=20,
    help="Maximum number of runs to display",
)
def list_runs(status: Optional[str], user_id: Optional[str], limit: int) -> None:
    """
    List execution runs with optional filtering.
    """
    try:
        registry = RunRegistry()
        runs = registry.list_runs(status=status, user_id=user_id, limit=limit)

        if not runs:
            click.echo("No runs found")
            return

        click.echo(f"\nFound {len(runs)} run(s):\n")

        for run in runs:
            status_color = {
                "created": "blue",
                "running": "yellow",
                "completed": "green",
                "failed": "red",
                "cancelled": "magenta",
            }.get(run["status"], "white")

            click.secho(f"Run ID: {run['run_id']}", bold=True)
            click.echo(f"  Status: ", nl=False)
            click.secho(run["status"], fg=status_color)
            click.echo(f"  Project: {run.get('project_name', 'N/A')}")
            click.echo(f"  Engine: {run.get('execution_engine', 'N/A')}")
            click.echo(f"  Created: {run['created_at']}")
            if run.get("completed_at"):
                click.echo(f"  Completed: {run['completed_at']}")
            click.echo()

    except Exception as e:
        click.secho(f"Error listing runs: {e}", fg="red")
        sys.exit(1)


@cli.command()
@click.option(
    "--run-id",
    required=True,
    type=str,
    help="Run ID to query",
)
def run_info(run_id: str) -> None:
    """
    Display detailed information about a specific run.
    """
    try:
        registry = RunRegistry()
        run = registry.get_run(run_id)

        click.secho(f"\nRun Details: {run_id}", bold=True, fg="cyan")
        click.echo("=" * 60)

        for key, value in run.items():
            if value is not None:
                click.echo(f"{key:20s}: {value}")

        # Get steps
        steps = registry.get_run_steps(run_id)
        if steps:
            click.echo(f"\n{'Steps':<20s}:")
            for step in steps:
                status_color = {
                    "started": "yellow",
                    "completed": "green",
                    "failed": "red",
                }.get(step["status"], "white")

                click.echo(f"  {step['step_order']}. {step['step_name']:<15s} ", nl=False)
                click.secho(step["status"], fg=status_color)
                if step.get("duration_seconds"):
                    click.echo(f"     Duration: {step['duration_seconds']:.2f}s")

    except Exception as e:
        click.secho(f"Error retrieving run info: {e}", fg="red")
        sys.exit(1)


@cli.command()
@click.option(
    "--run-id-a",
    required=True,
    type=str,
    help="First run ID",
)
@click.option(
    "--run-id-b",
    required=True,
    type=str,
    help="Second run ID",
)
def compare(run_id_a: str, run_id_b: str) -> None:
    """
    Compare two runs for reproducibility testing.

    Useful for validators to verify consistency.
    """
    try:
        registry = RunRegistry()
        run_a = registry.get_run(run_id_a)
        run_b = registry.get_run(run_id_b)

        click.secho(f"\nComparing runs:", bold=True, fg="cyan")
        click.echo(f"  A: {run_id_a}")
        click.echo(f"  B: {run_id_b}\n")

        # Compare key fields
        fields_to_compare = [
            "status",
            "config_hash",
            "data_hash",
            "project_name",
            "execution_engine",
        ]

        differences = []
        for field in fields_to_compare:
            val_a = run_a.get(field)
            val_b = run_b.get(field)

            if val_a == val_b:
                click.secho(f"  {field:20s}: MATCH", fg="green")
            else:
                click.secho(f"  {field:20s}: DIFFER", fg="red")
                differences.append(field)
                click.echo(f"    A: {val_a}")
                click.echo(f"    B: {val_b}")

        click.echo()
        if differences:
            click.secho(f"Runs differ in {len(differences)} field(s)", fg="red", bold=True)
        else:
            click.secho("Runs are identical", fg="green", bold=True)

    except Exception as e:
        click.secho(f"Error comparing runs: {e}", fg="red")
        sys.exit(1)


@cli.command()
@click.option(
    "--template",
    type=click.Choice(["beginner", "intermediate", "advanced"]),
    default="intermediate",
    help="Template complexity level",
)
@click.option(
    "--output",
    type=click.Path(),
    default="config.yml",
    help="Output configuration file path",
)
def init(template: str, output: str) -> None:
    """
    Initialize new project with template configuration.
    """
    try:
        # TODO: Load template from templates directory
        click.secho(f"Creating {template} configuration template...", fg="yellow")
        click.echo(f"Output: {output}")

        # For now, show message
        click.secho("Template generation not yet implemented", fg="yellow")
        click.echo("Please manually create config.yml using URD Appendix B as reference")

    except Exception as e:
        click.secho(f"Error initializing project: {e}", fg="red")
        sys.exit(1)


@cli.command()
def version() -> None:
    """Display version information."""
    click.echo(f"CR_Score version {__version__}")
    click.echo("Enterprise Scorecard Development Platform")


if __name__ == "__main__":
    cli()
