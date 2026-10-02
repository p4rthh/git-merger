import asyncio
from importlib import import_module

import click


def _load_backend_callable(module_name: str, function_name: str):
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            raise click.ClickException(
                f"{module_name} is not implemented yet."
            ) from error
        raise click.ClickException(
            f"Cannot load {module_name}: missing dependency {error.name!r}."
        ) from error

    try:
        return getattr(module, function_name)
    except AttributeError as error:
        raise click.ClickException(
            f"{module_name} does not provide {function_name}."
        ) from error


@click.group()
def cli():
    """Git Merger - Semantic merge conflict resolver."""
    pass


@cli.command()
@click.option(
    "--repo-path",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option("--branch-a", required=True)
@click.option("--branch-b", required=True)
@click.option("--file-path", required=True)
@click.option("--max-iterations", default=5, show_default=True, type=int)
def resolve(
    repo_path: str,
    branch_a: str,
    branch_b: str,
    file_path: str,
    max_iterations: int,
):
    """Resolve a merge conflict in a file."""
    resolve_merge_conflict = _load_backend_callable(
        "git_merger.orchestrator", "resolve_merge_conflict"
    )

    result = asyncio.run(
        resolve_merge_conflict(
            repo_path=repo_path,
            branch_a=branch_a,
            branch_b=branch_b,
            file_path=file_path,
            max_repair_iterations=max_iterations,
        )
    )

    click.echo(f"Status: {result.status}")

    if hasattr(result, "final_candidate") and result.final_candidate:
        click.echo("\nFinal candidate:\n")
        click.echo(result.final_candidate)

    if hasattr(result, "reason") and result.reason:
        click.echo(f"\nReason: {result.reason}")


@cli.command()
def demo():
    """Run benchmark scenarios."""
    run_demo = _load_backend_callable("git_merger.demo", "run_demo")
    result = run_demo()
    click.echo(result)


@cli.command()
def status():
    """Show the current agent state."""
    from git_merger.models import AgentState

    click.echo(f"Agent state: {AgentState.IDLE.name}")


if __name__ == "__main__":
    cli()