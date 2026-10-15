import asyncio
from importlib import import_module
import logging

logging.basicConfig(level=logging.INFO)

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
    required=False,
    type=click.Path(exists=True, file_okay=False),
    help="Path to the Git repository.",
)
@click.option("--branch-a", required=False, help="Branch A (ours).")
@click.option("--branch-b", required=False, help="Branch B (theirs).")
@click.option("--file-path", required=False, help="Relative file path within the repo.")
@click.option("--max-iterations", default=5, show_default=True, type=int, help="Max repair iterations.")
@click.option("--base", required=False, type=click.Path(exists=True), help="Ancestor file (%O in git merge driver).")
@click.option("--ours", required=False, type=click.Path(exists=True), help="Ours file (%A in git merge driver).")
@click.option("--theirs", required=False, type=click.Path(exists=True), help="Theirs file (%B in git merge driver).")
@click.option("--output", required=False, type=click.Path(), help="Output path (defaults to --ours).")
def resolve(
    repo_path: str | None,
    branch_a: str | None,
    branch_b: str | None,
    file_path: str | None,
    max_iterations: int,
    base: str | None,
    ours: str | None,
    theirs: str | None,
    output: str | None,
):
    """Resolve a merge conflict in a file or via Git custom merge driver."""
    # Custom Merge Driver Mode (%O %A %B)
    if base and ours and theirs:
        from pathlib import Path
        import sys

        out_file = Path(output) if output else Path(ours)
        base_file = Path(base)
        ours_file = Path(ours)
        theirs_file = Path(theirs)

        click.echo(f"git-merger: Running semantic merge driver on {ours}...")
        try:
            base_content = base_file.read_text(encoding="utf-8")
            ours_content = ours_file.read_text(encoding="utf-8")
            theirs_content = theirs_file.read_text(encoding="utf-8")

            # Try to invoke synthesizer directly if client available
            import os
            from openai import OpenAI
            from git_merger.models import BranchContext, FileVersion, MergeContext
            from git_merger.synthesizer import SemanticSynthesizer

            api_key = os.getenv("NEBIUS_API_KEY")
            if not api_key:
                click.echo("git-merger: NEBIUS_API_KEY not configured. Falling back to git standard conflict markers.", err=True)
                sys.exit(1)

            client = OpenAI(base_url="https://api.studio.nebius.ai/v1", api_key=api_key)
            context = MergeContext(
                file_path=str(out_file.name),
                base=FileVersion(content=base_content, commit_hash="base", commit_message="", author="", timestamp=""),
                ours=BranchContext(file_version=FileVersion(content=ours_content, commit_hash="ours", commit_message="", author="", timestamp=""), commit_log=[], diff_from_base="", changed_functions=[], full_diff_patch=""),
                theirs=BranchContext(file_version=FileVersion(content=theirs_content, commit_hash="theirs", commit_message="", author="", timestamp=""), commit_log=[], diff_from_base="", changed_functions=[], full_diff_patch=""),
                conflict_markers=None,
                ast_diff_summary={},
                shared_changed_functions=[],
            )

            synthesizer = SemanticSynthesizer(client)
            result = synthesizer.synthesize(context)
            out_file.write_text(result.candidate_code, encoding="utf-8")
            click.echo(f"git-merger: Successfully resolved and wrote merged code to {out_file}")
            sys.exit(0)
        except Exception as exc:
            click.echo(f"git-merger merge driver failed: {exc}", err=True)
            sys.exit(1)

    # Repository Branch Mode
    if not (repo_path and branch_a and branch_b and file_path):
        raise click.UsageError(
            "Either specify (--base, --ours, --theirs) for merge driver mode "
            "or (--repo-path, --branch-a, --branch-b, --file-path) for branch mode."
        )

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