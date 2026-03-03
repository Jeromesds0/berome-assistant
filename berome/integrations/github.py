"""GitHub integration – repos, commits, file reads, and more."""

from __future__ import annotations

import base64
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from github import Auth, Github, GithubException
from github.Repository import Repository

from berome.config import settings


@dataclass
class RepoInfo:
    name: str
    full_name: str
    url: str
    private: bool
    description: str
    default_branch: str
    stars: int
    forks: int


@dataclass
class FileContent:
    path: str
    content: str
    sha: str
    size: int
    encoding: str = "utf-8"


@dataclass
class CommitResult:
    sha: str
    message: str
    url: str
    files_changed: list[str] = field(default_factory=list)


class GitHubIntegration:
    """Wraps PyGithub to provide a clean, typed interface for Berome."""

    def __init__(self) -> None:
        if not settings.github_token:
            raise RuntimeError(
                "GITHUB_TOKEN is not set. Add it to your .env file."
            )
        auth = Auth.Token(settings.github_token)
        self._gh = Github(auth=auth)
        self._user = self._gh.get_user()

    # ── Repository management ─────────────────────────────────────────────────

    def list_repos(self, limit: int = 30) -> list[RepoInfo]:
        """List repositories for the authenticated user."""
        repos = []
        for repo in self._user.get_repos(sort="updated")[:limit]:
            repos.append(self._repo_info(repo))
        return repos

    def get_repo(self, repo_name: str) -> RepoInfo:
        """Fetch a single repo by name (owner/repo or just repo)."""
        full = repo_name if "/" in repo_name else f"{self._user.login}/{repo_name}"
        repo = self._gh.get_repo(full)
        return self._repo_info(repo)

    def create_repo(
        self,
        name: str,
        description: str = "",
        private: bool = False,
        auto_init: bool = True,
    ) -> RepoInfo:
        """Create a new GitHub repository."""
        repo = self._user.create_repo(
            name=name,
            description=description,
            private=private,
            auto_init=auto_init,
        )
        return self._repo_info(repo)

    def delete_repo(self, repo_name: str) -> bool:
        """Delete a repository (irreversible!)."""
        full = repo_name if "/" in repo_name else f"{self._user.login}/{repo_name}"
        try:
            repo = self._gh.get_repo(full)
            repo.delete()
            return True
        except GithubException:
            return False

    # ── File / content operations ─────────────────────────────────────────────

    def read_file(self, repo_name: str, file_path: str, ref: str = "") -> FileContent:
        """Read a file from a GitHub repository."""
        full = repo_name if "/" in repo_name else f"{self._user.login}/{repo_name}"
        repo = self._gh.get_repo(full)
        kwargs = {"path": file_path}
        if ref:
            kwargs["ref"] = ref
        content_file = repo.get_contents(**kwargs)
        raw = content_file.decoded_content.decode("utf-8", errors="replace")
        return FileContent(
            path=file_path,
            content=raw,
            sha=content_file.sha,
            size=content_file.size,
        )

    def list_directory(self, repo_name: str, dir_path: str = "", ref: str = "") -> list[dict]:
        """List contents of a directory in a repo."""
        full = repo_name if "/" in repo_name else f"{self._user.login}/{repo_name}"
        repo = self._gh.get_repo(full)
        kwargs = {"path": dir_path}
        if ref:
            kwargs["ref"] = ref
        items = repo.get_contents(**kwargs)
        if not isinstance(items, list):
            items = [items]
        return [
            {"name": i.name, "path": i.path, "type": i.type, "size": i.size}
            for i in items
        ]

    def create_or_update_file(
        self,
        repo_name: str,
        file_path: str,
        content: str,
        commit_message: str,
        branch: str = "",
    ) -> CommitResult:
        """Create or update a single file in a repo via the GitHub API."""
        full = repo_name if "/" in repo_name else f"{self._user.login}/{repo_name}"
        repo = self._gh.get_repo(full)
        branch = branch or repo.default_branch
        encoded = content.encode("utf-8")

        try:
            existing = repo.get_contents(file_path, ref=branch)
            result = repo.update_file(
                path=file_path,
                message=commit_message,
                content=encoded,
                sha=existing.sha,
                branch=branch,
            )
        except GithubException:
            result = repo.create_file(
                path=file_path,
                message=commit_message,
                content=encoded,
                branch=branch,
            )

        commit = result["commit"]
        return CommitResult(
            sha=commit.sha,
            message=commit_message,
            url=commit.html_url,
            files_changed=[file_path],
        )

    # ── Clone and push via Git CLI ────────────────────────────────────────────

    def clone_repo(self, repo_name: str, target_dir: Path) -> Path:
        """Clone a repo using the authenticated HTTPS URL."""
        full = repo_name if "/" in repo_name else f"{self._user.login}/{repo_name}"
        token = settings.github_token
        clone_url = f"https://{token}@github.com/{full}.git"
        target_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", clone_url, str(target_dir)],
            check=True,
            capture_output=True,
        )
        return target_dir

    def commit_and_push(
        self,
        repo_dir: Path,
        message: str,
        branch: str = "",
        files: list[str] | None = None,
    ) -> str:
        """Stage files, commit, and push to origin."""
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

        # Stage files (all by default)
        add_args = ["git", "-C", str(repo_dir), "add"] + (files or ["-A"])
        subprocess.run(add_args, check=True, env=env)

        subprocess.run(
            ["git", "-C", str(repo_dir), "commit", "-m", message],
            check=True,
            env=env,
        )

        push_args = ["git", "-C", str(repo_dir), "push"]
        if branch:
            push_args += ["origin", branch]
        result = subprocess.run(push_args, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"git push failed: {result.stderr}")
        return message

    # ── Branch management ─────────────────────────────────────────────────────

    def create_branch(self, repo_name: str, branch: str, from_branch: str = "") -> str:
        """Create a new branch from an existing one."""
        full = repo_name if "/" in repo_name else f"{self._user.login}/{repo_name}"
        repo = self._gh.get_repo(full)
        from_branch = from_branch or repo.default_branch
        source = repo.get_branch(from_branch)
        repo.create_git_ref(f"refs/heads/{branch}", source.commit.sha)
        return branch

    def create_pull_request(
        self,
        repo_name: str,
        title: str,
        body: str,
        head: str,
        base: str = "",
    ) -> dict:
        """Open a pull request."""
        full = repo_name if "/" in repo_name else f"{self._user.login}/{repo_name}"
        repo = self._gh.get_repo(full)
        base = base or repo.default_branch
        pr = repo.create_pull(title=title, body=body, head=head, base=base)
        return {"number": pr.number, "url": pr.html_url, "state": pr.state}

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _repo_info(repo: Repository) -> RepoInfo:
        return RepoInfo(
            name=repo.name,
            full_name=repo.full_name,
            url=repo.html_url,
            private=repo.private,
            description=repo.description or "",
            default_branch=repo.default_branch,
            stars=repo.stargazers_count,
            forks=repo.forks_count,
        )

    @property
    def username(self) -> str:
        return self._user.login
