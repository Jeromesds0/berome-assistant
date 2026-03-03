"""GitHub agent – executes GitHub operations as background tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from berome.agents.base import Agent, AgentTask
from berome.integrations.github import GitHubIntegration


class GitHubAgent(Agent):
    """
    Handles GitHub operations dispatched as tasks.

    Supported actions (set task.payload["action"]):
      list_repos   – list user repos
      get_repo     – get repo details (payload: repo)
      create_repo  – create a repo (payload: name, description, private)
      read_file    – read a file (payload: repo, path, ref?)
      list_dir     – list directory (payload: repo, path?, ref?)
      write_file   – create/update file (payload: repo, path, content, message, branch?)
      clone_repo   – clone to local dir (payload: repo, target_dir)
      push         – commit & push local dir (payload: repo_dir, message, branch?, files?)
      create_branch – create branch (payload: repo, branch, from_branch?)
      create_pr    – open PR (payload: repo, title, body, head, base?)
    """

    agent_type = "github"
    description = "Performs GitHub operations (repos, files, commits, PRs)"

    def __init__(self) -> None:
        super().__init__()
        self._gh: GitHubIntegration | None = None

    def _client(self) -> GitHubIntegration:
        if self._gh is None:
            self._gh = GitHubIntegration()
        return self._gh

    async def run(self, task: AgentTask) -> Any:
        action = task.payload.get("action", "")
        p = task.payload
        gh = self._client()

        if action == "list_repos":
            repos = gh.list_repos(limit=p.get("limit", 30))
            return [r.__dict__ for r in repos]

        if action == "get_repo":
            info = gh.get_repo(p["repo"])
            return info.__dict__

        if action == "create_repo":
            info = gh.create_repo(
                name=p["name"],
                description=p.get("description", ""),
                private=p.get("private", False),
                auto_init=p.get("auto_init", True),
            )
            return info.__dict__

        if action == "read_file":
            fc = gh.read_file(p["repo"], p["path"], p.get("ref", ""))
            return fc.__dict__

        if action == "list_dir":
            return gh.list_directory(p["repo"], p.get("path", ""), p.get("ref", ""))

        if action == "write_file":
            result = gh.create_or_update_file(
                repo_name=p["repo"],
                file_path=p["path"],
                content=p["content"],
                commit_message=p["message"],
                branch=p.get("branch", ""),
            )
            return result.__dict__

        if action == "clone_repo":
            target = Path(p["target_dir"])
            cloned = gh.clone_repo(p["repo"], target)
            return str(cloned)

        if action == "push":
            msg = gh.commit_and_push(
                repo_dir=Path(p["repo_dir"]),
                message=p["message"],
                branch=p.get("branch", ""),
                files=p.get("files"),
            )
            return msg

        if action == "create_branch":
            branch = gh.create_branch(p["repo"], p["branch"], p.get("from_branch", ""))
            return branch

        if action == "create_pr":
            return gh.create_pull_request(
                repo_name=p["repo"],
                title=p["title"],
                body=p.get("body", ""),
                head=p["head"],
                base=p.get("base", ""),
            )

        raise ValueError(f"Unknown GitHub action: {action!r}")
