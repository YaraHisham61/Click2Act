---
name: Memory file location
description: Always write memory files inside the local project .claude/memory/, never in ~/.claude
type: feedback
---

Always write memory files to `.claude/memory/` inside the project directory (e.g. `/home/aliaagheis/master-study/sw/Click2Act/.claude/memory/`), not to `~/.claude/projects/...`.

**Why:** User wants memory scoped and stored locally within the project repo, not in the global home directory.

**How to apply:** Every Write call for memory files must use the project-local path.
