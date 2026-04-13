---
name: Code editing output style
description: When user pastes code and asks to change it, print the result only — never read or edit files
type: feedback
---

When the user pastes code directly in the chat and asks to modify it, output the modified code as plain text in the response only. Do NOT read files, edit notebooks, or write to any file.

**Why:** User wants to copy the output themselves and decide where to paste it. Writing directly to notebooks or files is unwanted and disruptive.

**How to apply:** Any time code is pasted inline in the message + the request is to change/refactor/extend it → respond with the modified code block only, no file tools.
