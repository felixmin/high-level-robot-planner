---
name: claude-cli-review
description: Run Anthropic Claude through the local `claude` CLI to get a second-opinion review of code, configs, designs, prompts, or implementation plans. Use only when the user explicitly asks for Claude, explicitly asks for an external review, or explicitly asks for review through the CLI.
---

# Claude CLI Review

Use `scripts/run_claude_review.sh` rather than rewriting the `claude -p` command by hand.

## Workflow

1. Decide whether the review should be:
   - prompt-only: Claude reviews the prompt you provide and does not inspect local files
   - grounded: Claude can inspect local files or directories you explicitly add
2. Give Claude the full thing to review. Do one of these:
   - provide the complete plan, proposal, or text inline
   - point Claude at the exact code or directories to inspect
   - tell Claude to review the actual staged or unstaged changes rather than a summary
3. Write a concise prompt that includes:
   - the full artifact to review or exact file pointers
   - the exact review goal
   - the desired output shape
4. Prefer findings-first prompts. Ask for:
   - concrete risks or bugs first
   - unnecessary complexity second
   - a simplified recommendation last
5. Run `scripts/run_claude_review.sh` with either `--prompt-file <path>` or stdin.
6. If the sandboxed call hangs or cannot reach the backend, rerun the command outside the sandbox.
7. Summarize Claude's findings back to the user. Do not dump raw output unless the user asks.

## Prompt Pattern

Use this shape unless the user asks for something else:

```text
Review this <code/design/config> proposal.
Prioritize:
1. concrete risks
2. unnecessary complexity
3. the simplest design that preserves the intended behavior

Context:
- ...
- ...

Keep the answer concise and structured as findings first, then recommendation.
```

For grounded repo review, include exact paths and enough context for Claude to inspect the right area.
Do not send Claude a compressed paraphrase if the original plan, diff, or files can be provided directly.

## Commands

Prompt-only review from a file:

```bash
.codex/skills/claude-cli-review/scripts/run_claude_review.sh \
  --prompt-file /tmp/review_prompt.txt
```

Prompt-only review from stdin:

```bash
cat /tmp/review_prompt.txt | \
  .codex/skills/claude-cli-review/scripts/run_claude_review.sh
```

Grounded review with repo access:

```bash
.codex/skills/claude-cli-review/scripts/run_claude_review.sh \
  --tools default \
  --add-dir "$PWD" \
  --prompt-file /tmp/review_prompt.txt
```

Write the response to a file too:

```bash
.codex/skills/claude-cli-review/scripts/run_claude_review.sh \
  --prompt-file /tmp/review_prompt.txt \
  --output-file /tmp/review_output.txt
```

## Rules

- Only use this skill when the user explicitly asks for an external review or explicitly asks to ask Claude.
- Keep the default tool setting restrictive. Only enable Claude tools when the review needs file inspection.
- Add only the directories Claude actually needs with `--add-dir`.
- Always give Claude the full review target: complete plan, exact files, or the real staged/unstaged diff.
- Keep prompts concrete. Long narrative prompts usually produce worse review quality.
- Treat Claude as a second opinion, not the final authority.
- If Claude suggests a structural change, compare it against the local code before adopting it.

## Resources

- `scripts/run_claude_review.sh`
  Wrapper around `claude -p` that reads a review prompt from a file or stdin and optionally enables tool access or saves output.
