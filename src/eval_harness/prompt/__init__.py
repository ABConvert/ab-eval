"""Prompt tasks: benchmark a product's own LLM feature on frozen prompts it exported.

A coding case is a merged pull request graded by the team's tests. A prompt case is a frozen
(system, prompt) pair the product's own code produced, graded by a rubric judge and an
optional product-owned scorer. See docs/design/prompt-tasks.md and docs/prompt-tasks.md.
"""
