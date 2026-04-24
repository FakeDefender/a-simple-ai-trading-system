# AutoCodex User Manual

Generated from the embedded binary payload.

## Version

- Current version: `0.120.3`

## What Is New In This Version

- `/auto` now uses a Superpowers-first workflow as the primary path.
- Common control commands now have short aliases for daily use.
- Long legacy command names remain compatible for expert or existing workflows.
- `/auto status`, `/auto artifacts`, `/auto doctor`, and `/auto timeline` continue to surface workflow snapshot truth instead of hiding retained state.
- Long-running `/auto` and `repeat` work now uses grouped reconnect retries after dropped streams, so the runtime can stay alive through longer transient network failures.

## Core Entry Points

- `/auto`
  Resume the current auto workflow when there is recoverable state.
  If there is no recoverable state, reuse the latest meaningful user goal or show the usage guide.

- `/auto <goal>`
  Start from a natural-language goal.
  The router will prefer the bundled Superpowers workflow and attach the right skill path.

- `/auto <path.md>`
  Start from a document path.
  The router will inspect the document and choose the right workflow route.

## Short Control Commands

- `/auto s`
  Show current status.

- `/auto a`
  Show artifacts.

- `/auto d`
  Show doctor diagnostics.

- `/auto t`
  Show timeline.

- `/auto p`
  Pause the active auto run.

- `/auto c`
  Continue or resume the current auto run.

- `/auto x`
  Stop the current auto run and keep a recoverable stopped checkpoint when possible.

- `/auto n`
  Skip the current task and jump to the next task.

- `/auto g <n>`
  Jump to task number `<n>`.

- `/auto r`
  Reload the active auto profile without changing the current target when possible.

## Extended Commands

- `/auto preview <source-or-companion> [--skill <name>]`
  Preview the routing or plan state before running.

- `/auto reload-profile`
  Expert compatibility alias for profile reload.

- `/auto status`
- `/auto artifacts`
- `/auto doctor`
- `/auto timeline`
- `/auto pause`
- `/auto stop`
- `/auto skip`
- `/auto goto <n>`
- `/auto reload`
  Long compatibility aliases for the short commands above.

- `/auto validate <plan.exec.md> [--skill <name>]`
- `/auto normalize <source.md> [--skill <name>]`
- `/auto augment <source.md> [--skill <name>]`
  Legacy compatibility commands retained for migration and expert workflows.

## Repeat Mode

- `/re <rounds> [/<推进提示词>] [@<反思间隔> [/<反思提示词>]] [--skill <name>]`
  Start repeat mode with the short command.
  `<rounds>` counts only primary推进 rounds.
  `/<推进提示词>` overrides the repeated prompt for primary rounds.
  `@<反思间隔>` inserts one reflection round after every N primary rounds.
  `/<反思提示词>` overrides the reflection round prompt.
  If the primary prompt is omitted, repeat reuses the latest manual user prompt.
  If the reflection prompt is omitted, repeat uses the configured default reflection prompt: `反思+深查+补救，然后继续按TDD推进`.

- `/repeat <rounds> [/<推进提示词>] [@<反思间隔> [/<反思提示词>]] [--skill <name>]`
  Start repeat mode with the long command.

- `/re status`
- `/repeat status`
  Show repeat status, including long-running reconnect protection when active.

- `/re reload`
- `/repeat reload`
  Reload the active repeat profile in place.

- `/re stop`
- `/repeat stop`
  Stop the current repeat run.

- Examples:
  `/re 8 /继续推进`
  `/re 8 @2`
  `/re 8 /继续推进 @2 /反思+深查+补救`
  `/repeat 8 /继续推进 @2 /反思+深查+补救`

## Recommended Operator Flow

1. Start AutoCodex in your workspace.
2. Use one of:
   - `/auto`
   - `/auto <goal>`
   - `/auto <path.md>`
3. Use `/auto s` to confirm the active route, stage, next action, resume hint, and stream reconnect protection.
4. Use `/auto a`, `/auto d`, and `/auto t` whenever you need evidence instead of guessing.
5. Use `/auto p`, `/auto c`, `/auto n`, `/auto g <n>`, or `/auto x` to control execution.
6. Use `/auto r` when the embedded skill or runtime profile was updated and you want to refresh it in place.

## Practical Examples

- Start from a goal:
  `/auto refactor the login workflow and keep the rollout safe`

- Start from a document:
  `/auto D:/work/docs/login-refresh.md`

- Inspect the current state:
  `/auto s`

- Pause and continue:
  `/auto p`
  `/auto c`

- Skip or jump:
  `/auto n`
  `/auto g 12`

- Stop and inspect retained context:
  `/auto x`
  `/auto s`
  `/auto a`
  `/auto d`
  `/auto t`

## Files Released Beside The Executable

- `autocodex.toml`
  Runtime configuration.

- `AUTOCODEX-README.md`
  Lightweight product overview.

- `AUTOCODEX-MANUAL.md`
  This full operator manual.

## Notes

- This manual is generated from the binary on startup.
- If the embedded version changes, the manual is refreshed automatically.
- If you are already in a long-running session, reload only when you actually want the refreshed local profile files to be used.
- During long-running `/auto` or `repeat` sessions, a dropped stream may now retry in grouped waves instead of giving up after the first short retry budget.
- `/auto s` and `/repeat status` now show `stream_reconnect: long_running_automation (...)` while grouped reconnect protection is active, so you can verify the guardrail before a disconnect happens.
- When the footer status line is enabled, active `/auto` and `repeat` runs also add a compact badge like `reconnect 5x/180s` to the live HUD.
- While `/auto` or `repeat` is actively running, the default terminal title layout also appends a compact reconnect badge so long-running windows stay legible when unfocused without surprising custom title setups.
- When you see a message like `Reconnecting... 6/25 (group 2/5, cooldown 180s)`, that means the automation thread entered the next reconnect group and is intentionally waiting before retrying again.
