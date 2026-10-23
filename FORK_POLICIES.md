# Fork Policies (`lazydao/hermes-agent`)

## Core Changes and Live Gateway Restarts Require Explicit Approval

- Treat edits to Hermes core code and restarts of the live gateway (which serves
  every profile under multiplex) as separate high-impact actions. Before either
  action, report the exact scope, user-visible impact, verification plan, and
  rollback plan, then obtain explicit user approval for that action. A generic
  approval for the surrounding automation or incident work does not authorize
  core edits or gateway restarts.
- Prefer profile configuration, plugins, hooks, wrappers, or project-side
  coordination before changing Hermes core. If a core change is still needed,
  explain why those narrower extension points are insufficient before asking
  for approval.
- Immediately before restarting, re-check `active_agents` for every served
  profile. Do not restart while any profile has active work unless the user
  explicitly authorizes interrupting that work; after the restart verify the
  new process identity and the Feishu connection state of every profile.

## Reviewed Changes May Be Committed and Pushed

- For changes the user explicitly requested to implement or fix, unless the
  user limits the task to local changes only, review the actual diff before
  committing. The review may be performed by OpenCode, Claude, or the current
  executor. A self-review must explicitly examine scope, risk, verification
  evidence, and unresolved findings; passing tests alone is not a completed
  review.
- A clear passing review, the repository-required verification passing in the
  current run, a commit scope containing only the current task, confirmed
  ownership of the remote, and a safely synchronizable branch together
  authorize one commit and push for that task without another confirmation.
  If any condition is missing, do not commit or push. This convention does not
  authorize core edits, gateway restarts, merges, releases, deployments, force
  pushes, history rewrites, or inclusion of unrelated dirty work.

## Fork Upstream Sync Policy

This checkout is maintained through the `lazydao/hermes-agent` fork. Normal
upstream synchronization is gated by upstream release tags:

- Fetching or inspecting `upstream/main` is allowed, but do not merge, rebase,
  or cherry-pick it into `h3/main` merely because upstream main has advanced.
- Start an upstream synchronization only after `NousResearch/hermes-agent`
  publishes a release tag newer than the last evaluated release tag.
- Build a candidate branch from that release tag, reconcile the fork's local
  changes there, and validate all three Hermes profiles before promoting it to
  `h3/main` or restarting the live gateway.
- Selectively backport an untagged upstream fix only for a confirmed critical
  security or production issue and only after explicit user approval.
- The current evaluated release baseline is `v0.22.5`; update this baseline
  as part of each completed upstream synchronization.
