# Push to GitHub — run this auth line first

`gh` is installed but **not logged in**. In another terminal (same VM), run this;
it prompts without echoing the token:

```bash
read -rsp 'GitHub token: ' T && echo && printf '%s' "$T" | gh auth login --with-token && unset T && gh auth status
```

Token needs `repo` scope. Then tell the session it's done and it will commit +
push.

## What will be committed (from repo root, `main`, remote
`https://github.com/akbargherbal/yue2_lora_finetuning.git`)
- `context.md` — session-7 §17 (glitch list) + header/§1/§2/§7/§10/§15 updates
- `PLAN.md` — §4 tokenize command fixed (`--semantic --no-abc`)
- `bootstrap/setup.sh` — tokenizer head job + head check + cfg_scale patch apply
- `bootstrap/yue2_cfg_scale.patch` — new; wires `cfg_scale` into the YuE2 nodes
- `audition_planner.py` — new; base/checkpoint renderer
- `render_tokens_nar.py` — new; NAR-companion renderer
- `agent_notes/current.md`

**Not** committed: `ComfyUI/` (untracked clone, huge). Push will use
`git -c credential.helper='!gh auth git-credential'` so no git config is changed.
