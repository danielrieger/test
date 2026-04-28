# Implementation Plan — IMP Repository Fork

## Goal

Get a personal copy of the [IMP (Integrative Modeling Platform)](https://github.com/salilab/imp) codebase under `danielrieger/imp` for code analysis.

## License — No Issues

> [!TIP]
> IMP is licensed under the **GNU LGPL** (Lesser General Public License). This explicitly permits:
> - Forking and hosting your own copy
> - Reading, analyzing, and modifying the source code
> - Using it in your own (even proprietary) projects, as long as modifications *to IMP itself* remain LGPL
>
> **There are zero terms-of-use concerns with forking this repository.**

## Approach: GitHub Fork vs. Bare-Clone Mirror

| | **GitHub Fork** | **Bare-Clone Mirror** |
|---|---|---|
| Speed | Instant (server-side) | ~10 min (450 MB download + upload) |
| Disk usage | 0 bytes locally | ~450 MB temp |
| Upstream sync | Built-in "Sync fork" button | Manual |
| Full history | ✅ | ✅ |
| Independent repo | Linked to upstream | Fully independent |

> [!IMPORTANT]
> **Recommendation**: Use **GitHub Fork**. It's instant, requires no local disk space, preserves full history, and gives you a "Sync fork" button to pull upstream updates. The only difference is that GitHub labels it as a fork of `salilab/imp` — but this is purely cosmetic for your purposes.

## Proposed Steps

1. **Browser**: Navigate to `https://github.com/salilab/imp`
2. **Browser**: Click the **"Fork"** button
3. **Browser**: Set repository name to `imp`, owner to `danielrieger`, confirm
4. **Verify**: Check that `https://github.com/danielrieger/imp` exists

## Open Questions

> [!NOTE]
> If you prefer a **fully independent** copy (no "forked from salilab/imp" label), I can do the bare-clone mirror approach instead. Just let me know.
