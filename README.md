# migaku-notion

GitHub fork of [khatibomar/migoku](https://github.com/khatibomar/migoku): the repo root is the same Go migoku service (Docker, `localhost` REST API). The `sync/` directory adds a Python tool that mirrors your [Migaku](https://migaku.com) vocabulary into a Notion database (SQLite diff cache, pinyin for Mandarin).

This fork keeps upstream history and the fork network so you can open PRs to migoku while shipping the Notion workflow in one place.

Pure-Python successor (no Docker, no migoku): [migaku-notion-v2](https://github.com/gfsincere/migaku-notion-v2). v1 here still fits if you run migoku for reads.

## Install

```powershell
git clone https://github.com/gfsincere/migaku-notion.git
cd migaku-notion
docker compose up -d

cd sync
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python sync.py setup
python sync.py status
python sync.py sync --dry-run
python sync.py sync
```

You need: Migaku account, Notion internal integration connected to a parent page, Docker, Python 3.11+, Git. `setup` writes `sync/.env` (do not commit secrets).

## Fork-specific fixes

These are applied on top of upstream in this fork:

| Change | Reason |
|--------|--------|
| `Dockerfile` | Removed invalid `COPY ... /app/example` (folder is `examples/`). |
| `service.go` | Words cache key includes `limit` and `offset` so pagination is not wrong. |

Contributions that belong in core migoku should go to [khatibomar/migoku](https://github.com/khatibomar/migoku) as PRs when possible.

## What it does

1. Pull words from migoku (language and status filters).
2. Add tone-marked and numeric pinyin for Mandarin via `pypinyin`.
3. Merge fail rates from migoku difficult-words API.
4. Upsert rows in your Notion database.
5. Keep `sync/state.db` so only changed rows get Notion PATCHes on the next run.

Meaning column: yours to fill (e.g. Notion AI); sync does not overwrite it.

## Architecture

```
                 Migaku (account data)
                        │
                        │  migoku logs in, downloads SRS DB
                        ▼
                 migoku (Go, this repo root, Docker)
                 localhost:8080
                        │
                        │  REST: words, difficult, ...
                        ▼
              sync/sync.py (Python)
                        │
                        │  pinyin, merge stats, diff state.db
                        ▼
              Notion "Migaku Vocab" DB
```

## Commands

```powershell
cd sync
python sync.py status
python sync.py sync
python sync.py sync --full-refresh
python sync.py sync --status ALL
python sync.py sync --dry-run
python sync.py rebuild-cache
python sync.py chars
python sync.py chars --list
python sync.py export --csv vocab.csv
python sync.py export --xlsx vocab.xlsx
```

## Notion schema

| Property | Type | Notes |
|----------|------|--------|
| Word | Title | dictForm |
| Pinyin | Rich text | pypinyin for zh |
| Pinyin (numeric) | Rich text | zh |
| Meaning | Rich text | not overwritten by sync |
| Status | Select | KNOWN, LEARNING, etc. |
| Fail rate % | Number | from difficult list when present |
| Total reviews | Number | |
| Failed reviews | Number | |
| Part of speech | Rich text | often from difficult list only |
| Language | Select | |
| Last synced | Date | |
| Migaku key | Rich text | lang\|dictForm\|secondary |
| Sense # | Rich text | zh sense index |

## Limitations

- migoku exposes words and stats, not full dictionary glosses in the list endpoint; Meaning stays in Notion.
- Notion rate limit (~3 rps): first full sync of thousands of rows takes a while; later runs are fast thanks to `state.db`.
- Migaku may change endpoints; v2 targets the newer API. v1 read path depends on what migoku still supports.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Docker build fails on `example` path | Use this fork or apply the Dockerfile fix locally. |
| About half of words missing on sync | Use this fork or apply the words cache key fix in `service.go`. |
| Notion 404 on database | Connect the integration to the parent page (Connections). |
| Notion 401 | Check `NOTION_TOKEN` in `sync/.env`. |

## Credits

- [khatibomar/migoku](https://github.com/khatibomar/migoku)
- [Migaku](https://migaku.com)
- [pypinyin](https://github.com/mozillazg/python-pinyin)
- [Notion](https://notion.so)

## Support

[ko-fi.com/blacktonystark](https://ko-fi.com/blacktonystark)

## License

MIT.
