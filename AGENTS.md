# Repository Guidelines

## Project Structure & Module Organization
pp.py hosts the FastAPI core plus embedded Flask templating. Supporting services (i_calls.py, uth.py, common.py, prompts.py) live beside it; create new domains as sibling modules rather than inflating pp.py. HTML lives in 	emplates/ with feature folders (	emplates/chat, 	emplates/prompts), while hashed user assets, prompt landing pages, and SQLite files sit under data/ (notably data/Spark.db, data/users/<hash>/, data/static/). Automation scripts reside in 	ools/.

## Architecture & Key Components
The product is a FastAPI/Flask hybrid that powers an AI chat platform and SaaS prompt marketplace. Expect async SQLite (iosqlite), JWT auth, WebSocket chat, and per-prompt landing pages created via prompts.py. Keep marketplace assets aligned with the hash-based directory layout and reuse helpers like get_prompt_path().

## Build, Test, and Development Commands
- start_cmd.bat loads the SantaGPT2 conda environment; p.bat or python app.py starts dual-mode production.
- python app.py dev|tunnel|https selects HTTP, tunnel, or HTTPS-only startup (see 	est_server_modes.py).
- python -m venv .venv then .\.venv\Scripts\activate if you prefer virtualenv + pip install -r requirements.txt.
- python init_db.py seeds a fresh DB; python migration.py, python migration_chat_folders.py, or python migration_authentication_modes.py apply schema changes.
- python clear-audio-cache.py or python transferdata.py handle maintenance tasks.

## Coding Style & Naming Conventions
Target Python 3.11, four-space indentation, and PEP 8 spacing. Use snake_case for modules/functions, PascalCase for classes, and hyphenated Jinja filenames (index-stats.html). CLI output must remain ASCII-only to avoid Windows encoding failures. Favor type hints on async endpoints and commit shared helpers to common.py for reuse.

## Testing Guidelines
python test_server_modes.py performs smoke tests across startup modes after running python init_testDB.py. Broader coverage should live in a 	ests/ package using pytest or unittest with names like 	est_<feature>_<scenario>. Stub outbound AI calls with fakes or cassettes to keep tests deterministic.

## Commit & Pull Request Guidelines
History prefers present-tense, scoped commits (e.g., Update documentation with complete file structure). Document DB migrations, media asset updates, and environment impacts in PRs, link tracking issues, and include screenshots for template or CSS changes. Request review from the service owner when altering marketplace flows, 	emplates/, or 	ools/ scripts.

## Environment & Secrets
Keep credentials in .env and SSL keys in data/static/sec/. Do not commit regenerated SQLite binaries—produce migrations instead. Share rotation plans with Ops and scrub secrets before uploading logs.
## Real-Time Voice Calls (ElevenLabs)

- **Frontend overlay**: `data/static/js/chat/voice-call.js` controls the call panel (`fetchConfig`, `startCall`, `stopCall`, `completeSession`) and posts to the ElevenLabs API endpoints. It also refreshes the chat log once transcripts are saved.
- **Chat refresh helper**: `data/static/js/chat/chat.js` exposes `refreshActiveConversation()` which resets pagination and reloads the current conversation after a call.
- **Backend endpoints**: `app.py` defines `/api/conversations/{id}/elevenlabs/config`, `/session`, `/complete`, and `/stop`, plus the SDK proxy routes (`/sdk/elevenlabs-client.js*`) and the admin views under `/admin/elevenlabs-agents`.
- **Service layer**: `elevenlabs_service.py` owns configuration resolution, session status updates, transcript retrieval, and persistence (`save_transcript_to_db`). It relies on `tools/tts_load_balancer.get_elevenlabs_key()` for API keys and writes transcript turns into `MESSAGES`.
- **SDK caching**: `elevenlabs_sdk_proxy.py` downloads and caches the ElevenLabs ConvAI bundle so the frontend can load it from our domain.
- **Admin configuration**: `templates/admin_elevenlabs.html` plus the related handlers in `app.py` manage agent metadata, while `migration_elevenlabs_agents.py` seeds the `ELEVENLABS_AGENTS` table and adds `elevenlabs_session_id` / `elevenlabs_status` to `CONVERSATIONS`.

## CLI Editing Tips
- When PowerShell quoting or heredocs get in the way of multiline edits, spin up a temporary Python script to apply the change (write file, run once, delete). It keeps replacements predictable and avoids escaping headaches.
