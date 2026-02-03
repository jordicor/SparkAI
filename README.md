# SparkAI

**An open-source AI chat platform with real-time voice, WhatsApp integration, and a prompt marketplace.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Commits](https://img.shields.io/badge/Commits-576-orange.svg)]()

---

## What is SparkAI?

SparkAI is a production-ready AI chat platform that evolved from a prompt security testing tool into a full-featured marketplace. It supports **7 AI providers**, **15+ models**, **real-time voice calls**, **WhatsApp integration**, and **15 customizable themes**.

Built over 20 months with 576 commits, it's been used daily in production and recently underwent a comprehensive security audit.

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-AI Support** | OpenAI (GPT-4, GPT-5), Anthropic (Claude), Google (Gemini), xAI (Grok), OpenRouter (100+ models) |
| **Real-Time Voice** | ElevenLabs Agents for live voice conversations with AI |
| **WhatsApp Integration** | Chat with AI via text and voice notes through Twilio |
| **15 UI Themes** | Default, Dark, Terminal, Writer, Neumorphism, Frutiger Aero, and more |
| **Prompt Marketplace** | Create prompts as products with custom landing pages |
| **Media Generation** | DALL-E, Ideogram, Gemini images + VEO-3 video generation |
| **Export Options** | PDF and MP3 conversation exports with TTS voices |
| **Multi-Modal Auth** | Magic links, passwords, or both |

---

## The Story

### Origin: Santa Claus AI (2024)

SparkAI started as **TestMyPrompt** - a tool to test the security of AI prompts. The first use case? A [Santa Claus AI](https://github.com/jordicor/santa-claus-is-calling) chatbot that would call children as Santa. The challenge: ensuring Santa never breaks character, never reveals he's AI, and handles any jailbreak attempt.

I needed a platform to log all interactions for "forensic analysis" of where prompts failed. That became TestMyPrompt.

### Evolution: TestMyPrompt to SPARK (July 2024)

As OpenAI launched custom GPTs, I pivoted the project toward a prompt marketplace - a platform where creators could sell AI-powered products with their own landing pages.

During the early development phase, my nephew helped with testing, code reviews, and learning the ropes of Python backend development. Those few months of collaboration helped shape the foundation of what would become SPARK.

The development was intense: long coding days and extensive security auditing. There were breaks along the way (life happens), but the vision remained.

### Today: Open Source

After using SparkAI daily for personal projects, I'm releasing it as open source. It's 100% functional - every feature works. The codebase shows its history: some parts are intentionally over-engineered (exploring patterns, testing limits), some evolved organically. It's not a tutorial project - it's a production system with 576 commits of real development.

---

## Quick Start

### Prerequisites

- Python 3.10+
- Redis (optional, for caching)
- SQLite (included)

### Installation

```bash
# Clone the repository
git clone https://github.com/jordicor/sparkAI.git
cd sparkAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# At minimum: OPENAI_KEY, PEPPER, APP_SECRET_KEY

# Initialize database
python init_db.py

# Run the application
python app.py
```

The application will be available at `http://localhost:7789`.

### Required Environment Variables

```bash
# Core (required)
APP_SECRET_KEY=your-secret-key-here
PEPPER=your-pepper-for-password-hashing

# AI Providers (at least one)
OPENAI_KEY=sk-...
ANTHROPIC_KEY=sk-ant-...
GOOGLE_AI_KEY=AI...
XAI_API_KEY=xai-...
OPENROUTER_API_KEY=sk-or-...

# Voice (optional)
ELEVENLABS_KEY=...
DEEPGRAM_KEY=...

# WhatsApp (optional)
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
```

See `.env.example` for all available options.

---

## Architecture

SparkAI uses a **FastAPI + Flask hybrid** architecture:

```
                    ┌─────────────────────────────────────┐
                    │           Cloudflare CDN            │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │          Nginx Reverse Proxy        │
                    │    (auth_request for file access)   │
                    └─────────────────┬───────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
┌─────────▼─────────┐   ┌─────────────▼─────────────┐   ┌─────────▼─────────┐
│    FastAPI        │   │      WebSocket Server     │   │   Background      │
│  (REST + SSE)     │   │   (Real-time + TTS)       │   │  (Dramatiq)       │
└─────────┬─────────┘   └─────────────┬─────────────┘   └─────────┬─────────┘
          │                           │                           │
          └───────────────────────────┼───────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │      SQLite (WAL mode, pooled)      │
                    └─────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI + Flask (templating) |
| Database | SQLite with aiosqlite, WAL mode |
| Real-time | WebSockets, Server-Sent Events |
| Caching | Redis (optional, in-memory fallback) |
| Background | Dramatiq + Redis |
| File Storage | Hash-based directories (scalable to millions) |
| Authentication | JWT in httponly cookies, bcrypt + pepper |

For detailed architecture, see [docs/audit/ARCHITECTURE.md](docs/audit/ARCHITECTURE.md).

---

## AI Providers

SparkAI integrates with multiple AI providers through a unified interface:

| Provider | Models | Features |
|----------|--------|----------|
| **OpenAI** | GPT-4, GPT-4o, O1, GPT-5 | Chat, images (DALL-E), TTS |
| **Anthropic** | Claude 3.5 Sonnet, Haiku | Chat, thinking tokens, caching |
| **Google** | Gemini 2.5 | Chat, images, video (VEO-3) |
| **xAI** | Grok-2, Grok-4 | Chat |
| **OpenRouter** | 100+ models | Multi-provider access |
| **ElevenLabs** | TTS, real-time voice | Voice calls, transcription |
| **Deepgram** | Nova-2 | Speech-to-text |

---

## Features in Detail

### Real-Time Voice Calls

Chat with AI using your voice through ElevenLabs Agents. The system maintains conversation context and saves transcripts to your chat history.

### WhatsApp Integration

Connect SparkAI to WhatsApp via Twilio:
- Send text messages to AI
- Send voice notes (transcribed automatically)
- Receive text or audio responses

### Themes

15 themes with full CSS customization:

| Theme | Style |
|-------|-------|
| Default | Clean, professional |
| Dark | Dark mode |
| Terminal | Retro terminal |
| Writer | Distraction-free |
| Coder | Syntax-highlighted |
| Katari Shoji | Elegant gold/navy |
| Neumorphism | Soft shadows |
| Frutiger Aero | 2000s aesthetic |
| Memphis | Bold geometric |
| E-ink | Paper-like |
| Halloween, Christmas, Valentine's | Seasonal |

### Prompt Marketplace

Each prompt can have:
- Custom landing page
- Individual subdomain
- Image/description
- Voice configuration
- Access control (public/private)

### Media Generation

- **Images**: DALL-E, Ideogram, Gemini
- **Video**: Google VEO-3
- **Export**: PDF and MP3 with TTS voices

### AI Landing Page Wizard (Optional)

Generate and modify landing pages for your prompts using AI. The wizard uses Claude Code to create mobile-responsive pages with Tailwind CSS.

**Requirements:**
- Node.js 18+
- Claude Code CLI

**Installation:**
```bash
# Install Claude Code (see https://docs.anthropic.com/en/docs/claude-code/getting-started)
# On Windows (PowerShell as Admin):
irm https://claude.ai/install.ps1 | iex

# On macOS/Linux:
curl -fsSL https://claude.ai/install.sh | sh

# Verify installation
claude --version

# Configure (run once, follow prompts for API key)
claude
```

**Three operating modes:**
- **Create new**: Generate a complete landing page from scratch with the wizard (description, style, colors)
- **Modify existing**: Give Claude free-form instructions to improve your landing ("add FAQ section", "translate to English")
- **Start fresh**: Delete all files and generate from scratch

**Features:**
- 4 visual styles: Modern, Minimalist, Corporate, Creative
- Custom brand colors
- Multi-language support (English, Spanish)
- Auto-generated hero, features, testimonials, and CTA sections
- Iterative improvements via modify mode

---

## Project Stats

| Metric | Value |
|--------|-------|
| **Total Commits** | 576 |
| **Development Period** | May 2024 - January 2026 |
| **AI Providers** | 7 |
| **AI Models Supported** | 15+ |
| **UI Themes** | 15 |
| **Security Audit** | December 2025 (16 issues fixed) |

### Development Eras

**TestMyPrompt Era** (May - July 2024)
- 83 commits
- Foundation: Flask to FastAPI migration, JWT auth, WebSockets
- Use case: Prompt security testing

**SPARK Era** (July 2024 - Present)
- 493 commits
- Evolution: Marketplace, themes, real-time voice, video generation
- Vision: AI prompt marketplace platform

---

## Security

SparkAI underwent a comprehensive security audit in December 2025:

- **21 issues identified, 16 fixed**
- Path traversal prevention with `Path.is_relative_to()`
- JWT in httponly cookies (XSS protection)
- bcrypt + pepper for passwords
- Parameterized SQL queries everywhere
- SSRF protection for media URLs
- Rate limiting with Redis
- Decompression bomb protection

See the full audit in [docs/audit/](docs/audit/).

### Admin Access & Privacy

Platform administrators can access user conversations for:
- **Support**: Helping users with issues
- **Moderation**: Enforcing content policies
- **Legal compliance**: Responding to valid legal requests

**All admin access to user data is logged** in the `ADMIN_AUDIT_LOG` table, which records:
- Who accessed (admin ID)
- What was accessed (conversation/resource ID)
- When (timestamp)
- From where (IP address)

This audit trail ensures transparency and accountability. If you deploy SparkAI, ensure your Privacy Policy informs users about admin access capabilities.

---

## Configuration

### Nginx (Optional)

For production deployments with authenticated file serving, see [nginx/README.md](nginx/README.md).

**Windows users:** [Laragon](https://laragon.org/) provides an easy way to run Nginx and Redis locally with minimal configuration.

### Redis (Optional)

If Redis is available, SparkAI uses it for:
- Rate limiting
- Image token caching
- Background task queues

Without Redis, the system falls back to in-memory caching.

**Installation:**
- **Linux/Mac:** `apt install redis` or `brew install redis`
- **Windows:** Use [Laragon](https://laragon.org/) (includes Redis) or [Memurai](https://www.memurai.com/)

---

## Development

### Running in Development

```bash
python app.py
```

### Project Structure

```
sparkAI/
├── app.py              # Main application (FastAPI + Flask)
├── auth.py             # Authentication (JWT, magic links)
├── database.py         # Async SQLite operations
├── ai_calls.py         # AI provider integrations
├── common.py           # Configuration, utilities
├── prompts.py          # Marketplace logic
├── data/
│   ├── static/         # CSS, JS, images
│   │   ├── css/chat/   # Theme files
│   │   └── js/chat/    # Frontend modules
│   ├── users/          # User directories (hash-based)
│   └── Spark.db        # SQLite database
├── templates/          # Jinja2 templates
├── tools/              # PDF, MP3, video generation
├── nginx/              # Nginx configuration examples
└── docs/audit/         # Security audit documents
```

---

## Contributing

Contributions are welcome! Areas where help is appreciated:

- [ ] **Testing**: No automated tests yet (biggest gap)
- [ ] **Type hints**: Partial coverage, needs mypy
- [ ] **Documentation**: API docs beyond auto-gen
- [ ] **Themes**: New theme ideas

### Before Contributing

1. Read the [ARCHITECTURE.md](docs/audit/ARCHITECTURE.md)
2. Check existing issues
3. For large changes, open an issue first

---

## About the Repository

This repository is published without git history. The original development history (576 commits) remains in the local repository, but was not included in the public release to avoid potential exposure of API keys, internal comments, or other sensitive strings that may have been committed during 20 months of active development.

---

## Credits

**Developed by:** [Jordi Cor](https://github.com/jordicor)

**Early contributor:** My nephew, who helped during the TestMyPrompt era with testing, code reviews, and backend development.

**Built with AI assistance:** Claude, GPT, and Codex - the "vibe coding" approach where AI writes for speed and humans review for quality.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Related Projects

- [Santa Claus AI](https://github.com/jordicor/santa-claus-is-calling) - The project that started it all

---

> **Note:** This README is a work in progress. Contributors section will be updated as confirmations are received, and more storytelling about the development journey may be added.

---

*SparkAI: From testing Santa Claus prompts to a full AI marketplace.*
