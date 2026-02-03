# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
- **Start application**: `python app.py` or `p.bat`
- **Development environment setup**: Use `start_cmd.bat` to activate the SantaGPT2 conda environment

### Database Operations
- **Database path**: `data/Spark.db` (main SQLite database)
- **Initialize database**: `python init_db.py`
- **Database migrations**: `python migration.py` or `python migration_chat_folders.py`
- **Authentication migrations**: `python migration_authentication_modes.py`
- **Test database setup**: `python init_testDB.py`

### Utilities
- **Clear audio cache**: `python clear-audio-cache.py`
- **Transfer data**: `python transferdata.py`

## Architecture Overview

This is a Flask/FastAPI hybrid web application that provides an AI chat interface with multiple language models and a **SAAS Marketplace Platform** where users can create and sell AI prompts as individual products with custom landing pages.

### Core Architecture
- **Backend Framework**: FastAPI with Flask integration for templating
- **Database**: SQLite with async operations (aiosqlite)
- **WebSocket**: Real-time communication using FastAPI WebSocket
- **Authentication**: JWT-based authentication system
- **File Storage**: Local file system with organized directory structure

### Key Components

#### Main Application (`app.py`)
- FastAPI application with Flask template integration
- WebSocket connection management
- Static file serving and routing

#### Database Layer (`database.py`, `models.py`)
- Async SQLite operations with connection pooling
- Database models and schema management
- WAL mode optimization for concurrent access

#### AI Integration (`ai_calls.py`)
#### Real-Time Voice Calls (ElevenLabs)
- **Frontend overlay**: `data/static/js/chat/voice-call.js` mounts the voice call UI and wires SDK events to Spark. Key helpers include `fetchConfig()` (pulls `/api/conversations/{id}/elevenlabs/config`), `startCall()` / `handleConnected()` (bootstraps ConvAI sessions), `stopCall()` and `completeSession()` (requests `/elevenlabs/complete` then triggers the chat refresh), plus state management to lock the text input while a call is active.
- **Chat sync**: `data/static/js/chat/chat.js` exposes `refreshActiveConversation()` which resets pagination/abort controllers and reloads the active conversation so new transcript messages appear as soon as the overlay finishes saving.
- **Backend endpoints**: `app.py` provides `/api/conversations/{conversation_id}/elevenlabs/{config|session|complete|stop}` plus `/sdk/elevenlabs-client.js` and admin routes under `/admin/elevenlabs-agents`. Use these when debugging call setup or transcript completion flows.
- **Service layer**: `elevenlabs_service.py` wraps agent lookup, signed URL generation, context building, API polling (`check_conversation_status`, `fetch_full_transcript`), and persistence (`save_transcript_to_db`). It writes transcript turns into the `MESSAGES` table and updates `CONVERSATIONS.elevenlabs_session_id` / `elevenlabs_status`.
- **SDK proxy**: `elevenlabs_sdk_proxy.py` caches the ConvAI client bundle so the frontend can load `/sdk/elevenlabs-client.js` even if CDN access is flaky.
- **Admin tooling**: `templates/admin_elevenlabs.html`, the accompanying handlers in `app.py`, and `migration_elevenlabs_agents.py` let you configure which ElevenLabs agent/voice is mapped to each prompt.

- Multiple AI providers: OpenAI, Anthropic Claude, Google Gemini, xAI, OpenRouter
- Streaming response support
- Token usage tracking and limits

#### Authentication (`auth.py`)
- JWT token-based user authentication  
- Multi-modal authentication system (magic link only, password only, or both)
- User session management and password change permissions
- Protected route decorators

#### Common Utilities (`common.py`)
- Configuration management via environment variables
- Shared constants and helper functions
- Service cost calculations
- User hash generation for directory structure

#### Prompt Marketplace (`prompts.py`)
- SAAS marketplace system for selling AI prompts as products
- Landing page creation and management
- File structure management for prompt assets
- Prompt directory creation and path resolution

### Directory Structure

#### File System Architecture
The platform uses a hash-based directory structure for scalability:

**Users Directory (`data/users/`)**:
```
data/users/
└── {hash_prefix1}/ (first 3 chars of SHA1 hash)
    └── {hash_prefix2}/ (chars 4-7 of SHA1 hash)
        └── {user_hash}/ (full SHA1 hash of username + pepper)
            ├── files/ (conversation files)
            ├── profile/ (user profile data)
            └── prompts/ *** MARKETPLACE ***
                └── {prompt_id_prefix}/ (first 3 digits of prompt ID)
                    └── {prompt_id_suffix}_{sanitized_name}/
                        ├── home.html (main landing page)
                        ├── static/ (CSS, JS, images, audio)
                        └── templates/ (reusable components)
```

**Conversation Files (`data/static/files/`)**:
```
data/static/files/
└── {conversation_id_prefix1}/
    └── {conversation_id_prefix2}/
        ├── img/
        │   ├── bot/ (AI images: SHA1_256.webp, SHA1_fullsize.webp)
        │   └── user/ (User images: SHA1_256.webp, SHA1_fullsize.webp)
        ├── pdf/ (Conversation exports: prompt_name_timestamp.pdf)
        ├── mp3/ (TTS audio exports: prompt_name_timestamp.mp3)
        └── video/
            └── bot/ (AI videos: generated_video_convid_timestamp.mp4)
```

**User Profile Pictures (`data/users/{hash}/profile/`)**:
```
profile/
└── {user_hash}_{prompt_id}_{size}.webp (32, 64, 128, fullsize)
```

**Bot Avatars/Personalities (`data/users/{hash}/prompts/{id}/static/img/`)**:
```
static/img/
└── {prompt_id}_{name}_{timestamp}_{size}.webp (32, 64, 128, fullsize)
    # These are bot personality avatars that appear in webchat
    # Each prompt (personality) can have custom bot avatar
    # Displayed when users chat with that specific prompt/bot
```

#### Static Assets (`data/static/`)
- **CSS**: Unified theme system with 13 themes in `css/themes/` (see Theme System section below)
- **JavaScript**: Modular frontend code organized by feature
- **Images/Audio**: Media files and caching system

#### Templates (`templates/`)
- Jinja2 HTML templates organized by feature
- Admin interfaces for users, prompts, services, and LLMs
- Chat interface with real-time messaging

#### Tools (`tools/`)
- **Multimedia Generation**: AI video, image, PDF, and MP3 generation
- **Video Generation** (`generate_videos.py`): Google VEO-3 integration with local storage
- **Image Generation** (`generate_images.py`): DALL-E, Ideogram, Gemini support
- **PDF Export** (`download_pdf.py`): Complete conversation export to PDF
- **MP3 Export** (`download_mp3.py`): TTS-powered audio export of conversations
- **Background task processing** with Dramatiq
- **File processing utilities** and TTS integration

### Database Schema
- **Database Location**: `data/Spark.db`
- **Users**: User management and multi-modal authentication
- **User_Details**: Extended user settings, authentication modes, and permissions
- **Chats**: Chat conversations and folders  
- **Messages**: Individual chat messages with AI responses
- **Services/LLMs**: AI model configurations and pricing
- **Prompts**: System and user-defined prompts
- **Magic_Links**: Authentication tokens for magic link login

### Key Features
- **AI Chat System**: Multi-model AI chat with streaming responses (OpenAI, Claude, Gemini, xAI)
- **Prompt Marketplace**: SAAS platform where users create and sell AI prompts as individual products
- **Custom Landing Pages**: Each prompt gets its own customizable landing page with full assets
- **File Processing**: Upload and processing (images, audio, PDFs)
- **Text-to-speech integration**: Multiple TTS engines (ElevenLabs, OpenAI)
- **Chat folder organization system**: Drag-and-drop folder management
- **Multi-modal authentication**: Magic link, password, or both authentication modes
- **Granular permissions**: User roles and password change controls
- **Real-time WebSocket communication**: Live chat and updates
- **Scalable file system**: Hash-based directory structure for performance

### Environment Configuration
The application uses environment variables loaded from `.env` file for:
- API keys for various AI services
- Database configuration
- Security settings (JWT secrets, pepper for hashing)
- Feature flags (moderation API, semantic routing)
- Service limits (token limits, message size limits)

### Development Notes
- The codebase uses both sync and async patterns
- Database operations are primarily async using aiosqlite
- Static files are served directly by FastAPI
- Template rendering uses Flask's Jinja2 integration
- WebSocket connections are managed through a ConnectionManager class
- The application supports multiple themes switchable via CSS files
- **Hash-based file system**: Uses SHA1 hashing for scalable user directory structure
- **Marketplace architecture**: Each prompt is an independent product with full customization
- **Dual storage system**: Database for AI prompts, filesystem for marketplace landing pages

## Key Functions Quick Reference

### Session & Authentication
- **Session duration**: `auth.py:27` - `ACCESS_TOKEN_EXPIRE_MINUTES` (JWT token expiry)
- **User inactivity timeout**: `session-utils.js:14` - `_inactivityTimeout` (frontend activity detection)
- **Session validation**: `session-utils.js` - `SessionManager.validateSession()`
- **User authentication**: `auth.py` - `get_current_user()`, `create_access_token()`

### Chat Operations
- **Create new chat**: `chat.js` - `startNewConversation()`
- **Send message**: `chat.js` - `sendMessage()`
- **Load conversations**: `chat.js` - `loadConversations()`
- **Message handling**: `chat.js` - `addMessage()`

### WebSocket & Real-time Communication
- **WebSocket manager**: `models.py` - `ConnectionManager` (connect, disconnect, send_json, send_bytes)
- **WebSocket endpoint**: `app.py` - `/ws/{conversation_id}` endpoint for real-time chat communication

### Database Operations
- **Database location**: `data/Spark.db` - Main SQLite database file
- **Database connection**: `database.py` - `get_db_connection()` (async connection with readonly option)
- **User queries**: `auth.py` - `get_user_by_username()`, `get_user_by_id()`, `get_user_from_phone_number()`

### AI Model Integration
- **AI API calls**: `ai_calls.py` - `call_gpt_api()`, `call_claude_api()`, `call_gemini_api()`, `call_xai_api()`
- **Streaming responses**: `ai_calls.py` - `stream_response()` function
- **AI call router**: `ai_calls.py` - `/ai_call` endpoint

### File Handling & Upload
- **File upload handling**: `fileHandling.js` - `handleFileSelect()`, `processFiles()`
- **Image processing**: `save_images.py` - `save_image_locally()`, `resize_image()`, `generate_img_token()`
- **Image access**: `get_image.py` - JWT token-based image access system

### Prompt Marketplace System
- **User directory management**: `prompts.py` - `get_user_directory()`, `create_prompt_directory()`
- **Prompt path resolution**: `prompts.py` - `get_prompt_path()`, `get_prompt_templates_dir()`
- **Landing page creation**: Each prompt gets `home.html` and full static assets
- **Hash-based structure**: Uses SHA1(username + pepper) for scalable directory organization
- **Prompt ID formatting**: 7-digit zero-padded IDs (e.g., 12 becomes 0000012)
- **Asset management**: Complete CSS, JS, images, audio support per prompt
- **Database separation**: Prompt text (for AI) stored in DB, landing pages in filesystem

### User & Admin Management
- **User creation**: `app.py` - `add_user()`, `/create-user` endpoints
- **User deletion**: `app.py` - `delete_user()`, `delete_selected_users()`
- **Admin validation**: `app.py` - `is_admin()` function
- **Role management**: `models.py` - User class with role properties (`is_admin`, `is_manager`)

### Authentication & Security
- **Multi-modal authentication**: Users can use magic links, passwords, or both
- **Authentication modes**: `magic_link_only`, `password_only`, `magic_link_password`
- **Password change permissions**: `can_change_password` field controls user permissions
- **Magic link generation**: `auth.py` - `generate_magic_link()`
- **JWT token creation**: `auth.py` - `create_access_token()`
- **Password handling**: `auth.py` - `hash_password()`, `verify_password()`
- **User model methods**: `can_use_magic_link()`, `can_use_password()`, `should_show_change_password()`

### Background Tasks & Processing
- **Task management**: `tools/dramatiq_tasks.py` - Background task processing
- **TTS processing**: `tools/tts.py` - Text-to-speech functionality
- **Image generation**: `tools/generate_images.py` - AI image generation

### Frontend File Structure

#### Chat Interface Components
- **Main chat HTML**: `templates/chat/chat.html` - Chat interface template with sidebar, folders, and message areas
- **Chat JavaScript files** (`data/static/js/chat/`):
  - `chat.js` - Main chat functionality (messages, conversations, WebSocket)
  - `folders.js` - Folder management system (create, edit, delete, drag-drop)
  - `session-utils.js` - Session validation and timeout handling
  - `fileHandling.js` - File upload and image preview functionality
  - `audio.js` - Audio recording and playback
  - `config.js` - Configuration settings
  - `utils.js` - Utility functions
  - `main.js` - Main initialization

#### Theme System (Unified)

The application uses a **unified theme system** where all pages (except chat) share a single CSS file per theme. All 13 themes are **WCAG AA compliant** for contrast ratios.

**Directory Structure** (`data/static/css/`):
```
css/
├── themes/              # Unified theme system (13 themes)
│   ├── default.css      # Discord-like dark theme
│   ├── light.css        # Bootstrap light theme
│   ├── coder.css        # VS Code Dark+ theme
│   ├── terminal.css     # Hacker terminal theme
│   ├── writer.css       # Warm paper theme
│   ├── halloween.css    # Spooky orange theme
│   ├── xmas.css         # Festive red/green theme
│   ├── valentinesday.css# Romantic pink theme
│   ├── memphis.css      # 80s postmodern theme
│   ├── neumorphism.css  # Soft UI theme
│   ├── frutigeraero.css # Aero 2000s glassy theme
│   ├── eink.css         # E-reader black/white theme
│   ├── katarishoji.css  # Elegant gold/navy theme
│   └── _template.css    # Template for new themes
├── chat/                # Chat-specific themes (complex layout)
│   ├── chat-default.css
│   ├── chat-light.css
│   └── ... (13 theme files)
├── common.css           # Shared CSS variables and base styles
├── legacy/              # Old CSS files (for rollback, ~198 files)
└── *.css                # Other base files (skeleton.css, style.css)
```

**Theme Loading**: `theme-loader.js` automatically loads the correct theme:
- Admin/general pages → `themes/{theme}.css`
- Chat pages → `chat/chat-{theme}.css`

**CSS Variables**: All themes use standardized variables defined in each theme file:
```css
:root {
    --bg-primary, --bg-secondary, --bg-tertiary    /* Backgrounds */
    --text-primary, --text-secondary, --text-muted /* Text colors */
    --accent, --accent-hover, --accent-light       /* Brand color */
    --success, --danger, --warning, --info         /* Status colors */
    --border-color, --shadow-soft, --shadow-hover  /* Structure */
}
```

**Adding a New Theme**: Copy `_template.css`, update CSS variables, save as `{theme-name}.css`

#### Folder System Components
- **Folder section HTML**: Located in `templates/chat/chat.html` - Look for "folders-section" class
- **Folder modals HTML**: Located in `templates/chat/chat.html` - Look for "folderModal" and "moveChatModal" IDs
- **Folder JavaScript**: `data/static/js/chat/folders.js` - Complete folder functionality
- **Folder CSS**: Each chat theme file contains folder styling in the "Sistema de Carpetas (Folders)" section
- **Folder API endpoints**: 
  - `/api/chat-folders` - GET/POST folders
  - `/api/chat-folders/{id}` - PUT/DELETE specific folder
  - `/api/conversations/{id}/move-to-folder` - Move chat to folder

### User Platform & Encoding Considerations
- The user is using Claude Code under Windows 10 with CMD terminal
- **CRITICAL**: Windows CMD uses cp1252 encoding which CANNOT display:
  - Unicode characters (accented letters display as ??????)
  - Special symbols (★ ▲ ◆ etc. cause encoding errors)
  - Emojis (complete failure with charmap codec errors)
- **Code Guidelines**:
  - NEVER use emojis or unicode symbols in print statements or CLI output
  - Use only ASCII characters for terminal output
  - Use plain ASCII alternatives: * instead of ★, -> instead of →, etc.
  - For web interface, unicode/emojis are fine as browsers handle UTF-8
  - Test any CLI output with ASCII-only characters
- **Safe Characters**: Basic ASCII (a-z, A-Z, 0-9), standard punctuation (.,;:!?-+=*/<>[]{}())

- Always write your code, variables, and comments in US English. Talk to the user in Spanish from Spain.
