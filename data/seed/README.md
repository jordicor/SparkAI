# SparkAI Database Seed

This directory contains the seed data for initializing a fresh SparkAI installation.

## Contents

```
data/seed/
├── __init__.py          # Module init
├── seed_data.py         # Main seed script
├── README.md            # This file
├── prompts/             # Prompt text files (system prompts)
│   ├── spark.txt
│   ├── writer.txt
│   ├── coder.txt
│   ├── tutor.txt
│   ├── cole.txt
│   ├── creative.txt
│   ├── coach.txt
│   ├── nova_orion.txt
│   ├── ava.txt
│   ├── discursoman.txt
│   └── agente_chillon.txt
├── images/              # Prompt avatar images (4 sizes each)
│   ├── spark/
│   ├── writer/
│   ├── ...
│   └── agente_chillon/
└── landings/            # Landing page templates for each prompt
    ├── spark/
    │   ├── home.html    # Main landing page
    │   └── static/      # CSS, JS, additional images
    ├── writer/
    ├── coder/
    ├── tutor/
    ├── cole/
    ├── creative/
    ├── coach/
    ├── nova_orion/
    ├── ava/
    ├── discursoman/
    └── agente_chillon/
```

## Usage

### Prerequisites

1. Make sure the database schema exists. Run `init_db.py` if needed:
   ```bash
   python init_db.py
   ```

### Running the Seed

From the project root directory:

```bash
python -m data.seed.seed_data
```

Or:

```bash
python data/seed/seed_data.py
```

### Force Re-seed

If the database already has seed data and you want to re-seed:

```bash
python -m data.seed.seed_data --force
```

**Warning:** This will create duplicate data. Only use `--force` on a fresh database.

## What Gets Created

The seed script creates:

1. **User Roles** (3)
   - admin
   - manager
   - user

2. **Services** (6)
   - TTS-ELEVENLABS
   - TTS-OPENAI
   - STT
   - STT-ELEVENLABS
   - STT-DEEPGRAM
   - DALLE-2-256

3. **Voices** (28)
   - 2 OpenAI voices (Onyx, Fable)
   - 26 ElevenLabs voices

4. **LLM Models** (9)
   - 3 GPT models (gpt-4o-mini, gpt-4o, gpt-4-turbo)
   - 3 Claude models (haiku, sonnet, opus)
   - 2 Gemini models (flash, pro)
   - 1 xAI model (grok)

5. **Admin User**
   - Username: `admin`
   - Email: `admin@sparkai.local`
   - Auth: Magic link only (no password)
   - Full access to all features

6. **Prompts** (11) - Each with avatar images AND landing pages
   - Spark (general assistant)
   - Writer (writing assistant)
   - Coder (coding assistant)
   - Tutor (teaching assistant)
   - Cole (friend companion)
   - Creative (brainstorming)
   - Coach (personal development)
   - Nova-Orion (thinking partner)
   - AVA (dimensional traveler)
   - DiscursoMan (speech writer)
   - Agente Chillon (therapeutic anger release)

7. **Landing Pages** (11)
   - Each prompt gets a complete landing page (home.html + static assets)
   - Landing pages are mobile-responsive and ready to use
   - Accessible at `/p/{public_id}/{slug}/` after seed

## Customization

### Modifying Seed Data

Edit `seed_data.py` to:
- Change the admin user credentials
- Add/remove voices
- Add/remove LLM models
- Modify prompt configurations

### Adding New Prompts

1. Create a `.txt` file in `prompts/` with the prompt text
2. Add images to `images/{prompt_name}/` (4 sizes: 32, 64, 128, fullsize)
3. (Optional) Add landing page to `landings/{prompt_name}/` with:
   - `home.html` - Main landing page
   - `static/` - CSS, JS, images
4. Add an entry to the `PROMPTS` list in `seed_data.py`

**Tip:** Use the Landing Page Wizard in the admin panel to generate landing pages automatically with AI assistance.

### Changing Voice Assignments

Edit the `voice_id` field in the `PROMPTS` list in `seed_data.py`.

## Notes

- The seed creates a deterministic user hash based on the admin username
- Prompt IDs are fixed (1-11) for consistency
- Voice IDs are preserved from the original database
- The seed is idempotent for roles, services, voices, and LLMs (uses INSERT OR IGNORE)
