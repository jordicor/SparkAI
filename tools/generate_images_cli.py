#!/usr/bin/env python3
"""
Standalone CLI for image generation.

This script can be called independently without the full SPARK application context.
Useful for testing, scripting, and integration with other tools like Claude Code.

Examples:
    # Simple generation
    python generate_images_cli.py --prompt "A blue cat" --output cat.png

    # With POE/FLUX and references
    python generate_images_cli.py --prompt "Similar style" --engine poe --refs ref1.png --output result.png

    # With Gemini 3 Pro (best quality)
    python generate_images_cli.py --prompt "Photo" --engine gemini --model gemini-3-pro-image-preview -o photo.png
"""

import os
import sys
import orjson
import asyncio
import aiohttp
import argparse
import base64
import io
import re
from pathlib import Path
from typing import Optional, Union
from dotenv import load_dotenv

# Load environment from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# =============================================================================
# API KEYS
# =============================================================================
OPENAI_API_KEY = os.getenv('OPENAI_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_KEY')
POE_API_KEY = os.getenv('POE_API_KEY')

# =============================================================================
# CONFIGURATION
# =============================================================================
IMAGE_GENERATION_ENGINE = os.getenv('IMAGE_GENERATION_ENGINE', 'nano-banana').lower()
IMAGE_GENERATION_TIMEOUT = int(os.getenv('IMAGE_GENERATION_TIMEOUT', 120))

GEMINI_IMAGE_MODEL = os.getenv('GEMINI_IMAGE_MODEL', 'gemini-2.5-flash-image')
OPENAI_IMAGE_MODEL = os.getenv('OPENAI_IMAGE_MODEL', 'gpt-image-1')
POE_IMAGE_MODEL = os.getenv('POE_IMAGE_MODEL', 'flux2pro')

# =============================================================================
# MODEL CAPABILITIES
# =============================================================================
POE_MODEL_MAX_REFS = {
    "nanobananapro": 14,
    "flux2pro": 8,
    "flux2flex": 8,
    "fluxkontextpro": 1,
    "seedream40": 8,
    "Ideogram-v3": 3,
}

GEMINI_MODEL_MAX_REFS = {
    "gemini-3-pro-image-preview": 14,
    "gemini-2.5-flash-image": 3,
}

OPENAI_MODEL_MAX_REFS = {
    "gpt-image-1.5": 16,
    "gpt-image-1": 16,
    "gpt-image-1-mini": 16,
    "dall-e-3": 0,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _prepare_reference_image(image_source: Union[bytes, str, Path], max_size: int = 1024) -> Optional[str]:
    """Prepare a reference image for API consumption."""
    try:
        from PIL import Image

        img = None

        if isinstance(image_source, bytes):
            img = Image.open(io.BytesIO(image_source))
        elif isinstance(image_source, (str, Path)):
            path = Path(image_source)
            if path.exists():
                img = Image.open(path)
            elif isinstance(image_source, str) and image_source.startswith('data:image'):
                header, data = image_source.split(',', 1)
                img = Image.open(io.BytesIO(base64.b64decode(data)))

        if img is None:
            return None

        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size))

        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode()

    except Exception as e:
        print(f"Error preparing reference image: {e}", file=sys.stderr)
        return None


# =============================================================================
# GEMINI
# =============================================================================

async def generate_image_gemini(
    prompt: str,
    ratio: str = "1:1",
    reference_images: list = None,
    model: str = None
) -> tuple:
    """Generate image using Google Gemini with optional references."""
    selected_model = model or GEMINI_IMAGE_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{selected_model}:generateContent"

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    parts = []

    if reference_images:
        max_refs = GEMINI_MODEL_MAX_REFS.get(selected_model, 3)
        refs_added = 0
        for ref in reference_images[:max_refs]:
            b64_data = _prepare_reference_image(ref)
            if b64_data:
                parts.append({
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": b64_data
                    }
                })
                refs_added += 1
        if refs_added > 0:
            print(f"Added {refs_added} reference images to Gemini request", file=sys.stderr)

    ratio_instruction = f" Make sure the image has an aspect ratio of {ratio}." if ratio != "1:1" else ""

    if reference_images and len(reference_images) > 0:
        generation_prompt = f"""Use the reference images above to understand the visual style, characters, or elements to include.

Create a high-quality image: {prompt}{ratio_instruction}

Preserve the identity and appearance of any people shown in the references."""
    else:
        generation_prompt = f"Create a high-quality image: {prompt}{ratio_instruction}"

    parts.append({"text": generation_prompt})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE", "TEXT"]
        }
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers, timeout=IMAGE_GENERATION_TIMEOUT) as response:
            if response.status == 200:
                result = await response.json()

                if result.get('promptFeedback', {}).get('blockReason'):
                    raise Exception(f"Request blocked: {result['promptFeedback']['blockReason']}")

                candidates = result.get('candidates', [])
                if not candidates:
                    raise Exception("No candidates returned from Gemini")

                for part in candidates[0].get('content', {}).get('parts', []):
                    if 'inlineData' in part:
                        image_data = base64.b64decode(part['inlineData']['data'])
                        return image_data, prompt, ratio

                raise Exception("No image data in Gemini response")
            else:
                raise Exception(f"Gemini error: {await response.text()}")


# =============================================================================
# POE / FLUX
# =============================================================================

async def generate_image_poe(
    prompt: str,
    ratio: str = "16:9",
    reference_images: list = None,
    model: str = None
) -> tuple:
    """Generate image using Poe API (FLUX models) with optional references."""
    if not POE_API_KEY:
        raise Exception("POE_API_KEY not configured")

    from openai import OpenAI
    import requests

    client = OpenAI(
        api_key=POE_API_KEY,
        base_url="https://api.poe.com/v1",
    )

    selected_model = model or POE_IMAGE_MODEL
    print(f"Using POE model: {selected_model}", file=sys.stderr)

    content = []

    has_refs = reference_images and len(reference_images) > 0
    if reference_images:
        max_images = POE_MODEL_MAX_REFS.get(selected_model, 8)
        refs_added = 0
        for ref in reference_images[:max_images]:
            b64_data = _prepare_reference_image(ref)
            if b64_data:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}
                })
                refs_added += 1
        if refs_added > 0:
            print(f"Added {refs_added} reference images to POE request", file=sys.stderr)

    if has_refs:
        full_prompt = f"""REFERENCE IMAGES PROVIDED ABOVE - Use them to maintain visual consistency.

SCENE TO CREATE:
{prompt}

REQUIREMENTS:
- Aspect ratio: {ratio}
- High quality, professional image
- If people are shown in references, preserve their exact appearance
- DO NOT include any text in the image"""
    else:
        full_prompt = f"""{prompt}

Requirements:
- Aspect ratio: {ratio}
- High quality, professional image
- DO NOT include any text in the image"""

    content.append({"type": "text", "text": full_prompt})

    response = client.chat.completions.create(
        model=selected_model,
        messages=[{"role": "user", "content": content}],
        stream=False,
        extra_body={"aspect": ratio, "quality": "high"}
    )

    if response.choices and len(response.choices) > 0:
        response_content = response.choices[0].message.content
        image_url = None

        if isinstance(response_content, str):
            match = re.search(r'!\[.*?\]\((https?://[^\)]+)\)', response_content)
            if match:
                image_url = match.group(1)
            elif response_content.startswith('http'):
                image_url = response_content.strip()
            elif response_content.startswith('data:image'):
                image_url = response_content

        if image_url:
            if image_url.startswith('data:image'):
                header, data = image_url.split(',', 1)
                img_bytes = base64.b64decode(data)
            else:
                img_response = requests.get(image_url, timeout=60)
                img_bytes = img_response.content

            return img_bytes, prompt, ratio

    raise Exception("No image in POE response")


# =============================================================================
# OPENAI GPT-IMAGE
# =============================================================================

async def generate_image_openai(
    prompt: str,
    ratio: str = "16:9",
    reference_images: list = None,
    model: str = None
) -> tuple:
    """Generate image using OpenAI gpt-image models."""
    from openai import OpenAI
    import requests

    client = OpenAI(api_key=OPENAI_API_KEY)
    selected_model = model or OPENAI_IMAGE_MODEL

    size_map = {
        "1:1": "1024x1024",
        "16:9": "1536x1024",
        "9:16": "1024x1536",
        "4:3": "1536x1024",
        "3:4": "1024x1536",
    }
    size = size_map.get(ratio, "1536x1024")

    is_gpt_image = selected_model.startswith("gpt-image")

    response = client.images.generate(
        model=selected_model,
        prompt=prompt,
        size=size,
        quality="high" if is_gpt_image else "hd",
        n=1,
    )

    image_data = response.data[0]

    if hasattr(image_data, 'b64_json') and image_data.b64_json:
        img_bytes = base64.b64decode(image_data.b64_json)
    elif hasattr(image_data, 'url') and image_data.url:
        img_response = requests.get(image_data.url, timeout=60)
        img_bytes = img_response.content
    else:
        raise Exception("No image data in OpenAI response")

    return img_bytes, prompt, ratio


# =============================================================================
# UNIFIED GENERATION
# =============================================================================

async def generate_image(
    prompt: str,
    reference_images: list = None,
    engine: str = None,
    model: str = None
) -> tuple:
    """Generate an image using the specified engine."""
    selected_engine = (engine or IMAGE_GENERATION_ENGINE).lower()

    ratio_match = re.search(r'\b(\d+:\d+)\b', prompt)
    if ratio_match:
        ratio = ratio_match.group(1)
        prompt = re.sub(r'\b\d+:\d+\b', '', prompt).strip()
    else:
        ratio = "1:1"

    if selected_engine in ('nano-banana', 'gemini'):
        return await generate_image_gemini(prompt, ratio, reference_images, model)
    elif selected_engine == 'poe':
        return await generate_image_poe(prompt, ratio, reference_images, model)
    elif selected_engine == 'openai':
        return await generate_image_openai(prompt, ratio, reference_images, model)
    else:
        raise ValueError(f"Unsupported engine: {selected_engine}. Use: gemini, poe, or openai")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate images using AI providers (POE, Gemini, OpenAI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_images_cli.py --prompt "A blue cat" --output cat.png
  python generate_images_cli.py --prompt "Landing hero" --engine poe --ratio 16:9 -o hero.png
  python generate_images_cli.py --prompt "Similar style" --engine poe --refs ref.png -o result.png
  python generate_images_cli.py --prompt "Photo" --engine gemini --model gemini-3-pro-image-preview -o photo.png

Engines:
  gemini (default) : gemini-2.5-flash-image, gemini-3-pro-image-preview (up to 14 refs)
  poe              : flux2pro, nanobananapro, flux2flex, seedream40 (up to 14 refs)
  openai           : gpt-image-1, gpt-image-1.5 (up to 16 refs)
"""
    )

    parser.add_argument("--prompt", "-p", required=True, help="Text prompt for image generation")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--engine", "-e", default=None,
                        choices=["gemini", "nano-banana", "poe", "openai"],
                        help=f"Engine (default: {IMAGE_GENERATION_ENGINE})")
    parser.add_argument("--model", "-m", default=None, help="Specific model")
    parser.add_argument("--ratio", "-r", default="1:1",
                        choices=["1:1", "16:9", "9:16", "4:3", "3:4"],
                        help="Aspect ratio (default: 1:1)")
    parser.add_argument("--refs", "--references", nargs="+", default=None,
                        help="Reference image paths")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    parser.add_argument("--json", "-j", action="store_true", help="Output result as JSON")

    args = parser.parse_args()

    async def run():
        prompt = args.prompt
        if args.ratio and args.ratio not in prompt:
            prompt = f"{prompt} {args.ratio}"

        if not args.quiet and not args.json:
            engine_name = args.engine or IMAGE_GENERATION_ENGINE
            print(f"Generating with {engine_name}...", file=sys.stderr)

        image_bytes, revised_prompt, ratio = await generate_image(
            prompt=prompt,
            reference_images=args.refs,
            engine=args.engine,
            model=args.model
        )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            f.write(image_bytes)

        if args.json:
            result = {
                "success": True,
                "output": str(output_path.absolute()),
                "size": len(image_bytes),
                "ratio": ratio
            }
            print(orjson.dumps(result).decode())
        else:
            if not args.quiet:
                print(f"Saved: {output_path} ({len(image_bytes)} bytes)", file=sys.stderr)
            print(str(output_path.absolute()))

        return 0

    try:
        return asyncio.run(run())
    except Exception as e:
        if args.json:
            print(orjson.dumps({"success": False, "error": str(e)}).decode())
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
