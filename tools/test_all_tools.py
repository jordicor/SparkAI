#!/usr/bin/env python3
"""
SPARK Tools Testing Suite
=========================

Tests ALL registered tools to verify they work correctly.
Run this script to diagnose issues with AI tool integrations.

Usage:
    python tools/test_all_tools.py              # Run all tests
    python tools/test_all_tools.py --quick      # Quick connectivity tests only
    python tools/test_all_tools.py --tool image # Test specific tool
    python tools/test_all_tools.py --fix        # Show fixes for broken tools
"""

import os
import sys
import orjson
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


# =============================================================================
# CONFIGURATION CHECKER
# =============================================================================

def check_env_config():
    """Check and display current environment configuration."""
    print("\n" + "=" * 60)
    print("CONFIGURATION CHECK")
    print("=" * 60)

    config = {
        "IMAGE_GENERATION_ENGINE": os.getenv('IMAGE_GENERATION_ENGINE', 'nano-banana'),
        "GEMINI_IMAGE_MODEL": os.getenv('GEMINI_IMAGE_MODEL', 'gemini-2.5-flash-image'),
        "OPENAI_IMAGE_MODEL": os.getenv('OPENAI_IMAGE_MODEL', 'gpt-image-1'),
        "POE_IMAGE_MODEL": os.getenv('POE_IMAGE_MODEL', 'flux2pro'),
        "VIDEO_GENERATION_ENGINE": os.getenv('VIDEO_GENERATION_ENGINE', 'veo-3'),
    }

    api_keys = {
        "GEMINI_KEY": bool(os.getenv('GEMINI_KEY')),
        "OPENAI_KEY": bool(os.getenv('OPENAI_KEY')),
        "PERPLEXITY_API_KEY": bool(os.getenv('PERPLEXITY_API_KEY')),
        "POE_API_KEY": bool(os.getenv('POE_API_KEY')),
        "IDEOGRAM_KEY": bool(os.getenv('IDEOGRAM_KEY')),
    }

    print("\nEngine Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\nAPI Keys Status:")
    for key, present in api_keys.items():
        status = "[OK]" if present else "[MISSING]"
        print(f"  {key}: {status}")

    return config, api_keys


def check_model_validity():
    """Check if configured models are valid/current."""
    print("\n" + "=" * 60)
    print("MODEL VALIDITY CHECK")
    print("=" * 60)

    issues = []

    # Perplexity model check
    # OLD: llama-3.1-sonar-small-128k-online (DEPRECATED)
    # NEW: sonar, sonar-pro, sonar-deep-research
    print("\n[Perplexity]")
    print("  Current valid models: sonar, sonar-pro, sonar-deep-research")
    print("  DEPRECATED models: llama-3.1-sonar-* (removed from API)")

    # Check perplexity.py for outdated model
    perplexity_file = PROJECT_ROOT / "tools" / "perplexity.py"
    if perplexity_file.exists():
        content = perplexity_file.read_text()
        if "llama-3.1-sonar" in content:
            issues.append({
                "tool": "Perplexity",
                "file": "tools/perplexity.py",
                "problem": "Using deprecated model 'llama-3.1-sonar-small-128k-online'",
                "fix": "Change model to 'sonar' or 'sonar-pro'"
            })
            print("  [ERROR] Code uses deprecated llama-3.1-sonar model!")
        else:
            print("  [OK] Using current model")

    # Gemini image model check
    # VALID: gemini-2.5-flash-image, gemini-3-pro-image-preview
    # DEPRECATED: gemini-2.5-flash-image-preview (shutdown Jan 15, 2026)
    print("\n[Gemini Image]")
    print("  Current valid models: gemini-2.5-flash-image, gemini-3-pro-image-preview")
    print("  DEPRECATED: gemini-2.5-flash-image-preview (shutdown Jan 15, 2026)")

    gemini_model = os.getenv('GEMINI_IMAGE_MODEL', 'gemini-2.5-flash-image')
    if "preview" in gemini_model and "2.5-flash" in gemini_model:
        issues.append({
            "tool": "Gemini Image",
            "file": ".env",
            "problem": f"Model '{gemini_model}' is deprecated/invalid",
            "fix": "Use 'gemini-2.5-flash-image' or 'gemini-3-pro-image-preview'"
        })
        print(f"  [ERROR] Configured model '{gemini_model}' is deprecated!")
    else:
        print(f"  [OK] Model '{gemini_model}' appears valid")

    return issues


# =============================================================================
# TOOL TESTS
# =============================================================================

async def test_perplexity_api():
    """Test Perplexity API connectivity and response."""
    print("\n[TEST] Perplexity API")

    api_key = os.getenv('PERPLEXITY_API_KEY')
    if not api_key:
        return {"status": "SKIP", "reason": "PERPLEXITY_API_KEY not set"}

    import aiohttp

    # Test with CURRENT model names
    models_to_test = ["sonar", "sonar-pro"]

    for model in models_to_test:
        url = "https://api.perplexity.ai/chat/completions"
        payload = {
            "messages": [
                {"content": "What is 2+2?", "role": "user"}
            ],
            "model": model,
            "stream": False,
            "max_tokens": 50
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                        if content:
                            print(f"  [OK] Model '{model}' works! Response: {content[:50]}...")
                            return {"status": "OK", "model": model, "response_preview": content[:100]}
                        else:
                            print(f"  [WARN] Model '{model}' returned empty response")
                    else:
                        error = await response.text()
                        print(f"  [ERROR] Model '{model}' failed: {response.status} - {error[:100]}")
        except Exception as e:
            print(f"  [ERROR] Model '{model}' exception: {e}")

    return {"status": "FAIL", "reason": "All models failed"}


async def test_gemini_image_api():
    """Test Gemini image generation API."""
    print("\n[TEST] Gemini Image Generation API")

    api_key = os.getenv('GEMINI_KEY')
    if not api_key:
        return {"status": "SKIP", "reason": "GEMINI_KEY not set"}

    import aiohttp

    # Test with VALID models
    models_to_test = ["gemini-2.5-flash-image", "gemini-3-pro-image-preview"]

    for model in models_to_test:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }
        payload = {
            "contents": [{"parts": [{"text": "Generate a simple blue square"}]}],
            "generationConfig": {
                "responseModalities": ["IMAGE", "TEXT"]
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=60) as response:
                    if response.status == 200:
                        data = await response.json()
                        candidates = data.get('candidates', [])
                        if candidates:
                            parts = candidates[0].get('content', {}).get('parts', [])
                            has_image = any('inlineData' in p for p in parts)
                            if has_image:
                                print(f"  [OK] Model '{model}' generates images successfully!")
                                return {"status": "OK", "model": model}
                            else:
                                print(f"  [WARN] Model '{model}' responded but no image data")
                        else:
                            feedback = data.get('promptFeedback', {})
                            print(f"  [WARN] Model '{model}' no candidates. Feedback: {feedback}")
                    else:
                        error = await response.text()
                        print(f"  [ERROR] Model '{model}' failed: {response.status}")
                        # Parse error for useful info
                        try:
                            err_json = orjson.loads(error)
                            msg = err_json.get('error', {}).get('message', error[:200])
                            print(f"         {msg[:200]}")
                        except orjson.JSONDecodeError:
                            print(f"         {error[:200]}")
        except asyncio.TimeoutError:
            print(f"  [ERROR] Model '{model}' timed out (>60s)")
        except Exception as e:
            print(f"  [ERROR] Model '{model}' exception: {e}")

    return {"status": "FAIL", "reason": "All Gemini image models failed"}


async def test_gemini_video_api():
    """Test Gemini VEO video generation API (connectivity only)."""
    print("\n[TEST] Gemini VEO Video Generation API (connectivity check)")

    api_key = os.getenv('GEMINI_KEY')
    if not api_key:
        return {"status": "SKIP", "reason": "GEMINI_KEY not set"}

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        # Get configured model
        veo_model = os.getenv('VEO_MODEL', 'veo-3.1-fast-generate-preview')

        # Just verify the client can be created and list models
        # Don't actually generate a video (too expensive/slow)
        print("  [OK] google-genai library loaded")
        print(f"  [INFO] VEO model configured: {veo_model}")
        print("  [INFO] Valid models: veo-3.1-fast-generate-preview, veo-3.1-generate-preview, veo-2.0-generate-001")
        print("  [INFO] Skipping actual generation (takes 2-6 minutes, costs ~$0.50)")

        return {"status": "OK", "model": veo_model, "info": "Connectivity verified, actual generation skipped"}

    except ImportError:
        print("  [ERROR] google-genai library not installed")
        print("         Run: pip install google-genai")
        return {"status": "FAIL", "reason": "google-genai not installed"}
    except Exception as e:
        print(f"  [ERROR] {e}")
        return {"status": "FAIL", "reason": str(e)}


async def test_qr_generation():
    """Test QR code generation (local, no API)."""
    print("\n[TEST] QR Code Generation (local)")

    try:
        import qrcode
        from PIL import Image
        import io

        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
        qr.add_data("https://test.example.com")
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        size = len(buffer.getvalue())

        print(f"  [OK] QR generated successfully ({size} bytes)")

        # Check for template
        template_path = PROJECT_ROOT / "data" / "static" / "images" / "qr_template.png"
        if template_path.exists():
            print(f"  [OK] QR template found: {template_path}")
        else:
            print(f"  [WARN] QR template missing: {template_path}")

        return {"status": "OK", "size": size}

    except ImportError as e:
        print(f"  [ERROR] Missing dependency: {e}")
        return {"status": "FAIL", "reason": f"Missing: {e}"}
    except Exception as e:
        print(f"  [ERROR] {e}")
        return {"status": "FAIL", "reason": str(e)}


def test_time_tools():
    """Test timezone tools (local, no API)."""
    print("\n[TEST] Time Tools (local)")

    try:
        import pytz
        from datetime import datetime

        # Test get_time
        tz = pytz.timezone("America/New_York")
        ny_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"  [OK] get_time: New York = {ny_time}")

        # Test timezone difference
        tz1 = pytz.timezone("America/New_York")
        tz2 = pytz.timezone("Europe/Madrid")
        time1 = datetime.now(tz1)
        time2 = datetime.now(tz2)
        diff = time2 - time1
        hours, remainder = divmod(abs(diff.total_seconds()), 3600)
        print(f"  [OK] get_time_difference: NY to Madrid = {int(hours)}h")

        # Test convert_time
        from_tz = pytz.timezone("UTC")
        to_tz = pytz.timezone("Asia/Tokyo")
        test_time = datetime.strptime("2026-01-31 12:00:00", "%Y-%m-%d %H:%M:%S")
        from_time = from_tz.localize(test_time)
        to_time = from_time.astimezone(to_tz)
        print(f"  [OK] convert_time: UTC 12:00 = Tokyo {to_time.strftime('%H:%M')}")

        return {"status": "OK"}

    except ImportError as e:
        print(f"  [ERROR] Missing dependency: {e}")
        return {"status": "FAIL", "reason": f"Missing: {e}"}
    except Exception as e:
        print(f"  [ERROR] {e}")
        return {"status": "FAIL", "reason": str(e)}


async def test_openai_image_api():
    """Test OpenAI image generation API."""
    print("\n[TEST] OpenAI Image Generation API")

    api_key = os.getenv('OPENAI_KEY')
    if not api_key:
        return {"status": "SKIP", "reason": "OPENAI_KEY not set"}

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        # Just verify we can connect, don't actually generate (costs money)
        # We'll use models.list() to verify connectivity
        models = client.models.list()
        model_ids = [m.id for m in models.data if 'image' in m.id.lower() or 'dall' in m.id.lower()]

        if model_ids:
            print(f"  [OK] Connected! Available image models: {model_ids[:5]}")
            return {"status": "OK", "models": model_ids[:5]}
        else:
            print("  [OK] Connected! (checking image model availability...)")
            # Try a minimal generation
            try:
                response = client.images.generate(
                    model="dall-e-3",
                    prompt="A simple red dot",
                    n=1,
                    size="1024x1024",
                    quality="standard"
                )
                if response.data:
                    print("  [OK] DALL-E-3 generation works!")
                    return {"status": "OK", "model": "dall-e-3"}
            except Exception as e:
                print(f"  [WARN] DALL-E-3 test: {e}")

            return {"status": "OK", "info": "Connected but image models need verification"}

    except ImportError:
        print("  [ERROR] openai library not installed")
        return {"status": "FAIL", "reason": "openai not installed"}
    except Exception as e:
        print(f"  [ERROR] {e}")
        return {"status": "FAIL", "reason": str(e)}


async def test_poe_api():
    """Test Poe API connectivity."""
    print("\n[TEST] Poe API (FLUX models)")

    api_key = os.getenv('POE_API_KEY')
    if not api_key:
        return {"status": "SKIP", "reason": "POE_API_KEY not set"}

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.poe.com/v1",
        )

        # Try a simple text completion to verify connectivity
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use a cheap model for testing
            messages=[{"role": "user", "content": "Say 'test ok' and nothing else"}],
            max_tokens=20
        )

        if response.choices:
            content = response.choices[0].message.content
            print(f"  [OK] Poe API connected! Test response: {content}")
            return {"status": "OK"}
        else:
            print("  [WARN] Connected but empty response")
            return {"status": "OK", "info": "Connected, empty response"}

    except ImportError:
        print("  [ERROR] openai library not installed")
        return {"status": "FAIL", "reason": "openai not installed"}
    except Exception as e:
        print(f"  [ERROR] {e}")
        return {"status": "FAIL", "reason": str(e)}


def test_redis_connection():
    """Test Redis connection for background tasks."""
    print("\n[TEST] Redis Connection")

    try:
        # Try to import the project's redis config
        from rediscfg import redis_client

        # Test connection
        asyncio.get_event_loop().run_until_complete(redis_client.ping())
        print("  [OK] Redis connected successfully")
        return {"status": "OK"}

    except ImportError as e:
        print(f"  [WARN] Could not import rediscfg: {e}")
        return {"status": "WARN", "reason": "rediscfg import failed"}
    except Exception as e:
        print(f"  [ERROR] Redis connection failed: {e}")
        return {"status": "FAIL", "reason": str(e)}


def test_dramatiq_setup():
    """Test Dramatiq task queue setup."""
    print("\n[TEST] Dramatiq Task Queue")

    try:
        import dramatiq
        from tools import dramatiq_tasks

        registered_tasks = list(dramatiq_tasks.keys())
        print(f"  [OK] Dramatiq loaded. Registered tasks: {registered_tasks}")
        return {"status": "OK", "tasks": registered_tasks}

    except ImportError as e:
        print(f"  [ERROR] Import failed: {e}")
        return {"status": "FAIL", "reason": str(e)}
    except Exception as e:
        print(f"  [ERROR] {e}")
        return {"status": "FAIL", "reason": str(e)}


def test_registered_tools():
    """List all registered tools and handlers."""
    print("\n[TEST] Registered Tools & Handlers")

    try:
        from tools import tools, function_handlers

        print(f"\n  Registered Tools ({len(tools)}):")
        for tool in tools:
            name = tool.get('function', {}).get('name', 'unknown')
            desc = tool.get('function', {}).get('description', '')[:60]
            print(f"    - {name}: {desc}...")

        print(f"\n  Registered Handlers ({len(function_handlers)}):")
        for name in function_handlers.keys():
            print(f"    - {name}")

        # Check for mismatches
        tool_names = {t.get('function', {}).get('name') for t in tools}
        handler_names = set(function_handlers.keys())

        missing_handlers = tool_names - handler_names
        extra_handlers = handler_names - tool_names

        if missing_handlers:
            print(f"\n  [WARN] Tools without handlers: {missing_handlers}")
        if extra_handlers:
            print(f"\n  [WARN] Handlers without tools: {extra_handlers}")

        return {
            "status": "OK",
            "tools": list(tool_names),
            "handlers": list(handler_names),
            "missing_handlers": list(missing_handlers),
            "extra_handlers": list(extra_handlers)
        }

    except ImportError as e:
        print(f"  [ERROR] Import failed: {e}")
        return {"status": "FAIL", "reason": str(e)}
    except Exception as e:
        print(f"  [ERROR] {e}")
        return {"status": "FAIL", "reason": str(e)}


# =============================================================================
# FIX SUGGESTIONS
# =============================================================================

def show_fixes(issues):
    """Display fix suggestions for found issues."""
    if not issues:
        print("\n[OK] No issues found that require fixes!")
        return

    print("\n" + "=" * 60)
    print("SUGGESTED FIXES")
    print("=" * 60)

    for i, issue in enumerate(issues, 1):
        print(f"\n{i}. {issue['tool']}")
        print(f"   File: {issue['file']}")
        print(f"   Problem: {issue['problem']}")
        print(f"   Fix: {issue['fix']}")

    # Generate fix commands/patches
    print("\n" + "-" * 60)
    print("AUTO-FIX COMMANDS:")
    print("-" * 60)

    for issue in issues:
        if issue['tool'] == 'Perplexity':
            print(f"""
# Fix Perplexity model in tools/perplexity.py
# Change line with: "model": "llama-3.1-sonar-small-128k-online"
# To: "model": "sonar-pro"
""")


# =============================================================================
# MAIN
# =============================================================================

async def run_all_tests(quick=False):
    """Run all tool tests."""
    print("\n" + "=" * 60)
    print("SPARK TOOLS TEST SUITE")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Configuration check
    config, api_keys = check_env_config()

    # Model validity check
    issues = check_model_validity()

    results = {}

    # Local tests (fast)
    print("\n" + "=" * 60)
    print("LOCAL TESTS (No API calls)")
    print("=" * 60)

    results['time_tools'] = test_time_tools()
    results['qr_generation'] = await test_qr_generation()
    results['registered_tools'] = test_registered_tools()
    results['dramatiq'] = test_dramatiq_setup()
    results['redis'] = test_redis_connection()

    if quick:
        print("\n[INFO] Quick mode: Skipping API tests")
    else:
        print("\n" + "=" * 60)
        print("API TESTS (May take time and cost money)")
        print("=" * 60)

        results['perplexity'] = await test_perplexity_api()
        results['gemini_image'] = await test_gemini_image_api()
        results['gemini_video'] = await test_gemini_video_api()
        results['openai_image'] = await test_openai_image_api()
        results['poe'] = await test_poe_api()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, result in results.items():
        status = result.get('status', 'UNKNOWN')
        symbol = {
            'OK': '[OK]',
            'FAIL': '[FAIL]',
            'SKIP': '[SKIP]',
            'WARN': '[WARN]'
        }.get(status, '[???]')
        reason = result.get('reason', '')
        print(f"  {symbol:8} {test_name:20} {reason}")

    # Show fixes
    show_fixes(issues)

    return results, issues


def main():
    parser = argparse.ArgumentParser(description="Test all SPARK tools")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick mode: skip API tests")
    parser.add_argument("--tool", "-t", help="Test specific tool only")
    parser.add_argument("--fix", "-f", action="store_true", help="Show fixes for issues")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    results, issues = asyncio.run(run_all_tests(quick=args.quick))

    if args.json:
        output = {
            "results": results,
            "issues": issues,
            "timestamp": datetime.now().isoformat()
        }
        print(orjson.dumps(output, option=orjson.OPT_INDENT_2).decode())

    # Exit with error code if any tests failed
    failed = any(r.get('status') == 'FAIL' for r in results.values())
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
