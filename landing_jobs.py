#!/usr/bin/env python3
"""
Landing Jobs - Async job management for landing page generation.

Handles background execution of Claude Code for landing page generation/modification,
with JSON-based status tracking and PID monitoring.
"""

import os
import sys
import json
import uuid
import subprocess
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Literal

# Project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
JOBS_DIR = PROJECT_ROOT / "data" / "jobs" / "landing"
LANDING_WIZARD_PATH = PROJECT_ROOT / "landing_wizard.py"

# Ensure jobs directory exists
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Status types
JobStatus = Literal["pending", "running", "completed", "failed", "timeout"]


def generate_task_id() -> str:
    """Generate a unique task ID."""
    return uuid.uuid4().hex


def get_job_path(task_id: str) -> Path:
    """Get the path to a job's JSON file."""
    return JOBS_DIR / f"{task_id}.json"


def create_job(
    task_id: str,
    prompt_id: int,
    job_type: Literal["generate", "modify"],
    timeout_seconds: int = 300,
    request_data: dict = None
) -> dict:
    """
    Create a new job record.

    Args:
        task_id: Unique job identifier
        prompt_id: The prompt being processed
        job_type: "generate" or "modify"
        timeout_seconds: Max time before considering job timed out
        request_data: Original request parameters

    Returns:
        The job record dict
    """
    now = datetime.now(timezone.utc).isoformat()

    job = {
        "task_id": task_id,
        "prompt_id": prompt_id,
        "type": job_type,
        "status": "pending",
        "pid": None,
        "started_at": now,
        "updated_at": now,
        "completed_at": None,
        "timeout_seconds": timeout_seconds,
        "error": None,
        "files_created": None,
        "request": request_data or {}
    }

    job_path = get_job_path(task_id)
    with open(job_path, 'w', encoding='utf-8') as f:
        json.dump(job, f, indent=2)

    return job


def update_job(task_id: str, **updates) -> Optional[dict]:
    """
    Update a job record.

    Args:
        task_id: Job identifier
        **updates: Fields to update

    Returns:
        Updated job record or None if not found
    """
    job_path = get_job_path(task_id)

    if not job_path.exists():
        return None

    try:
        with open(job_path, 'r', encoding='utf-8') as f:
            job = json.load(f)

        job.update(updates)
        job["updated_at"] = datetime.now(timezone.utc).isoformat()

        with open(job_path, 'w', encoding='utf-8') as f:
            json.dump(job, f, indent=2)

        return job
    except Exception:
        return None


def get_job(task_id: str) -> Optional[dict]:
    """
    Get a job record, checking for timeout if still running.

    Args:
        task_id: Job identifier

    Returns:
        Job record or None if not found
    """
    job_path = get_job_path(task_id)

    if not job_path.exists():
        return None

    try:
        with open(job_path, 'r', encoding='utf-8') as f:
            job = json.load(f)

        # Check for timeout if job is still running
        if job["status"] == "running":
            job = _check_job_timeout(job)

        return job
    except Exception:
        return None


def _check_job_timeout(job: dict) -> dict:
    """
    Check if a running job has timed out.

    A job is considered timed out if:
    1. More than timeout_seconds have passed since updated_at
    2. AND the process (PID) is no longer running

    Args:
        job: The job record

    Returns:
        Updated job record (may be marked as timeout)
    """
    try:
        updated_at = datetime.fromisoformat(job["updated_at"].replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        elapsed = (now - updated_at).total_seconds()
        timeout = job.get("timeout_seconds", 300)

        if elapsed > timeout:
            # Check if process is still alive
            pid = job.get("pid")
            if pid and _is_process_running(pid):
                # Process still running, just update the timestamp
                # (Claude might still be working)
                return job
            else:
                # Process dead or no PID, mark as timeout
                job["status"] = "timeout"
                job["error"] = f"Job timed out after {int(elapsed)} seconds"
                job["completed_at"] = now.isoformat()
                job["updated_at"] = now.isoformat()

                # Save the updated status
                job_path = get_job_path(job["task_id"])
                with open(job_path, 'w', encoding='utf-8') as f:
                    json.dump(job, f, indent=2)
    except Exception:
        pass

    return job


def _is_process_running(pid: int) -> bool:
    """Check if a process with given PID is running."""
    try:
        if sys.platform == "win32":
            # Windows: use tasklist
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True
            )
            return str(pid) in result.stdout
        else:
            # Unix: send signal 0
            os.kill(pid, 0)
            return True
    except (OSError, subprocess.SubprocessError):
        return False


def delete_job(task_id: str) -> bool:
    """
    Delete a job record.

    Args:
        task_id: Job identifier

    Returns:
        True if deleted, False if not found
    """
    job_path = get_job_path(task_id)

    if job_path.exists():
        job_path.unlink()
        return True
    return False


def cleanup_old_jobs(max_age_hours: int = 24) -> int:
    """
    Delete job files older than max_age_hours.

    Args:
        max_age_hours: Maximum age in hours

    Returns:
        Number of jobs deleted
    """
    deleted = 0
    now = time.time()
    max_age_seconds = max_age_hours * 3600

    for job_file in JOBS_DIR.glob("*.json"):
        try:
            if now - job_file.stat().st_mtime > max_age_seconds:
                job_file.unlink()
                deleted += 1
        except Exception:
            pass

    return deleted


def get_active_job_for_prompt(prompt_id: int) -> Optional[dict]:
    """
    Check if there's already an active job for this prompt.

    Args:
        prompt_id: The prompt ID

    Returns:
        Active job record or None
    """
    for job_file in JOBS_DIR.glob("*.json"):
        try:
            with open(job_file, 'r', encoding='utf-8') as f:
                job = json.load(f)

            if job.get("prompt_id") == prompt_id and job.get("status") in ("pending", "running"):
                # Verify it's actually still running
                job = _check_job_timeout(job)
                if job.get("status") in ("pending", "running"):
                    return job
        except Exception:
            pass

    return None


# Worker script content - executed as subprocess
WORKER_SCRIPT = '''
import sys
import json
import os

# Add project root to path
sys.path.insert(0, r"{project_root}")

from landing_wizard import generate_landing, modify_landing, list_prompt_files
from landing_jobs import update_job

def run_job(task_id, job_type, prompt_dir, params):
    """Execute the landing wizard job and update status."""
    pid = os.getpid()

    # Mark as running with PID
    update_job(task_id, status="running", pid=pid)

    try:
        if job_type == "generate":
            result = generate_landing(
                prompt_dir=prompt_dir,
                user_description=params.get("description", ""),
                style=params.get("style", "modern"),
                primary_color=params.get("primary_color", "#3B82F6"),
                secondary_color=params.get("secondary_color", "#10B981"),
                language=params.get("language", "es"),
                timeout=params.get("timeout", 300),
                product_name=params.get("product_name", ""),
                ai_system_prompt=params.get("ai_system_prompt", ""),
                product_description=params.get("product_description", "")
            )
        else:  # modify
            result = modify_landing(
                prompt_dir=prompt_dir,
                instructions=params.get("instructions", ""),
                timeout=params.get("timeout", 300),
                product_name=params.get("product_name", ""),
                ai_system_prompt=params.get("ai_system_prompt", ""),
                product_description=params.get("product_description", "")
            )

        if result.get("success"):
            update_job(
                task_id,
                status="completed",
                files_created=result.get("files_created", []),
                completed_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
            )
        else:
            update_job(
                task_id,
                status="failed",
                error=result.get("error", "Unknown error"),
                completed_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
            )

    except Exception as e:
        update_job(
            task_id,
            status="failed",
            error=str(e),
            completed_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        )

if __name__ == "__main__":
    task_id = sys.argv[1]
    job_type = sys.argv[2]
    prompt_dir = sys.argv[3]
    params = json.loads(sys.argv[4])

    run_job(task_id, job_type, prompt_dir, params)
'''


def start_job(
    prompt_id: int,
    job_type: Literal["generate", "modify"],
    prompt_dir: str,
    params: dict,
    timeout_seconds: int = 300
) -> dict:
    """
    Start a background job for landing page generation/modification.

    Args:
        prompt_id: The prompt being processed
        job_type: "generate" or "modify"
        prompt_dir: Absolute path to prompt directory
        params: Parameters for the wizard function
        timeout_seconds: Max execution time

    Returns:
        Job record with task_id
    """
    # Check for existing active job
    existing = get_active_job_for_prompt(prompt_id)
    if existing:
        return {
            "success": False,
            "error": "A job is already running for this prompt",
            "existing_task_id": existing["task_id"]
        }

    # Create job record
    task_id = generate_task_id()
    job = create_job(
        task_id=task_id,
        prompt_id=prompt_id,
        job_type=job_type,
        timeout_seconds=timeout_seconds,
        request_data=params
    )

    # Write worker script to temp file
    worker_script = WORKER_SCRIPT.format(project_root=str(PROJECT_ROOT))
    worker_path = JOBS_DIR / f"worker_{task_id}.py"

    with open(worker_path, 'w', encoding='utf-8') as f:
        f.write(worker_script)

    # Prepare params JSON (escape for command line)
    params_json = json.dumps(params)

    try:
        # Launch worker process
        if sys.platform == "win32":
            # Windows: use CREATE_NO_WINDOW to hide console
            CREATE_NO_WINDOW = 0x08000000
            process = subprocess.Popen(
                [sys.executable, str(worker_path), task_id, job_type, prompt_dir, params_json],
                creationflags=CREATE_NO_WINDOW,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(PROJECT_ROOT)
            )
        else:
            # Unix: standard background process
            process = subprocess.Popen(
                [sys.executable, str(worker_path), task_id, job_type, prompt_dir, params_json],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                cwd=str(PROJECT_ROOT)
            )

        # Update job with PID
        update_job(task_id, pid=process.pid)

        return {
            "success": True,
            "task_id": task_id,
            "status": "pending",
            "prompt_id": prompt_id
        }

    except Exception as e:
        # Clean up on failure
        update_job(task_id, status="failed", error=str(e))
        if worker_path.exists():
            worker_path.unlink()

        return {
            "success": False,
            "error": str(e),
            "task_id": task_id
        }
