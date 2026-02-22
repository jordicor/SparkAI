import os
import time
import unittest
from typing import Any, Dict, List, Optional

# Keep module-level singleton tracker in memory mode during tests.
os.environ.setdefault("SECURITY_REDIS_MODE", "off")

from middleware.security import BaseSecurityBackend, InMemorySecurityBackend, SecurityConfig, SecurityTracker


class NoopEscalator:
    threshold = 1

    def status(self) -> Dict[str, Any]:
        return {"enabled": False}

    async def maybe_escalate(self, tracker, ip: str, block_count_for_ip: int, reason: str) -> Dict[str, Any]:
        return {"attempted": False, "status": "disabled"}

    async def delete_rule(self, rule_id: str) -> Dict[str, Any]:
        return {"deleted": False, "status": "disabled"}


class AlwaysFailRedisBackend(BaseSecurityBackend):
    name = "redis"

    async def ping(self) -> bool:
        raise RuntimeError("redis unavailable")

    async def is_blocked(self, ip: str) -> bool:
        raise RuntimeError("redis unavailable")

    async def block_ip(self, ip: str, hours: int, reason: str, source: str) -> Dict[str, Any]:
        raise RuntimeError("redis unavailable")

    async def unblock_ip(self, ip: str) -> bool:
        raise RuntimeError("redis unavailable")

    async def record_404(self, ip: str) -> None:
        raise RuntimeError("redis unavailable")

    async def get_404_count(self, ip: str, window_minutes: int) -> int:
        raise RuntimeError("redis unavailable")

    async def get_ip_block_count(self, ip: str) -> int:
        raise RuntimeError("redis unavailable")

    async def get_stats(self) -> Dict[str, Any]:
        raise RuntimeError("redis unavailable")

    async def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        raise RuntimeError("redis unavailable")

    async def is_cloudflare_escalated(self, ip: str) -> bool:
        raise RuntimeError("redis unavailable")

    async def mark_cloudflare_escalated(self, ip: str, details: Dict[str, Any]) -> None:
        raise RuntimeError("redis unavailable")

    async def get_blocked_ips(self, limit: int = 200) -> List[Dict[str, Any]]:
        raise RuntimeError("redis unavailable")

    async def get_cloudflare_escalation(self, ip: str) -> Optional[Dict[str, Any]]:
        raise RuntimeError("redis unavailable")

    async def clear_cloudflare_escalation(self, ip: str) -> bool:
        raise RuntimeError("redis unavailable")

    async def get_all_cloudflare_escalated_ips(self) -> List[str]:
        raise RuntimeError("redis unavailable")

    async def extend_block(self, ip: str, hours: int) -> bool:
        raise RuntimeError("redis unavailable")


class RecordingEscalator:
    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.calls: List[Dict[str, Any]] = []
        self.delete_calls: List[str] = []

    def status(self) -> Dict[str, Any]:
        return {"enabled": True, "threshold": self.threshold}

    async def maybe_escalate(self, tracker, ip: str, block_count_for_ip: int, reason: str) -> Dict[str, Any]:
        self.calls.append({
            "ip": ip,
            "block_count_for_ip": block_count_for_ip,
            "reason": reason,
        })
        details = {
            "rule_id": f"rule-{ip}",
            "mode": "block",
            "value": ip,
            "source": "created",
            "timestamp": "2026-02-06T00:00:00+00:00",
        }
        await tracker.mark_cloudflare_escalated(ip, details)
        return {"attempted": True, "status": "created", **details}

    async def delete_rule(self, rule_id: str) -> Dict[str, Any]:
        self.delete_calls.append(rule_id)
        return {"deleted": True, "rule_id": rule_id}


class SecurityTrackerAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_backend_state_is_visible_across_tracker_instances(self):
        shared_backend = InMemorySecurityBackend()

        tracker_a = SecurityTracker(
            mode="auto",
            redis_backend=shared_backend,
            memory_backend=InMemorySecurityBackend(),
            escalator=NoopEscalator(),
        )
        tracker_b = SecurityTracker(
            mode="auto",
            redis_backend=shared_backend,
            memory_backend=InMemorySecurityBackend(),
            escalator=NoopEscalator(),
        )

        await tracker_a.block_ip(
            ip="203.0.113.10",
            hours=1,
            reason="Too many 404s: 10 in 3 min",
            source="404_threshold",
        )

        self.assertTrue(await tracker_b.is_blocked("203.0.113.10"))

        stats = await tracker_b.get_stats()
        self.assertEqual(stats["active_backend"], "redis")

    async def test_auto_mode_falls_back_to_memory_when_redis_is_unavailable(self):
        tracker = SecurityTracker(
            mode="auto",
            redis_backend=AlwaysFailRedisBackend(),
            memory_backend=InMemorySecurityBackend(),
            escalator=NoopEscalator(),
        )

        await tracker.block_ip(
            ip="198.51.100.20",
            hours=1,
            reason="Instant block pattern: /.env",
            source="instant_pattern",
        )

        self.assertTrue(await tracker.is_blocked("198.51.100.20"))
        stats = await tracker.get_stats()
        self.assertEqual(stats["active_backend"], "memory")
        self.assertIn("redis unavailable", stats.get("last_backend_error", ""))

    async def test_block_events_counter_and_reason_breakdown_are_tracked(self):
        tracker = SecurityTracker(
            mode="off",
            memory_backend=InMemorySecurityBackend(),
            escalator=NoopEscalator(),
        )

        await tracker.block_ip(
            ip="203.0.113.30",
            hours=1,
            reason="Instant block pattern: /.git",
            source="instant_pattern",
        )
        await tracker.block_ip(
            ip="203.0.113.31",
            hours=1,
            reason="Too many 404s: 10 in 3 min",
            source="404_threshold",
        )

        stats = await tracker.get_stats()
        self.assertEqual(stats["block_events_total"], 2)
        self.assertEqual(stats["block_events_by_reason"].get("instant_pattern"), 1)
        self.assertEqual(stats["block_events_by_reason"].get("too_many_404s"), 1)

        events = await tracker.get_recent_events(limit=10)
        self.assertEqual(len(events), 2)
        self.assertIn("ip", events[0])
        self.assertIn("reason", events[0])

    async def test_manual_unblock_flow(self):
        tracker = SecurityTracker(
            mode="off",
            memory_backend=InMemorySecurityBackend(),
            escalator=NoopEscalator(),
        )

        await tracker.block_ip(
            ip="203.0.113.40",
            hours=1,
            reason="Manual block",
            source="manual",
        )
        self.assertTrue(await tracker.is_blocked("203.0.113.40"))

        result = await tracker.unblock_ip("203.0.113.40")
        self.assertTrue(result["unblocked"])
        self.assertFalse(await tracker.is_blocked("203.0.113.40"))

    async def test_blocked_ips_include_metadata_and_cloudflare_status(self):
        tracker = SecurityTracker(
            mode="off",
            memory_backend=InMemorySecurityBackend(),
            escalator=NoopEscalator(),
        )

        await tracker.block_ip(
            ip="203.0.113.77",
            hours=2,
            reason="Manual block",
            source="manual",
        )
        await tracker.mark_cloudflare_escalated(
            "203.0.113.77",
            {
                "rule_id": "test-rule-77",
                "source": "created",
                "timestamp": "2026-02-06T00:00:00+00:00",
            },
        )

        blocked_ips = await tracker.get_blocked_ips(limit=10)
        self.assertEqual(len(blocked_ips), 1)
        blocked = blocked_ips[0]
        self.assertEqual(blocked["ip"], "203.0.113.77")
        self.assertEqual(blocked["source"], "manual")
        self.assertTrue(blocked["cloudflare_synced"])
        self.assertEqual(blocked["cloudflare"]["rule_id"], "test-rule-77")

    async def test_retry_cloudflare_sync_uses_threshold_floor(self):
        escalator = RecordingEscalator(threshold=5)
        tracker = SecurityTracker(
            mode="off",
            memory_backend=InMemorySecurityBackend(),
            escalator=escalator,
        )

        await tracker.block_ip(
            ip="198.51.100.44",
            hours=1,
            reason="Manual block",
            source="manual",
        )

        result = await tracker.retry_cloudflare_sync("198.51.100.44", reason="Manual sync retry")
        self.assertEqual(result["status"], "created")
        last_call = escalator.calls[-1]
        self.assertEqual(last_call["ip"], "198.51.100.44")
        self.assertEqual(last_call["reason"], "Manual sync retry")
        self.assertGreaterEqual(last_call["block_count_for_ip"], 5)


    async def test_unblock_deletes_cloudflare_rule(self):
        escalator = RecordingEscalator(threshold=1)
        backend = InMemorySecurityBackend()
        tracker = SecurityTracker(
            mode="off",
            memory_backend=backend,
            escalator=escalator,
        )

        # Block triggers escalation (threshold=1)
        await tracker.block_ip(
            ip="203.0.113.90",
            hours=1,
            reason="Manual block",
            source="manual",
        )
        self.assertTrue(await tracker.is_cloudflare_escalated("203.0.113.90"))

        # Unblock should also call delete_rule
        result = await tracker.unblock_ip("203.0.113.90")
        self.assertTrue(result["unblocked"])
        self.assertTrue(result["cloudflare_deleted"])
        self.assertEqual(result["cloudflare_rule_id"], "rule-203.0.113.90")
        self.assertIn("rule-203.0.113.90", escalator.delete_calls)
        self.assertFalse(await tracker.is_cloudflare_escalated("203.0.113.90"))

    async def test_unblock_deletes_cloudflare_rule_even_if_local_block_missing(self):
        escalator = RecordingEscalator(threshold=1)
        backend = InMemorySecurityBackend()
        tracker = SecurityTracker(
            mode="off",
            memory_backend=backend,
            escalator=escalator,
        )

        await tracker.block_ip(
            ip="203.0.113.95",
            hours=1,
            reason="Manual block",
            source="manual",
        )
        self.assertTrue(await tracker.is_cloudflare_escalated("203.0.113.95"))

        # Simulate prior local cleanup/expiry while Cloudflare metadata still exists.
        backend._blocked_ips.pop("203.0.113.95", None)
        backend._blocked_meta.pop("203.0.113.95", None)

        result = await tracker.unblock_ip("203.0.113.95")
        self.assertFalse(result["unblocked"])
        self.assertTrue(result["cloudflare_deleted"])
        self.assertEqual(result["cloudflare_rule_id"], "rule-203.0.113.95")
        self.assertIn("rule-203.0.113.95", escalator.delete_calls)
        self.assertFalse(await tracker.is_cloudflare_escalated("203.0.113.95"))

    async def test_unblock_without_cloudflare_escalation(self):
        tracker = SecurityTracker(
            mode="off",
            memory_backend=InMemorySecurityBackend(),
            escalator=NoopEscalator(),
        )

        await tracker.block_ip(
            ip="203.0.113.91",
            hours=1,
            reason="Manual block",
            source="manual",
        )

        result = await tracker.unblock_ip("203.0.113.91")
        self.assertTrue(result["unblocked"])
        self.assertFalse(result["cloudflare_deleted"])
        self.assertIsNone(result["cloudflare_rule_id"])

    async def test_extend_block_updates_expiry_without_inflating_counters(self):
        backend = InMemorySecurityBackend()
        tracker = SecurityTracker(
            mode="off",
            memory_backend=backend,
            escalator=NoopEscalator(),
        )

        await tracker.block_ip(
            ip="203.0.113.92",
            hours=1,
            reason="Manual block",
            source="manual",
        )

        stats_before = await tracker.get_stats()
        events_before = await tracker.get_recent_events(limit=100)

        extended = await tracker.extend_block("203.0.113.92", 168)
        self.assertTrue(extended)

        stats_after = await tracker.get_stats()
        events_after = await tracker.get_recent_events(limit=100)

        # Counters must not change
        self.assertEqual(stats_before["block_events_total"], stats_after["block_events_total"])
        self.assertEqual(len(events_before), len(events_after))

        # Expiry should be extended (~168h from now)
        remaining = backend._blocked_ips["203.0.113.92"] - time.time()
        self.assertGreater(remaining, 3600 * 2)  # More than 2h = clearly extended from 1h

    async def test_opportunistic_cleanup_removes_expired_cf_escalations_on_new_block(self):
        escalator = RecordingEscalator(threshold=1)
        backend = InMemorySecurityBackend()
        tracker = SecurityTracker(
            mode="off",
            memory_backend=backend,
            escalator=escalator,
        )

        # Block and escalate
        await tracker.block_ip(
            ip="203.0.113.93",
            hours=1,
            reason="Manual block",
            source="manual",
        )
        self.assertTrue(await tracker.is_cloudflare_escalated("203.0.113.93"))

        # Simulate local block expiry by setting past timestamp
        backend._blocked_ips["203.0.113.93"] = 0

        old_interval = SecurityConfig.CF_CLEANUP_INTERVAL_SECONDS
        old_max_ips = SecurityConfig.CF_CLEANUP_MAX_IPS_PER_RUN
        old_budget = SecurityConfig.CF_CLEANUP_TIME_BUDGET_SECONDS
        try:
            SecurityConfig.CF_CLEANUP_INTERVAL_SECONDS = 1
            SecurityConfig.CF_CLEANUP_MAX_IPS_PER_RUN = 20
            SecurityConfig.CF_CLEANUP_TIME_BUDGET_SECONDS = 5.0
            tracker._last_cf_cleanup_ts = 0

            # New block event should trigger opportunistic cleanup.
            await tracker.block_ip(
                ip="203.0.113.96",
                hours=1,
                reason="Manual block",
                source="manual",
            )
        finally:
            SecurityConfig.CF_CLEANUP_INTERVAL_SECONDS = old_interval
            SecurityConfig.CF_CLEANUP_MAX_IPS_PER_RUN = old_max_ips
            SecurityConfig.CF_CLEANUP_TIME_BUDGET_SECONDS = old_budget

        # CF escalation should be cleaned up
        self.assertFalse(await tracker.is_cloudflare_escalated("203.0.113.93"))
        self.assertIn("rule-203.0.113.93", escalator.delete_calls)

    async def test_block_extends_duration_on_cf_escalation(self):
        escalator = RecordingEscalator(threshold=1)
        backend = InMemorySecurityBackend()
        tracker = SecurityTracker(
            mode="off",
            memory_backend=backend,
            escalator=escalator,
        )

        event = await tracker.block_ip(
            ip="203.0.113.94",
            hours=24,
            reason="Manual block",
            source="manual",
        )

        # Verify escalation happened and block was extended
        self.assertIn("cloudflare", event)
        remaining = backend._blocked_ips["203.0.113.94"] - time.time()
        # Must be more than 24h (original), should be ~168h
        self.assertGreater(remaining, 24 * 3600)

    async def test_record_request_ban_already_handled_skips_ban_evaluation(self):
        """When ban_already_handled=True, scoring still happens but no ban is triggered."""
        from middleware.ip_reputation import IPReputationManager, ReputationConfig

        manager = IPReputationManager()
        manager._initialized = True

        # Record with ban_already_handled=True and external_ban_until
        ban_until = time.time() + 86400
        result = manager.record_request(
            "10.0.0.1", 403, "/wp-admin",
            is_pattern_hit=True,
            ban_already_handled=True,
            external_ban_until=ban_until,
        )

        # Should return None (no ban evaluation)
        self.assertIsNone(result)

        # But score should still be recorded
        delta = manager._pending.get("10.0.0.1")
        self.assertIsNotNone(delta)
        self.assertEqual(delta.score_delta, ReputationConfig.SCORE_PATTERN_MATCH)
        self.assertEqual(delta.pattern_hits, 1)

        # External ban should be persisted
        self.assertEqual(delta.banned_until, ban_until)
        self.assertEqual(delta.times_banned_delta, 1)

    async def test_record_request_without_ban_already_handled_triggers_ban(self):
        """Default behavior: record_request evaluates ban rules when ban_already_handled=False."""
        from middleware.ip_reputation import IPReputationManager, ReputationConfig

        manager = IPReputationManager()
        manager._initialized = True

        # Score enough to trigger a ban (SCORE_PATTERN_MATCH = 50 >= BAN_SCORE_MODERATE = 50)
        result = manager.record_request(
            "10.0.0.2", 403, "/wp-admin",
            is_pattern_hit=True,
        )

        # Should trigger a ban
        self.assertIsNotNone(result)
        self.assertIn("ban_hours", result)
        self.assertIn("reason", result)

    async def test_block_ip_notifies_nginx_blocklist(self):
        """block_ip() should call nginx_blocklist_manager.add_ip()."""
        from unittest.mock import patch

        tracker = SecurityTracker(
            mode="off",
            memory_backend=InMemorySecurityBackend(),
            escalator=NoopEscalator(),
        )

        with patch("middleware.security.nginx_blocklist_manager") as mock_blocklist:
            await tracker.block_ip(
                ip="203.0.113.50",
                hours=1,
                reason="Test block",
                source="test",
            )
            mock_blocklist.add_ip.assert_called_once_with("203.0.113.50")

    async def test_unblock_ip_notifies_nginx_blocklist(self):
        """unblock_ip() should call nginx_blocklist_manager.remove_ip()."""
        from unittest.mock import patch

        tracker = SecurityTracker(
            mode="off",
            memory_backend=InMemorySecurityBackend(),
            escalator=NoopEscalator(),
        )

        await tracker.block_ip(
            ip="203.0.113.51",
            hours=1,
            reason="Test block",
            source="test",
        )

        with patch("middleware.security.nginx_blocklist_manager") as mock_blocklist:
            await tracker.unblock_ip("203.0.113.51")
            mock_blocklist.remove_ip.assert_called_once_with("203.0.113.51")


if __name__ == "__main__":
    unittest.main()
