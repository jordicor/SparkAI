"""
Lightweight async Twilio client using httpx.

Replaces asyncio.to_thread(twilio_client.messages.create, ...) with native async HTTP calls.
The sync twilio SDK is kept only for RequestValidator (local HMAC, no network I/O).

Twilio REST API docs:
- Messages: POST /2010-04-01/Accounts/{sid}/Messages.json
- Verify send: POST /v2/Services/{service_sid}/Verifications
- Verify check: POST /v2/Services/{service_sid}/VerificationCheck
"""

import httpx
import logging

logger = logging.getLogger(__name__)


class TwilioAPIError(Exception):
    """Raised when Twilio returns a non-2xx response."""

    def __init__(self, status_code: int, code: int, message: str):
        self.status_code = status_code
        self.code = code
        self.msg = message
        super().__init__(f"Twilio API error {code} (HTTP {status_code}): {message}")


class AsyncTwilioClient:
    """Async Twilio client for Messages and Verify V2 APIs."""

    _MESSAGES_URL = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    _VERIFY_URL = "https://verify.twilio.com/v2/Services/{service_sid}/Verifications"
    _VERIFY_CHECK_URL = "https://verify.twilio.com/v2/Services/{service_sid}/VerificationCheck"
    _MESSAGING_SERVICE_URL = "https://messaging.twilio.com/v1/Services/{service_sid}"

    def __init__(self, account_sid: str, auth_token: str):
        self.account_sid = account_sid
        self._auth = (account_sid, auth_token)
        self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Lazy-init the httpx client inside the running event loop."""
        if self._client is None:
            transport = httpx.AsyncHTTPTransport(
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
            self._client = httpx.AsyncClient(
                auth=self._auth,
                timeout=httpx.Timeout(30.0, connect=10.0),
                transport=transport,
                trust_env=False,
            )
        return self._client

    def _raise_on_error(self, response: httpx.Response) -> None:
        if response.status_code >= 400:
            try:
                body = response.json()
                code = body.get("code", 0)
                message = body.get("message", response.text)
            except Exception:
                code = 0
                message = response.text
            raise TwilioAPIError(
                status_code=response.status_code,
                code=code,
                message=message,
            )

    async def send_message(
        self,
        *,
        from_: str,
        to: str,
        body: str | None = None,
        media_url: list[str] | None = None,
    ) -> dict:
        """Send an SMS/WhatsApp message via Twilio REST API."""
        endpoint = self._MESSAGES_URL.format(sid=self.account_sid)
        form_data: dict = {"To": to, "From": from_}
        if body:
            form_data["Body"] = body
        if media_url:
            form_data["MediaUrl"] = media_url

        response = await self._get_client().post(endpoint, data=form_data)
        self._raise_on_error(response)
        return response.json()

    async def send_verification(
        self, service_sid: str, to: str, channel: str = "sms"
    ) -> dict:
        """Send a verification code via Twilio Verify V2."""
        endpoint = self._VERIFY_URL.format(service_sid=service_sid)
        response = await self._get_client().post(
            endpoint, data={"To": to, "Channel": channel}
        )
        self._raise_on_error(response)
        return response.json()

    async def check_verification(
        self, service_sid: str, to: str, code: str
    ) -> dict:
        """Check a verification code via Twilio Verify V2."""
        endpoint = self._VERIFY_CHECK_URL.format(service_sid=service_sid)
        response = await self._get_client().post(
            endpoint, data={"To": to, "Code": code}
        )
        self._raise_on_error(response)
        return response.json()

    async def get_messaging_service(self, service_sid: str) -> dict:
        """Fetch a Messaging Service configuration."""
        endpoint = self._MESSAGING_SERVICE_URL.format(service_sid=service_sid)
        response = await self._get_client().get(endpoint)
        self._raise_on_error(response)
        return response.json()

    async def update_messaging_service(
        self, service_sid: str, *, inbound_request_url: str
    ) -> dict:
        """Update a Messaging Service's inbound webhook URL."""
        endpoint = self._MESSAGING_SERVICE_URL.format(service_sid=service_sid)
        response = await self._get_client().post(
            endpoint, data={"InboundRequestUrl": inbound_request_url}
        )
        self._raise_on_error(response)
        return response.json()

    async def close(self) -> None:
        """Close the underlying httpx client and release connections."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
