"""
Email validation module for Aurvek registration system.

Provides robust email validation including:
- Format validation (RFC 5322 simplified)
- Disposable/temporary email domain blocking
- MX record verification
"""

import re
import logging
from typing import Tuple

import dns.resolver

logger = logging.getLogger(__name__)

# Known disposable/temporary email domains
# These services provide throwaway emails that shouldn't be used for registration
DISPOSABLE_DOMAINS = {
    # Popular temporary email services
    "tempmail.com",
    "temp-mail.org",
    "temp-mail.io",
    "tempail.com",
    "tempr.email",
    "10minutemail.com",
    "10minutemail.net",
    "10minmail.com",
    "20minutemail.com",
    "guerrillamail.com",
    "guerrillamail.org",
    "guerrillamail.net",
    "guerrillamail.biz",
    "guerrillamail.de",
    "sharklasers.com",
    "grr.la",
    "guerrillamailblock.com",
    "pokemail.net",
    "spam4.me",
    "mailinator.com",
    "mailinator.net",
    "mailinator.org",
    "mailinator2.com",
    "mailinater.com",
    "yopmail.com",
    "yopmail.fr",
    "yopmail.net",
    "throwaway.email",
    "throwawaymail.com",
    "fakeinbox.com",
    "fakemailgenerator.com",
    "dispostable.com",
    "disposableemailaddresses.com",
    "emailondeck.com",
    "getnada.com",
    "mohmal.com",
    "tempinbox.com",
    "burnermail.io",
    "mailnesia.com",
    "maildrop.cc",
    "mintemail.com",
    "mytrashmail.com",
    "trashmail.com",
    "trashmail.net",
    "trashmail.org",
    "wegwerfmail.de",
    "wegwerfmail.net",
    "spamgourmet.com",
    "mailcatch.com",
    "mailnull.com",
    "spamfree24.org",
    "antispam.de",
    "spambox.us",
    "trash-mail.at",
    "getairmail.com",
    "dropmail.me",
    "mailsac.com",
    "harakirimail.com",
    "33mail.com",
    "discard.email",
    "discardmail.com",
    "spambog.com",
    "spambog.de",
    "spambog.ru",
    "emailfake.com",
    "crazymailing.com",
    "tempsky.com",
    "inboxkitten.com",
    "anonymbox.com",
    "mailforspam.com",
}


def _validate_format(email: str) -> bool:
    """
    Validate email format using RFC 5322 simplified pattern.

    Args:
        email: Email address to validate

    Returns:
        True if format is valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def _verify_mx_records(domain: str) -> Tuple[bool, str]:
    """
    Verify that the domain has valid MX records (can receive email).

    Uses a short timeout to avoid blocking legitimate users.
    Fails open on timeouts and unexpected errors.

    Args:
        domain: Domain part of the email address

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        resolver = dns.resolver.Resolver()
        resolver.timeout = 3  # Short timeout
        resolver.lifetime = 5  # Total query lifetime

        # Try MX records first
        mx_records = resolver.resolve(domain, 'MX')
        if mx_records:
            logger.debug(f"MX records found for {domain}")
            return True, ""
        return False, "Email domain cannot receive mail"

    except dns.resolver.NXDOMAIN:
        logger.info(f"Domain does not exist: {domain}")
        return False, "Email domain does not exist"

    except dns.resolver.NoAnswer:
        # No MX records, try A record as fallback
        # Some domains use A records instead of MX for mail delivery
        try:
            resolver.resolve(domain, 'A')
            logger.debug(f"No MX but A record found for {domain}")
            return True, ""
        except Exception:
            logger.info(f"No MX or A records for {domain}")
            return False, "Email domain cannot receive mail"

    except dns.resolver.Timeout:
        # Fail open on timeout - don't block legitimate users
        logger.warning(f"DNS timeout for domain: {domain}")
        return True, ""

    except Exception as e:
        # Fail open on unexpected errors
        logger.error(f"MX verification error for {domain}: {e}")
        return True, ""


def validate_email_robust(email: str) -> Tuple[bool, str]:
    """
    Perform robust email validation:
    1. Format validation
    2. Disposable domain check
    3. MX record verification

    Args:
        email: Email address to validate

    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is empty string
    """
    if not email:
        return False, "Email address is required"

    email = email.strip().lower()

    # 1. Validate format
    if not _validate_format(email):
        return False, "Invalid email format"

    # Extract domain
    try:
        domain = email.split('@')[1]
    except IndexError:
        return False, "Invalid email format"

    # 2. Check for disposable domains
    if domain in DISPOSABLE_DOMAINS:
        logger.info(f"Registration attempt with disposable email domain: {domain}")
        return False, "Please use a permanent email address"

    # 3. Verify MX records
    has_mx, mx_error = _verify_mx_records(domain)
    if not has_mx:
        return False, mx_error

    return True, ""
