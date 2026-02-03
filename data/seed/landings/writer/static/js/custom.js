// Writer Landing Page - Custom JavaScript

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {

    // Mobile menu toggle
    initMobileMenu();

    // Smooth scrolling for anchor links
    initSmoothScroll();

    // Intersection Observer for scroll animations
    initScrollAnimations();

    // Add active state to navigation items
    initNavHighlight();

    // CTA button tracking (optional analytics)
    initCTATracking();
});

/**
 * Initialize mobile menu toggle functionality
 */
function initMobileMenu() {
    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');

    if (mobileMenuButton && mobileMenu) {
        mobileMenuButton.addEventListener('click', function() {
            mobileMenu.classList.toggle('hidden');

            // Update ARIA attributes for accessibility
            const isExpanded = !mobileMenu.classList.contains('hidden');
            mobileMenuButton.setAttribute('aria-expanded', isExpanded);

            // Animate menu icon (optional)
            const icon = mobileMenuButton.querySelector('svg');
            if (icon) {
                icon.style.transform = isExpanded ? 'rotate(90deg)' : 'rotate(0deg)';
            }
        });

        // Close mobile menu when clicking on a link
        const mobileMenuLinks = mobileMenu.querySelectorAll('a');
        mobileMenuLinks.forEach(link => {
            link.addEventListener('click', function() {
                mobileMenu.classList.add('hidden');
                mobileMenuButton.setAttribute('aria-expanded', 'false');
            });
        });

        // Close mobile menu when clicking outside
        document.addEventListener('click', function(event) {
            if (!mobileMenu.contains(event.target) &&
                !mobileMenuButton.contains(event.target) &&
                !mobileMenu.classList.contains('hidden')) {
                mobileMenu.classList.add('hidden');
                mobileMenuButton.setAttribute('aria-expanded', 'false');
            }
        });
    }
}

/**
 * Initialize smooth scrolling for anchor links
 */
function initSmoothScroll() {
    const anchorLinks = document.querySelectorAll('a[href^="#"]');

    anchorLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');

            // Skip if it's just "#"
            if (targetId === '#') return;

            const targetElement = document.querySelector(targetId);

            if (targetElement) {
                e.preventDefault();

                // Get navigation height for offset
                const nav = document.querySelector('nav');
                const navHeight = nav ? nav.offsetHeight : 0;

                // Scroll to target with offset
                const targetPosition = targetElement.getBoundingClientRect().top + window.pageYOffset - navHeight - 20;

                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });

                // Update URL without jumping
                history.pushState(null, null, targetId);
            }
        });
    });
}

/**
 * Initialize scroll-triggered animations using Intersection Observer
 */
function initScrollAnimations() {
    // Check if browser supports Intersection Observer
    if (!('IntersectionObserver' in window)) {
        return; // Skip animations on older browsers
    }

    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';

                // Stop observing after animation
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe all article elements and feature sections
    const animatedElements = document.querySelectorAll('article, .animate-on-scroll');

    animatedElements.forEach(element => {
        // Set initial state
        element.style.opacity = '0';
        element.style.transform = 'translateY(30px)';
        element.style.transition = 'opacity 0.6s ease, transform 0.6s ease';

        // Start observing
        observer.observe(element);
    });
}

/**
 * Highlight active navigation item based on scroll position
 */
function initNavHighlight() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('nav a[href^="#"]');

    if (sections.length === 0 || navLinks.length === 0) return;

    window.addEventListener('scroll', function() {
        let current = '';

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;

            if (window.pageYOffset >= sectionTop - 200) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('text-primary', 'font-semibold');

            if (link.getAttribute('href') === '#' + current) {
                link.classList.add('text-primary', 'font-semibold');
            }
        });
    });
}

/**
 * Track CTA button clicks (for analytics or conversion tracking)
 */
function initCTATracking() {
    const ctaButtons = document.querySelectorAll('a[href="register"]');

    ctaButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            // Log CTA click location
            const buttonText = this.textContent.trim();
            const section = this.closest('section')?.id || 'header';

            console.log('CTA Clicked:', {
                text: buttonText,
                section: section,
                timestamp: new Date().toISOString()
            });

            // Optional: Send to analytics service
            // Example: gtag('event', 'cta_click', { section: section });
            // Example: plausible('CTA Click', { props: { section: section } });
        });
    });
}

/**
 * Add loading state to buttons on click
 */
function addButtonLoadingState(button) {
    if (!button.classList.contains('btn-loading')) {
        button.classList.add('btn-loading');
        button.disabled = true;

        // Store original text
        button.dataset.originalText = button.textContent;

        // Remove loading state after navigation
        setTimeout(() => {
            button.classList.remove('btn-loading');
            button.disabled = false;
        }, 2000);
    }
}

/**
 * Keyboard navigation improvements
 */
document.addEventListener('keydown', function(e) {
    // Close mobile menu on Escape key
    if (e.key === 'Escape') {
        const mobileMenu = document.getElementById('mobile-menu');
        const mobileMenuButton = document.getElementById('mobile-menu-button');

        if (mobileMenu && !mobileMenu.classList.contains('hidden')) {
            mobileMenu.classList.add('hidden');
            if (mobileMenuButton) {
                mobileMenuButton.setAttribute('aria-expanded', 'false');
                mobileMenuButton.focus();
            }
        }
    }
});

/**
 * Lazy load images when they come into viewport (if needed)
 */
function initLazyLoading() {
    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver(function(entries) {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    if (img.dataset.src) {
                        img.src = img.dataset.src;
                        img.removeAttribute('data-src');
                        imageObserver.unobserve(img);
                    }
                }
            });
        });

        const lazyImages = document.querySelectorAll('img[data-src]');
        lazyImages.forEach(img => imageObserver.observe(img));
    }
}

/**
 * Detect if user prefers reduced motion
 */
function prefersReducedMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/**
 * Update scroll behavior based on user preferences
 */
if (prefersReducedMotion()) {
    document.documentElement.style.scrollBehavior = 'auto';
}

// Export functions for potential use in other scripts
window.WriterLanding = {
    addButtonLoadingState: addButtonLoadingState,
    prefersReducedMotion: prefersReducedMotion
};
