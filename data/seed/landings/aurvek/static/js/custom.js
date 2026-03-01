// Cyberpunk Landing Page - Custom JavaScript

// Smooth scroll for anchor links
document.addEventListener('DOMContentLoaded', function() {
    // Smooth scroll for navigation links
    const navLinks = document.querySelectorAll('a[href^="#"]');

    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();

            const targetId = this.getAttribute('href');
            if (targetId === '#') return;

            const targetElement = document.querySelector(targetId);

            if (targetElement) {
                const offsetTop = targetElement.offsetTop - 80; // Account for fixed nav

                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Intersection Observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-fade-in');
            }
        });
    }, observerOptions);

    // Observe all major sections
    const sections = document.querySelectorAll('section');
    sections.forEach(section => {
        observer.observe(section);
    });

    // Add parallax effect to grid background
    const gridBg = document.getElementById('grid-bg');

    window.addEventListener('scroll', function() {
        const scrolled = window.pageYOffset;
        const parallaxSpeed = 0.5;

        if (gridBg) {
            gridBg.style.transform = `perspective(500px) rotateX(60deg) translateZ(${scrolled * parallaxSpeed * 0.05}px)`;
        }
    });

    // Add hover effect to feature cards
    const featureCards = document.querySelectorAll('.cyber-card-feature');

    featureCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-8px)';
        });

        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });

    // Add click tracking for CTA buttons (for analytics)
    const ctaButtons = document.querySelectorAll('a[href="register"]');

    ctaButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Track CTA click event
            const buttonText = this.textContent.trim();
            console.log('CTA clicked:', buttonText);

            // You can add analytics tracking here
            // Example: gtag('event', 'cta_click', { button_text: buttonText });
        });
    });

    // Animate stats bars on scroll into view
    const statsBars = document.querySelectorAll('.cyber-bar');

    const statsObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animation = 'fillBar 1.5s ease-out forwards';
                statsObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });

    statsBars.forEach(bar => {
        statsObserver.observe(bar);
    });

    // Mobile menu toggle (if needed in future)
    const mobileMenuButton = document.querySelector('[data-mobile-menu-toggle]');
    const mobileMenu = document.querySelector('[data-mobile-menu]');

    if (mobileMenuButton && mobileMenu) {
        mobileMenuButton.addEventListener('click', function() {
            const isExpanded = this.getAttribute('aria-expanded') === 'true';
            this.setAttribute('aria-expanded', !isExpanded);
            mobileMenu.classList.toggle('hidden');
        });
    }

    // Add typing animation to hero typing indicator
    const typingIndicator = document.querySelector('.animate-pulse');

    if (typingIndicator) {
        // Simulate typing completion after 3 seconds
        setTimeout(() => {
            const parentElement = typingIndicator.parentElement;
            if (parentElement) {
                parentElement.style.opacity = '0';
                parentElement.style.transition = 'opacity 0.5s';

                setTimeout(() => {
                    parentElement.remove();
                }, 500);
            }
        }, 3000);
    }

    // Cyberpunk glitch effect on logo hover
    const logo = document.querySelector('.cyber-glow');

    if (logo) {
        logo.addEventListener('mouseenter', function() {
            this.style.animation = 'glitch 0.3s ease';
        });

        logo.addEventListener('animationend', function() {
            this.style.animation = '';
        });
    }

    // Performance monitoring - preload register page on hover
    const registerLinks = document.querySelectorAll('a[href="register"]');

    registerLinks.forEach(link => {
        link.addEventListener('mouseenter', function() {
            // Preload the register page for faster navigation
            const preloadLink = document.createElement('link');
            preloadLink.rel = 'prefetch';
            preloadLink.href = 'register';

            if (!document.querySelector(`link[href="register"]`)) {
                document.head.appendChild(preloadLink);
            }
        }, { once: true });
    });

    // Add accessibility - keyboard navigation enhancement
    document.addEventListener('keydown', function(e) {
        // Skip to main content with "/" key
        if (e.key === '/' && e.ctrlKey) {
            e.preventDefault();
            const mainContent = document.querySelector('main') || document.querySelector('#features');
            if (mainContent) {
                mainContent.focus();
                mainContent.scrollIntoView({ behavior: 'smooth' });
            }
        }
    });

    // Dynamic year in footer
    const yearElements = document.querySelectorAll('[data-year]');
    const currentYear = new Date().getFullYear();

    yearElements.forEach(element => {
        element.textContent = currentYear;
    });

    // Log page load time for performance monitoring
    window.addEventListener('load', function() {
        const loadTime = performance.now();
        console.log(`Page loaded in ${Math.round(loadTime)}ms`);
    });
});

// Utility function to debounce scroll events
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Optimized scroll handler
const handleScroll = debounce(function() {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

    // Add shadow to nav on scroll
    const nav = document.querySelector('nav');
    if (nav) {
        if (scrollTop > 50) {
            nav.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)';
        } else {
            nav.style.boxShadow = 'none';
        }
    }
}, 10);

window.addEventListener('scroll', handleScroll);

// Export functions for potential use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        debounce
    };
}
