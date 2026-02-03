/**
 * Agente Chillón - Custom Interactive Features
 * Rustic tavern experience enhancements
 */

(function() {
    'use strict';

    // Wait for DOM to be fully loaded
    document.addEventListener('DOMContentLoaded', function() {

        // Smooth scroll for anchor links
        initSmoothScroll();

        // Add parallax effect to hero section
        initParallaxEffect();

        // Animate elements on scroll
        initScrollAnimations();

        // Add hover sound effect simulation (visual feedback)
        initButtonEffects();

        // Track CTA clicks for analytics (if needed)
        trackCTAClicks();

        // Add rustic shake effect on certain elements
        initRusticEffects();
    });

    /**
     * Smooth scrolling for anchor links
     */
    function initSmoothScroll() {
        const links = document.querySelectorAll('a[href^="#"]');

        links.forEach(link => {
            link.addEventListener('click', function(e) {
                const href = this.getAttribute('href');

                // Skip empty anchors
                if (href === '#') return;

                e.preventDefault();

                const target = document.querySelector(href);
                if (target) {
                    const headerOffset = 80;
                    const elementPosition = target.getBoundingClientRect().top;
                    const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

                    window.scrollTo({
                        top: offsetPosition,
                        behavior: 'smooth'
                    });
                }
            });
        });
    }

    /**
     * Parallax effect for hero section
     */
    function initParallaxEffect() {
        const hero = document.querySelector('main > section:first-child');

        if (!hero) return;

        let ticking = false;

        window.addEventListener('scroll', function() {
            if (!ticking) {
                window.requestAnimationFrame(function() {
                    const scrolled = window.pageYOffset;
                    const heroHeight = hero.offsetHeight;

                    // Only apply parallax while hero is visible
                    if (scrolled < heroHeight) {
                        const parallaxSpeed = 0.5;
                        hero.style.transform = `translateY(${scrolled * parallaxSpeed}px)`;
                        hero.style.opacity = 1 - (scrolled / heroHeight) * 0.5;
                    }

                    ticking = false;
                });

                ticking = true;
            }
        });
    }

    /**
     * Animate elements when they come into view
     */
    function initScrollAnimations() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver(function(entries) {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                    observer.unobserve(entry.target);
                }
            });
        }, observerOptions);

        // Observe article elements
        const articles = document.querySelectorAll('article');
        articles.forEach((article, index) => {
            // Initial state
            article.style.opacity = '0';
            article.style.transform = 'translateY(30px)';
            article.style.transition = `opacity 0.6s ease ${index * 0.1}s, transform 0.6s ease ${index * 0.1}s`;

            observer.observe(article);
        });
    }

    /**
     * Add interactive effects to buttons
     */
    function initButtonEffects() {
        const ctaButtons = document.querySelectorAll('a[href="register"]');

        ctaButtons.forEach(button => {
            // Add ripple effect on click
            button.addEventListener('click', function(e) {
                const ripple = document.createElement('span');
                ripple.style.position = 'absolute';
                ripple.style.borderRadius = '50%';
                ripple.style.background = 'rgba(255, 255, 255, 0.6)';
                ripple.style.width = '20px';
                ripple.style.height = '20px';
                ripple.style.animation = 'ripple 0.6s ease-out';
                ripple.style.pointerEvents = 'none';

                const rect = button.getBoundingClientRect();
                ripple.style.left = (e.clientX - rect.left - 10) + 'px';
                ripple.style.top = (e.clientY - rect.top - 10) + 'px';

                button.appendChild(ripple);

                setTimeout(() => ripple.remove(), 600);
            });

            // Add shake effect on hover
            button.addEventListener('mouseenter', function() {
                this.style.animation = 'shake 0.3s ease';
            });

            button.addEventListener('animationend', function() {
                this.style.animation = '';
            });
        });
    }

    /**
     * Track CTA button clicks
     */
    function trackCTAClicks() {
        const ctaButtons = document.querySelectorAll('a[href="register"]');

        ctaButtons.forEach((button, index) => {
            button.addEventListener('click', function() {
                // Log to console (replace with actual analytics if needed)
                console.log(`CTA clicked: Button ${index + 1} - ${button.textContent.trim()}`);

                // Optional: Send to analytics service
                // if (typeof gtag !== 'undefined') {
                //     gtag('event', 'cta_click', {
                //         'event_category': 'engagement',
                //         'event_label': button.textContent.trim()
                //     });
                // }
            });
        });
    }

    /**
     * Add rustic shake and glow effects
     */
    function initRusticEffects() {
        // Add CSS for shake animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes shake {
                0%, 100% { transform: translateX(0); }
                25% { transform: translateX(-5px) rotate(-1deg); }
                75% { transform: translateX(5px) rotate(1deg); }
            }

            @keyframes ripple {
                to {
                    width: 200px;
                    height: 200px;
                    opacity: 0;
                }
            }

            @keyframes glow {
                0%, 100% {
                    box-shadow: 0 0 5px rgba(143, 91, 48, 0.5),
                                0 0 10px rgba(143, 91, 48, 0.3);
                }
                50% {
                    box-shadow: 0 0 10px rgba(143, 91, 48, 0.8),
                                0 0 20px rgba(143, 91, 48, 0.5),
                                0 0 30px rgba(143, 91, 48, 0.3);
                }
            }
        `;
        document.head.appendChild(style);

        // Add glow effect to important elements
        const glowElements = document.querySelectorAll('h1, h2');
        glowElements.forEach(element => {
            element.addEventListener('mouseenter', function() {
                this.style.animation = 'glow 2s ease-in-out infinite';
            });

            element.addEventListener('mouseleave', function() {
                this.style.animation = '';
            });
        });
    }

    /**
     * Add "Back to Top" button functionality
     */
    function initBackToTop() {
        const backToTop = document.createElement('button');
        backToTop.innerHTML = '↑';
        backToTop.className = 'fixed bottom-8 right-8 bg-rustic text-white w-12 h-12 rounded-full shadow-lg opacity-0 transition-opacity duration-300 z-50';
        backToTop.setAttribute('aria-label', 'Back to top');
        backToTop.style.display = 'none';

        document.body.appendChild(backToTop);

        // Show/hide based on scroll position
        window.addEventListener('scroll', function() {
            if (window.pageYOffset > 300) {
                backToTop.style.display = 'block';
                setTimeout(() => backToTop.style.opacity = '1', 10);
            } else {
                backToTop.style.opacity = '0';
                setTimeout(() => backToTop.style.display = 'none', 300);
            }
        });

        // Scroll to top on click
        backToTop.addEventListener('click', function() {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }

    // Initialize back to top button
    initBackToTop();

    /**
     * Add easter egg - Konami code for special effect
     */
    function initEasterEgg() {
        const konamiCode = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'b', 'a'];
        let konamiIndex = 0;

        document.addEventListener('keydown', function(e) {
            if (e.key === konamiCode[konamiIndex]) {
                konamiIndex++;

                if (konamiIndex === konamiCode.length) {
                    // Easter egg activated
                    document.body.style.animation = 'shake 0.5s ease';

                    const message = document.createElement('div');
                    message.textContent = '🍺 TAVERN MODE ACTIVATED! 🍺';
                    message.className = 'fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-rustic text-white px-8 py-4 rounded-lg text-2xl font-bold z-50 shadow-2xl';
                    document.body.appendChild(message);

                    setTimeout(() => {
                        message.style.opacity = '0';
                        message.style.transition = 'opacity 1s ease';
                        setTimeout(() => message.remove(), 1000);
                    }, 2000);

                    konamiIndex = 0;
                }
            } else {
                konamiIndex = 0;
            }
        });
    }

    // Initialize easter egg
    initEasterEgg();

    /**
     * Lazy load images if any are added
     */
    function initLazyLoad() {
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

    // Initialize lazy loading
    initLazyLoad();

})();
