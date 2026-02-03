// Custom JavaScript for Coach landing page

document.addEventListener('DOMContentLoaded', function() {

    // Mobile menu toggle
    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');

    if (mobileMenuButton && mobileMenu) {
        mobileMenuButton.addEventListener('click', function() {
            mobileMenu.classList.toggle('hidden');
            mobileMenu.classList.toggle('active');

            // Update ARIA expanded attribute
            const isExpanded = !mobileMenu.classList.contains('hidden');
            mobileMenuButton.setAttribute('aria-expanded', isExpanded);
        });

        // Close mobile menu when clicking on a link
        const mobileMenuLinks = mobileMenu.querySelectorAll('a');
        mobileMenuLinks.forEach(link => {
            link.addEventListener('click', function() {
                mobileMenu.classList.add('hidden');
                mobileMenu.classList.remove('active');
                mobileMenuButton.setAttribute('aria-expanded', 'false');
            });
        });
    }

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const href = this.getAttribute('href');

            // Only prevent default for hash links (not just "#")
            if (href !== '#' && href.length > 1) {
                e.preventDefault();
                const target = document.querySelector(href);

                if (target) {
                    const navHeight = document.querySelector('nav').offsetHeight;
                    const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - navHeight - 20;

                    window.scrollTo({
                        top: targetPosition,
                        behavior: 'smooth'
                    });
                }
            }
        });
    });

    // Navbar background on scroll
    const nav = document.querySelector('nav');
    let lastScroll = 0;

    window.addEventListener('scroll', function() {
        const currentScroll = window.pageYOffset;

        // Add shadow when scrolled
        if (currentScroll > 50) {
            nav.classList.add('shadow-lg');
        } else {
            nav.classList.remove('shadow-lg');
        }

        lastScroll = currentScroll;
    });

    // Intersection Observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
                // Optionally unobserve after animation
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe all sections and articles for reveal animations
    const revealElements = document.querySelectorAll('section, article');
    revealElements.forEach(element => {
        element.classList.add('reveal');
        observer.observe(element);
    });

    // Add loaded class to images when they load
    const images = document.querySelectorAll('img[loading="lazy"]');
    images.forEach(img => {
        if (img.complete) {
            img.classList.add('loaded');
        } else {
            img.addEventListener('load', function() {
                this.classList.add('loaded');
            });
        }
    });

    // Animate stats/numbers on scroll (if you add stats counters)
    function animateValue(element, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const value = Math.floor(progress * (end - start) + start);
            element.textContent = value.toLocaleString();
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }

    // Track CTA button clicks (for analytics)
    const ctaButtons = document.querySelectorAll('a[href="register"]');
    ctaButtons.forEach((button, index) => {
        button.addEventListener('click', function() {
            // Log to console (replace with actual analytics tracking)
            console.log('CTA clicked:', {
                position: index,
                text: this.textContent.trim(),
                timestamp: new Date().toISOString()
            });

            // If you're using Google Analytics:
            // gtag('event', 'cta_click', {
            //     'event_category': 'engagement',
            //     'event_label': this.textContent.trim(),
            //     'value': index
            // });
        });
    });

    // Testimonial interaction tracking
    const testimonialCards = document.querySelectorAll('#testimonials article');
    testimonialCards.forEach((card, index) => {
        card.addEventListener('mouseenter', function() {
            // Optional: track which testimonials users read
            console.log('Testimonial viewed:', index);
        });
    });

    // FAQ accordion behavior (optional enhancement)
    const faqItems = document.querySelectorAll('section article');
    faqItems.forEach(item => {
        const heading = item.querySelector('h3');
        if (heading) {
            heading.style.cursor = 'pointer';
            heading.addEventListener('click', function() {
                const content = item.querySelector('p');
                if (content) {
                    content.classList.toggle('hidden');
                }
            });
        }
    });

    // Add keyboard navigation support
    document.addEventListener('keydown', function(e) {
        // ESC key closes mobile menu
        if (e.key === 'Escape' && mobileMenu && !mobileMenu.classList.contains('hidden')) {
            mobileMenu.classList.add('hidden');
            mobileMenu.classList.remove('active');
            if (mobileMenuButton) {
                mobileMenuButton.setAttribute('aria-expanded', 'false');
            }
        }
    });

    // Performance optimization: Lazy load background images
    const lazyBackgrounds = document.querySelectorAll('[data-bg]');
    const bgObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const element = entry.target;
                element.style.backgroundImage = `url(${element.dataset.bg})`;
                bgObserver.unobserve(element);
            }
        });
    });

    lazyBackgrounds.forEach(bg => {
        bgObserver.observe(bg);
    });

    // Add "Back to Top" button functionality
    const backToTopButton = document.createElement('button');
    backToTopButton.innerHTML = '&uarr;';
    backToTopButton.className = 'fixed bottom-8 right-8 bg-primary text-white w-12 h-12 rounded-full shadow-lg hover:bg-primary-dark transition-all opacity-0 pointer-events-none z-50';
    backToTopButton.setAttribute('aria-label', 'Back to top');
    document.body.appendChild(backToTopButton);

    window.addEventListener('scroll', function() {
        if (window.pageYOffset > 500) {
            backToTopButton.style.opacity = '1';
            backToTopButton.style.pointerEvents = 'auto';
        } else {
            backToTopButton.style.opacity = '0';
            backToTopButton.style.pointerEvents = 'none';
        }
    });

    backToTopButton.addEventListener('click', function() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });

    // Track scroll depth (for analytics)
    let maxScrollDepth = 0;
    window.addEventListener('scroll', function() {
        const scrollPercentage = (window.pageYOffset / (document.documentElement.scrollHeight - window.innerHeight)) * 100;
        if (scrollPercentage > maxScrollDepth) {
            maxScrollDepth = Math.floor(scrollPercentage);

            // Log milestone scroll depths
            if ([25, 50, 75, 90].includes(maxScrollDepth)) {
                console.log('Scroll depth:', maxScrollDepth + '%');

                // If using Google Analytics:
                // gtag('event', 'scroll_depth', {
                //     'event_category': 'engagement',
                //     'event_label': maxScrollDepth + '%',
                //     'value': maxScrollDepth
                // });
            }
        }
    });

    // Prevent layout shift by setting aspect ratios
    const images_to_optimize = document.querySelectorAll('img:not([width]):not([height])');
    images_to_optimize.forEach(img => {
        img.addEventListener('load', function() {
            if (!this.hasAttribute('width')) {
                this.style.aspectRatio = `${this.naturalWidth} / ${this.naturalHeight}`;
            }
        });
    });

    // Add visible focus indicators for keyboard navigation
    document.body.addEventListener('keydown', function(e) {
        if (e.key === 'Tab') {
            document.body.classList.add('keyboard-nav');
        }
    });

    document.body.addEventListener('mousedown', function() {
        document.body.classList.remove('keyboard-nav');
    });

    // Console welcome message
    console.log('%c👋 Welcome to Coach!', 'font-size: 20px; font-weight: bold; color: #000080;');
    console.log('%cReady to transform your life? Start your journey at the register link above.', 'font-size: 14px; color: #666;');

});

// Utility function to detect if element is in viewport
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

// Utility function for throttling scroll events
function throttle(func, wait) {
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

// Export functions if using modules (optional)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        isInViewport,
        throttle
    };
}
