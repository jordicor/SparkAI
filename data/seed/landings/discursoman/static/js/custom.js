// DiscursoMan Landing Page - Interactive Features

// Smooth scroll for anchor links
document.addEventListener('DOMContentLoaded', function() {

    // Smooth scroll for navigation links
    const navLinks = document.querySelectorAll('a[href^="#"]');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;

            const targetSection = document.querySelector(targetId);
            if (targetSection) {
                const navHeight = document.querySelector('nav').offsetHeight;
                const targetPosition = targetSection.offsetTop - navHeight;

                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Add scroll-based animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-fade-in-up');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe feature cards, testimonials, and pricing cards
    const animatedElements = document.querySelectorAll(
        '.grid > div, section > div > div, .space-y-8 > div'
    );
    animatedElements.forEach(el => {
        observer.observe(el);
    });

    // Navbar scroll effect
    const navbar = document.querySelector('nav');
    let lastScroll = 0;

    window.addEventListener('scroll', function() {
        const currentScroll = window.pageYOffset;

        // Add shadow on scroll
        if (currentScroll > 10) {
            navbar.classList.add('shadow-lg');
        } else {
            navbar.classList.remove('shadow-lg');
        }

        // Optional: Hide navbar on scroll down, show on scroll up
        // Uncomment if you want this behavior
        /*
        if (currentScroll > lastScroll && currentScroll > 100) {
            navbar.style.transform = 'translateY(-100%)';
        } else {
            navbar.style.transform = 'translateY(0)';
        }
        */

        lastScroll = currentScroll;
    });

    // Stats counter animation
    const statsNumbers = document.querySelectorAll('.text-4xl.font-bold');
    const statsObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateNumber(entry.target);
                statsObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });

    statsNumbers.forEach(stat => {
        if (stat.textContent.match(/\d/)) {
            statsObserver.observe(stat);
        }
    });

    // Animate numbers
    function animateNumber(element) {
        const text = element.textContent;
        const hasPlus = text.includes('+');
        const hasPercent = text.includes('%');
        const number = parseInt(text.replace(/[^0-9]/g, ''));

        if (isNaN(number)) return;

        const duration = 2000;
        const steps = 60;
        const increment = number / steps;
        let current = 0;

        const timer = setInterval(() => {
            current += increment;
            if (current >= number) {
                current = number;
                clearInterval(timer);
            }

            let displayText = Math.floor(current).toLocaleString('es-ES');
            if (hasPlus) displayText += '+';
            if (hasPercent) displayText += '%';

            element.textContent = displayText;
        }, duration / steps);
    }

    // CTA button click tracking (placeholder for analytics)
    const ctaButtons = document.querySelectorAll('a[href="#comenzar"], a[href="#precios"]');
    ctaButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Add analytics tracking here
            console.log('CTA clicked:', this.textContent.trim());
        });
    });

    // Mobile menu toggle (if needed in future)
    // You can add a hamburger menu for mobile here

    // Form validation placeholder (if forms are added later)
    function validateForm(formElement) {
        const inputs = formElement.querySelectorAll('input[required], textarea[required]');
        let isValid = true;

        inputs.forEach(input => {
            if (!input.value.trim()) {
                isValid = false;
                input.classList.add('border-red-500');
            } else {
                input.classList.remove('border-red-500');
            }
        });

        return isValid;
    }

    // Lazy loading for images (if images are added)
    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver(function(entries) {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    if (img.dataset.src) {
                        img.src = img.dataset.src;
                        img.classList.remove('img-loading');
                        imageObserver.unobserve(img);
                    }
                }
            });
        });

        document.querySelectorAll('img[data-src]').forEach(img => {
            imageObserver.observe(img);
        });
    }

    // Parallax effect on hero section (subtle)
    const heroSection = document.querySelector('section');
    if (heroSection) {
        window.addEventListener('scroll', function() {
            const scrolled = window.pageYOffset;
            const parallaxElements = heroSection.querySelectorAll('.parallax-layer');

            parallaxElements.forEach((el, index) => {
                const speed = (index + 1) * 0.1;
                el.style.transform = `translateY(${scrolled * speed}px)`;
            });
        });
    }

    // Add ripple effect to buttons
    function createRipple(event) {
        const button = event.currentTarget;
        const ripple = document.createElement('span');
        const diameter = Math.max(button.clientWidth, button.clientHeight);
        const radius = diameter / 2;

        ripple.style.width = ripple.style.height = `${diameter}px`;
        ripple.style.left = `${event.clientX - button.offsetLeft - radius}px`;
        ripple.style.top = `${event.clientY - button.offsetTop - radius}px`;
        ripple.classList.add('ripple');

        const rippleContainer = button.querySelector('.ripple');
        if (rippleContainer) {
            rippleContainer.remove();
        }

        button.appendChild(ripple);
    }

    const buttons = document.querySelectorAll('button, a.bg-primary, a.bg-secondary');
    buttons.forEach(button => {
        button.addEventListener('click', createRipple);
    });

    // Easter egg: Konami code
    let konamiCode = [];
    const konamiSequence = [38, 38, 40, 40, 37, 39, 37, 39, 66, 65];

    document.addEventListener('keydown', function(e) {
        konamiCode.push(e.keyCode);
        konamiCode.splice(-konamiSequence.length - 1, konamiCode.length - konamiSequence.length);

        if (konamiCode.join('').includes(konamiSequence.join(''))) {
            activateEasterEgg();
        }
    });

    function activateEasterEgg() {
        document.body.style.animation = 'rainbow 2s linear infinite';
        setTimeout(() => {
            document.body.style.animation = '';
        }, 5000);
    }

    // Accessibility: Skip to main content
    const skipLink = document.createElement('a');
    skipLink.href = '#caracteristicas';
    skipLink.textContent = 'Saltar al contenido principal';
    skipLink.className = 'sr-only focus:not-sr-only focus:absolute focus:top-0 focus:left-0 bg-primary text-white px-4 py-2 z-50';
    document.body.insertBefore(skipLink, document.body.firstChild);

    // Performance: Reduce animations on low-end devices
    if (navigator.hardwareConcurrency <= 2) {
        document.body.classList.add('reduce-motion');
    }

    // Console message for developers
    console.log('%c¡Hola, Desarrollador! 👋', 'font-size: 20px; font-weight: bold; color: #3b82f6;');
    console.log('%c¿Interesado en cómo funciona DiscursoMan? Visita nuestra documentación.', 'font-size: 14px; color: #10b981;');

    // Track time on page (for analytics)
    let timeOnPage = 0;
    setInterval(() => {
        timeOnPage++;
        // Send to analytics every 30 seconds
        if (timeOnPage % 30 === 0) {
            console.log('Time on page:', timeOnPage, 'seconds');
            // window.analytics.track('time_on_page', { seconds: timeOnPage });
        }
    }, 1000);

    // Detect if user is about to leave (exit intent)
    document.addEventListener('mouseleave', function(e) {
        if (e.clientY <= 0) {
            // User is moving mouse out of the page
            // Show exit intent popup or special offer
            console.log('Exit intent detected');
            // showExitIntentModal();
        }
    });

    // Preload critical resources
    function preloadResource(url, type) {
        const link = document.createElement('link');
        link.rel = 'preload';
        link.href = url;
        link.as = type;
        document.head.appendChild(link);
    }

    // Service worker registration (for PWA features)
    if ('serviceWorker' in navigator) {
        // Uncomment when service worker is implemented
        // navigator.serviceWorker.register('/sw.js').then(function(registration) {
        //     console.log('Service Worker registered:', registration);
        // });
    }

    // Add keyboard navigation improvements
    document.addEventListener('keydown', function(e) {
        // Escape key closes modals
        if (e.key === 'Escape') {
            const modals = document.querySelectorAll('.modal');
            modals.forEach(modal => {
                modal.classList.add('hidden');
            });
        }
    });

    // Log page performance metrics
    window.addEventListener('load', function() {
        if (window.performance) {
            const perfData = window.performance.timing;
            const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
            console.log('Page load time:', pageLoadTime, 'ms');
        }
    });

    console.log('DiscursoMan landing page initialized successfully!');
});

// Utility function: Debounce
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

// Utility function: Throttle
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}