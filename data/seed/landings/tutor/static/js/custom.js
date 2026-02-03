// Custom JavaScript for Tutor landing page

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {

    // Mobile menu toggle
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');
    const mobileMenu = document.getElementById('mobile-menu');

    if (mobileMenuBtn && mobileMenu) {
        mobileMenuBtn.addEventListener('click', function() {
            mobileMenu.classList.toggle('hidden');

            // Update aria-expanded attribute for accessibility
            const isExpanded = !mobileMenu.classList.contains('hidden');
            mobileMenuBtn.setAttribute('aria-expanded', isExpanded);
        });

        // Close mobile menu when clicking on a link
        const mobileLinks = mobileMenu.querySelectorAll('a');
        mobileLinks.forEach(link => {
            link.addEventListener('click', function() {
                mobileMenu.classList.add('hidden');
                mobileMenuBtn.setAttribute('aria-expanded', 'false');
            });
        });

        // Close mobile menu when clicking outside
        document.addEventListener('click', function(event) {
            const isClickInsideMenu = mobileMenu.contains(event.target);
            const isClickOnButton = mobileMenuBtn.contains(event.target);

            if (!isClickInsideMenu && !isClickOnButton && !mobileMenu.classList.contains('hidden')) {
                mobileMenu.classList.add('hidden');
                mobileMenuBtn.setAttribute('aria-expanded', 'false');
            }
        });
    }

    // Smooth scroll with offset for fixed navigation
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');

            // Skip if href is just "#"
            if (targetId === '#') return;

            const targetElement = document.querySelector(targetId);

            if (targetElement) {
                e.preventDefault();

                const navHeight = document.querySelector('nav').offsetHeight;
                const targetPosition = targetElement.offsetTop - navHeight - 20;

                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Add fade-in animation to sections on scroll
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const fadeInObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
                fadeInObserver.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe all feature cards and testimonials
    const animatedElements = document.querySelectorAll('article, section > div > div');
    animatedElements.forEach(el => {
        fadeInObserver.observe(el);
    });

    // Add active state to navigation links based on scroll position
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('nav a[href^="#"]');

    function updateActiveNavLink() {
        const scrollPosition = window.scrollY + 100;

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            const sectionId = section.getAttribute('id');

            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                navLinks.forEach(link => {
                    link.classList.remove('text-primary', 'font-semibold');
                    if (link.getAttribute('href') === `#${sectionId}`) {
                        link.classList.add('text-primary', 'font-semibold');
                    }
                });
            }
        });
    }

    // Throttle scroll events for better performance
    let scrollTimeout;
    window.addEventListener('scroll', function() {
        if (scrollTimeout) {
            window.cancelAnimationFrame(scrollTimeout);
        }
        scrollTimeout = window.requestAnimationFrame(function() {
            updateActiveNavLink();
            updateNavBackground();
        });
    });

    // Add background to nav on scroll
    function updateNavBackground() {
        const nav = document.querySelector('nav');
        if (window.scrollY > 50) {
            nav.classList.add('shadow-md');
        } else {
            nav.classList.remove('shadow-md');
        }
    }

    // Add hover effect to CTA buttons
    const ctaButtons = document.querySelectorAll('a[href="register"]');
    ctaButtons.forEach(button => {
        button.classList.add('pulse-on-hover');
    });

    // Animate stats numbers when they come into view
    const statsSection = document.querySelector('section.bg-secondary');
    if (statsSection) {
        const statsObserver = new IntersectionObserver(function(entries) {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const statNumbers = entry.target.querySelectorAll('.text-5xl');
                    statNumbers.forEach((stat, index) => {
                        setTimeout(() => {
                            stat.classList.add('stat-number');
                            stat.style.opacity = '0';
                            stat.style.transform = 'translateY(20px)';

                            setTimeout(() => {
                                stat.style.transition = 'all 0.6s ease';
                                stat.style.opacity = '1';
                                stat.style.transform = 'translateY(0)';
                            }, 50);
                        }, index * 100);
                    });
                    statsObserver.unobserve(entry.target);
                }
            });
        }, observerOptions);

        statsObserver.observe(statsSection);
    }

    // Add keyboard navigation support
    document.addEventListener('keydown', function(e) {
        // Close mobile menu with Escape key
        if (e.key === 'Escape' && mobileMenu && !mobileMenu.classList.contains('hidden')) {
            mobileMenu.classList.add('hidden');
            mobileMenuBtn.setAttribute('aria-expanded', 'false');
            mobileMenuBtn.focus();
        }
    });

    // Track CTA button clicks (for analytics, if needed)
    ctaButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            // Log to console for demo purposes
            console.log('CTA clicked:', this.textContent, 'from section:', this.closest('section')?.id || 'header');

            // Here you would send to analytics service
            // Example: gtag('event', 'click', { 'event_category': 'CTA', 'event_label': this.textContent });
        });
    });

    // Add visual feedback for form validation (if forms are added later)
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const submitBtn = this.querySelector('[type="submit"]');
            if (submitBtn) {
                submitBtn.classList.add('opacity-50', 'cursor-not-allowed');
                submitBtn.disabled = true;
                submitBtn.textContent = 'Processing...';
            }
        });
    });

    // Easter egg: Log welcome message
    console.log('%cWelcome to Tutor! 🎓', 'font-size: 20px; font-weight: bold; color: #ff8040;');
    console.log('%cReady to learn smarter, not harder?', 'font-size: 14px; color: #004000;');

    // Preload images on hover for better UX
    const links = document.querySelectorAll('a[href="register"]');
    links.forEach(link => {
        link.addEventListener('mouseenter', function() {
            // Preload the registration page in the background
            const prefetchLink = document.createElement('link');
            prefetchLink.rel = 'prefetch';
            prefetchLink.href = 'register';
            document.head.appendChild(prefetchLink);
        }, { once: true });
    });

    // Add copy-to-clipboard functionality for any code blocks (if added later)
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(block => {
        const button = document.createElement('button');
        button.textContent = 'Copy';
        button.className = 'copy-btn';
        button.addEventListener('click', function() {
            navigator.clipboard.writeText(block.textContent).then(() => {
                button.textContent = 'Copied!';
                setTimeout(() => {
                    button.textContent = 'Copy';
                }, 2000);
            });
        });
        block.parentElement.style.position = 'relative';
        block.parentElement.appendChild(button);
    });

    console.log('Tutor landing page loaded successfully!');
});
