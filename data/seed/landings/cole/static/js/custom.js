// Tinder-style Landing Page JavaScript for Cole

document.addEventListener('DOMContentLoaded', function() {

    // ========================================
    // CARD SWIPE FUNCTIONALITY
    // ========================================

    let currentCardIndex = 0;
    const cards = document.querySelectorAll('.tinder-card');
    const indicators = document.querySelectorAll('.indicator-dot');
    const nopeBtn = document.getElementById('nopeBtn');
    const superBtn = document.getElementById('superBtn');
    const likeBtn = document.getElementById('likeBtn');

    let startX = 0;
    let startY = 0;
    let currentX = 0;
    let currentY = 0;
    let isDragging = false;

    if (cards.length > 0) {
        const activeCard = cards[currentCardIndex];

        // Touch/Mouse Events for Swiping
        activeCard.addEventListener('mousedown', startDrag);
        activeCard.addEventListener('touchstart', startDrag);

        document.addEventListener('mousemove', drag);
        document.addEventListener('touchmove', drag);

        document.addEventListener('mouseup', endDrag);
        document.addEventListener('touchend', endDrag);

        function startDrag(e) {
            if (e.target.closest('.tinder-btn')) return; // Don't drag when clicking buttons

            isDragging = true;
            const card = cards[currentCardIndex];

            if (e.type === 'touchstart') {
                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
            } else {
                startX = e.clientX;
                startY = e.clientY;
            }

            card.style.transition = 'none';
        }

        function drag(e) {
            if (!isDragging) return;

            const card = cards[currentCardIndex];

            if (e.type === 'touchmove') {
                currentX = e.touches[0].clientX;
                currentY = e.touches[0].clientY;
            } else {
                currentX = e.clientX;
                currentY = e.clientY;
            }

            const deltaX = currentX - startX;
            const deltaY = currentY - startY;
            const rotation = deltaX * 0.1;

            card.style.transform = `translateX(${deltaX}px) translateY(${deltaY}px) rotate(${rotation}deg)`;

            // Visual feedback
            if (Math.abs(deltaX) > 50) {
                if (deltaX > 0) {
                    card.classList.add('swiping-right');
                    card.classList.remove('swiping-left');
                } else {
                    card.classList.add('swiping-left');
                    card.classList.remove('swiping-right');
                }
            } else {
                card.classList.remove('swiping-right', 'swiping-left');
            }
        }

        function endDrag(e) {
            if (!isDragging) return;

            isDragging = false;
            const card = cards[currentCardIndex];
            const deltaX = currentX - startX;

            card.style.transition = 'all 0.6s cubic-bezier(0.23, 1, 0.32, 1)';

            // Swipe threshold
            if (Math.abs(deltaX) > 100) {
                if (deltaX > 0) {
                    swipeRight();
                } else {
                    swipeLeft();
                }
            } else {
                // Snap back
                card.style.transform = '';
                card.classList.remove('swiping-right', 'swiping-left');
            }
        }
    }

    // Button Click Handlers
    if (nopeBtn) {
        nopeBtn.addEventListener('click', function(e) {
            e.preventDefault();
            swipeLeft();
        });
    }

    if (superBtn) {
        superBtn.addEventListener('click', function(e) {
            e.preventDefault();
            // Show info or navigate to FAQ
            const faqSection = document.getElementById('why-cole');
            if (faqSection) {
                faqSection.scrollIntoView({ behavior: 'smooth' });
            }
        });
    }

    function swipeLeft() {
        const card = cards[currentCardIndex];
        card.classList.add('swiped-left');
        card.classList.remove('swiping-left');

        setTimeout(() => {
            nextCard();
        }, 600);
    }

    function swipeRight() {
        const card = cards[currentCardIndex];
        card.classList.add('swiped-right');
        card.classList.remove('swiping-right');

        setTimeout(() => {
            nextCard();
        }, 600);
    }

    function nextCard() {
        if (currentCardIndex < cards.length - 1) {
            currentCardIndex++;
            updateIndicators();

            // Remove event listeners from old card
            const oldCard = cards[currentCardIndex - 1];
            oldCard.removeEventListener('mousedown', startDrag);
            oldCard.removeEventListener('touchstart', startDrag);

            // Add event listeners to new card
            const newCard = cards[currentCardIndex];
            newCard.addEventListener('mousedown', startDrag);
            newCard.addEventListener('touchstart', startDrag);
        } else {
            // All cards swiped - scroll to CTA
            setTimeout(() => {
                const ctaSection = document.querySelector('section.bg-gradient-to-br.from-pink-500');
                if (ctaSection) {
                    ctaSection.scrollIntoView({ behavior: 'smooth' });
                }
            }, 300);
        }
    }

    function updateIndicators() {
        indicators.forEach((dot, index) => {
            if (index === currentCardIndex) {
                dot.classList.add('active');
            } else {
                dot.classList.remove('active');
            }
        });
    }

    // Indicator click handlers
    indicators.forEach((dot, index) => {
        dot.addEventListener('click', () => {
            if (index > currentCardIndex) {
                // Swipe through cards to reach the clicked one
                const card = cards[currentCardIndex];
                card.classList.add('swiped-left');

                setTimeout(() => {
                    currentCardIndex = index;
                    updateIndicators();
                }, 300);
            }
        });
    });

    // ========================================
    // SMOOTH SCROLL FOR ANCHOR LINKS
    // ========================================

    const smoothScrollLinks = document.querySelectorAll('a[href^="#"]');

    smoothScrollLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href.startsWith('#')) {
                e.preventDefault();
                const targetId = href.substring(1);
                const targetElement = document.getElementById(targetId);

                if (targetElement) {
                    targetElement.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });

    // ========================================
    // HEADER SCROLL EFFECT
    // ========================================

    const header = document.getElementById('header');
    let lastScroll = 0;

    window.addEventListener('scroll', function() {
        const currentScroll = window.pageYOffset;

        if (currentScroll > 50) {
            header.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.1)';
        } else {
            header.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.05)';
        }

        lastScroll = currentScroll;
    });

    // ========================================
    // INTERSECTION OBSERVER FOR ANIMATIONS
    // ========================================

    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe FAQ items
    const faqItems = document.querySelectorAll('.faq-details');
    faqItems.forEach((item, index) => {
        item.style.opacity = '0';
        item.style.transform = 'translateY(20px)';
        item.style.transition = `opacity 0.5s ease ${index * 0.1}s, transform 0.5s ease ${index * 0.1}s`;
        observer.observe(item);
    });

    // ========================================
    // CTA BUTTON RIPPLE EFFECT
    // ========================================

    const ctaButtons = document.querySelectorAll('a[href="register"]');
    ctaButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;

            ripple.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                left: ${x}px;
                top: ${y}px;
                background: rgba(255, 255, 255, 0.5);
                border-radius: 50%;
                transform: scale(0);
                animation: ripple-animation 0.6s ease-out;
                pointer-events: none;
            `;

            this.style.position = 'relative';
            this.style.overflow = 'hidden';
            this.appendChild(ripple);

            setTimeout(() => ripple.remove(), 600);
        });
    });

    // ========================================
    // KEYBOARD NAVIGATION ENHANCEMENT
    // ========================================

    document.addEventListener('keydown', function(e) {
        if (e.key === 'Tab') {
            document.body.classList.add('keyboard-nav');
        }

        // Arrow keys for card navigation
        if (e.key === 'ArrowLeft') {
            swipeLeft();
        } else if (e.key === 'ArrowRight' && likeBtn) {
            window.location.href = 'register';
        }
    });

    document.addEventListener('mousedown', function() {
        document.body.classList.remove('keyboard-nav');
    });

    // ========================================
    // PERFORMANCE OPTIMIZATION
    // ========================================

    // Add will-change property to animated elements
    cards.forEach(card => {
        card.style.willChange = 'transform, opacity';
    });

    // Remove will-change after animation completes
    setTimeout(() => {
        cards.forEach(card => {
            card.style.willChange = 'auto';
        });
    }, 3000);

    // ========================================
    // CONSOLE EASTER EGG
    // ========================================

    console.log('%c👋 Hey there!', 'color: #ec4899; font-size: 24px; font-weight: bold;');
    console.log('%c💕 Looks like you\'re checking out Cole\'s profile!', 'color: #ef4444; font-size: 16px;');
    console.log('%c🎨 Interested in how this works? Swipe right to get started!', 'color: #6b7280; font-size: 14px; font-style: italic;');
    console.log('%c\n(Psst... try swiping the cards with your mouse or touch!)', 'color: #9ca3af; font-size: 12px;');

});

// Add CSS for keyboard navigation and animations
const style = document.createElement('style');
style.textContent = `
    .keyboard-nav *:focus-visible {
        outline: 3px solid #ec4899 !important;
        outline-offset: 2px !important;
    }

    @keyframes ripple-animation {
        to {
            transform: scale(2);
            opacity: 0;
        }
    }

    /* Smooth transitions for everything */
    * {
        -webkit-tap-highlight-color: transparent;
    }

    /* Prevent text selection during drag */
    .tinder-card.dragging {
        user-select: none;
        -webkit-user-select: none;
    }
`;
document.head.appendChild(style);
