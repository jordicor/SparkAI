// Nova-Orion - Multi-Dimensional Interactive Experience

// Smooth scroll function
function scrollToSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Dimensional Canvas Animation
class DimensionalCanvas {
    constructor() {
        this.canvas = document.getElementById('dimensional-canvas');
        this.particles = [];
        this.connections = [];
        this.mouseX = 0;
        this.mouseY = 0;
        this.init();
    }

    init() {
        // Create particle effect using CSS
        this.createParticles();
        this.animate();
        this.setupMouseTracking();
    }

    createParticles() {
        const particleCount = 50;
        for (let i = 0; i < particleCount; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.cssText = `
                position: absolute;
                width: ${Math.random() * 4 + 1}px;
                height: ${Math.random() * 4 + 1}px;
                background: ${Math.random() > 0.5 ? 'rgba(0, 255, 255, 0.6)' : 'rgba(0, 255, 128, 0.6)'};
                border-radius: 50%;
                pointer-events: none;
                box-shadow: 0 0 10px currentColor;
            `;

            const particle_data = {
                element: particle,
                x: Math.random() * window.innerWidth,
                y: Math.random() * window.innerHeight,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                size: Math.random() * 4 + 1
            };

            this.particles.push(particle_data);
            this.canvas.appendChild(particle);
        }
    }

    setupMouseTracking() {
        document.addEventListener('mousemove', (e) => {
            this.mouseX = e.clientX;
            this.mouseY = e.clientY;
        });
    }

    animate() {
        this.particles.forEach((particle) => {
            // Update position
            particle.x += particle.vx;
            particle.y += particle.vy;

            // Bounce off edges
            if (particle.x < 0 || particle.x > window.innerWidth) {
                particle.vx *= -1;
            }
            if (particle.y < 0 || particle.y > window.innerHeight) {
                particle.vy *= -1;
            }

            // Mouse attraction
            const dx = this.mouseX - particle.x;
            const dy = this.mouseY - particle.y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance < 200) {
                particle.vx += dx * 0.00005;
                particle.vy += dy * 0.00005;
            }

            // Apply velocity damping
            particle.vx *= 0.99;
            particle.vy *= 0.99;

            // Update DOM
            particle.element.style.transform = `translate(${particle.x}px, ${particle.y}px)`;
        });

        requestAnimationFrame(() => this.animate());
    }
}

// Initialize dimensional canvas
const dimensionalCanvas = new DimensionalCanvas();

// Intersection Observer for scroll animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe all sections
document.querySelectorAll('section').forEach((section) => {
    section.style.opacity = '0';
    section.style.transform = 'translateY(30px)';
    section.style.transition = 'opacity 1s ease, transform 1s ease';
    observer.observe(section);
});

// Observe cards
document.querySelectorAll('.manifesto-card, .dimension-card, .experience-card').forEach((card, index) => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(30px)';
    card.style.transition = `opacity 0.6s ease ${index * 0.1}s, transform 0.6s ease ${index * 0.1}s`;
    observer.observe(card);
});

// Parallax effect for hero section
let lastScrollY = window.scrollY;

window.addEventListener('scroll', () => {
    const scrollY = window.scrollY;
    const hero = document.getElementById('hero');

    if (hero) {
        const parallaxSpeed = 0.5;
        hero.style.transform = `translateY(${scrollY * parallaxSpeed}px)`;
    }

    lastScrollY = scrollY;
});

// Navigation background on scroll
const nav = document.querySelector('nav');
window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
        nav.style.background = 'rgba(0, 0, 0, 0.8)';
        nav.style.backdropFilter = 'blur(20px)';
    } else {
        nav.style.background = 'rgba(0, 0, 0, 0.3)';
        nav.style.backdropFilter = 'blur(10px)';
    }
});

// Cursor trail effect
class CursorTrail {
    constructor() {
        this.trails = [];
        this.maxTrails = 20;
        this.init();
    }

    init() {
        document.addEventListener('mousemove', (e) => {
            this.createTrail(e.clientX, e.clientY);
        });
    }

    createTrail(x, y) {
        const trail = document.createElement('div');
        trail.style.cssText = `
            position: fixed;
            width: 8px;
            height: 8px;
            background: radial-gradient(circle, rgba(0, 255, 255, 0.6), transparent);
            border-radius: 50%;
            pointer-events: none;
            z-index: 9999;
            transform: translate(-50%, -50%);
            left: ${x}px;
            top: ${y}px;
            animation: trailFade 1s ease-out forwards;
        `;

        document.body.appendChild(trail);

        setTimeout(() => {
            trail.remove();
        }, 1000);
    }
}

// Add trail animation CSS
const trailStyle = document.createElement('style');
trailStyle.textContent = `
    @keyframes trailFade {
        0% {
            transform: translate(-50%, -50%) scale(1);
            opacity: 1;
        }
        100% {
            transform: translate(-50%, -50%) scale(0);
            opacity: 0;
        }
    }
`;
document.head.appendChild(trailStyle);

// Initialize cursor trail
const cursorTrail = new CursorTrail();

// Button ripple effect
document.querySelectorAll('a[href="register"], button').forEach((button) => {
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
            animation: ripple 0.6s ease-out;
            pointer-events: none;
        `;

        this.style.position = 'relative';
        this.style.overflow = 'hidden';
        this.appendChild(ripple);

        setTimeout(() => ripple.remove(), 600);
    });
});

// Add ripple animation CSS
const rippleStyle = document.createElement('style');
rippleStyle.textContent = `
    @keyframes ripple {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
`;
document.head.appendChild(rippleStyle);

// Typing effect for hero subtitle (optional enhancement)
function typeWriter(element, text, speed = 50) {
    let i = 0;
    element.textContent = '';

    function type() {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    }

    type();
}

// Add glitch effect to cards on hover
document.querySelectorAll('.manifesto-card, .dimension-card').forEach((card) => {
    card.addEventListener('mouseenter', function() {
        this.style.animation = 'cardGlitch 0.3s ease';
    });

    card.addEventListener('animationend', function() {
        this.style.animation = '';
    });
});

const glitchStyle = document.createElement('style');
glitchStyle.textContent = `
    @keyframes cardGlitch {
        0%, 100% { transform: translate(0, 0); }
        25% { transform: translate(-2px, 2px); }
        50% { transform: translate(2px, -2px); }
        75% { transform: translate(-2px, -2px); }
    }
`;
document.head.appendChild(glitchStyle);

// Performance optimization: Disable animations on low-end devices
if (navigator.hardwareConcurrency && navigator.hardwareConcurrency < 4) {
    document.body.classList.add('reduced-motion');
    const reducedMotionStyle = document.createElement('style');
    reducedMotionStyle.textContent = `
        .reduced-motion *,
        .reduced-motion *::before,
        .reduced-motion *::after {
            animation-duration: 0.01ms !important;
            transition-duration: 0.01ms !important;
        }
    `;
    document.head.appendChild(reducedMotionStyle);
}

// Accessibility: Skip to main content
const skipLink = document.createElement('a');
skipLink.href = '#hero';
skipLink.textContent = 'Skip to main content';
skipLink.style.cssText = `
    position: absolute;
    top: -100px;
    left: 0;
    background: #00ffff;
    color: #000;
    padding: 8px 16px;
    text-decoration: none;
    z-index: 10000;
    transition: top 0.3s;
`;
skipLink.addEventListener('focus', () => {
    skipLink.style.top = '0';
});
skipLink.addEventListener('blur', () => {
    skipLink.style.top = '-100px';
});
document.body.prepend(skipLink);

// Console easter egg
console.log('%c🌌 NOVA-ORION 🌌', 'color: #00ffff; font-size: 24px; font-weight: bold; text-shadow: 0 0 10px #00ffff;');
console.log('%cCurious by nature. Think WITH you, not FOR you.', 'color: #00ff80; font-size: 14px;');
console.log('%cInterested in the code? We think that is pretty cool.', 'color: #fff; font-size: 12px;');

// Log page load time for performance monitoring
window.addEventListener('load', () => {
    const loadTime = window.performance.timing.domContentLoadedEventEnd - window.performance.timing.navigationStart;
    console.log(`%cPage loaded in ${loadTime}ms`, 'color: #00ffff; font-weight: bold;');
});