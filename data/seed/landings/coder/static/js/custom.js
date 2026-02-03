// Terminal OS Landing Page - Custom JavaScript

// Boot Sequence Configuration
const bootMessages = [
    'CODER_OS BOOTLOADER v2.1.0',
    'Initializing system components...',
    'Loading kernel modules................ [OK]',
    'Mounting file systems................ [OK]',
    'Starting security protocols.......... [OK]',
    'Initializing AI engine............... [OK]',
    'Loading language models.............. [OK]',
    '  - Python support................... [OK]',
    '  - JavaScript/TypeScript............ [OK]',
    '  - Go support....................... [OK]',
    '  - Rust support..................... [OK]',
    '  - C/C++ support.................... [OK]',
    'Starting code analysis service....... [OK]',
    'Enabling security scanner............ [OK]',
    'Loading best practices database...... [OK]',
    'System initialization complete.',
    '',
    'Welcome to CODER_OS',
    'Type "help" for available commands',
    ''
];

// Boot Sequence Animation
function bootSequence() {
    const bootTextElement = document.getElementById('boot-text');
    const mainOS = document.getElementById('main-os');
    const bootOverlay = document.getElementById('boot-sequence');

    let messageIndex = 0;
    let charIndex = 0;
    let currentLine = null;

    function typeNextChar() {
        if (messageIndex >= bootMessages.length) {
            // Boot complete - show main OS
            setTimeout(() => {
                bootOverlay.classList.add('fade-out');
                setTimeout(() => {
                    bootOverlay.style.display = 'none';
                    mainOS.classList.remove('hidden');
                    initializeMainOS();
                }, 1000);
            }, 500);
            return;
        }

        if (charIndex === 0) {
            // Create new line
            currentLine = document.createElement('div');
            currentLine.className = 'text-green-400 font-mono text-sm';
            bootTextElement.appendChild(currentLine);
        }

        const currentMessage = bootMessages[messageIndex];

        if (charIndex < currentMessage.length) {
            currentLine.textContent += currentMessage[charIndex];
            charIndex++;

            // Random typing speed for realistic effect
            const speed = Math.random() * 30 + 10;
            setTimeout(typeNextChar, speed);
        } else {
            // Move to next message
            messageIndex++;
            charIndex = 0;

            // Pause between lines
            setTimeout(typeNextChar, 100);
        }
    }

    // Start boot sequence
    typeNextChar();
}

// Initialize Main OS Interface
function initializeMainOS() {
    // Smooth scroll for navigation
    initSmoothScroll();

    // Terminal panel interactions
    initTerminalPanels();

    // Add matrix rain effect (subtle)
    addMatrixRain();

    // Parallax scroll effects
    initParallaxEffects();
}

// Smooth Scroll for Navigation Links
function initSmoothScroll() {
    const links = document.querySelectorAll('a[href^="#"]');

    links.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);

            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Terminal Panel Interactions
function initTerminalPanels() {
    const panels = document.querySelectorAll('.terminal-panel');

    panels.forEach(panel => {
        panel.addEventListener('click', () => {
            const targetSection = panel.getAttribute('data-panel');

            // Flash effect
            panel.style.backgroundColor = 'rgba(0, 255, 0, 0.2)';
            setTimeout(() => {
                panel.style.backgroundColor = '';
            }, 200);

            // Navigate to section
            if (targetSection) {
                const section = document.getElementById(targetSection);
                if (section) {
                    section.scrollIntoView({ behavior: 'smooth' });
                }
            }
        });

        // Hover sound effect simulation (visual feedback)
        panel.addEventListener('mouseenter', () => {
            panel.style.boxShadow = '0 0 20px rgba(0, 255, 0, 0.5)';
        });

        panel.addEventListener('mouseleave', () => {
            panel.style.boxShadow = '';
        });
    });
}

// Matrix Rain Effect (Subtle Background)
function addMatrixRain() {
    const matrixContainer = document.createElement('div');
    matrixContainer.style.position = 'fixed';
    matrixContainer.style.top = '0';
    matrixContainer.style.left = '0';
    matrixContainer.style.width = '100%';
    matrixContainer.style.height = '100%';
    matrixContainer.style.pointerEvents = 'none';
    matrixContainer.style.zIndex = '1';
    matrixContainer.style.overflow = 'hidden';
    matrixContainer.style.opacity = '0.1';

    document.body.appendChild(matrixContainer);

    // Create falling characters
    const chars = '01';
    const columns = Math.floor(window.innerWidth / 20);

    for (let i = 0; i < Math.min(columns, 30); i++) {
        if (Math.random() > 0.7) { // Only 30% of columns have rain
            createMatrixColumn(matrixContainer, i * 20, chars);
        }
    }
}

function createMatrixColumn(container, x, chars) {
    const span = document.createElement('span');
    span.className = 'matrix-char';
    span.style.left = `${x}px`;
    span.style.animationDuration = `${Math.random() * 10 + 10}s`;
    span.style.animationDelay = `${Math.random() * 5}s`;
    span.textContent = chars[Math.floor(Math.random() * chars.length)];

    container.appendChild(span);

    // Recreate character when animation ends
    span.addEventListener('animationend', () => {
        span.style.animationDelay = '0s';
    });
}

// Parallax Scroll Effects
function initParallaxEffects() {
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;

        // Parallax effect on hero section
        const hero = document.getElementById('hero');
        if (hero) {
            const gridPattern = hero.querySelector('.grid-pattern');
            if (gridPattern) {
                gridPattern.style.transform = `translateY(${scrolled * 0.5}px)`;
            }
        }
    });
}

// Add typing cursor to main title
function addTypingCursor() {
    const titles = document.querySelectorAll('.glitch-text');
    titles.forEach(title => {
        const cursor = document.createElement('span');
        cursor.className = 'cursor';
        cursor.textContent = '_';
        cursor.style.animation = 'blink 1s step-end infinite';
        title.appendChild(cursor);
    });
}

// Keyboard Shortcuts
function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + K to focus on CTA
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const cta = document.querySelector('a[href="register"]');
            if (cta) {
                cta.focus();
                cta.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }

        // ESC to scroll to top
        if (e.key === 'Escape') {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    });
}

// Terminal Command Easter Egg
function initTerminalEasterEgg() {
    let keySequence = '';
    const secretCode = 'help';

    document.addEventListener('keypress', (e) => {
        keySequence += e.key.toLowerCase();

        // Keep only last few characters
        if (keySequence.length > secretCode.length) {
            keySequence = keySequence.slice(-secretCode.length);
        }

        if (keySequence === secretCode) {
            showTerminalHelp();
            keySequence = '';
        }
    });
}

function showTerminalHelp() {
    const helpText = `
╔═══════════════════════════════════════╗
║     CODER_OS COMMAND REFERENCE        ║
╚═══════════════════════════════════════╝

Available Commands:
  start     - Initialize coding session
  features  - View feature list
  security  - View security protocols
  help      - Show this message
  register  - Create account

System Status: ONLINE
Security Level: MAXIMUM
Ready for input...
    `;

    // Create modal overlay
    const modal = document.createElement('div');
    modal.style.position = 'fixed';
    modal.style.top = '0';
    modal.style.left = '0';
    modal.style.width = '100%';
    modal.style.height = '100%';
    modal.style.backgroundColor = 'rgba(0, 0, 0, 0.95)';
    modal.style.zIndex = '10000';
    modal.style.display = 'flex';
    modal.style.alignItems = 'center';
    modal.style.justifyContent = 'center';
    modal.style.padding = '20px';

    const content = document.createElement('pre');
    content.style.color = '#00ff00';
    content.style.fontFamily = 'monospace';
    content.style.fontSize = '14px';
    content.style.border = '2px solid #00ff00';
    content.style.padding = '20px';
    content.style.maxWidth = '600px';
    content.style.width = '100%';
    content.textContent = helpText;

    modal.appendChild(content);
    document.body.appendChild(modal);

    // Close on click
    modal.addEventListener('click', () => {
        modal.remove();
    });

    // Close on ESC
    const closeOnEsc = (e) => {
        if (e.key === 'Escape') {
            modal.remove();
            document.removeEventListener('keydown', closeOnEsc);
        }
    };
    document.addEventListener('keydown', closeOnEsc);
}

// Glitch Effect on Scroll
function initScrollGlitchEffect() {
    const glitchElements = document.querySelectorAll('.glitch-text');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animation = 'flicker 0.3s';
                setTimeout(() => {
                    entry.target.style.animation = 'flicker 2s infinite';
                }, 300);
            }
        });
    }, { threshold: 0.5 });

    glitchElements.forEach(el => observer.observe(el));
}

// System Status Monitor (Fake Stats)
function initSystemMonitor() {
    const statusElements = document.querySelectorAll('.status-badge');

    setInterval(() => {
        statusElements.forEach(el => {
            // Random flicker effect
            if (Math.random() > 0.95) {
                el.style.opacity = '0.5';
                setTimeout(() => {
                    el.style.opacity = '1';
                }, 100);
            }
        });
    }, 2000);
}

// Mobile Menu Toggle (if needed)
function initMobileMenu() {
    const header = document.querySelector('header');
    let lastScroll = 0;

    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;

        if (currentScroll > lastScroll && currentScroll > 100) {
            // Scrolling down
            header.style.transform = 'translateY(-100%)';
        } else {
            // Scrolling up
            header.style.transform = 'translateY(0)';
        }

        lastScroll = currentScroll;
    });
}

// CTA Button Pulse on Scroll
function initCTAPulse() {
    const ctaButtons = document.querySelectorAll('a[href="register"]');

    window.addEventListener('scroll', () => {
        const scrollPercent = (window.pageYOffset / (document.documentElement.scrollHeight - window.innerHeight)) * 100;

        if (scrollPercent > 80) {
            ctaButtons.forEach(btn => {
                btn.style.animation = 'pulse-slow 1s ease-in-out infinite';
            });
        }
    });
}

// Performance Monitor (Development Only)
function logPerformance() {
    if (window.performance && window.performance.timing) {
        window.addEventListener('load', () => {
            setTimeout(() => {
                const perfData = window.performance.timing;
                const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
                console.log(`%c[CODER_OS] Page loaded in ${pageLoadTime}ms`, 'color: #00ff00; font-weight: bold;');
            }, 0);
        });
    }
}

// Initialize Everything on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('%c[CODER_OS] Initializing system...', 'color: #00ff00; font-weight: bold; font-size: 16px;');

    // Start boot sequence
    bootSequence();

    // Initialize keyboard shortcuts
    initKeyboardShortcuts();

    // Initialize easter egg
    initTerminalEasterEgg();

    // Initialize scroll effects
    initScrollGlitchEffect();

    // Initialize CTA pulse
    initCTAPulse();

    // Log performance
    logPerformance();

    console.log('%c[CODER_OS] System ready. Type "help" for commands.', 'color: #00ff00; font-weight: bold;');
});

// Handle visibility change (tab switching)
document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
        console.log('%c[CODER_OS] Welcome back.', 'color: #00ff00;');
    }
});

// Console styling for branding
console.log('%c' + `
╔═══════════════════════════════════════╗
║           CODER_OS v2.1.0             ║
║     Professional Coding Assistant     ║
╚═══════════════════════════════════════╝
`, 'color: #00ff00; font-family: monospace;');
