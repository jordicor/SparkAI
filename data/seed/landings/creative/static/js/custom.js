// Creative Landing Page - Neural Network Interactive Canvas
// Author: Claude Code
// Description: Dynamic neural network visualization that responds to mouse movement

class NeuralNetwork {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.nodes = [];
        this.connections = [];
        this.mouse = { x: 0, y: 0 };
        this.nodeCount = 80;
        this.maxDistance = 150;
        this.primaryColor = '#33ff4d';
        this.secondaryColor = '#ff8080';

        this.init();
        this.animate();
    }

    init() {
        this.resize();
        window.addEventListener('resize', () => this.resize());
        window.addEventListener('mousemove', (e) => this.updateMouse(e));
        this.createNodes();
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = document.body.scrollHeight;
    }

    createNodes() {
        this.nodes = [];
        for (let i = 0; i < this.nodeCount; i++) {
            this.nodes.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                radius: Math.random() * 2 + 1,
                energy: 0
            });
        }
    }

    updateMouse(e) {
        this.mouse.x = e.clientX;
        this.mouse.y = e.clientY + window.scrollY;
    }

    drawNode(node) {
        const gradient = this.ctx.createRadialGradient(
            node.x, node.y, 0,
            node.x, node.y, node.radius * 3
        );

        const color = node.energy > 0.5 ? this.primaryColor : this.secondaryColor;
        const alpha = 0.3 + (node.energy * 0.7);

        gradient.addColorStop(0, `${color}${Math.floor(alpha * 255).toString(16).padStart(2, '0')}`);
        gradient.addColorStop(1, 'transparent');

        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        this.ctx.fill();

        // Draw glow for energized nodes
        if (node.energy > 0.7) {
            this.ctx.shadowBlur = 20;
            this.ctx.shadowColor = color;
            this.ctx.fillStyle = color;
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, node.radius * 0.5, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.shadowBlur = 0;
        }
    }

    drawConnection(node1, node2, distance) {
        const opacity = 1 - (distance / this.maxDistance);
        const energy = (node1.energy + node2.energy) / 2;

        const gradient = this.ctx.createLinearGradient(
            node1.x, node1.y,
            node2.x, node2.y
        );

        gradient.addColorStop(0, `${this.primaryColor}${Math.floor(opacity * energy * 128).toString(16).padStart(2, '0')}`);
        gradient.addColorStop(0.5, `${this.secondaryColor}${Math.floor(opacity * energy * 128).toString(16).padStart(2, '0')}`);
        gradient.addColorStop(1, `${this.primaryColor}${Math.floor(opacity * energy * 128).toString(16).padStart(2, '0')}`);

        this.ctx.strokeStyle = gradient;
        this.ctx.lineWidth = energy * 2;
        this.ctx.beginPath();
        this.ctx.moveTo(node1.x, node1.y);
        this.ctx.lineTo(node2.x, node2.y);
        this.ctx.stroke();

        // Draw energy pulse
        if (energy > 0.8) {
            const pulsePos = (Date.now() % 1000) / 1000;
            const pulseX = node1.x + (node2.x - node1.x) * pulsePos;
            const pulseY = node1.y + (node2.y - node1.y) * pulsePos;

            this.ctx.fillStyle = this.primaryColor;
            this.ctx.shadowBlur = 10;
            this.ctx.shadowColor = this.primaryColor;
            this.ctx.beginPath();
            this.ctx.arc(pulseX, pulseY, 3, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.shadowBlur = 0;
        }
    }

    updateNodes() {
        this.nodes.forEach(node => {
            // Update position
            node.x += node.vx;
            node.y += node.vy;

            // Bounce off edges
            if (node.x < 0 || node.x > this.canvas.width) node.vx *= -1;
            if (node.y < 0 || node.y > this.canvas.height) node.vy *= -1;

            // Keep within bounds
            node.x = Math.max(0, Math.min(this.canvas.width, node.x));
            node.y = Math.max(0, Math.min(this.canvas.height, node.y));

            // Calculate distance to mouse
            const dx = this.mouse.x - node.x;
            const dy = this.mouse.y - node.y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            // Update energy based on proximity to mouse
            if (distance < 200) {
                node.energy = Math.min(1, node.energy + 0.05);

                // Attract to mouse
                const force = (200 - distance) / 200;
                node.vx += (dx / distance) * force * 0.02;
                node.vy += (dy / distance) * force * 0.02;
            } else {
                node.energy = Math.max(0, node.energy - 0.02);
            }

            // Apply friction
            node.vx *= 0.99;
            node.vy *= 0.99;
        });
    }

    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        this.updateNodes();

        // Draw connections
        for (let i = 0; i < this.nodes.length; i++) {
            for (let j = i + 1; j < this.nodes.length; j++) {
                const dx = this.nodes[i].x - this.nodes[j].x;
                const dy = this.nodes[i].y - this.nodes[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < this.maxDistance) {
                    this.drawConnection(this.nodes[i], this.nodes[j], distance);
                }
            }
        }

        // Draw nodes on top
        this.nodes.forEach(node => this.drawNode(node));

        requestAnimationFrame(() => this.animate());
    }
}

// Custom Cursor
class CustomCursor {
    constructor() {
        this.cursor = document.createElement('div');
        this.cursor.id = 'customCursor';
        document.body.appendChild(this.cursor);

        this.init();
    }

    init() {
        document.addEventListener('mousemove', (e) => {
            this.cursor.style.left = e.clientX + 'px';
            this.cursor.style.top = e.clientY + 'px';
        });

        document.addEventListener('mousedown', () => {
            this.cursor.classList.add('active');
        });

        document.addEventListener('mouseup', () => {
            this.cursor.classList.remove('active');
        });
    }
}

// Particle Trail Effect
class ParticleTrail {
    constructor() {
        this.particles = [];
        this.init();
    }

    init() {
        document.addEventListener('mousemove', (e) => {
            if (Math.random() > 0.8) {
                this.createParticle(e.clientX, e.clientY);
            }
        });
    }

    createParticle(x, y) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = x + 'px';
        particle.style.top = y + 'px';
        particle.style.width = Math.random() * 8 + 4 + 'px';
        particle.style.height = particle.style.width;

        document.body.appendChild(particle);

        setTimeout(() => {
            particle.remove();
        }, 1000);
    }
}

// Scroll Reveal Animation
class ScrollReveal {
    constructor() {
        this.elements = document.querySelectorAll('.process-step, .testimonial-card');
        this.init();
    }

    init() {
        this.checkVisibility();
        window.addEventListener('scroll', () => this.checkVisibility());
    }

    checkVisibility() {
        this.elements.forEach(element => {
            const rect = element.getBoundingClientRect();
            const isVisible = rect.top < window.innerHeight * 0.8;

            if (isVisible) {
                element.classList.add('visible');
            }
        });
    }
}

// Smooth Scroll for Anchor Links
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href === '#') return;

            e.preventDefault();
            const target = document.querySelector(href);

            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Parallax Effect for Hero
function initParallax() {
    window.addEventListener('scroll', () => {
        const scrolled = window.scrollY;
        const parallaxElements = document.querySelectorAll('.idea-spark');

        parallaxElements.forEach((element, index) => {
            const speed = 0.05 * (index + 1);
            element.style.transform = `translateY(${scrolled * speed}px)`;
        });
    });
}

// Feature Card Tilt Effect
function initFeatureCardTilt() {
    const cards = document.querySelectorAll('.feature-card');

    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            const rotateX = (y - centerY) / 10;
            const rotateY = (centerX - x) / 10;

            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-8px)`;
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = '';
        });
    });
}

// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Neural Network Canvas
    const canvas = document.getElementById('neuralCanvas');
    if (canvas) {
        new NeuralNetwork(canvas);
    }

    // Custom Cursor
    new CustomCursor();

    // Particle Trail
    new ParticleTrail();

    // Scroll Reveal
    new ScrollReveal();

    // Smooth Scroll
    initSmoothScroll();

    // Parallax Effect
    initParallax();

    // Feature Card Tilt
    initFeatureCardTilt();

    // Update canvas height on scroll
    window.addEventListener('scroll', () => {
        const canvas = document.getElementById('neuralCanvas');
        if (canvas && canvas.height < document.body.scrollHeight) {
            canvas.height = document.body.scrollHeight;
        }
    });
});

// Performance optimization: Reduce animation on low-end devices
if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    document.body.style.animation = 'none';
}
