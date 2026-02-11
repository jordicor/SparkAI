// theme.js
class ThemeManager {
    constructor(templatePath) {
        this.templatePath = templatePath;
        this.defaultThemeLoaded = false;
        this.init();
    }

    // Helper function to generate static URL
    getStaticUrl(path) {
        if (window.CDN_CONFIG && window.CDN_CONFIG.enabled) {
            // Remove /static from path if present since CDN baseUrl already includes it
            if (path.startsWith('/static/')) {
                path = path.substring(7); // Remove '/static' (7 characters)
            } else if (path.startsWith('/static')) {
                path = path.substring(7); // Remove '/static' (7 characters)
            }
            
            // Ensure path starts with /
            if (!path.startsWith('/')) {
                path = '/' + path;
            }
            
            return window.CDN_CONFIG.baseUrl + path;
        }
        return path;
    }

    init() {
        document.addEventListener('DOMContentLoaded', () => {
            const currentTheme = localStorage.getItem('theme') || 'default';
            this.setTheme(currentTheme);
            this.initializePromptHandling();
        });
    }

    getTemplatePath() {
        const pathSegments = this.templatePath.split('/');
        
        const templateName = pathSegments[pathSegments.length - 1];
        
        const basePath = pathSegments.join('/');
        
        return {
            templateName: templateName,
            basePath: basePath
        };
    }

    loadThemeCSS(theme) {
        if (theme === 'default' && this.defaultThemeLoaded) {
            console.error('Default theme failed to load. Using inline fallback styles.');
            return;
        }

        let linkElement = document.getElementById('theme-css');
        const { templateName, basePath } = this.getTemplatePath();
        const cssUrl = this.getStaticUrl(`/static/css/${basePath}/${templateName}-${theme}.css`);
        
        if (!linkElement || linkElement.href !== cssUrl) {
            const newLink = document.createElement('link');
            newLink.rel = 'stylesheet';
            newLink.id = 'theme-css';
            newLink.href = cssUrl;

            newLink.onload = () => {
                if (linkElement) {
                    linkElement.remove();
                }
                linkElement = newLink;
                if (theme === 'default') {
                    this.defaultThemeLoaded = true;
                }
            };

            newLink.onerror = () => {
                console.error('Error loading theme:', theme);
                this.fallbackToRootPath(theme, newLink);
            };

            if (linkElement) {
                linkElement.parentNode.insertBefore(newLink, linkElement);
            } else {
                document.head.appendChild(newLink);
            }
        }

        if (theme === 'xmas') {
            this.createSnowflakes();
        }
    }

    fallbackToRootPath(theme, linkElement) {
        const { templateName } = this.getTemplatePath();
        const rootCssUrl = this.getStaticUrl(`/static/css/${templateName}/${templateName}-${theme}.css`);
        
        linkElement.href = rootCssUrl;
        
        linkElement.onerror = () => {
            if (theme !== 'default') {
                this.loadThemeCSS('default');
            } else {
                this.defaultThemeLoaded = true;
                this.applyFallbackStyles();
            }
        };
    }

    applyFallbackStyles() {
        // Basic fallback styles in case default theme fails to load
        const style = document.createElement('style');
        style.textContent = `
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f0f0f0;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 15px;
                background-color: #fff;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
        `;
        document.head.appendChild(style);
    }

    setTheme(theme) {
        // If theme is null, undefined or empty string, use 'default'
        if (!theme) {
            theme = 'default';
        }
        this.loadThemeCSS(theme);
        localStorage.setItem('theme', theme);
    }

    createSnowflakes() {
        const numberOfSnowflakes = 50;
        for (let i = 0; i < numberOfSnowflakes; i++) {
            this.createSnowflake();
        }
    }

    createSnowflake() {
        const snowflake = document.createElement('div');
        snowflake.className = 'snowflake';
        snowflake.innerHTML = '❄';
        snowflake.style.left = Math.random() * 100 + 'vw';
        snowflake.style.animationDuration = (Math.random() * 3 + 2) + 's';
        snowflake.style.opacity = Math.random();
        snowflake.style.fontSize = (Math.random() * 10 + 10) + 'px';
        
        document.body.appendChild(snowflake);
        
        snowflake.addEventListener('animationend', () => {
            snowflake.remove();
            this.createSnowflake();
        });
    }

    initializePromptHandling() {
        const promptSelect = document.getElementById("prompt");
        if (promptSelect) {
            const initialPromptId = promptSelect.value;
            if (typeof updateButtons === 'function') {
                updateButtons(initialPromptId);
            }
        }

        const cacheOptions = document.getElementById('cacheOptions');
        if (cacheOptions) {
            cacheOptions.style.display = 'none';
        }
    }
}

// Export for use as module
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ThemeManager;
}