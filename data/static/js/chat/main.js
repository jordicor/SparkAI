/* main.js */

document.addEventListener('DOMContentLoaded', function() {
    // Restore conversation ID after theme change reload
    const restoreConvId = localStorage.getItem('restoreConversationId');
    if (restoreConvId && typeof currentConversationId !== 'undefined') {
        currentConversationId = parseInt(restoreConvId, 10);
        localStorage.removeItem('restoreConversationId');
    }

    var fileInput = document.getElementById('image-files');
    var previewsContainer = document.getElementById('image-previews');
    const dropZone = document.body; // Use whole body as drop zone
    const dropZoneOverlay = document.getElementById('drop-zone-overlay');
    var toggleButton = document.querySelector('.btn-toggle-sidebar');
    var sidebar = document.getElementById('sidebar');
    var userBalanceElement = document.getElementById("user-balance");
    var userBalance = userBalanceElement ? parseInt(userBalanceElement.getAttribute("data-balance")) : 0;	
    var messageInputContainer = document.getElementById("form-message");
    var insufficientBalanceMessage = document.getElementById("insufficient-balance-message");
    var chatContent = document.querySelector('.chat-window');
    attachedFiles = []
    const manager = new TextareaImagePreviewManager();

    // Save instance to window in case we need to access it later
    window.textareaImagePreviewManager = manager;
    const messageText = document.getElementById('message-text');

    messageText.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = this.scrollHeight + 'px';
    });
    // Theme initialization is now handled by themeManager.js

    // For the Prompt select
    var promptDropdown = document.getElementById('promptDropdown');
    if (promptDropdown) {
        promptDropdown.addEventListener('change', withSession(function(e) {
            const selectedValue = e.target.value;
            updateSelection('/chat', {
                form_type: 'prompt',
                prompt_id: selectedValue
            });
        }));
    } else {
        console.error('[DEBUG] Prompt dropdown NOT found');
    }

    // For the LLM select
    var llmDropdown = document.getElementById('llmDropdown');
    if (llmDropdown) {
        llmDropdown.addEventListener('change', withSession(function(e) {
            const selectedValue = e.target.value;
            updateSelection('/chat', {
                form_type: 'llm',
                type_of_model: selectedValue
            });
        }));
    } else {
        console.error('[DEBUG] LLM dropdown NOT found');
    }

    // For code coloring
    hljs.highlightAll();

    function updateSelection(url, data) {
        secureFetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            if (!data.success) {
                console.error('Error updating selection');
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }

    function checkBalanceAndHideInput() {
        if (userBalance === 0) {
            messageInputContainer.style.display = 'none';
            insufficientBalanceMessage.style.display = 'block';
        }
    }
    checkBalanceAndHideInput();    

    function toggleSidebar() {
        sidebar.classList.toggle('active');
        dropZoneOverlay.classList.toggle('active');
    }

    function closeSidebar() {
        sidebar.classList.remove('active');
        dropZoneOverlay.classList.remove('active');
    }

    window.closeSidebar = closeSidebar;
    
    if (toggleButton) {
        toggleButton.addEventListener('click', function(event) {
            event.preventDefault();
            toggleSidebar();
        });
    } else {
        console.error('Toggle button not found');
    }

    if (dropZoneOverlay) {
        dropZoneOverlay.addEventListener('click', closeSidebar);
    } else {
        console.error('Overlay not found');
    }

    if (chatContent) {
        chatContent.addEventListener('click', function(event) {
            if (sidebar.classList.contains('active')) {
                closeSidebar();
            }
        });
    } else {
        console.error('Chat content not found');
    }

    let touchStartX = 0;
    let touchEndX = 0;

    document.addEventListener('touchstart', e => {
        if (isMobile()) {
            touchStartX = e.changedTouches[0].screenX;
        }
    });
    document.addEventListener('touchend', e => {
        if (isMobile()) {
            touchEndX = e.changedTouches[0].screenX;
            handleSwipe();
        }
    });

    function handleSwipe() {
        const swipeDistance = touchEndX - touchStartX;
        if (swipeDistance > 100 && !sidebar.classList.contains('active')) {
            toggleSidebar();
        } else if (swipeDistance < -100 && sidebar.classList.contains('active')) {
            closeSidebar();
        }
    }

    // Close sidebar on mobile devices when loading
    if (isMobile()) {
        closeSidebar();
    }

    // Show loading indicator at start
    document.getElementById('loading-indicator').style.display = 'block';

    // Load conversations
    loadConversations(false, true).then(() => {
        // Enable controls and hide loading indicator when all initial loads are complete
        enableInputControls();
        document.getElementById('loading-indicator').style.display = 'none';
    }).catch(error => {
        console.error('Error during initial load:', error);
        // Make sure to enable controls even if there's an error
        enableInputControls();
        document.getElementById('loading-indicator').style.display = 'none';
    });

    if (Config.adminView) {
        var sendButton = document.querySelector('#form-message button[type="submit"]');
        messageText.classList.add('hidden');
        sendButton.classList.add('hidden');
    }

    // New code for "New Chat" split button
    var newChatMainBtn = document.getElementById('new-chat-main-btn');
    
    if (newChatMainBtn) {
        newChatMainBtn.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Check session before creating new conversation (force check for critical action)
            SessionManager.validateSession(true).then((isValid) => {
                if (isValid) {
                    // Session is valid, create new conversation
                    startNewConversation();
                } else {
                    // Session invalid, modal already shown by validateSession
                }
            });
        });
    } else {
        console.error('Main New Chat button not found');
    }

    // Theme is now handled by themeManager.js

    document.getElementById('form-message').onsubmit = function(e) {
        e.preventDefault();
        var messageText = document.getElementById('message-text').value;
        /*messageText = encodeForHTML(messageText);*/
        
        // Check session before sending message (force check for critical action)
        SessionManager.validateSession(true).then((isValid) => {
            if (isValid) {
                // Session is valid, send message and clear form
                try {
                    sendMessage(messageText);
                    document.getElementById('message-text').value = ''; 
                    Config.attachedFiles = [];
                } catch (error) {
                    console.error('Error sending message:', error);
                    // Don't clear form if there was an error
                }
            } else {
                // Session invalid, modal already shown by validateSession
                // Don't clear the form so user can copy their message
            }
        });
    };

    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            const sendButton = document.getElementById('send-button');
            if (sendButton.innerText === 'Stop') {
                stopReceivingStream();
            } else {
                const imagePreviews = document.getElementById('image-previews');
                const previewImages = imagePreviews.querySelectorAll('img');
                if (previewImages.length > 0) {
                    const lastImage = previewImages[previewImages.length - 1];
                    imagePreviews.removeChild(lastImage);
                    attachedFiles.pop();
                    if (previewImages.length === 1) {
                        imagePreviews.classList.add('hidden');
                    }
                }
            }
        }
    });

    document.getElementById('message-text').addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            document.getElementById('form-message').dispatchEvent(new Event('submit', {cancelable: true}));
        }
    });

    let dragTimeout;

    if (dropZone) {
        dropZone.addEventListener('dragover', function(e) {
            e.preventDefault();
            e.stopPropagation();
            clearTimeout(dragTimeout);
            dropZoneOverlay.classList.remove('d-none');
            dropZoneOverlay.classList.add('d-flex');
        });
    
        dropZone.addEventListener('dragleave', function(e) {
            e.preventDefault();
            e.stopPropagation();
            dragTimeout = setTimeout(function() {
                dropZoneOverlay.classList.add('d-none');
                dropZoneOverlay.classList.remove('d-flex');
            }, 100);
        });
    
        dropZone.addEventListener('drop', function(e) {
            e.preventDefault();
            e.stopPropagation();
            clearTimeout(dragTimeout);
            dropZoneOverlay.classList.add('d-none');
            dropZoneOverlay.classList.remove('d-flex');
    
            const files = e.dataTransfer.files;
            handleDroppedFiles(files);
            setTimeout(() => {
                document.getElementById('message-text').focus();
            }, 0);
        });
    }
    
    function handleDroppedFiles(files) {
        const imagePreviews = document.getElementById('image-previews');
        for (const file of files) {
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    img.className = 'preview-image';
                    imagePreviews.appendChild(img);
                };
                reader.readAsDataURL(file);
                attachedFiles.push(file);
            }
        }
    
        if (attachedFiles.length > 0) {
            imagePreviews.classList.remove('hidden');
        }
        setTimeout(() => {
            document.getElementById('message-text').focus();
        }, 0);
    }
});

let attachedFiles = [];

function checkSession(callback) {
    fetch('/api/check-session')
        .then(response => {
            return response.json();
        })
        .then(data => {
            if (data.expired) {
                window.location.href = '/login';
            } else {
                callback();
            }
        })
        .catch(error => {
            console.error("Error checking session:", error);
        });
}

function isMobile() {
    return window.innerWidth < 769;
}

// This function is defined in chat.js and overwritten when that file loads
function startNewConversation() {
    // Stub function - overwritten by chat.js
}

class TextareaImagePreviewManager {
    constructor() {
        this.textarea = document.getElementById('message-text');
        this.previewsContainer = document.getElementById('image-previews');
        this.messageInput = document.querySelector('.message-input');
        this.formMessage = document.getElementById('form-message');
        
        if (!this.textarea || !this.previewsContainer || !this.messageInput) {
            console.error('Required elements not found');
            return;
        }
        
        this.init();
    }
    
    init() {
        // Set up initial styles
        this.setupStyles();

        // Set up observers
        this.setupObservers();

        // Set up event listeners
        this.setupEventListeners();

        // Initial update
        this.updatePreviewsPosition();
    }

    setupStyles() {
        // Styles for message container
        Object.assign(this.messageInput.style, {
            position: 'relative',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'flex-end'
        });

        // Styles for previews container
        Object.assign(this.previewsContainer.style, {
            position: 'absolute',
            left: '0',
            right: '0',
            height: 'auto',
            minHeight: '100px',
            maxHeight: '150px',
            overflowY: 'auto',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'flex-start',
            flexWrap: 'wrap',
            gap: '10px',
            padding: '10px',
            backgroundColor: 'rgba(47, 49, 54, 0.95)',
            zIndex: '10',
            transition: 'bottom 0.2s ease'
        });

        // Styles for the form
        if (this.formMessage) {
            Object.assign(this.formMessage.style, {
                width: '100%',
                maxWidth: '48rem',
                margin: '0 auto',
                position: 'relative'
            });
        }

        // Ensure textarea has necessary styles
        Object.assign(this.textarea.style, {
            maxHeight: '33vh',
            minHeight: '44px',
            overflowY: 'auto',
            resize: 'none',
            lineHeight: '1.4'
        });
    }

    setupObservers() {
        // Observe changes in textarea size
        this.resizeObserver = new ResizeObserver(() => {
            this.updatePreviewsPosition();
        });
        this.resizeObserver.observe(this.textarea);

        // Observe changes in previews container
        this.mutationObserver = new MutationObserver(() => {
            this.updatePreviewsPosition();
            this.updatePreviewsVisibility();
        });
        
        this.mutationObserver.observe(this.previewsContainer, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['class']
        });
    }

    setupEventListeners() {
        // Update on window resize
        window.addEventListener('resize', () => this.updatePreviewsPosition());

        // Update when typing
        this.textarea.addEventListener('input', () => this.updatePreviewsPosition());

        // Update when pasting content
        this.textarea.addEventListener('paste', () => {
            setTimeout(() => this.updatePreviewsPosition(), 0);
        });
    }
    
    updatePreviewsPosition() {
        if (this.previewsContainer.classList.contains('hidden')) {
            return;
        }
        
        const textareaRect = this.textarea.getBoundingClientRect();
        const textareaHeight = this.textarea.offsetHeight;
        const bottomPosition = textareaHeight + 20; // 20px padding

        this.previewsContainer.style.bottom = `${bottomPosition}px`;
    }
    
    updatePreviewsVisibility() {
        const hasImages = this.previewsContainer.children.length > 0;
        const isHidden = this.previewsContainer.classList.contains('hidden');
        
        if (hasImages && isHidden) {
            this.previewsContainer.classList.remove('hidden');
            this.updatePreviewsPosition();
        } else if (!hasImages && !isHidden) {
            this.previewsContainer.classList.add('hidden');
        }
    }

    // Method to clean up observers if necessary
    destroy() {
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
        }
        if (this.mutationObserver) {
            this.mutationObserver.disconnect();
        }
    }
}

function updatePreviewsPosition() {
    if (window.textareaImagePreviewManager) {
        window.textareaImagePreviewManager.updatePreviewsPosition();
    }
}