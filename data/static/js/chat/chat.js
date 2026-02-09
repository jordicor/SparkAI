/* chat.js */

let offset = 0;
let limit = 10;
let initialLoad = true;
let isLoading = false;
let allMessagesLoaded = false;
let allConversationsLoaded = false;
let lowestLoadedId = Infinity;
let currentAbortController = null;
let currentTimer = null;
let lastSelectedConversationId = null;
var botname = "bot";
const limitMessage = 25;
const processedMessageIds = new Set();
let currentThinkingBudget = 0;
let isCurrentConversationLocked = false;

// Auto-scroll state management
let isUserScrolledUp = false;
const SCROLL_THRESHOLD = 100; // px from bottom to consider "locked" to bottom

function isNearBottom(element) {
    return element.scrollHeight - element.scrollTop - element.clientHeight < SCROLL_THRESHOLD;
}

function scrollToBottomIfNeeded() {
    const chatWindow = document.getElementById('chat-window');
    if (!isUserScrolledUp) {
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
}

// Web Search Toggle state
let webSearchEnabled = true;   // User preference (default ON)
let webSearchAllowed = true;   // Prompt allows web search

// =============================================
// API Key Mode Manager
// =============================================
const ApiKeyManager = {
    mode: typeof apiKeyMode !== 'undefined' ? apiKeyMode : 'both_prefer_own',
    canSend: typeof canSendMessages !== 'undefined' ? canSendMessages : true,
    requiresOwn: typeof requiresOwnKeys !== 'undefined' ? requiresOwnKeys : false,
    hasOwn: typeof hasOwnKeys !== 'undefined' ? hasOwnKeys : false,

    /**
     * Check if user can send messages based on API key configuration
     * @returns {boolean}
     */
    canSendMessages: function() {
        return this.canSend;
    },

    /**
     * Update the API key status by fetching from server
     * @returns {Promise<void>}
     */
    async refreshStatus() {
        try {
            const response = await fetch('/api/user/api-key-status');
            if (response.ok) {
                const data = await response.json();
                this.mode = data.mode;
                this.canSend = data.can_send_messages;
                this.requiresOwn = data.requires_own_keys;
                this.hasOwn = data.has_own_keys;

                // Update UI based on new status
                this.updateUI();
            }
        } catch (error) {
            console.error('Failed to refresh API key status:', error);
        }
    },

    /**
     * Update UI elements based on current API key status
     */
    updateUI() {
        const banner = document.getElementById('api-keys-required-banner');
        const inputContainer = document.getElementById('message-input-container');

        if (this.canSend) {
            // Hide banner and enable input
            if (banner) banner.style.display = 'none';
            if (inputContainer) inputContainer.removeAttribute('data-disabled');
        } else {
            // Show banner and disable input
            if (banner) banner.style.display = 'block';
            if (inputContainer) inputContainer.setAttribute('data-disabled', 'true');
        }
    },

    /**
     * Handle API key error response from server
     * @param {Object} errorData - Error data from server
     */
    handleApiKeyError(errorData) {
        if (errorData.error === 'api_keys_required' || errorData.action === 'configure_api_keys') {
            // Show a notification to the user
            const shouldRedirect = confirm(
                'You need to configure your API keys to use AI services. ' +
                'Would you like to go to the API credentials page?'
            );
            if (shouldRedirect) {
                window.location.href = '/api-credentials';
            }
        }
    }
};

function addMessage(author, message, timestampInfo = null, isTemporary = false, messageObj = null, prepend = false, container = null, messageId = null) {
    var divMessage = document.createElement('div');
    divMessage.classList.add('message', 'text-white', author === 'user' ? 'user' : 'bot');
    if (messageId) {
        divMessage.dataset.messageId = messageId;
    }

    var messageContentContainer = document.createElement('div');
    messageContentContainer.classList.add('message-content-container');

    var avatarContainer = createAvatar(author);
    messageContentContainer.appendChild(avatarContainer);

    var messageContent = document.createElement('div');
    messageContent.classList.add('message-content');

    if (isTemporary) {
        divMessage.classList.add('temporary-message');
    }

    let messageText = '';

    if (messageObj) {
        if (messageObj.type === 'text') {
            messageText = messageObj.text;
            var divText = document.createElement('p');
            if (author === 'user') {
                divText.classList.add('preserve-whitespace');
                divText.textContent = messageText;
            } else {
                let escapedText = escapeHTML(messageText);
                let processedHTML = marked.parse(escapedText);
                let parsedMarkdown = unescapeCodeBlocks(processedHTML);
                parsedMarkdown = formatCodeBlocks(parsedMarkdown);
                divText.innerHTML = parsedMarkdown;

                divText.querySelectorAll('pre code').forEach((el) => {
                    hljs.highlightElement(el);
                });
            }
            messageContent.appendChild(divText);
        } else if (messageObj.type === 'image_url') {
            var imgElement = document.createElement('img');
            imgElement.src = messageObj.url;
            imgElement.alt = messageObj.alt || '';
            imgElement.loading = 'lazy';
            imgElement.style.maxWidth = '256px';
            imgElement.style.maxHeight = '256px';
            imgElement.style.objectFit = 'contain';
            imgElement.style.cursor = 'pointer';
            imgElement.dataset.fullsize = messageObj.url.replace('_256.webp', '_fullsize.webp');
            imgElement.dataset.messageId = messageId;
            imgElement.onclick = function() {
                imageHandler.showFullsize(this.dataset.fullsize, this.dataset.messageId);
            };
            messageContent.appendChild(imgElement);
        } else if (messageObj.type === 'video_url') {
            var videoElement = document.createElement('video');
            videoElement.src = messageObj.url;
            videoElement.controls = true;
            videoElement.style.maxWidth = '100%';
            videoElement.style.maxHeight = '480px';
            videoElement.style.width = 'auto';
            videoElement.style.height = 'auto';
            videoElement.preload = 'metadata';
            
            // Add poster image if available
            if (messageObj.poster) {
                videoElement.poster = messageObj.poster;
            }
            
            // Add accessibility attributes
            if (messageObj.alt) {
                videoElement.setAttribute('aria-label', messageObj.alt);
                videoElement.title = messageObj.alt;
            }
            
            messageContent.appendChild(videoElement);
        }
    } else {
        messageText = String(message);
        var divText = document.createElement('p');
        if (author === 'user') {
            divText.classList.add('preserve-whitespace');
            divText.textContent = messageText;
        } else {
            let escapedText = escapeHTML(messageText);
            let processedHTML = marked.parse(escapedText);
            parsedMarkdown = unescapeCodeBlocks(processedHTML);
            divText.innerHTML = parsedMarkdown;

            divText.querySelectorAll('pre code').forEach((el) => {
                hljs.highlightElement(el);
            });
        }
        messageContent.appendChild(divText);
    }   

    if (timestampInfo) {
    
        var infoContainer = document.createElement('div');
        infoContainer.classList.add('message-info');
    
        var iconContainer = document.createElement('div');
        iconContainer.classList.add('icon-container');
    
        const audioIcon = document.createElement('i');
        audioIcon.classList.add('fa', 'fa-volume-up');
        audioIcon.style.cursor = 'pointer';
        audioIcon.style.display = 'inline';
    
        audioIcon.dataset.id = currentConversationId;
        audioIcon.onclick = function() {
            textToSpeech(messageText, user_id, currentConversationId, audioIcon, author);
        };
    
        const bookmarkIcon = document.createElement('i');
        bookmarkIcon.classList.add('fas', 'fa-bookmark', 'bookmark-icon');
        bookmarkIcon.style.cursor = 'pointer';
        bookmarkIcon.style.display = 'none';
    
        if (messageObj && messageObj.is_bookmarked) {
            bookmarkIcon.classList.add('bookmarked');
            bookmarkIcon.style.display = 'inline';
        }
    
        const conversationId = messageObj && messageObj.conversation_id ? messageObj.conversation_id : currentConversationId;
        bookmarkIcon.dataset.conversationId = conversationId;
    
        bookmarkIcon.onclick = function() {
            const messageElement = this.closest('.message');
            const resolvedMessageId = messageElement ? messageElement.dataset.messageId : null;
            if (resolvedMessageId) {
                toggleBookmark(resolvedMessageId, currentConversationId, this);
            } else {
                console.error('Could not find message ID to mark as favorite');
            }
        };
    
        const copyIcon = document.createElement('i');
        copyIcon.classList.add('fas', 'fa-copy', 'copy-icon');
        copyIcon.style.cursor = 'pointer';
        copyIcon.style.display = 'none';
        copyIcon.onclick = function() {
            copyToClipboard(messageText, copyIcon);
        };
    
        iconContainer.appendChild(audioIcon);
        iconContainer.appendChild(bookmarkIcon);
        iconContainer.appendChild(copyIcon);
    
        if (author === 'bot') {
            const rollbackIcon = document.createElement('i');
            rollbackIcon.classList.add('fas', 'fa-undo', 'rollback-icon');
            rollbackIcon.style.cursor = 'pointer';
            rollbackIcon.style.display = 'none';
            rollbackIcon.title = 'Start over from here';
            if (messageId) {
                rollbackIcon.setAttribute('data-message-id', messageId);
            }
            rollbackIcon.onclick = function() {
                rollbackConversation(this.getAttribute('data-message-id'), currentConversationId);
            };
    
            const chatTitle = document.querySelector('.chatbot-info h4').textContent;
            if (chatTitle !== "My Bookmarks") {
                iconContainer.appendChild(rollbackIcon);
            }
        }

        // Add arrow icon if we are in "My Bookmarks"
        if (isMyBookmarksView() && conversationId) {
            const goToConversationIcon = document.createElement('i');
            goToConversationIcon.classList.add('fas', 'fa-arrow-right', 'go-to-conversation-icon');
            goToConversationIcon.style.cursor = 'pointer';
            goToConversationIcon.style.display = 'inline';
            goToConversationIcon.title = 'Go to conversation';

            goToConversationIcon.onclick = function() {
                continueConversation(conversationId, 'Chat', 'machine', false, messageId);
            };

            iconContainer.appendChild(goToConversationIcon);
        }
    
        var timeSpan = document.createElement('span');
    
        // Determine the correct date based on whether it's a new or loaded message
        var messageDate;
        if (timestampInfo.isNewMessage) {
            messageDate = new Date(timestampInfo.timestamp.originalUtc);
        } else {
            // Assume timestampInfo is directly the date string from database
            if (typeof timestampInfo.timestamp.originalUtc === 'string') {
                messageDate = new Date(timestampInfo.timestamp.originalUtc.replace(' ', 'T') + 'Z');
            } else {
                console.error('timestampInfo.timestamp.originalUtc is not a string:', timestampInfo.timestamp.originalUtc);
                timeSpan.textContent = 'Invalid date';
            }
        }


        // Verify if date is valid
        if (isNaN(messageDate.getTime())) {
            console.error('Invalid date:', timestampInfo);
            timeSpan.textContent = 'Invalid date';
        } else {
            // Convert UTC to local time
            var localDate = new Date(messageDate.toLocaleString('en-US', { timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone }));
    
            // Format the time in the user's local time zone
            var formattedTime = localDate.toLocaleTimeString(undefined, {
                hour: '2-digit',
                minute: '2-digit',
                hour12: false
            });
            timeSpan.textContent = formattedTime;
    
            // Format full date for title
            var formattedDate = localDate.toLocaleString(undefined, {
                year: 'numeric',
                month: '2-digit',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit',
                hour12: false
            });
            timeSpan.title = formattedDate;
    
        }
    
        infoContainer.appendChild(iconContainer);
        infoContainer.appendChild(timeSpan);
        messageContent.appendChild(infoContainer);
    }

    messageContentContainer.appendChild(messageContent);
    divMessage.appendChild(messageContentContainer);

    if (container) {
        container.appendChild(divMessage);
    } else {
        var chatMessagesContainer = document.getElementById('chat-messages-container');
        if (prepend) {
            chatMessagesContainer.insertBefore(divMessage, chatMessagesContainer.firstChild);
        } else {
            chatMessagesContainer.appendChild(divMessage);
            scrollToBottomIfNeeded();
        }
    }

    divMessage.addEventListener('mouseover', function() {
        const messageId = this.dataset.messageId;
        this.querySelectorAll('.fa-bookmark, .fa-copy, .fa-undo, .fa-arrow-right').forEach(icon => {
            if (!icon.classList.contains('bookmarked')) {
                if (icon.classList.contains('fa-bookmark') && !messageId) {
                    icon.style.display = 'none';
                    return;
                }
                icon.style.display = 'inline';
            }
        });
    });

    divMessage.addEventListener('mouseout', function() {
        this.querySelectorAll('.fa-bookmark, .fa-copy, .fa-undo, .fa-arrow-right').forEach(icon => {
            if (!icon.classList.contains('bookmarked')) {
                icon.style.display = 'none';
            }
        });
    });
    initializeNewImages(divMessage);

    return divMessage;
}

function rollbackConversation(messageId, conversationId) {
    if (!messageId) {
        console.error('No messageId provided for rollback');
        return;
    }
    if (confirm('Are you sure you want to roll back the conversation to this point? All messages after this will be deleted.')) {
        fetch(`/api/conversations/${conversationId}/rollback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message_id: messageId })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Remove messages from the interface
                const chatMessagesContainer = document.getElementById('chat-messages-container');
                const messages = Array.from(chatMessagesContainer.querySelectorAll('.message'));
                let foundIndex = -1;
                
                if (messageId) {
                    // Search for the message both by data-message-id and by a child element with data-message-id
                    foundIndex = messages.findIndex(msg => 
                        msg.dataset.messageId === messageId || 
                        msg.querySelector(`[data-message-id="${messageId}"]`)
                    );
                }
                
                if (foundIndex !== -1) {
                    // Delete messages after found index
                    for (let i = messages.length - 1; i > foundIndex; i--) {
                        messages[i].remove();
                    }
                    
                    // Restart variables
                    offset = foundIndex + 1;
                    allMessagesLoaded = false;
                    isCurrentConversationEmpty = false;
                    
                } else {
                    console.error('Message not found for rollback');
                }
                
            } else {
                console.error('Error rolling back conversation:', data.error);
            }
        })
        .catch(error => {
            console.error('Error rolling back conversation:', error);
        });
    }
}


function formatCodeBlocks(html) {
    const codeRegex = /<pre><code class="language-(\w+)">([\s\S]*?)<\/code><\/pre>/g;
    return html.replace(codeRegex, function(match, language, code) {
        // Remove whitespace at start and end of code
        code = code.trim();
        // Remove common indentation
        const lines = code.split('\n');
        const commonIndent = lines.reduce((min, line) => {
            const indent = line.match(/^\s*/)[0].length;
            return line.trim() ? Math.min(min, indent) : min;
        }, Infinity);
        const dedentedCode = lines.map(line => line.slice(commonIndent)).join('\n');
        
        // Create HTML without spaces or line breaks between tags
        return `<div class="code-block"style="display:grid"><div class="code-header"><span class="code-language">${language}</span><button class="copy-button" onclick="copyCode(this)"><i class="fas fa-copy"></i></button></div><pre><code class="language-${language}">${dedentedCode}</code></pre></div>`;
    });
}

function copyCode(button) {
    const codeBlock = button.closest('.code-block');
    const code = codeBlock.querySelector('code').innerText;
    navigator.clipboard.writeText(code).then(() => {
        button.innerHTML = '<i class="fas fa-check"></i>';
        setTimeout(() => {
            button.innerHTML = '<i class="fas fa-copy"></i>';
        }, 2000);
    });
}

function copyToClipboard(text, icon) {
    navigator.clipboard.writeText(text).then(function() {
        icon.classList.remove('fa-copy');
        icon.classList.add('fa-check');
        setTimeout(function() {
            icon.classList.remove('fa-check');
            icon.classList.add('fa-copy');
        }, 2000);
    }).catch(function(err) {
        console.error('Error copying text: ', err);
    });
}


function showNoChatTemplate() {
    var chatWindow = document.getElementById('chat-window');
    chatWindow.innerHTML = '<div class="no-chat-message">There is no selected chat or the chat has been deleted.</div>'; 
    document.getElementById('message-text').disabled = true;
    document.querySelector('#form-message button[type="submit"]').disabled = true;
}

function sendMessage(messageText) {
    // Check if user can send messages based on API key configuration
    if (!ApiKeyManager.canSendMessages()) {
        ApiKeyManager.handleApiKeyError({ error: 'api_keys_required', action: 'configure_api_keys' });
        return;
    }

    if (currentConversationId === null) {
        console.error('No conversation selected');
        return;
    }
    if (!messageText.trim()) {
        return;
    }

    // Reset auto-scroll state when user sends a message
    isUserScrolledUp = false;

    // Send the compressed message with pako
    const messageText_raw = messageText;

    let timestamp = new Date().toISOString();

    let userMessageElement;
    addMessage(
        'user',
        messageText_raw,
        { timestamp: convertToLocalTime(timestamp), isNewMessage: true },
        false,
        { type: 'text', text: messageText_raw },
        false,
        null,
        null,
        (element) => {
            userMessageElement = element;
        }
    );

    // Get the user message ID
    //getLastMessageId(userMessageElement)

    addLoadingIndicator();
    document.getElementById('message-text').disabled = true;

    var files = document.getElementById('image-files').files;
    var formData = new FormData();

    // Compress message with pako (maximum compression level)
    const compressedMessage = pako.deflate(messageText_raw, { level: 9 });
    formData.append('text_compressed', new Blob([compressedMessage], { type: 'application/octet-stream' }));
    formData.append('is_compressed', 'true'); // Indicate that message is compressed
    
    // Add thinking budget tokens if set
    if (currentThinkingBudget > 0) {
        formData.append('thinking_budget_tokens', currentThinkingBudget);
    }

    var imagePreviews = document.getElementById('image-previews');
    imagePreviews.innerHTML = '';
    
    processFiles(attachedFiles, formData, imagePreviews);
    attachedFiles = [];

    controller = new AbortController();
    const signal = controller.signal;

    toggleSendButton('Stop');

    secureFetch(`/api/conversations/${currentConversationId}/messages`, {
        method: 'POST',
        body: formData,
        signal: signal,
    })
    .then(response => {
        if (!response) {
            // secureFetch returned null (session expired)
            return null;
        }
        if (response.status === 401) {
            return response.json().then(data => {
                if (data.redirect) {
                    window.location.href = data.redirect;
                    return null;
                }
            });
        }
        if (response.status === 403) {
            // Conversation is locked (e.g. watchdog force-lock)
            // Remove the user message we optimistically added to the DOM
            if (userMessageElement) userMessageElement.remove();
            isCurrentConversationLocked = true;
            const lockedBanner = document.getElementById('locked-conversation-banner');
            if (lockedBanner) lockedBanner.style.display = 'flex';
            const msgInput = document.getElementById('message-text');
            msgInput.placeholder = 'This conversation is locked';
            msgInput.disabled = true;
            const submitBtn = document.querySelector('#form-message button[type="submit"]');
            if (submitBtn) submitBtn.disabled = true;
            document.getElementById('loading-indicator').style.display = 'none';
            toggleSendButton('Send');
            // Reload conversation to show the system lock message
            loadConversation(currentConversationId);
            return null;
        }
        return response.body.getReader();
    })
    .then(reader => {
        if (!reader) return; // If there was redirection, reader will be undefined
        
        const decoder = new TextDecoder();
        let buffer = '';
        let botMessageText = '';
        let endConversation = false;
        let newMessageId = null;
        let newUserMessageId = null;
        let updatedChatName = null;

        // Create the bot message element before starting to read the stream
        const botMessageElement = document.createElement('div');
        botMessageElement.classList.add('message', 'text-white', 'bot');

        // Create the container for the avatar and the message
        const messageContentContainer = document.createElement('div');
        messageContentContainer.classList.add('message-content-container');

        // Create the avatar container
        const avatarContainer = createAvatar('bot');

        // Add avatar to message container
        messageContentContainer.appendChild(avatarContainer);

        // Create the div for the message content
        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');

        const botMessageParagraph = document.createElement('p');
        messageContent.appendChild(botMessageParagraph);

        // Add message content to message container
        messageContentContainer.appendChild(messageContent);

        // Add message container to main message div
        botMessageElement.appendChild(messageContentContainer);

        document.getElementById('chat-messages-container').appendChild(botMessageElement);

        // Add icons to bot message
        const infoContainer = document.createElement('div');
        infoContainer.classList.add('message-info');
        const iconContainer = document.createElement('div');
        iconContainer.classList.add('icon-container');

        const audioIcon = createIcon('fa-volume-up', 'inline', () => textToSpeech(marked.parse(botMessageText), user_id, currentConversationId, audioIcon, 'bot'));
        const bookmarkIcon = createIcon('fa-bookmark', 'none', function() {
            const messageElement = this.closest('.message');
            const messageId = messageElement ? messageElement.dataset.messageId : null;
            if (messageId) {
                toggleBookmark(messageId, currentConversationId, this);
            } else {
                console.error('Could not find message ID to mark as favorite');
            }
        });
        const copyIcon = createIcon('fa-copy', 'none', () => copyToClipboard(botMessageText, copyIcon));
        const rollbackIcon = createIcon('fa-undo', 'none', () => rollbackConversation(newMessageId, currentConversationId));            

        iconContainer.appendChild(audioIcon);
        iconContainer.appendChild(bookmarkIcon);
        iconContainer.appendChild(copyIcon);
        iconContainer.appendChild(rollbackIcon);

        const timeSpan = document.createElement('span');
        infoContainer.appendChild(iconContainer);
        infoContainer.appendChild(timeSpan);
        messageContent.appendChild(infoContainer);

        function createIcon(iconClass, styleDisplay, onClickFunction, messageId = null) {
            const icon = document.createElement('i');
            icon.classList.add('fas', iconClass);
            icon.style.cursor = 'pointer';
            icon.style.display = styleDisplay;
            if (messageId && iconClass === 'fa-undo') {
                icon.setAttribute('data-message-id', messageId);
                icon.onclick = function() {
                    rollbackConversation(this.getAttribute('data-message-id'), currentConversationId);
                };
            } else {
                icon.onclick = onClickFunction;
            }
            return icon;
        }

        removeLoadingIndicator();
        // Reset text field to original size
        document.getElementById('message-text').style.height = 'auto';
        addLoadingIndicator();
        scrollToBottomIfNeeded();

        function readStream() {
            return reader.read().then(({ done, value }) => {
				if (done) {
					toggleSendButton('Send');
					initializeNewImages(botMessageElement);
					if (updatedChatName) {
						updateActiveChatName(updatedChatName);
					}
					return;
				}				
                const decoder = new TextDecoder('utf-8');
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
        
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6).trim();
						
                        if (data === '[DONE]') {
                            toggleSendButton('Send');
                            initializeNewImages(botMessageElement);
                            return;
                        }
                        try {
                            const parsedData = JSON.parse(data);

                            // Handle actions from AI welfare system
                            if (parsedData.action === 'end_conversation') {
                                endConversation = true;
                                isCurrentConversationLocked = true;
                                // Update UI to show locked state
                                const selectedChat = document.querySelector(`.list-group-item[data-conversation-id="${currentConversationId}"]`);
                                if (selectedChat) {
                                    selectedChat.dataset.locked = 'true';
                                    selectedChat.classList.add('conversation-locked');
                                    // Update icon in sidebar
                                    const nameSpan = selectedChat.querySelector('.chat-name');
                                    if (nameSpan && !nameSpan.querySelector('.fa-comment-slash')) {
                                        const chatText = nameSpan.textContent;
                                        nameSpan.innerHTML = `<i class="fas fa-comment-slash" title="This conversation is locked"></i> ${chatText}`;
                                    }
                                }
                                // Show locked banner
                                const lockedBanner = document.getElementById('locked-conversation-banner');
                                if (lockedBanner) lockedBanner.style.display = 'flex';
                            }
                            // Note: pass_turn action sends '🚩' as content, which displays
                            // as a normal bot message. No special handling needed.

                            if (parsedData.updated_chat_name) {
                                updateActiveChatName(parsedData.updated_chat_name);
                            } else if (parsedData.video_content) {
                                // Handle video content - render as video element
                                try {
                                    const videoData = JSON.parse(parsedData.video_content);
                                    if (videoData[0]?.type === 'video_url') {
                                        const videoObj = videoData[0].video_url;
                                        botMessageParagraph.innerHTML = '';
                                        const videoElement = document.createElement('video');
                                        videoElement.src = videoObj.url;
                                        videoElement.controls = true;
                                        videoElement.style.maxWidth = '100%';
                                        videoElement.style.maxHeight = '480px';
                                        videoElement.style.width = 'auto';
                                        videoElement.style.height = 'auto';
                                        videoElement.preload = 'metadata';
                                        if (videoObj.alt) {
                                            videoElement.setAttribute('aria-label', videoObj.alt);
                                            videoElement.title = videoObj.alt;
                                        }
                                        botMessageParagraph.appendChild(videoElement);
                                        botMessageText = '';
                                    }
                                } catch (e) {
                                    console.error('Error parsing video content:', e);
                                }
                                scrollToBottomIfNeeded();
                            } else if (parsedData.content) {
                                // Handle replace_last for progress updates
                                if (parsedData.replace_last) {
                                    botMessageText = parsedData.content;
                                } else {
                                    botMessageText += parsedData.content;
                                }

                                let processedHTML = marked.parse(botMessageText);
                                processedHTML = formatCodeBlocks(processedHTML);
                                botMessageParagraph.innerHTML = processedHTML;

                                // Apply highlight.js to code blocks
                                botMessageElement.querySelectorAll('pre code').forEach((block) => {
                                    hljs.highlightElement(block);
                                });

                                // Initialize newly added images
                                initializeNewImages(botMessageElement);

                                scrollToBottomIfNeeded();
                            } else if (parsedData.message_ids) {
								if (parsedData.message_ids.bot) {
									newMessageId = parsedData.message_ids.bot;
								}
                                if (parsedData.message_ids.user) {
                                    newUserMessageId = parsedData.message_ids.user;
                                }
							}


                        } catch (error) {
                            console.error('Error parsing JSON:', error);
                        }
                    }
                }
                return readStream();
            });
        }

        // Add timestamp and icons to bot message
        let botTimestamp = new Date();
        let localBotTimestamp = botTimestamp.toLocaleString(undefined, {
            year: 'numeric',
            month: 'numeric',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            hour12: false,
            timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone
        });
        
        timeSpan.textContent = botTimestamp.toLocaleString(undefined, {
            hour: '2-digit',
            minute: '2-digit',
            hour12: false
        });
        timeSpan.title = localBotTimestamp;

        return readStream().then(() => {
            removeLoadingIndicator();
            document.getElementById('message-text').disabled = false;
            document.querySelector('#form-message button[type="submit"]').disabled = false;
            document.getElementById('message-text').focus();
            scrollToBottomIfNeeded();

            isCurrentConversationEmpty = false;
            attachedFiles = [];
            if (endConversation) {
                document.getElementById('message-text').disabled = true;
                document.getElementById('send-button').disabled = true;
                toggleSendButton('Send');
                document.getElementById('send-button').onclick = null;
            } else {
                toggleSendButton('Send');
            }

            // Update the bot message ID
            if (newMessageId) {
                updateMessageId(newMessageId, botMessageElement);
            }
            if (newUserMessageId && userMessageElement) {
                updateMessageId(newUserMessageId, userMessageElement);
            }

            // Add event listeners to show/hide icons
            botMessageElement.addEventListener('mouseover', function() {
                const messageId = this.dataset.messageId;
                this.querySelectorAll('.fa-volume-up, .fa-bookmark, .fa-copy, .fa-undo').forEach(icon => {
                    if (!icon.classList.contains('bookmarked')) {
                        if (icon.classList.contains('fa-bookmark') && !messageId) {
                            icon.style.display = 'none';
                            return;
                        }
                        icon.style.display = 'inline';
                    }
                });
            });

            botMessageElement.addEventListener('mouseout', function() {
                this.querySelectorAll('.fa-volume-up, .fa-bookmark, .fa-copy, .fa-undo').forEach(icon => {
                    if (!icon.classList.contains('bookmarked')) {
                        icon.style.display = 'none';
                    }
                });
            });
        });
    })
    .catch(error => {
        console.error('Error:', error);
        removeLoadingIndicator();
        toggleSendButton('Send');
        document.getElementById('message-text').disabled = false;
        document.querySelector('#form-message button[type="submit"]').disabled = false;
        document.getElementById('message-text').focus();

        // Reset text field to original size in case of error
        document.getElementById('message-text').style.height = 'auto';
    });
}

function updateMessageId(messageId, element) {
    if (element) {
        const isBotMessage = element.classList.contains('bot');
        element.dataset.messageId = messageId;
        const rollbackIcon = element.querySelector('.fa-undo');
        if (rollbackIcon) {
            rollbackIcon.setAttribute('data-message-id', messageId);
            rollbackIcon.style.display = 'inline';
            rollbackIcon.onclick = function() {
                rollbackConversation(messageId, currentConversationId);
            };
        } else if (isBotMessage) {
        }
    } else {
    }
}

function getLastMessageId(element = null) {
    return fetch(`/api/conversations/${currentConversationId}/last_message_id`)
        .then(response => response.json())
        .then(data => {
            if (data.message_id) {
                if (element) {
                    updateMessageId(data.message_id, element);
                }
                return data.message_id;
            } else {
                return null;
            }
        })
        .catch(error => {
            console.error('Error fetching last message ID:', error);
            return null;
        });
}

function createAvatar(author) {
    const avatarContainer = document.createElement('div');
    avatarContainer.classList.add('avatar');

    if (author === 'user') {
        if (userProfilePicture) {
            var avatarImg = document.createElement('img');
            avatarImg.src = userProfilePicture;
            avatarImg.alt = username;
            avatarImg.title = username;
            avatarImg.classList.add('avatar-img');
            avatarContainer.appendChild(avatarImg);
        } else {
            var userInitial = username.charAt(0).toUpperCase();
            avatarContainer.textContent = userInitial;
            avatarContainer.title = username;
        }
    } else {
        if (botProfilePicture) {
            var avatarImg = document.createElement('img');
            avatarImg.src = botProfilePicture;
            avatarImg.alt = botname;
            avatarImg.title = botname;
            avatarImg.classList.add('avatar-img');
            avatarContainer.appendChild(avatarImg);
        } else {
            var botInitial = (botname && botname.length > 0) ? botname.charAt(0).toUpperCase() : 'B';
            avatarContainer.textContent = botInitial;
            avatarContainer.title = botname || 'Bot';
        }
    }

    return avatarContainer;
}

function updateActiveChatName(newName) {
    
    // Search for the active chat anywhere (main list or folders)
    const activeChatElement = document.querySelector('.active-chat');
    
    if (activeChatElement) {
        const chatNameSpan = activeChatElement.querySelector('.chat-name');
        
        if (chatNameSpan) {
            const oldName = chatNameSpan.textContent;
            chatNameSpan.textContent = newName;
        } else {
            console.error('Element span.chat-name not found within active chat');
        }
    } else {
        console.error('Active chat element not found');
    }

    // Update chat title at the top
    const chatTitle = document.querySelector('.chatbot-info h4');
    if (chatTitle) {
        const dateOptions = { year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' };
        const formattedStartDate = new Date(startDate).toLocaleDateString(undefined, dateOptions);
        chatTitle.textContent = `${newName}`;
        chatTitle.title = `Created: ${formattedStartDate}`;
    }
}

const loadedConversationIds = new Set();

function addConversationElement(conversation, chatName, currentConversationId, isNew = false) {
    //console.log(`Adding conversation: ${conversation.id} External: ${conversation.external_platform} Chat Name: ${chatName}`);
    
    // Ensure we always have a valid name
    chatName = chatName || `Chat ${conversation.id}`;

    // Check if element already exists for this conversation
    const existingElement = document.querySelector(`[data-conversation-id="${conversation.id}"]`);
    if (existingElement) {
        // Preserve external state if it already existed
        if (!conversation.external_platform && existingElement.dataset.externalPlatform) {
            conversation.external_platform = existingElement.dataset.externalPlatform;
        }
        updateSingleConversation(existingElement, conversation, document.querySelector('#external-chats-container'), document.querySelector('#dynamic-chats-container'));
        return;
    }

    if (loadedConversationIds.has(conversation.id)) {
        return; // Skip if already loaded
    }
    
    loadedConversationIds.add(conversation.id);

    const dynamicChatsContainer = document.querySelector('#dynamic-chats-container');
    const externalChatsContainer = document.querySelector('#external-chats-container');
    const conversationElement = document.createElement('a');
    conversationElement.href = '#';
    conversationElement.classList.add('list-group-item', 'list-group-item-action');
    conversationElement.dataset.conversationId = conversation.id;
    conversationElement.dataset.machine = conversation.machine;
    conversationElement.dataset.llmModel = conversation.llm_model || '';
    conversationElement.dataset.locked = conversation.locked ? 'true' : 'false';
    conversationElement.dataset.webSearchAllowed = conversation.web_search_allowed !== false ? 'true' : 'false';
    if (conversation.locked) {
        conversationElement.classList.add('conversation-locked');
    }
    if (conversation.external_platform) {
        conversationElement.dataset.externalPlatform = conversation.external_platform;
    }
    if (conversation.id === currentConversationId) {
        conversationElement.classList.add('active-chat');
    }

    // Create conversation element content
    const nameSpan = document.createElement('span');
    nameSpan.className = 'chat-name';
    if (conversation.external_platform) {
        const iconClass = getExternalPlatformIcon(conversation.external_platform);
        nameSpan.innerHTML = `<i class="${iconClass}"></i> ${chatName}`;
    } else if (conversation.locked) {
        nameSpan.innerHTML = `<i class="fas fa-comment-slash" title="This conversation is locked"></i> ${chatName}`;
    } else {
        nameSpan.textContent = chatName;
    }
    conversationElement.appendChild(nameSpan);

    // Create and add menu
    const chatMenu = createChatMenu(conversation);
    conversationElement.appendChild(chatMenu);

    // Handle conversation addition
    if (conversation.external_platform) {
        //console.log(`External conversation detected: ${conversation.external_platform}`);
        // Move current external chat (if exists) to dynamic container
        const currentExternalChat = externalChatsContainer.firstElementChild;
        if (currentExternalChat) {
            currentExternalChat.classList.remove('active-chat');
            dynamicChatsContainer.insertBefore(currentExternalChat, dynamicChatsContainer.firstChild);
        }

        // Clear external container
        externalChatsContainer.innerHTML = '';
        
        // Add new conversation element to externalChatsContainer
        externalChatsContainer.appendChild(conversationElement);
        document.querySelector('.external-section').style.display = 'block';
    } else {
        if (isNew) {
            dynamicChatsContainer.insertBefore(conversationElement, dynamicChatsContainer.firstChild);
        } else {
            dynamicChatsContainer.appendChild(conversationElement);
        }
    }
    setupConversationElementListeners(conversationElement);
}

function createChatMenu(conversation) {
    const chatMenu = document.createElement('div');
    chatMenu.classList.add('chat-menu');

    const ellipsisIcon = document.createElement('i');
    ellipsisIcon.classList.add('fas', 'fa-ellipsis-h');
    chatMenu.appendChild(ellipsisIcon);

    const chatMenuContent = document.createElement('div');
    chatMenuContent.classList.add('chat-menu-content');
    chatMenu.appendChild(chatMenuContent);

    // Rename option
    const renameLink = createMenuLink('fa-edit', 'Rename', () => renameConversation(conversation.id));
    chatMenuContent.appendChild(renameLink);

    // Download as MP3 option
    const downloadAudioLink = createMenuLink('fa-music', 'Download MP3', () => downloadAudio(conversation.id));
    chatMenuContent.appendChild(downloadAudioLink);

    // Download as PDF option
    const downloadPdfLink = createMenuLink('fa-download', 'Download PDF', () => downloadPDF(conversation.id));
    chatMenuContent.appendChild(downloadPdfLink);

    // Delete option
    const deleteLink = createMenuLink('fa-trash-alt', 'Delete', () => deleteConversation(conversation.id), 'text-danger');
    chatMenuContent.appendChild(deleteLink);

    // Lock/Unlock option (admin only)
    if (typeof isAdmin !== 'undefined' && isAdmin) {
        const isLocked = conversation.locked;
        const lockIcon = isLocked ? 'fa-lock-open' : 'fa-lock';
        const lockText = isLocked ? 'Unlock' : 'Lock';
        const lockLink = createMenuLink(lockIcon, lockText, () => toggleLockConversation(conversation.id, !isLocked));
        chatMenuContent.appendChild(lockLink);
    }

    // Add separator
    const separator = document.createElement('div');
    separator.classList.add('menu-separator');
    chatMenuContent.appendChild(separator);

    // WhatsApp option
    const whatsappLink = createPlatformLink('whatsapp', conversation);
    chatMenuContent.appendChild(whatsappLink);

    // Click handler for the chat-menu-content div
    chatMenu.addEventListener('click', (e) => {
        e.stopPropagation();

        const isCurrentlyOpen = chatMenuContent.style.display === 'block';

        // Close all open menus first
        document.querySelectorAll('.chat-menu-content').forEach(menu => {
            menu.style.display = 'none';
            menu.classList.remove('menu-above');
        });

        // Toggle current menu
        if (!isCurrentlyOpen) {
            chatMenuContent.style.display = 'block';

            // Check if menu overflows viewport bottom
            const menuRect = chatMenuContent.getBoundingClientRect();
            const viewportHeight = window.innerHeight;

            if (menuRect.bottom > viewportHeight) {
                chatMenuContent.classList.add('menu-above');
            } else {
                chatMenuContent.classList.remove('menu-above');
            }
        }
    });

    // Handler to close menu when clicking outside
    document.addEventListener('click', (e) => {
        if (!chatMenu.contains(e.target) && chatMenuContent.style.display === 'block') {
            chatMenuContent.style.display = 'none';
            chatMenuContent.classList.remove('menu-above');
        }
    });

    // Prevent menu from closing when clicking inside
    chatMenuContent.addEventListener('click', (e) => {
        e.stopPropagation();
    });

    return chatMenu;
}

const renameConversation = withSession(function(conversationId) {
    // Close menu immediately
    const chatMenu = document.querySelector(`[data-conversation-id="${conversationId}"] .chat-menu`);
    if (chatMenu) {
        const chatMenuContent = chatMenu.querySelector('.chat-menu-content');
        if (chatMenuContent) {
            chatMenuContent.style.display = 'none';
        }
    }

    const conversationElement = document.querySelector(`[data-conversation-id="${conversationId}"]`);
    const nameSpan = conversationElement.querySelector('.chat-name');
    const currentName = nameSpan.textContent.trim();

    // Create an input to edit the name
    const input = document.createElement('input');
    input.type = 'text';
    input.value = currentName;
    input.classList.add('rename-input');
    input.maxLength = 256; // Limit to 256 characters

    // Replace the span with the input
    nameSpan.replaceWith(input);
    input.focus();

    // Function to cancel editing
    function cancelEdit() {
        input.replaceWith(nameSpan);
    }

    // Function to save new name
    async function saveNewName() {
        const newName = input.value.trim().substring(0, 256); // Ensure maximum 256 characters
        if (newName && newName !== currentName) {
            try {
                const response = await secureFetch(`/api/conversations/${conversationId}/rename`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ new_name: newName }),
                });

                if (response.ok) {
                    nameSpan.textContent = newName;
                    updateActiveChatName(newName);
                } else {
                    console.error('Error renaming conversation');
                }
            } catch (error) {
                console.error('Error sending rename request:', error);
            }
        }
        cancelEdit();
    }

    // Handle the event of pressing Enter or ESC
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            saveNewName();
        } else if (e.key === 'Escape') {
            cancelEdit();
        }
    });

    // Handle the event of losing focus
    document.addEventListener('click', function onClickOutside(e) {
        if (e.target !== input && !input.contains(e.target)) {
            saveNewName();
            document.removeEventListener('click', onClickOutside);
        }
    });

    // Prevent the click inside the input from propagating the event
    input.addEventListener('click', (e) => {
        e.stopPropagation();
    });
});

function createMenuLink(iconClass, text, onClick, additionalClass = '') {
    const link = document.createElement('a');
    link.href = '#';
    link.classList.add('menu-link');
    if (additionalClass) link.classList.add(additionalClass);
    link.innerHTML = `<i class="fas ${iconClass}"></i> ${text}`;
    link.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        onClick();
    });
    return link;
}

function getExternalPlatformIcon(platform) {
    switch (platform.toLowerCase()) {
        case 'whatsapp':
            return 'fab fa-whatsapp';
        case 'telegram':
            return 'fab fa-telegram';
        default:
            return 'fas fa-external-link-alt';
    }
}

function createPlatformLink(platform, conversation) {
    const isAssigned = conversation.external_platform === platform;
    
    if (platform === 'whatsapp' && isAssigned) {
        // Create WhatsApp submenu container
        const container = document.createElement('div');
        container.className = 'whatsapp-menu-container';
        
        // Main WhatsApp option
        const mainLink = document.createElement('a');
        mainLink.href = '#';
        mainLink.classList.add('platform-link');
        mainLink.innerHTML = `<i class="fab fa-whatsapp"></i> Remove from WhatsApp`;
        mainLink.addEventListener('click', function(e) {
            e.stopPropagation();
            toggleExternalPlatform(conversation.id, platform, isAssigned);
        });
        
        // Mode separator
        const modeSeparator = document.createElement('div');
        modeSeparator.classList.add('menu-separator');
        
        // Voice mode option
        const voiceModeLink = document.createElement('a');
        voiceModeLink.href = '#';
        voiceModeLink.classList.add('platform-link', 'whatsapp-mode-option');
        voiceModeLink.innerHTML = `<i class="fas fa-microphone"></i> <span class="mode-text">Voice Mode</span> <span class="mode-check" style="display: none;">✓</span>`;
        voiceModeLink.addEventListener('click', function(e) {
            e.stopPropagation();
            changeWhatsAppMode(conversation.id, 'voice');
        });
        
        // Text mode option
        const textModeLink = document.createElement('a');
        textModeLink.href = '#';
        textModeLink.classList.add('platform-link', 'whatsapp-mode-option');
        textModeLink.innerHTML = `<i class="fas fa-keyboard"></i> <span class="mode-text">Text Mode</span> <span class="mode-check" style="display: none;">✓</span>`;
        textModeLink.addEventListener('click', function(e) {
            e.stopPropagation();
            changeWhatsAppMode(conversation.id, 'text');
        });
        
        container.appendChild(mainLink);
        container.appendChild(modeSeparator);
        container.appendChild(voiceModeLink);
        container.appendChild(textModeLink);
        
        // Load current mode and update checkmarks
        loadCurrentWhatsAppMode(conversation.id, voiceModeLink, textModeLink);
        
        return container;
    } else {
        // Regular platform link for other cases
        const link = document.createElement('a');
        link.href = '#';
        link.classList.add('platform-link');
        const icon = platform === 'whatsapp' ? 'fa-whatsapp' : 'fa-telegram';
        const action = isAssigned ? 'Remove from' : 'Use for';
        link.innerHTML = `<i class="fab ${icon}"></i> ${action} ${platform.charAt(0).toUpperCase() + platform.slice(1)}`;
        link.addEventListener('click', function(e) {
            e.stopPropagation();
            toggleExternalPlatform(conversation.id, platform, isAssigned);
        });
        return link;
    }
}

const toggleExternalPlatform = withSession(function(conversationId, platform, isAssigned) {
    const action = isAssigned ? 'remove' : 'add';
    const visibleCount = getVisibleConversationsCount();
    secureFetch(`/api/conversations/${conversationId}/external-platform`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ platform, action, visible_count: visibleCount })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateConversationElement(conversationId, data.updatedConversations.find(conv => conv.id === parseInt(conversationId)), data.updatedConversations);
        } else {
            console.error('Error updating external platform:', data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
});

function updateConversationElement(conversationId, updatedConversation, allConversations) {
    const externalChatsContainer = document.querySelector('#external-chats-container');
    const dynamicChatsContainer = document.querySelector('#dynamic-chats-container');

    // First, remove all existing instances of updated conversation
    document.querySelectorAll(`[data-conversation-id="${conversationId}"]`).forEach(el => el.remove());

    // Then, update or create conversation element
    const element = document.createElement('a');
    updateSingleConversation(element, updatedConversation, externalChatsContainer, dynamicChatsContainer);

    // Update other conversations if necessary
    allConversations.forEach(conv => {
        if (conv.id !== conversationId) {
            const existingElement = document.querySelector(`[data-conversation-id="${conv.id}"]`);
            if (existingElement) {
                updateSingleConversation(existingElement, conv, externalChatsContainer, dynamicChatsContainer);
            } else {
                const newElement = document.createElement('a');
                updateSingleConversation(newElement, conv, externalChatsContainer, dynamicChatsContainer);
            }
        }
    });

    if (updatedConversation.external_platform) {
        // Move to external section
        externalChatsContainer.innerHTML = '';
        externalChatsContainer.appendChild(element);
        document.querySelector('.external-section').style.display = 'block';
    } else {
        // Move to dynamic container if not external
        dynamicChatsContainer.appendChild(element);
    }
    // Sort conversations in dynamic container
    sortDynamicChats(dynamicChatsContainer);

    // Hide external section if empty
    if (externalChatsContainer.children.length === 0) {
        document.querySelector('.external-section').style.display = 'none';
    } else {
        document.querySelector('.external-section').style.display = 'block';
    }
}

function getVisibleConversationsCount() {
    const externalCount = document.querySelector('#external-chats-container').children.length;
    const dynamicCount = document.querySelector('#dynamic-chats-container').children.length;
    return externalCount + dynamicCount;
}

function sortDynamicChats(dynamicChatsContainer) {
    const chats = Array.from(dynamicChatsContainer.children);
    chats.sort((a, b) => {
        const idA = parseInt(a.dataset.conversationId);
        const idB = parseInt(b.dataset.conversationId);
        return idB - idA; // Descending order, change to idA - idB for ascending order
    });
    
    chats.forEach(chat => dynamicChatsContainer.appendChild(chat));
}

function updateSingleConversation(element, conversationData, externalContainer, dynamicContainer) {
    // Ensure we always have a valid name
    const chatName = conversationData.chat_name || `Chat ${conversationData.id}`;
    let conversationContent = '';
    let targetContainer;

    if (conversationData.external_platform) {
        const iconClass = getExternalPlatformIcon(conversationData.external_platform);
        conversationContent = `<i class="${iconClass}"></i> ${chatName}`;
        targetContainer = externalContainer;
    } else {
        conversationContent = chatName;
        targetContainer = dynamicContainer;
    }

    // Update existing element or create a new one
    if (!(element instanceof HTMLElement)) {
        element = document.createElement('a');
    }
    element.href = '#';
    element.className = 'list-group-item list-group-item-action';
    if (element.classList.contains('active-chat')) {
        element.classList.add('active-chat');
    }
    element.dataset.conversationId = conversationData.id;
    element.dataset.machine = conversationData.machine || 'undefined';

    element.innerHTML = conversationContent;

    const chatMenu = createChatMenu(conversationData);
    element.appendChild(chatMenu);

    if (element.parentElement !== targetContainer) {
        if (targetContainer === externalContainer) {
            const currentExternalChat = externalContainer.firstElementChild;
            if (currentExternalChat) {
                currentExternalChat.classList.remove('active-chat');
                dynamicContainer.insertBefore(currentExternalChat, dynamicContainer.firstChild);
            }
            externalContainer.innerHTML = '';
            externalContainer.appendChild(element);
        } else {
            dynamicContainer.appendChild(element); 
        }
    } else if (!element.parentElement) {
        targetContainer.appendChild(element);
    }

    setupConversationElementListeners(element);
}

function setupConversationElementListeners(element) {
    element.removeEventListener('click', conversationClickHandler);
    element.addEventListener('click', conversationClickHandler);
}

function conversationClickHandler(e) {
    if (!e.target.closest('.chat-menu')) {
        var conversationId = this.getAttribute('data-conversation-id');
        var chatNameElement = this.querySelector('.chat-name');
        var chatName = chatNameElement ? chatNameElement.textContent.trim() : `Chat ${conversationId}`;
        var machine = this.getAttribute('data-machine');
        
        if (conversationId) {
            
            // Remove active-chat class from ALL chats everywhere
            document.querySelectorAll('.active-chat').forEach(el => {
                el.classList.remove('active-chat');
            });
            
            // Add active-chat class to this element
            this.classList.add('active-chat');
            
            // Update global selectedChat variable
            window.selectedChat = this;
            
            continueConversation(conversationId, chatName, machine);
        }
    }
}

function getPlatformFromElement(element) {
    const whatsappIcon = element.querySelector('.fa-whatsapp');
    const telegramIcon = element.querySelector('.fa-telegram');
    if (whatsappIcon) return 'whatsapp';
    if (telegramIcon) return 'telegram';
    return null;
}

function deactivateChat() {
    // Hide the input box and buttons
    document.getElementById('message-input-container').style.display = 'none';
    
    // Add translucent layer over chat-window
    const chatWindow = document.getElementById('window-chat');
    let overlay = document.getElementById('chat-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'chat-overlay';
        chatWindow.appendChild(overlay);
    }
    overlay.style.display = 'block';

    // Update global state
    currentConversationId = null;
}

function removeOverlay() {
    const overlay = document.getElementById('chat-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
    document.getElementById('message-input-container').style.display = 'flex';
}


function removeConversationElement(conversationId) {
    const conversationElement = document.querySelector(`[data-conversation-id="${conversationId}"]`);
    if (conversationElement) {
        conversationElement.remove();
    }
}

function loadConversations(loadMore = false, isInit = false) {
    if (allConversationsLoaded && !isInit) return Promise.resolve();

    // Use embedded data on initial load (avoids HTTP request)
    if (isInit && !loadMore && typeof embeddedInitialConversations !== 'undefined' && embeddedInitialConversations !== null) {
        const conversations = embeddedInitialConversations;
        embeddedInitialConversations = null; // Clear to avoid reuse
        return processConversations(conversations, loadMore, isInit);
    }

    var url = `/api/conversations?user_id=${user_id}&limit=${limit}`;
    if (lowestLoadedId !== Infinity) {
        url += `&max_id=${lowestLoadedId - 1}`;
    }

    // By default, only load conversations not in folders (loose conversations)
    // folder_id parameter when null/undefined will return conversations not in folders
    // This ensures the main chat list only shows loose conversations

    return fetch(url)
        .then(response => response.json())
        .then(conversations => processConversations(conversations, loadMore, isInit))
        .catch(error => {
            console.error('Error loading conversations:', error);
            enableInputControls();
            document.getElementById('loading-indicator').style.display = 'none';
        });
}

function processConversations(conversations, loadMore, isInit) {
    if (conversations.length === 0) {
        allConversationsLoaded = true;
        document.getElementById('load-more-button').style.display = 'none';
        if (isInit) {
            return startNewConversation();
        }
        return Promise.resolve();
    }

    conversations.forEach(conversation => {
        addConversationElement(conversation, conversation.chat_name, currentConversationId);
        lowestLoadedId = Math.min(lowestLoadedId, conversation.id);
    });

    if (conversations.length < limit) {
        allConversationsLoaded = true;
        document.getElementById('load-more-button').style.display = 'none';
    } else {
        document.getElementById('load-more-button').style.display = 'block';
    }

    if (loadMore && conversations.length > 0) {
        const firstNewElement = document.querySelector(`[data-conversation-id="${conversations[0].id}"]`);
        if (firstNewElement) {
            firstNewElement.scrollIntoView({ behavior: 'smooth' });
        }
    }

    if (isInit && currentConversationId !== null) {
        const existingConversation = conversations.find(conv => conv.id === currentConversationId);
        if (existingConversation) {
            return continueConversation(currentConversationId, existingConversation.chat_name, existingConversation.machine, isInit, null, existingConversation);
        } else {
            // Load the most recent conversation instead of creating a new one
            if (conversations && conversations.length > 0) {
                const mostRecent = conversations[0]; // Conversations are ordered by date
                return continueConversation(mostRecent.id, mostRecent.chat_name, mostRecent.machine, isInit, null, mostRecent);
            }
            // Only create new chat if there are no conversations at all
            return startNewConversation();
        }
    }

    return Promise.resolve();
}

function loadMoreConversations() {
    loadConversations(true);
}

function continueConversation(conversationId, chatName, machine, isInit = false, targetMessageId = null, conversationData = null) {
    removeOverlay();

    // Check if this conversation is already loaded (compare with current conversation ID)
    if (currentConversationId &&
        currentConversationId.toString() === conversationId.toString() &&
        !isInit) {
        return Promise.resolve();
    }

    const selectedChat = document.querySelector(`[data-conversation-id="${conversationId}"]`);   

    offset = 0;
    allMessagesLoaded = false;
    isLoading = false;
    localStorage.setItem('activeConversationId', conversationId);
    currentConversationId = conversationId;
    const dateOptions = { year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' };
    const formattedStartDate = new Date(startDate).toLocaleDateString(undefined, dateOptions);

    // Extract only chat name, ignoring menu
    let chatTitleText = "New Chat";
    if (selectedChat && selectedChat.firstChild) {
        chatTitleText = selectedChat.firstChild.textContent.trim();
    } else if (chatName) {
        chatTitleText = chatName;
    }
    document.querySelector('.chatbot-info h4').textContent = `${chatTitleText}`;
    
    // Deactivate all chats and My Bookmarks
    document.querySelectorAll('.list-group-item-action').forEach(function(item) {
        item.classList.remove('active-chat', 'previously-active-chat', 'active-bookmarks');
    });

    // Activate the selected chat
    if (selectedChat) {
        selectedChat.classList.add('active-chat');
    }
    // Disable input controls before loading messages
    disableInputControls();

    return new Promise((resolve) => {
        loadMessages(conversationId, false, targetMessageId).then(() => {
            setupInfiniteScroll(conversationId);

            // Show and correctly configure the input box and other chat-related elements
            const messageInputContainer = document.getElementById('message-input-container');
            messageInputContainer.style.display = 'flex';
            messageInputContainer.style.justifyContent = 'center';

            const formMessage = document.getElementById('form-message');
            formMessage.style.display = 'flex';
            formMessage.style.justifyContent = 'center';
            formMessage.style.width = '100%';
            formMessage.style.maxWidth = '48rem';

            const messageText = document.getElementById('message-text');
            messageText.focus();

            if (typeof window.closeSidebar === 'function') {
                window.closeSidebar();
            }

            // Get llm_model from conversation element or passed data
            const llmModel = selectedChat?.dataset?.llmModel || conversationData?.llm_model || null;
            updateChatHeader(conversationId, chatTitleText, llmModel);

            // Check if conversation is locked (from DOM element or passed data)
            isCurrentConversationLocked = selectedChat?.dataset?.locked === 'true' || conversationData?.locked === true;
            const lockedBanner = document.getElementById('locked-conversation-banner');

            if (isCurrentConversationLocked) {
                // Show locked banner and disable input (but not loading indicator)
                if (lockedBanner) lockedBanner.style.display = 'flex';
                messageText.placeholder = 'This conversation is locked';
                messageText.disabled = true;
                document.querySelector('#form-message button[type="submit"]').disabled = true;
                document.getElementById('loading-indicator').style.display = 'none';
            } else {
                // Hide locked banner and enable input
                if (lockedBanner) lockedBanner.style.display = 'none';
                enableInputControls();
                messageText.placeholder = 'Type a message...';
                messageText.disabled = false;
            }
            
            initializeCurrentMessageIndex();
            showPromptInfo();

            // Initialize web search toggle control with data from conversation element or passed data
            let webSearchAllowedByPrompt = null;
            if (selectedChat?.dataset?.webSearchAllowed !== undefined) {
                webSearchAllowedByPrompt = selectedChat.dataset.webSearchAllowed === 'true';
            } else if (conversationData?.web_search_allowed !== undefined) {
                webSearchAllowedByPrompt = conversationData.web_search_allowed;
            }
            initWebSearchControl(conversationId, webSearchAllowedByPrompt);

            resolve();
        });
    });
}

function isMyBookmarksView() {
    return document.querySelector('.chatbot-info h4').textContent === "My Bookmarks";
}

function updateChatHeader(conversationId, chatName, llmModel = null) {
    const chatTitle = document.getElementById('chat-title');
    const chatModel = document.getElementById('chat-model');
    const chatTitleAvatar = document.getElementById('chat-title-avatar');
    const dateOptions = { year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' };
    const formattedStartDate = new Date(startDate).toLocaleDateString(undefined, dateOptions);

    chatTitleAvatar.innerHTML = ''; // Clear the container
    const botAvatar = createAvatar('bot');
    chatTitleAvatar.appendChild(botAvatar);

    // Check if we are in "My Bookmarks"
    if (chatName === "My Bookmarks") {
        chatTitle.textContent = chatName;
        chatTitle.title = ''; // No hover
        chatModel.textContent = ''; // No model
        chatTitleAvatar.style.display = 'none';
    } else {
        chatTitleAvatar.style.display = 'block';

        // Set clean title (without date or prompt)
        chatTitle.textContent = chatName;
        chatTitle.title = `Created: ${formattedStartDate}`; // Show date on hover

        // Use provided model or fetch from API as fallback
        if (llmModel) {
            chatModel.textContent = llmModel;
            if (window.modelSelector) {
                window.modelSelector.updateCurrentModel(llmModel);
            }
        } else {
            // Fallback: fetch from API (for new conversations without cached data)
            fetch(`/api/conversations/${conversationId}/details`)
                .then(response => response.json())
                .then(data => {
                    const modelInfo = data.model || 'Unknown Model';
                    chatModel.textContent = modelInfo;

                    // Update model selector state if it exists
                    if (window.modelSelector) {
                        window.modelSelector.updateCurrentModel(modelInfo);
                    }
                })
                .catch(error => {
                    console.error('Error fetching conversation details:', error);
                    chatModel.textContent = '';
                });
        }
    }
}



function convertToLocalTime(utcTimestamp) {
    
    const date = new Date(utcTimestamp + 'Z');  // Add 'Z' to force UTC

    const localTimeString = date.toLocaleString('en-CA', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        hour12: false,
        timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone
    });
    
    const formattedTime = localTimeString.replace(/(\d+)\/(\d+)\/(\d+)/, '$3/$2/$1');
    
    return {
        originalUtc: utcTimestamp,
        localTime: formattedTime
    };
}

function processMessage(message, container, prepend = false) {
    const timestamps = convertToLocalTime(message.date);
    let messageObj;

    try {
        const parsedMessage = JSON.parse(message.message);
        if (Array.isArray(parsedMessage)) {
            parsedMessage.forEach(item => {
                if (item.type === 'text') {
                    messageObj = {
                        type: 'text',
                        text: String(item.text),
                        is_bookmarked: message.is_bookmarked,
                        conversation_id: message.conversation_id
                    };
                } else if (item.type === 'image_url') {
                    messageObj = {
                        type: 'image_url',
                        url: item.image_url.url,
                        alt: item.image_url.alt,
                        is_bookmarked: message.is_bookmarked,
                        conversation_id: message.conversation_id
                    };
                } else if (item.type === 'video_url') {
                    messageObj = {
                        type: 'video_url',
                        url: item.video_url.url,
                        alt: item.video_url.alt,
                        mime_type: item.video_url.mime_type,
                        poster: item.video_url.poster,
                        is_bookmarked: message.is_bookmarked,
                        conversation_id: message.conversation_id
                    };
                } else if (item.type === 'image' && item.source.type === 'base64') {
                    messageObj = {
                        type: 'image_url',
                        url: item.source.data,
                        is_bookmarked: message.is_bookmarked,
                        conversation_id: message.conversation_id
                    };
                }

                addMessage(
                    message.type,
                    null,
                    { timestamp: timestamps, isNewMessage: false },
                    false,
                    messageObj,
                    prepend,
                    container,
                    message.id
                );
            });
        } else {
            messageObj = {
                type: 'text',
                text: String(parsedMessage),
                is_bookmarked: message.is_bookmarked,
                conversation_id: message.conversation_id
            };
            addMessage(
                message.type,
                null,
                { timestamp: timestamps, isNewMessage: false },
                false,
                messageObj,
                prepend,
                container,
                message.id
            );
        }
    } catch (e) {
        messageObj = {
            type: 'text',
            text: String(message.message),
            is_bookmarked: message.is_bookmarked,
            conversation_id: message.conversation_id
        };
        addMessage(
            message.type,
            null,
            { timestamp: timestamps, isNewMessage: false },
            false,
            messageObj,
            prepend,
            container,
            message.id
        );
    }

    // Mark the message as processed
    processedMessageIds.add(message.id);
}

async function loadMessages(conversationId, prepend = false, targetMessageId = null, attempt = 0) {
    if (currentAbortController) {
        currentAbortController.abort();
    }

    currentAbortController = new AbortController();
    const signal = currentAbortController.signal;
    
    if (isLoading || allMessagesLoaded) return Promise.resolve();
    isLoading = true;

    disableInputControls();
    
    const chatWindow = document.getElementById('chat-window');
    const chatMessagesContainer = document.getElementById('chat-messages-container');
    const scrollHeight = chatWindow.scrollHeight;
    const scrollTop = chatWindow.scrollTop;
    
    try {
        const response = await secureFetch(`/api/conversations/${conversationId}/messages?limit=${limitMessage}&offset=${offset}`, { signal });
        if (!response) {
            // secureFetch returned null (session expired, modal already shown)
            isLoading = false;
            enableInputControls();
            return null;
        }

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        const messages = data.messages;
        const conversationInfo = data.conversation_info;
        
        if (messages.length < limitMessage) {
            allMessagesLoaded = true;
        }
        
        if (!prepend) {
            chatMessagesContainer.innerHTML = '';
        }
        
        const tempDiv = document.createElement('div');
        
		// name of the prompt (bot)
        botname = conversationInfo.prompt_name;

		// prompt description (bot)
		promptDescription = conversationInfo.prompt_description

        // Update the bot's profile picture
        botProfilePicture = conversationInfo.bot_profile_picture;

        let targetMessageFound = false;

        messages.forEach(message => {
            processMessage(message, tempDiv, prepend);
            if (targetMessageId && message.id === targetMessageId) {
                targetMessageFound = true;
            }
        });
        
        if (prepend) {
            chatMessagesContainer.insertBefore(tempDiv, chatMessagesContainer.firstChild);
            const newScrollHeight = chatWindow.scrollHeight;
            chatWindow.scrollTop = newScrollHeight - scrollHeight + scrollTop;
        } else {
            chatMessagesContainer.appendChild(tempDiv);
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }
        
        offset += messages.length;
        isCurrentConversationEmpty = messages.length === 0 && offset === 0;

        if (targetMessageId && !targetMessageFound && !allMessagesLoaded) {
            // Try to load more messages
            return loadMessages(conversationId, prepend, targetMessageId, attempt + 1);
        } else if (targetMessageId && targetMessageFound) {
            setTimeout(() => {
                highlightAndScrollToMessage(targetMessageId);
            }, 100);
        } else if (!targetMessageId) {
            enableInputControls();
        }
    } catch (error) {
        if (error.name === 'AbortError') {
        } else if (error.message === 'Session expired') {
            // Session validation was already handled by secureFetch, no need to log as error
        } else {
            console.error('Error loading messages:', error);
        }
    } finally {
        isLoading = false;
        enableInputControls();
    }
}

function refreshActiveConversation() {
    if (!currentConversationId) {
        return Promise.resolve();
    }

    // Reset pagination so the next fetch pulls fresh messages.
    offset = 0;
    allMessagesLoaded = false;
    isLoading = false;

    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }

    processedMessageIds.clear();

    return loadMessages(currentConversationId, false);
}

window.refreshActiveConversation = refreshActiveConversation;

function highlightAndScrollToMessage(messageId) {
    const targetMessage = document.querySelector(`[data-message-id="${messageId}"]`);
    if (targetMessage) {
        targetMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
        targetMessage.classList.add('highlight');
        setTimeout(() => {
            targetMessage.classList.remove('highlight');
        }, 2000);
    } else {
        console.error('Message with specified ID not found:', messageId);
    }
}

function enableInputControls() {
    // Don't enable if conversation is locked
    if (isCurrentConversationLocked) {
        document.getElementById('loading-indicator').style.display = 'none';
        return;
    }

    document.getElementById('message-text').disabled = false;
    document.querySelector('#form-message button[type="submit"]').disabled = false;
    document.getElementById('image-files').disabled = false;
    if (document.querySelector('.attach-icon')) {
        document.querySelector('.attach-icon').style.pointerEvents = 'auto';
    }
    if (document.getElementById('audio-button')) {
        document.getElementById('audio-button').style.pointerEvents = 'auto';
    }
    document.getElementById('loading-indicator').style.display = 'none';
}

function disableInputControls() {
    document.getElementById('message-text').disabled = true;
    document.querySelector('#form-message button[type="submit"]').disabled = true;
    document.getElementById('image-files').disabled = true;
    if (document.querySelector('.attach-icon')) {
        document.querySelector('.attach-icon').style.pointerEvents = 'none';
    }
    if (document.getElementById('audio-button')) {
        document.getElementById('audio-button').style.pointerEvents = 'none';
    }
    document.getElementById('loading-indicator').style.display = 'block';
}


function setupInfiniteScroll(conversationId) {
    const chatWindow = document.getElementById('chat-window');
    chatWindow.onscroll = function() {
        // Infinite scroll: load older messages when at top
        if (chatWindow.scrollTop === 0 && !isLoading && !allMessagesLoaded) {
            loadMessages(conversationId, true);
        }

        // Auto-scroll management: detect user scroll intent
        if (isNearBottom(chatWindow)) {
            isUserScrolledUp = false;
        } else {
            isUserScrolledUp = true;
        }
    };
}

let isCurrentConversationEmpty = true;
let isFirstCall = true;

function startNewConversation(promptId = null) {
    if (admin_view) {
        return Promise.resolve();
    }

    if (!isFirstCall && isCurrentConversationEmpty) {
        return Promise.resolve();
    }

    
    // If promptId is not passed, get it from the dropdown
    if (promptId === null) {
        const promptDropdown = document.getElementById('promptDropdown');
        if (promptDropdown) {
            promptId = promptDropdown.value;
        }
    }
    
    let body = {};
    if (promptId !== null) {
        body.prompt_id = promptId;
    }

    return secureFetch('/api/conversations/new', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(body)
    })
    .then(response => {
        if (!response) {
            // secureFetch returned null (likely session expired)
            throw new Error('Session expired');
        }
        return response.json();
    })
    .then(data => {
        startDate = new Date(); 
        addConversationElement(data, data.name, null, true);
        return continueConversation(data.id, data.name, data.machine);
    })
    .then(() => {
        isCurrentConversationEmpty = true; 
        isFirstCall = false;
    })
    .catch(error => {
        if (error.message === 'Session expired') {
            // Session validation was already handled by secureFetch, no need to log as error
            return;
        }
        console.error('Error starting a new conversation:', error);
    });
}

function stopReceivingStream(event) {
    if (event) {
        event.preventDefault();
    }
    
    fetch(`/api/conversations/${currentConversationId}/stop`, {
        method: 'POST'
    }).then(response => {
        if (response.ok) {
        } else {
            console.error('Server failed to acknowledge stop request.');
        }
    }).catch(error => {
        console.error('Error sending stop request:', error);
    });

    toggleSendButton();
    removeLoadingIndicator();
    document.getElementById('message-text').disabled = true;
    document.getElementById('message-text').value = '';
}

function toggleBookmark(messageId, conversationId, bookmarkIcon) {
    const isCurrentlyBookmarked = bookmarkIcon.classList.contains('bookmarked');
    const action = isCurrentlyBookmarked ? 'remove' : 'add';

    // Get the correct conversationId from the data-conversationId attribute of the icon
    const correctConversationId = bookmarkIcon.dataset.conversationId || conversationId;

    fetch(`/api/conversations/${correctConversationId}/bookmark`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
            message_id: messageId,
            action: action 
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            if (action === 'add') {
                bookmarkIcon.classList.add('fa-check');
                setTimeout(() => {
                    bookmarkIcon.classList.remove('fa-check');
                    bookmarkIcon.classList.add('bookmarked');
                    bookmarkIcon.style.display = 'inline';
                }, 1000);
            } else {
                bookmarkIcon.classList.add('fade-out');
                setTimeout(() => {
                    bookmarkIcon.classList.remove('bookmarked');
                    bookmarkIcon.classList.remove('fade-out');
                    bookmarkIcon.style.display = 'none';
                    
                    if (document.querySelector('.chatbot-info h4').textContent === "My Bookmarks") {
                        bookmarkIcon.closest('.message').remove();
                    }
                }, 300);
            }
        } else {
            console.error(`Error ${action === 'add' ? 'adding' : 'removing'} message ${action === 'add' ? 'to' : 'from'} favorites:`, data.error);
        }
    })
    .catch(error => {
        console.error(`Error ${action === 'add' ? 'adding' : 'removing'} message ${action === 'add' ? 'to' : 'from'} favorites:`, error);
    });
}

function loadBookmarkedMessages() {
    const myBookmarksBtn = document.getElementById('my-bookmarks-btn');
    if (myBookmarksBtn.classList.contains('active-bookmarks')) {
        return;
    }

    fetch(`/api/bookmarks`)
    .then(response => response.json())
    .then(messages => {
        const chatMessagesContainer = document.getElementById('chat-messages-container');
        chatMessagesContainer.innerHTML = '';
        
        const chatTitle = document.querySelector('.chatbot-info h4');
        const chatModel = document.getElementById('chat-model');
        chatTitle.textContent = "My Bookmarks";
        chatTitle.title = ''; // No hover for bookmarks
        chatModel.textContent = ''; // No model for bookmarks
        
        messages.forEach(message => {
            processMessage(message, chatMessagesContainer);
        });

        // Remove all rollback icons
        document.querySelectorAll('.rollback-icon').forEach(icon => icon.remove());

        document.getElementById('message-input-container').style.display = 'none';

        // Change the active chat to previously-active-chat
        const activeChat = document.querySelector('.list-group-item-action.active-chat');
        if (activeChat) {
            activeChat.classList.remove('active-chat');
            activeChat.classList.add('previously-active-chat');
        }
        
        // Deactivate other chats, but keep the previously-active-chat
        document.querySelectorAll('.list-group-item-action').forEach(function(item) {
            if (!item.classList.contains('previously-active-chat')) {
                item.classList.remove('active-chat', 'active-bookmarks');
            }
        });
        
        // Activate My Bookmarks button
        myBookmarksBtn.classList.add('active-bookmarks');
    })
    .catch(error => console.error('Error loading bookmarked messages:', error));
}

document.getElementById('my-bookmarks-btn').addEventListener('click', function(e) {
    e.preventDefault();
    loadBookmarkedMessages();
});

document.addEventListener('click', function(e) {
    if (e.target && e.target.classList.contains('list-group-item-action')) {
        var conversationId = e.target.getAttribute('data-conversation-id');
        var chatName = e.target.textContent.trim();
        var machine = e.target.getAttribute('data-machine'); 
        if (conversationId) {
			// Check if there are open websockets before trying to stop the audio
			if (ws && ws.readyState === WebSocket.OPEN) {
				stopAudioAndWebSocket();
			}
            continueConversation(conversationId, chatName, machine);
            
            // Deactivate My Bookmarks
            document.getElementById('my-bookmarks-btn').classList.remove('active-bookmarks');
        }
    } else if (e.target && (e.target.id === 'my-bookmarks-btn' || e.target.closest('#my-bookmarks-btn'))) {
        loadBookmarkedMessages();
    }
});

// Call this function when page loads or messages are updated
//document.addEventListener('DOMContentLoaded', initializeCurrentMessageIndex);

/* Vertical navigation of messages */

let currentMessageIndex = -1;
let isAtBottomOfMessage = false;


function getMessages() {
    return Array.from(document.getElementById('chat-messages-container').getElementsByClassName('message'));
}

function isMessageLong(message) {
    const chatWindow = document.getElementById('chat-window');
    return message.offsetHeight > chatWindow.clientHeight;
}

function createNumberBubble(button, number) {
    const bubble = document.createElement('div');
    bubble.className = 'number-bubble';
    bubble.textContent = number;

    const rect = button.getBoundingClientRect();
    
    // Calculate initial position
    let leftPosition = rect.left - 40;
    
    // Make sure the bubble does not leave the screen on the left
    if (leftPosition < 10) {
        leftPosition = 10;
    }
    
    bubble.style.left = `${leftPosition}px`;
    bubble.style.top = `${rect.top + rect.height / 2}px`;

    document.body.appendChild(bubble);

    setTimeout(() => {
        document.body.removeChild(bubble);
    }, 900);
}

function scrollToTop() {
    const messages = getMessages();
    //console.log('scrollToTop - Total messages:', messages.length);
    if (messages.length > 0) {
        if (currentMessageIndex === 0 && document.getElementById('chat-window').scrollTop === 0) {
            shakeMessage(messages[0]);
        } else {
            currentMessageIndex = 0;
            isAtBottomOfMessage = false;
            scrollToMessage(currentMessageIndex);
            createNumberBubble(document.querySelector('.nav-button[title="Go to the very top"]'), 1);
        }
    } else {
    }
}

function scrollToBottom() {
    const messages = getMessages();
    //console.log('scrollToBottom - Total messages:', messages.length);
    if (messages.length > 0) {
        const lastIndex = messages.length - 1;
        const lastMessage = messages[lastIndex];
        if (currentMessageIndex === lastIndex && isAtBottomOfMessage) {
            shakeMessage(lastMessage);
        } else {
            currentMessageIndex = lastIndex;
            isAtBottomOfMessage = isMessageLong(lastMessage);
            scrollToMessage(currentMessageIndex);
            createNumberBubble(document.querySelector('.nav-button[title="Go to the very bottom"]'), messages.length);
        }
    } else {
    }
}

function scrollUp() {
    const messages = getMessages();
    if (currentMessageIndex >= 0 && currentMessageIndex < messages.length) {
        const currentMessage = messages[currentMessageIndex];
        const chatWindow = document.getElementById('chat-window');
        const messageTop = currentMessage.offsetTop - chatWindow.offsetTop;
        
        if (chatWindow.scrollTop > messageTop) {
            // If we are not at the beginning of the current message, scroll up
            chatWindow.scrollTo({top: messageTop, behavior: 'smooth'});
            createNumberBubble(document.querySelector('.nav-button[title="Go one message up"]'), currentMessageIndex + 1);
        } else if (currentMessageIndex > 0) {
            // If we are at the beginning of the current message, go to the previous message
            currentMessageIndex--;
            scrollToMessage(currentMessageIndex);
            createNumberBubble(document.querySelector('.nav-button[title="Go one message up"]'), currentMessageIndex + 1);
        } else {
            // If we are on the first message and at the beginning, shake the message
            shakeMessage(currentMessage);
        }
    } else {
        initializeCurrentMessageIndex();
        if (currentMessageIndex > 0) {
            scrollUp();
        }
    }
}

function scrollDown() {
    const messages = getMessages();
    //console.log('scrollDown - Current index:', currentMessageIndex, 'Total messages:', messages.length);
    if (currentMessageIndex < messages.length - 1) {
        currentMessageIndex++;
        isAtBottomOfMessage = false;
        scrollToMessage(currentMessageIndex);
        createNumberBubble(document.querySelector('.nav-button[title="Go one message down"]'), currentMessageIndex + 1);
    } else if (currentMessageIndex === messages.length - 1) {
        const lastMessage = messages[messages.length - 1];
        if (isMessageLong(lastMessage) && !isAtBottomOfMessage) {
            isAtBottomOfMessage = true;
            scrollToMessage(currentMessageIndex);
        } else {
            shakeMessage(lastMessage);
        }
    } else {
        initializeCurrentMessageIndex();
        if (currentMessageIndex < messages.length - 1) {
            scrollDown();
        }
    }
}

function scrollToMessage(index) {
    const messages = getMessages();
    //console.log('scrollToMessage - Scrolling to index:', index);
    if (index >= 0 && index < messages.length) {
        const message = messages[index];
        if (isAtBottomOfMessage && isMessageLong(message)) {
            message.scrollIntoView({behavior: 'smooth', block: 'end'});
        } else {
            message.scrollIntoView({behavior: 'smooth', block: 'start'});
        }
        highlightMessage(message);
    } else {
    }
}

function highlightMessage(message) {
    message.classList.add('highlight');
    setTimeout(() => {
        message.classList.remove('highlight');
    }, 1000);
}

function shakeMessage(message) {
    if (message) {
        message.classList.add('shake');
        setTimeout(() => {
            message.classList.remove('shake');
        }, 500);
    } else {
    }
}

function initializeCurrentMessageIndex() {
    const chatWindow = document.getElementById('chat-window');
    const messages = getMessages();
    //console.log('initializeCurrentMessageIndex - Total messages:', messages.length);
    
    if (messages.length > 0) {
        const scrollTop = chatWindow.scrollTop;
        const containerHeight = chatWindow.clientHeight;

        for (let i = 0; i < messages.length; i++) {
            const messageTop = messages[i].offsetTop - chatWindow.offsetTop;
            const messageBottom = messageTop + messages[i].offsetHeight;

            if (messageBottom > scrollTop) {
                currentMessageIndex = i;
                isAtBottomOfMessage = (i === messages.length - 1) && 
                                      (messageBottom <= scrollTop + containerHeight) && 
                                      isMessageLong(messages[i]);
                //console.log('Current message index set to:', currentMessageIndex, 'isAtBottomOfMessage:', isAtBottomOfMessage);
                return;
            }
        }

        currentMessageIndex = messages.length - 1;
        isAtBottomOfMessage = true;
        //console.log('Scroll at bottom, setting to last message:', currentMessageIndex);
    } else {
        currentMessageIndex = -1;
        isAtBottomOfMessage = false;
        //console.log('No messages found, setting index to -1');
    }
}

async function handleResponse(response) {
    removeLoadingIndicator();
    switch (response.status) {
        case 402:
            showInsufficientBalancePopup("transcribe audio");
            break;
        case 204:
            break;
        case 500:
            const data = await response.json();
            NotificationModal.error('Server Error', data.error);
            break;
        default:
            if (response.ok) {
                const data = await response.json();
                if (data["prompt"]) {
                    document.getElementById('message-text').value = data["prompt"];
                    document.getElementById('send-button').click();
                }
            } else {
            }
            break;
    }
}

function showPromptInfo() {
    const chatMessagesContainer = document.getElementById('chat-messages-container');
    const existingPromptInfo = chatMessagesContainer.querySelector('.prompt-info');

    if (existingPromptInfo) {
        existingPromptInfo.remove();
    }

    const promptInfo = document.createElement('div');
    promptInfo.classList.add('prompt-info');

    const infoContainer = document.createElement('div');
    infoContainer.classList.add('prompt-info-container');

    const imageSection = document.createElement('div');
    imageSection.classList.add('prompt-image-section');
    imageSection.style.position = 'relative';

    const initialContainer = document.createElement('div');
    initialContainer.classList.add('prompt-initial');
    const initial = (botname || 'Assistant').charAt(0).toUpperCase();
    initialContainer.textContent = initial;
    initialContainer.title = botname || 'Assistant';
    imageSection.appendChild(initialContainer);

    if (botProfilePicture) {
        const img = document.createElement('img');
        img.src = botProfilePicture.replace('_32', '_128');
        img.style.position = 'absolute';
        img.style.top = '0';
        img.style.left = '0';
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'cover';

        img.style.cursor = 'pointer';
        img.dataset.fullsize = botProfilePicture.replace('_32', '_fullsize');
        img.onclick = function() {
            imageHandler.showFullsize(this.dataset.fullsize, null);
        };

        imageSection.appendChild(img);
    }
    
    const textSection = document.createElement('div');
    textSection.classList.add('prompt-text-section');
    
    const promptName = document.createElement('h3');
    promptName.classList.add('prompt-name');
    promptName.textContent = botname || 'Assistant';
    
    textSection.appendChild(promptName);

	const description = document.createElement('p');
	description.classList.add('prompt-description');
	description.textContent = promptDescription;
	textSection.appendChild(description);

    infoContainer.appendChild(imageSection);
    infoContainer.appendChild(textSection);
    promptInfo.appendChild(infoContainer);

    if (chatMessagesContainer.firstChild) {
        chatMessagesContainer.insertBefore(promptInfo, chatMessagesContainer.firstChild);
    } else {
        chatMessagesContainer.appendChild(promptInfo);
    }
}

// Call this function when page loads or messages are updated
//document.addEventListener('DOMContentLoaded', initializeCurrentMessageIndex);

// Add listener to update index when scrolling
document.getElementById('chat-window').addEventListener('scroll', initializeCurrentMessageIndex);

// Model Selector functionality
class ModelSelector {
    constructor() {
        this.dropdownMenu = document.getElementById('model-dropdown-menu');
        this.dropdownIcon = document.getElementById('model-dropdown-icon');
        this.modelSelectorContainer = document.querySelector('.model-selector-container');
        this.dropdownContent = document.getElementById('model-dropdown-content');
        this.chatModel = document.getElementById('chat-model');
        this.currentModel = null;
        this.currentLlmId = null;
        
        
        if (this.dropdownMenu) {
            // Initialize dropdown menu
        }
        
        this.init();
    }
    
    init() {
        // Initialize event listeners
        if (this.modelSelectorContainer) {
            this.modelSelectorContainer.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleDropdown();
            });
        }
        
        // Close dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!this.dropdownMenu.contains(e.target) && !this.modelSelectorContainer.contains(e.target)) {
                this.closeDropdown();
            }
        });
        
        // Populate models on page load
        this.populateModels();
    }
    
    populateModels() {
        
        if (!window.availableModels || !this.dropdownContent) {
            return;
        }
        
        // Group models by machine
        const groupedModels = {};
        window.availableModels.forEach(model => {
            if (!groupedModels[model.machine]) {
                groupedModels[model.machine] = [];
            }
            groupedModels[model.machine].push(model);
        });
        
        
        let html = '';
        
        // Sort machines: put GPT first, then alphabetically
        const sortedMachines = Object.keys(groupedModels).sort((a, b) => {
            if (a === 'GPT') return -1;
            if (b === 'GPT') return 1;
            return a.localeCompare(b);
        });
        
        sortedMachines.forEach((machine, groupIndex) => {
            if (groupIndex > 0) {
                html += '<div style="height: 4px;"></div>'; // Separator
            }
            
            html += `<div class="model-group">`;
            html += `<div class="model-group-header">${machine}</div>`;
            
            // Sort models within each group
            const sortedModels = groupedModels[machine].sort((a, b) => a.model.localeCompare(b.model));
            
            sortedModels.forEach(model => {
                // const visionBadge = model.vision ? '<span class="vision-badge">Vision</span>' : '';
                html += `
                    <div class="model-item" data-llm-id="${model.id}" data-model="${model.model}">
                        <span>${model.model}</span>
                    </div>
                `;
            });
            
            html += '</div>';
        });
        
        this.dropdownContent.innerHTML = html;
        
        
        // Add click listeners to model items
        this.dropdownContent.querySelectorAll('.model-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.stopPropagation();
                const llmId = parseInt(item.dataset.llmId);
                const modelName = item.dataset.model;
                this.selectModel(llmId, modelName);
            });
        });
    }
    
    updateCurrentModel(modelName) {
        this.currentModel = modelName;
        
        // Find the llm_id for this model
        if (window.availableModels) {
            const modelData = window.availableModels.find(m => m.model === modelName);
            if (modelData) {
                this.currentLlmId = modelData.id;
            }
        }
        
        // Update UI to show current model
        this.updateModelDisplay();
        
        // Update thinking tokens visibility
        if (window.updateThinkingTokensVisibility) {
            setTimeout(() => {
                window.updateThinkingTokensVisibility();
            }, 100);
        }
    }
    
    updateModelDisplay() {
        // Update current model highlighting in dropdown
        this.dropdownContent.querySelectorAll('.model-item').forEach(item => {
            item.classList.remove('current');
            if (item.dataset.model === this.currentModel) {
                item.classList.add('current');
            }
        });
    }
    
    async selectModel(llmId, modelName) {
        if (!currentConversationId || llmId === this.currentLlmId) {
            this.closeDropdown();
            return;
        }
        
        try {
            // Show loading state
            this.chatModel.textContent = 'Updating...';
            
            const response = await fetch(`/api/conversations/${currentConversationId}/model`, {
                method: 'PATCH',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'include',
                body: JSON.stringify({
                    llm_id: llmId
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                // Update local state
                this.currentModel = result.model;
                this.currentLlmId = llmId;
                
                // Update UI
                this.chatModel.textContent = result.model;
                this.updateModelDisplay();
                
                // Update thinking tokens visibility for new model
                if (window.updateThinkingTokensVisibility) {
                    window.updateThinkingTokensVisibility();
                }
                
                // Show success feedback
                this.showSuccess();
                
            } else {
                throw new Error('Failed to update model');
            }
            
        } catch (error) {
            console.error('Error updating model:', error);
            
            // Restore original model name
            this.chatModel.textContent = this.currentModel || 'Unknown Model';
            
            // Show error feedback
            this.showError();
        }
        
        this.closeDropdown();
    }
    
    showSuccess() {
        // Briefly highlight the model selector
        this.modelSelectorContainer.style.backgroundColor = 'rgba(139, 128, 116, 0.2)';
        setTimeout(() => {
            this.modelSelectorContainer.style.backgroundColor = '';
        }, 600);
    }
    
    showError() {
        // Briefly show error state
        this.modelSelectorContainer.style.backgroundColor = 'rgba(244, 67, 54, 0.1)';
        this.chatModel.style.color = '#f44336';
        
        setTimeout(() => {
            this.modelSelectorContainer.style.backgroundColor = '';
            this.chatModel.style.color = '';
        }, 2000);
    }
    
    toggleDropdown() {
        if (this.dropdownMenu.classList.contains('show')) {
            this.closeDropdown();
        } else {
            this.openDropdown();
        }
    }
    
    openDropdown() {
        
        // Don't open if no conversation is active
        if (!currentConversationId) {
            return;
        }
        
        this.dropdownMenu.classList.add('show');
        this.dropdownIcon.classList.add('expanded');
        
        
        
        this.updateModelDisplay(); // Ensure current model is highlighted
    }
    
    closeDropdown() {
        this.dropdownMenu.classList.remove('show');
        this.dropdownIcon.classList.remove('expanded');
    }
}

// Initialize model selector when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.modelSelector = new ModelSelector();
});

// WhatsApp Mode Management Functions
async function loadCurrentWhatsAppMode(conversationId, voiceModeLink, textModeLink) {
    try {
        const response = await secureFetch(`/api/whatsapp-mode/${conversationId}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (response && response.ok) {
            const data = await response.json();
            const currentMode = data.mode || 'text'; // Default to text mode
            
            // Update checkmarks
            updateModeCheckmarks(voiceModeLink, textModeLink, currentMode);
        }
    } catch (error) {
        console.error('Error loading WhatsApp mode:', error);
        // Default to text mode if error
        updateModeCheckmarks(voiceModeLink, textModeLink, 'text');
    }
}

function updateModeCheckmarks(voiceModeLink, textModeLink, currentMode) {
    const voiceCheck = voiceModeLink.querySelector('.mode-check');
    const textCheck = textModeLink.querySelector('.mode-check');
    
    
    if (currentMode === 'voice') {
        if (voiceCheck) voiceCheck.style.display = 'inline';
        if (textCheck) textCheck.style.display = 'none';
    } else {
        if (voiceCheck) voiceCheck.style.display = 'none';
        if (textCheck) textCheck.style.display = 'inline';
    }
}

const changeWhatsAppMode = withSession(async function(conversationId, newMode) {
    const modeText = newMode === 'voice' ? 'Voice Mode' : 'Text Mode';
    
    // Show confirmation modal
    showGenericModal(
        'Confirm Mode Change',
        `Are you sure you want to switch to ${modeText}?`,
        'Confirm',
        'Cancel',
        'btn-primary',
        async function() {
            try {
                const response = await secureFetch(`/api/whatsapp-mode/${conversationId}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ mode: newMode })
                });
                
                if (response && response.ok) {
                    const data = await response.json();
                    
                    // Show success message
                    showGenericModal(
                        'Mode Changed',
                        `The mode has been changed to ${modeText} successfully.`,
                        'OK',
                        '',
                        'btn-success'
                    );
                    
                    // Update all WhatsApp menus for this conversation
                    updateWhatsAppModeInAllMenus(conversationId, newMode);
                } else {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Error changing mode');
                }
            } catch (error) {
                console.error('Error changing WhatsApp mode:', error);
                showGenericModal(
                    'Error',
                    `Could not change mode: ${error.message}`,
                    'OK',
                    '',
                    'btn-danger'
                );
            }
        }
    );
});

function updateWhatsAppModeInAllMenus(conversationId, newMode) {
    // Find all WhatsApp menu containers for this conversation
    const chatElements = document.querySelectorAll(`[data-conversation-id="${conversationId}"]`);
    
    
    chatElements.forEach(chatElement => {
        const whatsappContainer = chatElement.querySelector('.whatsapp-menu-container');
        if (whatsappContainer) {
            const allModeOptions = whatsappContainer.querySelectorAll('.whatsapp-mode-option');
            
            // More specific selection: find by icon
            let voiceModeLink = null;
            let textModeLink = null;
            
            allModeOptions.forEach(option => {
                const icon = option.querySelector('i');
                if (icon && icon.classList.contains('fa-microphone')) {
                    voiceModeLink = option;
                } else if (icon && icon.classList.contains('fa-keyboard')) {
                    textModeLink = option;
                }
            });
            
            if (voiceModeLink && textModeLink) {
                updateModeCheckmarks(voiceModeLink, textModeLink, newMode);
            } else {
            }
        } else {
        }
    });
}

// Initialize thinking tokens control
function initializeThinkingTokensControl() {
    const control = document.getElementById('thinking-tokens-control');
    if (!control) return;
    
    const icon = control.querySelector('.thinking-tokens-icon');
    const popup = document.getElementById('thinking-tokens-popup');
    const slider = document.getElementById('thinking-tokens-slider');
    const input = document.getElementById('thinking-tokens-input');
    const display = document.getElementById('thinking-tokens-display');
    const applyBtn = document.getElementById('thinking-tokens-apply');
    const presetBtns = document.querySelectorAll('.preset-btn');
    
    // Check if current model supports thinking tokens
    function updateThinkingTokensVisibility() {
        const currentModel = document.getElementById('chat-model')?.textContent || '';
        // Support Claude 3.5+ Sonnet, Claude 4 Opus, and any Claude 3.7 or 4.x models
        const modelLower = currentModel.toLowerCase();
        const isSupported = 
            modelLower.includes('claude-sonnet-4') ||     // Claude 3.5 Sonnet (4-20250514)
            modelLower.includes('claude-opus-4') ||       // Claude 4 Opus
            modelLower.includes('claude-3.7') ||          // Any Claude 3.7 model
            modelLower.includes('claude-3-7') ||          // Alternative format
            modelLower.includes('claude-4') ||            // Any Claude 4.x model
            (modelLower.includes('claude') && modelLower.includes('sonnet') && modelLower.includes('4')); // Catch variations

        if (isSupported) {
            control.style.display = 'flex';
        } else {
            control.style.display = 'none';
            currentThinkingBudget = 0;
        }
    }
    
    // Toggle popup
    icon.addEventListener('click', (e) => {
        e.stopPropagation();
        
        if (popup.style.display === 'none' || window.getComputedStyle(popup).display === 'none') {
            popup.style.display = 'block';
        } else {
            popup.style.display = 'none';
        }
        
    });
    
    // Close popup when clicking outside
    document.addEventListener('click', (e) => {
        if (!control.contains(e.target)) {
            popup.style.display = 'none';
        }
    });
    
    // Update value display
    function updateDisplay(value) {
        value = parseInt(value);
        if (value === 0) {
            display.textContent = 'Off';
            icon.classList.remove('active');
        } else {
            display.textContent = value.toLocaleString();
            icon.classList.add('active');
        }
        
        // Update active preset button
        presetBtns.forEach(btn => {
            if (parseInt(btn.dataset.value) === value) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }
    
    // Slider change
    slider.addEventListener('input', (e) => {
        const value = e.target.value;
        input.value = value;
        updateDisplay(value);
    });
    
    // Input change
    input.addEventListener('input', (e) => {
        let value = parseInt(e.target.value) || 0;
        value = Math.max(0, Math.min(128000, value));
        e.target.value = value;
        
        // Update slider if within range
        if (value <= 20000) {
            slider.value = value;
        }
        updateDisplay(value);
    });
    
    // Preset buttons
    presetBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const value = btn.dataset.value;
            input.value = value;
            if (value <= 20000) {
                slider.value = value;
            }
            updateDisplay(value);
        });
    });
    
    // Apply button
    applyBtn.addEventListener('click', (e) => {
        e.preventDefault();
        currentThinkingBudget = parseInt(input.value) || 0;
        popup.style.display = 'none';
        
        // Show confirmation
        const originalText = applyBtn.textContent;
        applyBtn.textContent = 'Applied!';
        setTimeout(() => {
            applyBtn.textContent = originalText;
        }, 1000);
    });
    
    // Listen for model changes
    const modelDropdown = document.getElementById('model-dropdown-content');
    if (modelDropdown) {
        modelDropdown.addEventListener('click', () => {
            setTimeout(updateThinkingTokensVisibility, 100);
        });
    }
    
    // Make function globally accessible
    window.updateThinkingTokensVisibility = updateThinkingTokensVisibility;

    // Initial check
    updateThinkingTokensVisibility();
}

// =============================================
// Web Search Toggle Control
// =============================================

function initWebSearchControl(conversationId, webSearchAllowedByPrompt = null) {
    const control = document.getElementById('web-search-control');
    const button = document.getElementById('web-search-button');

    if (!control || !button) return;

    if (!conversationId) {
        control.classList.remove('visible');
        return;
    }

    // Use passed data or fall back to default (true = allowed)
    webSearchAllowed = webSearchAllowedByPrompt !== null ? webSearchAllowedByPrompt : true;
    // Use global user preference from template
    webSearchEnabled = typeof webSearchUserEnabled !== 'undefined' ? webSearchUserEnabled : true;

    if (webSearchAllowed) {
        // Prompt allows web search - show the toggle
        control.classList.add('visible');
        updateWebSearchButtonState();
    } else {
        // Prompt disables web search - hide the toggle completely
        control.classList.remove('visible');
    }
}

function updateWebSearchButtonState() {
    const button = document.getElementById('web-search-button');
    if (!button) return;

    if (webSearchEnabled) {
        button.classList.add('active');
        button.title = 'Web Search: ON (click to disable)';
    } else {
        button.classList.remove('active');
        button.title = 'Web Search: OFF (click to enable)';
    }
}

async function toggleWebSearch() {
    if (!webSearchAllowed) return;

    try {
        const response = await secureFetch('/api/user/web-search-toggle', {
            method: 'POST'
        });
        if (!response.ok) return;
        const data = await response.json();
        webSearchEnabled = data.web_search_enabled;
        // Update global so it persists when switching conversations
        webSearchUserEnabled = webSearchEnabled;
        updateWebSearchButtonState();
    } catch (error) {
        console.error('Error toggling web search:', error);
    }
}

function initWebSearchEventListeners() {
    const button = document.getElementById('web-search-button');
    if (button) {
        button.addEventListener('click', toggleWebSearch);
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function() {
    initializeThinkingTokensControl();
    initWebSearchEventListeners();
    // Check again after a short delay to ensure model info is loaded
    setTimeout(() => {
        if (window.updateThinkingTokensVisibility) {
            window.updateThinkingTokensVisibility();
        }
    }, 500);
});

// Make loadMessages globally accessible for voice-call.js
window.loadMessages = loadMessages;

