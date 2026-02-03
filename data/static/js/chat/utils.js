/* utils.js */

const Config = {
    isAudioPlaying: false,
    currentAudio: null,
    currentAudioIcon: null,
    currentStopIcon: null,
    attachedFiles: [],
    have_vision: true,
    mediaRecorder: null,
    audioChunks: []
};

function startCountdown(targetElementId, durationInSeconds) {
    let endTime = localStorage.getItem("countdownEndTime");
    let now = new Date().getTime();

    if (!endTime || now > endTime) {
        endTime = now + durationInSeconds * 1000;
        localStorage.setItem("countdownEndTime", endTime);
    }

    let countdownFunction = setInterval(function() {
        now = new Date().getTime();
        let timeleft = endTime - now;
        
        let hours = Math.floor((timeleft % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
        let minutes = Math.floor((timeleft % (1000 * 60 * 60)) / (1000 * 60));
        let seconds = Math.floor((timeleft % (1000 * 60)) / 1000);
        
        document.getElementById(targetElementId).innerHTML = hours + "h " + minutes + "m " + seconds + "s ";
        
        if (timeleft < 0) {
            clearInterval(countdownFunction);
            document.getElementById(targetElementId).innerHTML = "EXPIRED";
            localStorage.removeItem("countdownEndTime");
        }
    }, 1000);
}


function escapeHTML(text) {
    var div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function encodeForHTML(str) {
    return str.replace(/[\u00A0-\u9999<>\&"']/gim, function(i) {
        return {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;',
        }[i] || '&#' + i.charCodeAt(0) + ';';
    });
}

function unescapeHTML(html) {
    const doc = new DOMParser().parseFromString(html, 'text/html');
    return doc.documentElement.textContent;
}

function unescapeCodeBlocks(html) {
	// Create temporary div
	var div = document.createElement('div');
	div.innerHTML = html;

	// Find all <code> and <code class="..."> elements
	div.querySelectorAll('code').forEach(function(element) {
		element.innerHTML = unescapeHTML(element.innerHTML);
	});

	return div.innerHTML;
}


function removeWaitingMessage() {
    var temporaryMessages = document.querySelectorAll('.temporary-message');
    temporaryMessages.forEach(function(message) {
        message.remove();
    });
}    

// Form submit handler moved to main.js to avoid conflicts
// document.getElementById('form-message').onsubmit = function(e) { ... };

function showGenericModal(title, message, confirmText, cancelText, confirmClass, onConfirm, hideOnConfirm = true) {
    const modalElement = document.getElementById('genericModal');
    const modal = new bootstrap.Modal(modalElement);
    const confirmBtn = document.getElementById('genericModalConfirmBtn');
    const cancelBtn = document.getElementById('genericModalCancelBtn');
    const modalLabel = document.getElementById('genericModalLabel');
    const modalBody = document.getElementById('genericModalBody');

    // Reset button state and content
    confirmBtn.style.display = ''; // Show confirm button
    confirmBtn.disabled = false; // Enable confirm button
    confirmBtn.className = `btn ${confirmClass}`; // Apply style class
    confirmBtn.textContent = confirmText; // Set confirm button text

    cancelBtn.style.display = ''; // Show cancel button
    cancelBtn.textContent = cancelText; // Set cancel button text

    modalLabel.textContent = title; // Set modal title
    modalBody.textContent = message; // Set modal message

    // Assign onclick event to confirm button
    confirmBtn.onclick = () => {
        if (hideOnConfirm) {
            modal.hide();
        }
        onConfirm(modal);
    };

    // Assign onclick event to cancel button
    cancelBtn.onclick = () => {
        modal.hide();
    };

    // Assign keyboard handlers
    modalElement.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            confirmBtn.click();
        } else if (event.key === 'Escape') {
            event.preventDefault();
            cancelBtn.click();
        }
    }, { once: true });

    modal.show();
}


function deleteConversation(conversationId) {
    showGenericModal(
        'Confirm Deletion',
        'Are you sure you want to delete this conversation?',
        'Delete',
        'Cancel',
        'btn-danger',
        withSession(() => {
            // Close dropdown menu before proceeding
            const conversationElement = document.querySelector(`[data-conversation-id="${conversationId}"]`);
            if (conversationElement) {
                const chatMenu = conversationElement.querySelector('.chat-menu');
                const bootstrapDropdown = bootstrap.Dropdown.getInstance(chatMenu.querySelector('.dropdown-toggle'));
                if (bootstrapDropdown) {
                    bootstrapDropdown.hide();
                }
            }

            // First, remove association with all external platforms
            removeAllExternalPlatformAssociations(conversationId)
                .then(() => {
                    // Then, delete conversation using secureFetch
                    return secureFetch(`/api/conversations/${conversationId}`, { 
                        method: 'DELETE'
                    });
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Could not delete the conversation');
                    }
                    return response.json();
                })
                .then(data => {
                    const conversationElement = document.querySelector(`[data-conversation-id="${conversationId}"]`);
                    const wasExternal = conversationElement && conversationElement.closest('#external-chats-container');
                    
                    removeConversationElement(conversationId);

                    if (wasExternal) {
                        updateExternalSection();
                    }

                    if (currentConversationId == conversationId) {
                        deactivateChat();
                        isCurrentConversationEmpty = false;
                    }
                })
                .catch(error => {
                    console.error('Error deleting the chat:', error);
                    NotificationModal.error('Delete Error', 'An error occurred while deleting the chat. Please try again.');
                });
        })
    );
}

function toggleLockConversation(conversationId, lock) {
    const action = lock ? 'lock' : 'unlock';
    const actionCapitalized = lock ? 'Lock' : 'Unlock';

    showGenericModal(
        `Confirm ${actionCapitalized}`,
        `Are you sure you want to ${action} this conversation?`,
        actionCapitalized,
        'Cancel',
        lock ? 'btn-warning' : 'btn-success',
        withSession(() => {
            secureFetch(`/api/conversations/${conversationId}/lock`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ lock: lock })
            })
            .then(response => {
                if (!response.ok) throw new Error(`Could not ${action} the conversation`);
                return response.json();
            })
            .then(data => {
                // Update UI
                const conversationElement = document.querySelector(`[data-conversation-id="${conversationId}"]`);
                if (conversationElement) {
                    conversationElement.dataset.locked = lock ? 'true' : 'false';
                    if (lock) {
                        conversationElement.classList.add('conversation-locked');
                    } else {
                        conversationElement.classList.remove('conversation-locked');
                    }

                    // Update icon in sidebar
                    const nameSpan = conversationElement.querySelector('.chat-name');
                    if (nameSpan) {
                        const chatText = nameSpan.textContent.replace(/^\s*/, '');
                        if (lock) {
                            nameSpan.innerHTML = `<i class="fas fa-comment-slash" title="This conversation is locked"></i> ${chatText}`;
                        } else {
                            // Remove lock icon
                            const lockIcon = nameSpan.querySelector('.fa-comment-slash');
                            if (lockIcon) lockIcon.remove();
                        }
                    }
                }

                // Update current conversation state if this is the active one
                if (currentConversationId == conversationId) {
                    isCurrentConversationLocked = lock;
                    const lockedBanner = document.getElementById('locked-conversation-banner');
                    const messageText = document.getElementById('message-text');

                    if (lock) {
                        if (lockedBanner) lockedBanner.style.display = 'flex';
                        if (messageText) {
                            messageText.placeholder = 'This conversation is locked';
                            messageText.disabled = true;
                        }
                        document.querySelector('#form-message button[type="submit"]').disabled = true;
                    } else {
                        if (lockedBanner) lockedBanner.style.display = 'none';
                        if (messageText) {
                            messageText.placeholder = 'Type a message...';
                            messageText.disabled = false;
                        }
                        document.querySelector('#form-message button[type="submit"]').disabled = false;
                    }
                }

                NotificationModal.success(`Conversation ${actionCapitalized}ed`, `The conversation has been ${action}ed successfully.`);
            })
            .catch(error => {
                console.error(`Error ${action}ing the chat:`, error);
                NotificationModal.error(`${actionCapitalized} Error`, `An error occurred while ${action}ing the chat. Please try again.`);
            });
        })
    );
}

function removeAllExternalPlatformAssociations(conversationId) {
    return secureFetch(`/api/conversations/${conversationId}/external-platform`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
            action: 'remove',
            platform: 'all'
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to remove external platform associations');
        }
        return response.json();
    });
}

function updateExternalSection() {
    const externalChatsContainer = document.querySelector('#external-chats-container');
    const externalSection = document.querySelector('.external-section');
    
    if (externalChatsContainer.children.length === 0) {
        externalSection.style.display = 'none';
    } else {
        externalSection.style.display = 'block';
    }
}

function downloadPDF(conversationId) {
    showGenericModal(
        'Download PDF',
        'Do you want to download this conversation as PDF?',
        'Download',
        'Cancel',
        'btn-primary',
        withSession((modal) => {
            const confirmBtn = document.getElementById('genericModalConfirmBtn');
            const cancelBtn = document.getElementById('genericModalCancelBtn');
            const modalLabel = document.getElementById('genericModalLabel');
            const modalBody = document.getElementById('genericModalBody');

            // Disable confirm button to prevent multiple clicks
            confirmBtn.disabled = true;
            confirmBtn.textContent = 'Processing...';

            // Make AJAX request to endpoint
            secureFetch(`/download-pdf/${conversationId}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Request error');
                    }
                    return response.json();
                })
                .then(data => {
                    // Update modal content with response message
                    modalLabel.textContent = 'PDF Generation Started';
                    modalBody.textContent = data.message;
                    confirmBtn.style.display = 'none'; // Hide confirm button
                    cancelBtn.textContent = 'Close';
                })
                .catch(error => {
                    console.error('Error:', error);
                    modalLabel.textContent = 'Error';
                    modalBody.textContent = 'An error occurred while starting PDF generation.';
                    confirmBtn.style.display = 'none'; // Hide confirm button
                    cancelBtn.textContent = 'Close';
                })
                .finally(() => {
                    // Re-enable confirm button in case of retry
                    confirmBtn.disabled = false;
                    confirmBtn.textContent = 'Download';
                });
        }),
        false // Don't hide modal when clicking "Confirm"
    );
}



function downloadAudio(conversationId) {
    showGenericModal(
        'Download MP3',
        'Do you want to download this conversation as MP3?',
        'Download',
        'Cancel',
        'btn-primary',
        withSession((modal) => {
            const confirmBtn = document.getElementById('genericModalConfirmBtn');
            const cancelBtn = document.getElementById('genericModalCancelBtn');
            const modalLabel = document.getElementById('genericModalLabel');
            const modalBody = document.getElementById('genericModalBody');

            // Disable confirm button to prevent multiple clicks
            confirmBtn.disabled = true;
            confirmBtn.textContent = 'Processing...';

            // Make AJAX request to endpoint
            secureFetch(`/download-mp3/${conversationId}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Request error');
                    }
                    return response.json();
                })
                .then(data => {
                    // Update modal content with response message
                    modalLabel.textContent = 'MP3 Generation Started';
                    modalBody.textContent = data.message;
                    confirmBtn.style.display = 'none'; // Hide confirm button
                    cancelBtn.textContent = 'Close';
                })
                .catch(error => {
                    console.error('Error:', error);
                    modalLabel.textContent = 'Error';
                    modalBody.textContent = 'An error occurred while starting MP3 generation.';
                    confirmBtn.style.display = 'none'; // Hide confirm button
                    cancelBtn.textContent = 'Close';
                })
                .finally(() => {
                    // Re-enable confirm button in case of retry
                    confirmBtn.disabled = false;
                    confirmBtn.textContent = 'Download';
                });
        }),
        false // Don't hide modal when clicking "Confirm"
    );
}

function serveMp3(conversationId) {
    secureFetch(`/serve-mp3/${conversationId}`, {
        method: 'GET'
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('MP3 not available');
            }
            return response.blob();
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `conversation_${conversationId}.mp3`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
        })
        .catch(error => {
            console.error('Error:', error);
            NotificationModal.error('MP3 Unavailable', 'The MP3 file is not available yet or an error occurred.');
        });
}


function toggleSendButton(state) {
    const sendButton = document.getElementById('send-button');
    if (state === 'Stop') {
        sendButton.innerText = 'Stop';
        sendButton.onclick = stopReceivingStream;
    } else {
        sendButton.innerText = 'Send';
        sendButton.onclick = handleSendButtonClick;
    }
}

function handleSendButtonClick(event) {
    event.preventDefault();
    const messageText = document.getElementById('message-text').value;
    
    // Check session before sending message (force check for critical action)
    SessionManager.validateSession(true).then((isValid) => {
        if (isValid) {
            // Session is valid, send message and clear form
            sendMessage(messageText);
            document.getElementById('message-text').value = '';
            Config.attachedFiles = [];
        } else {
            // Session invalid, modal already shown by validateSession
            // Don't clear the form so user can copy their message
        }
    });
}

function addLoadingIndicator() {
    var chatWindow = document.getElementById('chat-window');
    var loadingIndicator = document.createElement('div');
    loadingIndicator.classList.add('loading-indicator', 'temporary-message');
    loadingIndicator.innerHTML = `
        <div class="spinner-border text-primary" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
    `;
    chatWindow.appendChild(loadingIndicator);

    // Ensure loading indicator is completely visible
    loadingIndicator.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function removeLoadingIndicator() {
    var chatWindow = document.getElementById('chat-window');
    var loadingIndicator = chatWindow.querySelector('.loading-indicator');
    if (loadingIndicator) {
        chatWindow.removeChild(loadingIndicator);
    }
}

function showInsufficientBalancePopup(failedAction) {
    document.getElementById('failedAction').textContent = failedAction;

    $('#insufficientBalancePopup').modal({
      keyboard: false, 
      backdrop: 'static'
    });
}

function closePopup() {
    var myModalEl = document.getElementById('insufficientBalancePopup');
    var modal = bootstrap.Modal.getInstance(myModalEl); 
    modal.hide();
}

    //console.log("Initializing imageHandler");
    const imageHandler = {
        images: [],
        init: function(images) {
            //console.log("Initializing with images:", images);
            this.images = images;
            this.setupEventListeners();
            this.initializeImages();
        },
        setupEventListeners: function() {
            //console.log("Setting up event listeners");
            const fullsizeContainer = document.getElementById('fullsizeContainer');
            if (fullsizeContainer) {
                fullsizeContainer.addEventListener('click', (event) => {
                    if (event.target === fullsizeContainer) {
                        this.closeFullsize();
                    }
                });
            } else {
                console.error("Element 'fullsizeContainer' not found");
            }
            document.addEventListener('keydown', this.handleEscapeKey.bind(this));
        },
        initializeImages: function() {
            //console.log("Initializing images");
            const images = document.querySelectorAll('#chat-messages-container .message-content img');
            images.forEach(img => {
                if (!img.dataset.initialized) {
                    img.dataset.fullsize = img.src.replace('_256.webp', '_fullsize.webp');
                    img.onclick = () => this.showFullsize(img.dataset.fullsize, img.dataset.messageId);
                    img.dataset.initialized = 'true';
                }
            });
        },
		
		showFullsize: function(url, messageId) {
			const fullsizeContainer = document.getElementById('fullsizeContainer');
			const fullsizeImage = document.getElementById('fullsizeImage');
			const downloadButton = document.getElementById('downloadButton');
			const deleteButton = document.getElementById('deleteButton');

			if (!fullsizeContainer || !fullsizeImage || !downloadButton || !deleteButton) {
				console.error("One or more required elements not found");
				return;
			}

			// Modify URL to get fullsize version
			const fullsizeUrl = url.replace('_256.webp', '_fullsize.webp');

			// Hide current image
			fullsizeImage.style.display = 'none';

			// Show loading indicator
			const loadingIndicator = document.createElement('div');
			loadingIndicator.className = 'loading-indicator';
			loadingIndicator.innerHTML = '<div class="spinner-border text-light" role="status"><span class="visually-hidden">Loading...</span></div>';
			fullsizeContainer.appendChild(loadingIndicator);

			// Show container
			fullsizeContainer.style.display = 'block';

			// Change image source
			fullsizeImage.src = fullsizeUrl;

			// Wait for image to fully load
			fullsizeImage.onload = function() {
				// Hide loading indicator
				fullsizeContainer.removeChild(loadingIndicator);

				// Show new image
				fullsizeImage.style.display = 'block';
			};

			fullsizeImage.onerror = function() {
				console.error("Error loading image");
				fullsizeContainer.removeChild(loadingIndicator);
				fullsizeImage.style.display = 'block'; // Show error image if exists
			};

			// Hide or show download/delete buttons as appropriate
			// Only show for message images (with messageId), not for avatar/profile images
			if (!messageId) {
				downloadButton.style.display = 'none';
				deleteButton.style.display = 'none';
			} else {
				downloadButton.style.display = '';
				deleteButton.style.display = '';
			}

			downloadButton.onclick = (e) => {
				e.preventDefault();
				this.downloadImage(fullsizeUrl);
			};

			deleteButton.onclick = () => {
				if (messageId) {
					this.deleteImage(messageId);
				} else {
					NotificationModal.warning('Cannot Delete', 'This image cannot be deleted.');
				}
			};
		},

		
        closeFullsize: function() {
            //console.log("Closing fullsize image");
            const fullsizeContainer = document.getElementById('fullsizeContainer');
            if (fullsizeContainer) {
                fullsizeContainer.style.display = 'none';
            } else {
                console.error("Element 'fullsizeContainer' not found");
            }
        },
        handleEscapeKey: function(event) {
            if (event.key === 'Escape') {
                this.closeFullsize();
            }
        },
        downloadImage: function(url) {
            const link = document.createElement('a');
            link.href = url;
            const randomName = Math.random().toString(36).substring(2, 8);
            link.download = `${randomName}.png`;
            link.target = '_blank';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        },

		deleteImage: function(messageId) {
			showGenericModal(
				'Confirm Deletion',
				'Are you sure you want to delete this image?',
				'Delete',
				'Cancel',
				'btn-danger',
				() => {
					fetch(`/api/delete-image/${messageId}`, {
						method: 'DELETE',
					})
					.then(response => response.json())
					.then(data => {
						if (data.success) {
							this.closeFullsize();
							this.onImagesDeleted([messageId]);
							this.images = this.images.filter(img => img.id !== messageId);
							
							// Find image element
							const imgElement = document.querySelector(`img[data-message-id="${messageId}"]`);
							if (imgElement) {
								const messageElement = imgElement.closest('.message');
								if (messageElement) {
									// Determine if it's a user or bot message
									const isUserMessage = messageElement.classList.contains('user');
									
									// Create correct message structure
									const messageContentContainer = document.createElement('div');
									messageContentContainer.classList.add('message-content-container');
									
									const messageContent = document.createElement('div');
									messageContent.classList.add('message-content');
									
									const deletedText = document.createElement('p');
									deletedText.textContent = '[image deleted]';
									
									// Build message structure
									messageContent.appendChild(deletedText);
									messageContentContainer.appendChild(messageContent);
									
									// Replace message content
									messageElement.innerHTML = '';
									messageElement.appendChild(messageContentContainer);
								}
							}
						} else {
							console.error('Error deleting image:', data.error);
						}
					})
					.catch(error => console.error('Error:', error));
				}
			);
		},
		

        onImagesDeleted: function(deletedIds) {
            // Images deleted callback
        }
    };
    
    document.addEventListener('DOMContentLoaded', function() {
        imageHandler.init([]);
    });
    
    // Function to initialize newly added images
    function initializeNewImages(container) {
        const images = container.querySelectorAll('.message-content img:not([data-initialized])');
        images.forEach(img => {
            img.dataset.fullsize = img.src.replace('_256.webp', '_fullsize.webp');
            img.onclick = () => imageHandler.showFullsize(img.dataset.fullsize, img.dataset.messageId);
            img.dataset.initialized = 'true';
        });
    }
    
    // Mutation observer to detect new images added to DOM
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'childList') {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        initializeNewImages(node);
                    }
                });
            }
        });
    });
    
    // Configure and start observer
    const config = { childList: true, subtree: true };
    const targetNode = document.getElementById('chat-messages-container');
    observer.observe(targetNode, config);

    
