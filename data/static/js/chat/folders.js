/* folders.js - Chat Folders Management */

let chatFolders = [];
let currentEditingFolderId = null;
let currentMovingConversationId = null;
let currentSelectedFolderId = null; // Track currently selected folder for new chat creation

// Initialize folders functionality
async function initializeFolders() {
    await loadChatFolders();
    setupFoldersEventListeners();
    setupDragAndDrop();
}

// Setup event listeners for folders
function setupFoldersEventListeners() {
    // Add folder button
    document.getElementById('add-folder-btn')?.addEventListener('click', showAddFolderModal);
    
    // Save folder button
    document.getElementById('saveFolderBtn')?.addEventListener('click', saveFolderHandler);
    
    // Color presets
    document.querySelectorAll('.color-preset').forEach(preset => {
        preset.addEventListener('click', (e) => {
            const color = e.target.dataset.color;
            document.getElementById('folderColor').value = color;
        });
    });
    
    // Remove from folder button
    document.getElementById('removeFromFolderBtn')?.addEventListener('click', removeFromFolderHandler);
}

// Show add folder modal
function showAddFolderModal() {
    currentEditingFolderId = null;
    document.getElementById('folderModalLabel').textContent = 'Add Folder';
    document.getElementById('folderName').value = '';
    document.getElementById('folderColor').value = '#3B82F6';
    
    const modal = new bootstrap.Modal(document.getElementById('folderModal'));
    modal.show();
}

// Show edit folder modal
function showEditFolderModal(folderId) {
    const folder = chatFolders.find(f => f.id === folderId);
    if (!folder) return;
    
    currentEditingFolderId = folderId;
    document.getElementById('folderModalLabel').textContent = 'Edit Folder';
    document.getElementById('folderName').value = folder.name;
    document.getElementById('folderColor').value = folder.color;
    
    const modal = new bootstrap.Modal(document.getElementById('folderModal'));
    modal.show();
}

// Save folder handler
const saveFolderHandler = withSession(async function() {
    const name = document.getElementById('folderName').value.trim();
    const color = document.getElementById('folderColor').value;
    
    if (!name) {
        NotificationModal.toast('Folder name is required', 'error');
        return;
    }
    
    try {
        let response;
        if (currentEditingFolderId) {
            // Edit existing folder
            response = await secureFetch(`/api/chat-folders/${currentEditingFolderId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ name, color })
            });
        } else {
            // Create new folder
            response = await secureFetch('/api/chat-folders', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ name, color })
            });
        }
        
        const result = await response.json();
        
        if (response.ok) {
            NotificationModal.toast(result.message || 'Folder saved successfully', 'success');
            await loadChatFolders();
            
            // Update existing chat menus
            updateAllChatMenus();
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('folderModal'));
            modal.hide();
        } else {
            NotificationModal.toast(result.error || 'Failed to save folder', 'error');
        }
    } catch (error) {
        console.error('Error saving folder:', error);
        NotificationModal.toast('Failed to save folder', 'error');
    }
});

// Load chat folders
async function loadChatFolders() {
    try {
        // Use embedded data if available (avoids HTTP request on initial load)
        if (typeof embeddedChatFolders !== 'undefined' && embeddedChatFolders !== null) {
            chatFolders = embeddedChatFolders;
            embeddedChatFolders = null; // Clear to avoid reuse on subsequent calls
            renderChatFolders();
            return true;
        }

        // Fallback to API fetch
        const response = await fetch('/api/chat-folders');
        const result = await response.json();

        if (response.ok) {
            chatFolders = result.folders || [];
            renderChatFolders();
            return true;
        } else {
            console.error('Failed to load folders:', result.error);
            return false;
        }
    } catch (error) {
        console.error('Error loading folders:', error);
        return false;
    }
}

// Render chat folders in sidebar
function renderChatFolders() {
    const container = document.getElementById('chat-folders-container');
    if (!container) return;
    
    // Clear but preserve the 'New Folder' button
    const newFolderBtn = container.querySelector('.new-folder-item');
    container.innerHTML = '';
    
    // Re-add the 'New Folder' button first
    if (!newFolderBtn) {
        const newFolderElement = document.createElement('div');
        newFolderElement.className = 'new-folder-item';
        newFolderElement.id = 'add-folder-btn';
        newFolderElement.title = 'New Folder';
        newFolderElement.innerHTML = `
            <i class="fas fa-folder-plus"></i>
            <span>New Folder</span>
        `;
        newFolderElement.addEventListener('click', showAddFolderModal);
        container.appendChild(newFolderElement);
    } else {
        container.appendChild(newFolderBtn);
    }
    
    // Add all folders
    chatFolders.forEach(folder => {
        const folderElement = createFolderElement(folder);
        container.appendChild(folderElement);
    });
}

// Create folder element
function createFolderElement(folder) {
    const folderDiv = document.createElement('div');
    folderDiv.className = 'folder-item d-flex justify-content-between align-items-center mb-1 p-2 rounded';
    folderDiv.style.cursor = 'pointer';
    folderDiv.dataset.folderId = folder.id;
    folderDiv.dataset.folderExpanded = 'false';
    
    // Folder content (left side)
    const folderContent = document.createElement('div');
    folderContent.className = 'd-flex align-items-center flex-grow-1';
    
    // Get folder icon based on state
    const iconClass = folderDiv.dataset.folderExpanded === 'true' ? 'fa-folder-open' : 'fa-folder';

    // Chat count for tooltip
    const chatCount = folder.conversation_count || 0;
    const chatCountText = chatCount === 1 ? '1 chat' : `${chatCount} chats`;

    folderContent.innerHTML = `
        <i class="fas ${iconClass} folder-icon me-2" style="color: ${folder.color}; font-size: 16px;"></i>
        <span class="folder-name">${escapeHTML(folder.name)}</span>
    `;

    // Set tooltip with chat count
    folderDiv.title = chatCountText;
    
    // Actions dropdown (right side) - now hidden by default, shown on hover
    const actionsDiv = document.createElement('div');
    actionsDiv.className = 'folder-actions dropdown';
    actionsDiv.innerHTML = `
        <button class="btn btn-sm btn-ghost dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">
            <i class="fas fa-ellipsis-v"></i>
        </button>
        <ul class="dropdown-menu dropdown-menu-dark">
            <li><a class="dropdown-item" href="#" onclick="showEditFolderModal(${folder.id})">
                <i class="fas fa-edit"></i> Edit
            </a></li>
            <li><a class="dropdown-item text-danger" href="#" onclick="deleteFolderHandler(${folder.id})">
                <i class="fas fa-trash"></i> Delete
            </a></li>
        </ul>
    `;
    
    // Click handler for folder (expand/collapse chats in folder)
    folderContent.addEventListener('click', () => toggleFolderChats(folder.id));
    
    folderDiv.appendChild(folderContent);
    folderDiv.appendChild(actionsDiv);
    
    // Add container for folder's chats (initially hidden)
    const folderChatsContainer = document.createElement('div');
    folderChatsContainer.className = 'folder-chats-container mt-2';
    folderChatsContainer.id = `folder-chats-${folder.id}`;
    folderChatsContainer.style.display = 'none';
    
    const folderWrapper = document.createElement('div');
    folderWrapper.appendChild(folderDiv);
    folderWrapper.appendChild(folderChatsContainer);
    
    return folderWrapper;
}

// Toggle folder chats visibility
function toggleFolderChats(folderId) {
    const container = document.getElementById(`folder-chats-${folderId}`);
    const folderItem = document.querySelector(`[data-folder-id="${folderId}"]`);
    if (!container || !folderItem) return;
    
    // Update selected folder tracking
    const wasExpanded = container.style.display !== 'none';
    
    // Collapse all other folders first
    document.querySelectorAll('.folder-chats-container').forEach(folderContainer => {
        folderContainer.style.display = 'none';
    });
    
    // Remove active styling from all folders and update icons
    document.querySelectorAll('.folder-item').forEach(item => {
        item.classList.remove('folder-selected');
        item.dataset.folderExpanded = 'false';
        const icon = item.querySelector('.folder-icon');
        if (icon) {
            icon.classList.remove('fa-folder-open');
            icon.classList.add('fa-folder');
        }
    });
    
    if (wasExpanded) {
        // If this folder was expanded, collapse it and deselect
        container.style.display = 'none';
        currentSelectedFolderId = null;
        folderItem.dataset.folderExpanded = 'false';
        const icon = folderItem.querySelector('.folder-icon');
        if (icon) {
            icon.classList.remove('fa-folder-open');
            icon.classList.add('fa-folder');
        }
        updateNewChatButtonState();
    } else {
        // Expand this folder and select it
        container.style.display = 'block';
        currentSelectedFolderId = folderId;
        
        // Add active styling to selected folder and update icon
        folderItem.classList.add('folder-selected');
        folderItem.dataset.folderExpanded = 'true';
        const icon = folderItem.querySelector('.folder-icon');
        if (icon) {
            icon.classList.remove('fa-folder');
            icon.classList.add('fa-folder-open');
        }
        
        loadFolderChats(folderId);
        updateNewChatButtonState();
    }
}

// Update conversation count for a specific folder without reloading everything
async function updateFolderConversationCount(folderId) {
    try {
        const response = await secureFetch(`/api/conversations?user_id=1&folder_id=${folderId}`);
        if (!response) {
            // secureFetch returned null (session expired, modal already shown)
            return;
        }
        const folderConversations = await response.json();
        
        // Find the folder element and update its count
        const folderElement = document.querySelector(`[data-folder-id="${folderId}"]`);
        if (folderElement) {
            const countElement = folderElement.querySelector('.conversation-count');
            if (countElement) {
                countElement.textContent = `(${folderConversations.length})`;
            }
        }
    } catch (error) {
        if (error.message === 'Session expired') {
            // Session validation was already handled by secureFetch, no need to log as error
        } else {
            console.error('Error updating folder conversation count:', error);
        }
    }
}

// Load chats for a specific folder
async function loadFolderChats(folderId, onComplete = null) {
    try {
        const response = await secureFetch(`/api/conversations?user_id=${user_id}&folder_id=${folderId}`);
        if (!response) {
            // secureFetch returned null (session expired, modal already shown)
            return;
        }
        const conversations = await response.json();
        
        const container = document.getElementById(`folder-chats-${folderId}`);
        if (!container) return;
        
        container.innerHTML = '';
        
        conversations.forEach(conversation => {
            const chatElement = createFolderChatElement(conversation);
            container.appendChild(chatElement);
        });
        
        // Execute callback if provided (after DOM is updated)
        if (onComplete && typeof onComplete === 'function') {
            onComplete();
        }
    } catch (error) {
        if (error.message === 'Session expired') {
            // Session validation was already handled by secureFetch, no need to log as error
        } else {
            console.error('Error loading folder chats:', error);
        }
    }
}

// Create chat element for folder
function createFolderChatElement(conversation) {
    const chatElement = document.createElement('a');
    chatElement.href = '#';
    chatElement.className = 'list-group-item list-group-item-action folder-chat-item';
    chatElement.dataset.conversationId = conversation.id;
    
    const chatName = conversation.chat_name || `Chat ${conversation.id}`;
    
    // Create the chat content
    const chatContent = document.createElement('div');
    chatContent.className = 'chat-content-container d-flex justify-content-between align-items-center';
    
    const chatInfo = document.createElement('div');
    chatInfo.className = 'chat-content';
    chatInfo.innerHTML = `
        <div class="chat-name">${escapeHTML(chatName)}</div>
        <small class="text-muted">${new Date(conversation.start_date).toLocaleDateString()}</small>
    `;
    
    // Create the complete chat menu (same as main chats)
    const chatMenu = createChatMenuForFolder(conversation);
    
    chatContent.appendChild(chatInfo);
    chatContent.appendChild(chatMenu);
    chatElement.appendChild(chatContent);
    
    // Click handler to continue conversation
    chatElement.addEventListener('click', (e) => {
        if (!e.target.closest('.chat-menu')) {
            // Remove active-chat class from ALL chats everywhere
            document.querySelectorAll('.active-chat').forEach(el => {
                el.classList.remove('active-chat');
            });
            
            // Add active-chat class to this element
            chatElement.classList.add('active-chat');
            
            // Update selectedChat global variable so active chat detection works
            window.selectedChat = chatElement;
            
            continueConversation(conversation.id, chatName);
        }
    });
    
    return chatElement;
}

// Create chat menu for folder chats (similar to main chat menu)
function createChatMenuForFolder(conversation) {
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

    // Download MP3 option
    const downloadAudioLink = createMenuLink('fa-music', 'Download MP3', () => downloadAudio(conversation.id));
    chatMenuContent.appendChild(downloadAudioLink);

    // Download PDF option
    const downloadPdfLink = createMenuLink('fa-download', 'Download PDF', () => downloadPDF(conversation.id));
    chatMenuContent.appendChild(downloadPdfLink);

    // Delete option
    const deleteLink = createMenuLink('fa-trash-alt', 'Delete', () => deleteConversation(conversation.id), 'text-danger');
    chatMenuContent.appendChild(deleteLink);

    // Add separator
    const separator = document.createElement('div');
    separator.classList.add('menu-separator');
    chatMenuContent.appendChild(separator);

    // WhatsApp option
    const whatsappLink = createPlatformLink('whatsapp', conversation);
    chatMenuContent.appendChild(whatsappLink);
    
    // Add folder options section
    addFolderOptionsToMenu(chatMenuContent, conversation.id);

    let isMenuOpen = false;

    // Click handler for chat-menu-content div
    chatMenu.addEventListener('click', (e) => {
        e.stopPropagation();
        isMenuOpen = !isMenuOpen;
        chatMenuContent.style.display = isMenuOpen ? 'block' : 'none';
    });

    // Handler to close menu when clicking outside
    document.addEventListener('click', (e) => {
        if (!chatMenu.contains(e.target) && isMenuOpen) {
            isMenuOpen = false;
            chatMenuContent.style.display = 'none';
        }
    });

    return chatMenu;
}

// Delete folder handler
const deleteFolderHandler = withSession(function(folderId) {
    const folder = chatFolders.find(f => f.id === folderId);
    if (!folder) return;

    const confirmMessage = `Are you sure you want to delete the folder "${folder.name}"? All chats in this folder will be moved to the main chat list.`;

    NotificationModal.confirm('Delete Folder', confirmMessage, async () => {
        try {
            const response = await secureFetch(`/api/chat-folders/${folderId}`, {
                method: 'DELETE'
            });

            const result = await response.json();

            if (response.ok) {
                NotificationModal.toast(result.message || 'Folder deleted successfully', 'success');
                loadChatFolders();
                loadConversations(false, false);
            } else {
                NotificationModal.toast(result.error || 'Failed to delete folder', 'error');
            }
        } catch (error) {
            console.error('Error deleting folder:', error);
            NotificationModal.toast('Failed to delete folder', 'error');
        }
    }, null, { type: 'error', confirmText: 'Delete' });
});

// Show move chat modal
function showMoveChatModal(conversationId) {
    currentMovingConversationId = conversationId;
    
    // Populate folders list
    const foldersList = document.getElementById('foldersList');
    if (!foldersList) return;
    
    foldersList.innerHTML = '';
    
    chatFolders.forEach(folder => {
        const folderOption = document.createElement('button');
        folderOption.className = 'list-group-item list-group-item-action d-flex align-items-center';
        folderOption.innerHTML = `
            <div class="folder-color-indicator me-2" style="background-color: ${folder.color}; width: 12px; height: 12px; border-radius: 50%;"></div>
            <span>${escapeHTML(folder.name)}</span>
        `;
        
        folderOption.addEventListener('click', () => moveChatToFolder(conversationId, folder.id));
        foldersList.appendChild(folderOption);
    });
    
    const modal = new bootstrap.Modal(document.getElementById('moveChatModal'));
    modal.show();
}

// Move chat to folder
const moveChatToFolder = withSession(async function(conversationId, folderId) {
    try {
        const response = await secureFetch(`/api/conversations/${conversationId}/move-to-folder`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ folder_id: folderId })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            NotificationModal.toast(result.message || 'Chat moved successfully', 'success');
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('moveChatModal'));
            modal.hide();
            
            // Refresh UI
            loadChatFolders();
            loadConversations(false, false);
        } else {
            NotificationModal.toast(result.error || 'Failed to move chat', 'error');
        }
    } catch (error) {
        console.error('Error moving chat:', error);
        NotificationModal.toast('Failed to move chat', 'error');
    }
});

// Remove from folder handler
function removeFromFolderHandler() {
    if (!currentMovingConversationId) return;
    moveChatToFolder(currentMovingConversationId, null);
}

// Setup drag and drop functionality
function setupDragAndDrop() {
    // Make chat items draggable
    makeChatItemsDraggable();
    
    // Make folders drop targets
    makeFoldersDroppable();
}

// Make chat items draggable
function makeChatItemsDraggable() {
    // Function to add drag attributes to existing chat items
    const addDragToChats = () => {
        const chatItems = document.querySelectorAll('[data-conversation-id]:not(.folder-chat-item)');
        chatItems.forEach(chatItem => {
            if (!chatItem.hasAttribute('draggable')) {
                chatItem.setAttribute('draggable', 'true');
                chatItem.addEventListener('dragstart', handleDragStart);
                chatItem.addEventListener('dragend', handleDragEnd);
                
                // Set draggable cursor but don't add visual drag handle
                chatItem.style.cursor = 'grab';
            }
        });
        
        // Enhance existing chat menus with folder options
        chatItems.forEach(chatItem => {
            if (!chatItem.hasAttribute('data-folder-menu-enhanced')) {
                chatItem.setAttribute('data-folder-menu-enhanced', 'true');
                enhanceExistingChatMenu(chatItem);
            }
        });
    };
    
    // Initial setup
    addDragToChats();
    
    // Setup observer for new chat items
    const observer = new MutationObserver((mutations) => {
        let shouldUpdate = false;
        mutations.forEach(mutation => {
            if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                shouldUpdate = true;
            }
        });
        
        if (shouldUpdate) {
            // Wait a bit for the original menu to be created
            setTimeout(() => {
                addDragToChats();
            }, 200);
        }
    });
    
    const chatContainer = document.getElementById('dynamic-chats-container');
    if (chatContainer) {
        observer.observe(chatContainer, { childList: true, subtree: true });
    }
}

// Make folders drop targets
function makeFoldersDroppable() {
    // Setup drop zones when folders are rendered
    const setupDropZones = () => {
        const folderItems = document.querySelectorAll('.folder-item');
        folderItems.forEach(folderItem => {
            folderItem.addEventListener('dragover', handleDragOver);
            folderItem.addEventListener('drop', handleDrop);
            folderItem.addEventListener('dragenter', handleDragEnter);
            folderItem.addEventListener('dragleave', handleDragLeave);
        });
        
        // Also make the main chats area a drop zone (to remove from folders)
        const mainChatsArea = document.querySelector('.sidebar-section:not(.folders-section) h3');
        if (mainChatsArea) {
            const dropZone = document.createElement('div');
            dropZone.className = 'main-chats-drop-zone';
            dropZone.textContent = 'Drop here to remove from folder';
            dropZone.style.cssText = 'display: none; padding: 10px; border: 2px dashed var(--sidebar-border-color); border-radius: 6px; margin: 5px 0; text-align: center; font-size: 0.8rem; color: var(--sidebar-text-color); opacity: 0.7;';
            
            mainChatsArea.parentNode.insertBefore(dropZone, mainChatsArea.nextSibling);
            
            dropZone.addEventListener('dragover', handleDragOver);
            dropZone.addEventListener('drop', (e) => handleDrop(e, null)); // null means remove from folder
            dropZone.addEventListener('dragenter', (e) => {
                e.preventDefault();
                dropZone.style.display = 'block';
                dropZone.style.backgroundColor = 'rgba(114, 137, 218, 0.1)';
            });
            dropZone.addEventListener('dragleave', (e) => {
                if (!dropZone.contains(e.relatedTarget)) {
                    dropZone.style.backgroundColor = '';
                }
            });
        }
    };
    
    // Setup initially and after folder updates
    setupDropZones();
    
    // Re-setup when folders are updated
    const originalRenderChatFolders = renderChatFolders;
    renderChatFolders = function() {
        originalRenderChatFolders.call(this);
        setupDropZones();
    };
}

// Drag and drop event handlers
let draggedConversationId = null;

function handleDragStart(e) {
    draggedConversationId = e.target.dataset.conversationId;
    e.target.style.opacity = '0.5';
    e.target.style.cursor = 'grabbing';
    
    // Show drop zones
    document.querySelectorAll('.folder-item').forEach(folder => {
        folder.classList.add('drop-zone-active');
    });
    
    const dropZone = document.querySelector('.main-chats-drop-zone');
    if (dropZone) {
        dropZone.style.display = 'block';
    }
    
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/html', e.target.outerHTML);
}

function handleDragEnd(e) {
    e.target.style.opacity = '';
    e.target.style.cursor = 'grab';
    draggedConversationId = null;
    
    // Hide drop zones
    document.querySelectorAll('.folder-item').forEach(folder => {
        folder.classList.remove('drop-zone-active', 'drag-over');
    });
    
    const dropZone = document.querySelector('.main-chats-drop-zone');
    if (dropZone) {
        dropZone.style.display = 'none';
        dropZone.style.backgroundColor = '';
    }
}

function handleDragOver(e) {
    if (draggedConversationId) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
    }
}

function handleDragEnter(e) {
    if (draggedConversationId) {
        e.preventDefault();
        e.target.closest('.folder-item')?.classList.add('drag-over');
    }
}

function handleDragLeave(e) {
    const folderItem = e.target.closest('.folder-item');
    if (folderItem && !folderItem.contains(e.relatedTarget)) {
        folderItem.classList.remove('drag-over');
    }
}

function handleDrop(e, targetFolderId = null) {
    e.preventDefault();
    
    if (!draggedConversationId) return;
    
    // Get target folder ID if not provided
    if (targetFolderId === null) {
        const folderItem = e.target.closest('.folder-item');
        if (folderItem) {
            targetFolderId = parseInt(folderItem.dataset.folderId);
        }
        // If targetFolderId is still null, it means we're dropping on main area (remove from folder)
    }
    
    // Move the conversation
    moveChatToFolderDragDrop(draggedConversationId, targetFolderId);
    
    // Clean up UI
    document.querySelectorAll('.folder-item').forEach(folder => {
        folder.classList.remove('drag-over');
    });
}

// Move chat to folder via drag and drop
const moveChatToFolderDragDrop = withSession(async function(conversationId, folderId) {
    try {
        const response = await secureFetch(`/api/conversations/${conversationId}/move-to-folder`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ folder_id: folderId })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            NotificationModal.toast(
                folderId ? 'Chat moved to folder successfully' : 'Chat removed from folder successfully', 
                'success'
            );
            
            // Refresh UI
            loadChatFolders();
            loadConversations(false, false);
        } else {
            NotificationModal.toast(result.error || 'Failed to move chat', 'error');
        }
    } catch (error) {
        console.error('Error moving chat:', error);
        NotificationModal.toast('Failed to move chat', 'error');
    }
});

// Enhance existing chat menu with folder options
function enhanceExistingChatMenu(chatItem) {
    const conversationId = chatItem.dataset.conversationId;
    if (!conversationId) return;
    
    // Function to add folder options
    const addFolderOptions = () => {
        const chatMenu = chatItem.querySelector('.chat-menu');
        const chatMenuContent = chatMenu?.querySelector('.chat-menu-content');
        
        if (chatMenuContent && !chatMenuContent.querySelector('.folders-menu-section')) {
            // Add folder options to existing menu
            addFolderOptionsToMenu(chatMenuContent, conversationId);
            return true;
        }
        return false;
    };
    
    // Try immediately first
    if (!addFolderOptions()) {
        // Wait for the chat menu to be created by the original script
        setTimeout(() => {
            addFolderOptions();
        }, 100);
    }
}

// Check if a conversation is currently in a folder
function isConversationInFolder(conversationId) {
    // Simple and direct: check if the conversation exists as a folder-chat-item
    const folderChatItem = document.querySelector(`.folder-chat-item[data-conversation-id="${conversationId}"]`);
    const isInFolder = !!folderChatItem;
    
    return isInFolder;
}

// Add folder options to existing chat menu
function addFolderOptionsToMenu(menuContent, conversationId) {
    // Create folder section
    const folderSection = document.createElement('div');
    folderSection.className = 'folders-menu-section';
    
    // Add separator before folder options
    const separator = document.createElement('div');
    separator.classList.add('menu-separator');
    folderSection.appendChild(separator);
    
    // Add "Move to Folder" submenu
    if (chatFolders.length > 0) {
        const moveToFolderLink = createFolderMenuLink('fa-folder-open', 'Move to Folder', () => {
            showMoveChatModal(conversationId);
        });
        folderSection.appendChild(moveToFolderLink);
    }
    
    // Add "Remove from Folder" option only if chat is in a folder
    const isInFolder = isConversationInFolder(conversationId);
    if (isInFolder) {
        const removeFromFolderLink = createFolderMenuLink('fa-times', 'Remove from Folder', () => {
            moveChatToFolderDragDrop(conversationId, null);
        });
        folderSection.appendChild(removeFromFolderLink);
    }
    
    // Try to insert before WhatsApp section, or at the end
    const existingLinks = menuContent.querySelectorAll('.menu-link');
    let insertBeforeElement = null;
    
    // Look for WhatsApp or other platform links
    for (let link of existingLinks) {
        if (link.textContent.toLowerCase().includes('whatsapp') || 
            link.textContent.toLowerCase().includes('telegram')) {
            // Find the separator before this link
            let prevElement = link.previousElementSibling;
            while (prevElement && !prevElement.classList.contains('menu-separator')) {
                prevElement = prevElement.previousElementSibling;
            }
            if (prevElement) {
                insertBeforeElement = prevElement;
                break;
            }
        }
    }
    
    if (insertBeforeElement) {
        menuContent.insertBefore(folderSection, insertBeforeElement);
    } else {
        menuContent.appendChild(folderSection);
    }
}

// Create a menu link similar to the original createMenuLink function
function createFolderMenuLink(iconClass, text, onClick, additionalClass = '') {
    const link = document.createElement('a');
    link.href = '#';
    link.classList.add('menu-link');
    if (additionalClass) {
        link.classList.add(additionalClass);
    }
    
    const icon = document.createElement('i');
    icon.classList.add('fas', iconClass);
    
    const textNode = document.createTextNode(` ${text}`);
    
    link.appendChild(icon);
    link.appendChild(textNode);
    
    link.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        onClick();
        
        // Close the menu after clicking
        const menuContent = link.closest('.chat-menu-content');
        if (menuContent) {
            menuContent.style.display = 'none';
        }
    });
    
    return link;
}

// Update all existing chat menus with current folder options
function updateAllChatMenus() {
    const chatItems = document.querySelectorAll('[data-conversation-id]:not(.folder-chat-item)');
    chatItems.forEach(chatItem => {
        const menuContent = chatItem.querySelector('.chat-menu-content');
        if (menuContent) {
            // Remove existing folder section
            const existingSection = menuContent.querySelector('.folders-menu-section');
            if (existingSection) {
                existingSection.remove();
            }
            
            // Re-add folder options with updated data
            const conversationId = chatItem.dataset.conversationId;
            if (conversationId) {
                addFolderOptionsToMenu(menuContent, conversationId);
            }
        }
    });
}

// Initialize when everything is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeFoldersDelayed);
} else {
    initializeFoldersDelayed();
}

function initializeFoldersDelayed() {
    // Wait for chat.js to be fully loaded and initialized
    setTimeout(async () => {
        await initializeFolders();
        // Also enhance any existing chat items that might already be loaded
        updateAllChatMenus();
        
        // Setup new chat button integration
        setupNewChatIntegration();
    }, 500);
}

// Update new chat button state based on selected folder
function updateNewChatButtonState() {
    const newChatBtn = document.getElementById('new-chat-main-btn');
    if (!newChatBtn) return;
    
    if (currentSelectedFolderId) {
        // Update button text to show it will create in folder
        const folder = chatFolders.find(f => f.id === currentSelectedFolderId);
        if (folder) {
            const originalText = newChatBtn.getAttribute('data-original-text') || newChatBtn.textContent;
            if (!newChatBtn.getAttribute('data-original-text')) {
                newChatBtn.setAttribute('data-original-text', originalText);
            }
            newChatBtn.innerHTML = `<i class="fas fa-comment-alt"></i> New Chat in "${folder.name}"`;
            newChatBtn.classList.add('btn-folder-selected');
        }
    } else {
        // Restore original button text
        const originalText = newChatBtn.getAttribute('data-original-text');
        if (originalText) {
            newChatBtn.textContent = originalText;
        }
        newChatBtn.classList.remove('btn-folder-selected');
    }
}

// Setup integration with new chat functionality
function setupNewChatIntegration() {
    // Store reference to original startNewConversation function if it exists
    if (typeof window.startNewConversation === 'function') {
        window.originalStartNewConversation = window.startNewConversation;
    }

    // Store reference to original updateActiveChatName function
    if (typeof window.updateActiveChatName === 'function') {
        window.originalUpdateActiveChatName = window.updateActiveChatName;
    }
    
    // Override updateActiveChatName to also update folder chats
    window.updateActiveChatName = function(newName) {
        // Call original function first
        if (typeof window.originalUpdateActiveChatName === 'function') {
            window.originalUpdateActiveChatName(newName);
        }
        
        // Also update in folder chats if applicable
        const folderChatsContainers = document.querySelectorAll('.folder-chats-container');
        folderChatsContainers.forEach(container => {
            const activeChatElement = container.querySelector('.folder-chat-item.active-chat');
            if (activeChatElement) {
                const chatNameSpan = activeChatElement.querySelector('.chat-name');
                if (chatNameSpan) {
                    chatNameSpan.textContent = newName;
                }
            }
        });
        
        // Update folder conversation counts since chat names might affect counts
        loadChatFolders();
    };
    
    // Override startNewConversation to support folders
    window.startNewConversation = function(promptId = null) {
        
        // If no folder selected, use original function
        if (!currentSelectedFolderId && typeof window.originalStartNewConversation === 'function') {
            return window.originalStartNewConversation(promptId);
        }
        
        // Admin check
        if (admin_view) {
            return Promise.resolve();
        }

        if (!isFirstCall && isCurrentConversationEmpty) {
            return Promise.resolve();
        }
        
        let body = {};
        if (promptId !== null) {
            body.prompt_id = promptId;
        }
        
        // Add folder_id if a folder is selected
        if (currentSelectedFolderId) {
            body.folder_id = currentSelectedFolderId;
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
            
            if (currentSelectedFolderId) {
                // If created in folder, reload folder chats and then activate the new chat
                loadFolderChats(currentSelectedFolderId, () => {
                    // Callback executed after folder chats are loaded
                    const newFolderChat = document.querySelector(`.folder-chat-item[data-conversation-id="${data.id}"]`);
                    
                    if (newFolderChat) {
                        // Remove active from all other chats
                        document.querySelectorAll('.active-chat').forEach(el => {
                            el.classList.remove('active-chat');
                        });
                        
                        // Mark this chat as active
                        newFolderChat.classList.add('active-chat');
                        window.selectedChat = newFolderChat;
                    }
                });

                // Update only the conversation count for this specific folder
                updateFolderConversationCount(currentSelectedFolderId);
                
                // Don't add to main conversation list - it's in a folder
            } else {
                // This shouldn't happen since we call original function above
                addConversationElement(data, data.name, null, true);
            }
            
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
            console.error('Error creating conversation:', error);
            // Fall back to original function if available
            if (typeof window.originalStartNewConversation === 'function') {
                return window.originalStartNewConversation(promptId);
            }
        });
    };
}

// Function to deselect current folder (useful for other parts of the UI)
function deselectCurrentFolder() {
    currentSelectedFolderId = null;
    
    // Collapse all folders
    document.querySelectorAll('.folder-chats-container').forEach(container => {
        container.style.display = 'none';
    });
    
    // Remove active styling
    document.querySelectorAll('.folder-item').forEach(item => {
        item.classList.remove('folder-selected');
    });
    
    updateNewChatButtonState();
}