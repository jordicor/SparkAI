/**
 * API Credentials Manager
 * Manages user API keys for AI providers with multiple storage modes
 */

class UserCredentialsManager {
    constructor() {
        this.STORAGE_KEY = 'aurvek_user_api_keys';
        this.storageMode = 'session'; // 'session' | 'persistent' | 'server'
        this.providers = ['openai', 'anthropic', 'google', 'xai', 'elevenlabs'];
        this.init();
    }

    /**
     * Initialize the credentials manager
     */
    init() {
        // Load storage mode preference
        const savedMode = localStorage.getItem('aurvek_credentials_storage_mode');
        if (savedMode) {
            this.storageMode = savedMode;
        }

        // Also check if there are keys in localStorage (persistent mode)
        const persistentData = localStorage.getItem(this.STORAGE_KEY);
        if (persistentData) {
            const data = JSON.parse(persistentData);
            if (data.storageMode) {
                this.storageMode = data.storageMode;
            }
        }

        // Load existing keys into form
        this.loadKeysToForm();
    }

    /**
     * Get the appropriate storage based on mode
     * @returns {Storage} localStorage or sessionStorage
     */
    getStorage() {
        return this.storageMode === 'persistent' ? localStorage : sessionStorage;
    }

    /**
     * Set the storage mode
     * @param {string} mode - 'session' | 'persistent' | 'server'
     */
    async setStorageMode(mode) {
        const oldMode = this.storageMode;
        this.storageMode = mode;

        // Save mode preference
        localStorage.setItem('aurvek_credentials_storage_mode', mode);

        // If switching away from server mode, we might want to keep server keys
        // If switching to server mode, we might want to migrate local keys

        if (oldMode !== 'server' && mode === 'server') {
            // Migrating to server - upload current local keys
            const localKeys = this.getLocalKeys();
            if (Object.keys(localKeys.keys).length > 0) {
                await this.saveAllToServer(localKeys.keys);
            }
        } else if (oldMode === 'server' && mode !== 'server') {
            // Migrating from server - download keys to local storage
            const serverKeys = await this.getAllFromServer();
            if (serverKeys && Object.keys(serverKeys).length > 0) {
                this.saveLocalKeys(serverKeys);
            }
        } else if (oldMode === 'session' && mode === 'persistent') {
            // Move from session to persistent
            const sessionData = sessionStorage.getItem(this.STORAGE_KEY);
            if (sessionData) {
                localStorage.setItem(this.STORAGE_KEY, sessionData);
                sessionStorage.removeItem(this.STORAGE_KEY);
            }
        } else if (oldMode === 'persistent' && mode === 'session') {
            // Move from persistent to session
            const persistentData = localStorage.getItem(this.STORAGE_KEY);
            if (persistentData) {
                sessionStorage.setItem(this.STORAGE_KEY, persistentData);
                localStorage.removeItem(this.STORAGE_KEY);
            }
        }

        // Update the stored data with new mode
        const storage = this.getStorage();
        const data = this.getAllLocalData();
        data.storageMode = mode;
        storage.setItem(this.STORAGE_KEY, JSON.stringify(data));
    }

    /**
     * Get all local data from storage
     * @returns {Object} Storage data object
     */
    getAllLocalData() {
        const storage = this.getStorage();
        const stored = storage.getItem(this.STORAGE_KEY);
        return stored ? JSON.parse(stored) : { storageMode: this.storageMode, keys: {} };
    }

    /**
     * Get local keys only (not from server)
     * @returns {Object} Keys object
     */
    getLocalKeys() {
        return this.getAllLocalData();
    }

    /**
     * Save keys to local storage
     * @param {Object} keys - Keys object to save
     */
    saveLocalKeys(keys) {
        const storage = this.getStorage();
        const data = { storageMode: this.storageMode, keys };
        storage.setItem(this.STORAGE_KEY, JSON.stringify(data));
    }

    /**
     * Set a key for a provider
     * @param {string} provider - Provider name
     * @param {string} key - API key
     */
    async setKey(provider, key) {
        if (this.storageMode === 'server') {
            return await this.saveToServer(provider, key);
        }

        const data = this.getAllLocalData();
        if (key) {
            data.keys[provider] = key;
        } else {
            delete data.keys[provider];
        }
        this.getStorage().setItem(this.STORAGE_KEY, JSON.stringify(data));
        return { success: true };
    }

    /**
     * Get a key for a provider
     * @param {string} provider - Provider name
     * @returns {Promise<string|null>} The API key or null
     */
    async getKey(provider) {
        if (this.storageMode === 'server') {
            return await this.getFromServer(provider);
        }

        const data = this.getAllLocalData();
        return data.keys[provider] || null;
    }

    /**
     * Get all keys (for sending with requests)
     * @returns {Promise<Object>} Keys object
     */
    async getAllKeys() {
        if (this.storageMode === 'server') {
            return await this.getAllFromServer();
        }
        return this.getAllLocalData().keys;
    }

    /**
     * Test an API key
     * @param {string} provider - Provider name
     * @param {string} key - API key to test
     * @returns {Promise<Object>} Test result
     */
    async testKey(provider, key) {
        try {
            const response = await fetch('/api/test-api-key', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                credentials: 'include',
                body: JSON.stringify({ provider, key })
            });
            return await response.json();
        } catch (error) {
            return { success: false, message: error.message };
        }
    }

    /**
     * Save a key to the server (server mode)
     * @param {string} provider - Provider name
     * @param {string} key - API key
     * @returns {Promise<Object>} Save result
     */
    async saveToServer(provider, key) {
        try {
            const response = await fetch('/api/user-credentials', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                credentials: 'include',
                body: JSON.stringify({ provider, key })
            });
            const result = await response.json();

            // Handle not_allowed error (system_only mode)
            if (response.status === 403 && result.error === 'not_allowed') {
                this.showNotAllowedError();
                return { success: false, message: result.message, notAllowed: true };
            }

            return result;
        } catch (error) {
            return { success: false, message: error.message };
        }
    }

    /**
     * Show error when user is not allowed to configure keys
     */
    showNotAllowedError() {
        NotificationModal.info('System Keys Only', 'Your account is configured to use system API keys only. You cannot configure your own keys.');
    }

    /**
     * Save all keys to server
     * @param {Object} keys - Keys object
     * @returns {Promise<Object>} Save result
     */
    async saveAllToServer(keys) {
        try {
            const response = await fetch('/api/user-credentials/batch', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                credentials: 'include',
                body: JSON.stringify({ keys })
            });
            const result = await response.json();

            // Handle not_allowed error (system_only mode)
            if (response.status === 403 && result.error === 'not_allowed') {
                this.showNotAllowedError();
                return { success: false, message: result.message, notAllowed: true };
            }

            return result;
        } catch (error) {
            return { success: false, message: error.message };
        }
    }

    /**
     * Get a key from the server
     * @param {string} provider - Provider name
     * @returns {Promise<string|null>} The API key or null
     */
    async getFromServer(provider) {
        try {
            const response = await fetch(`/api/user-credentials/${provider}`, {
                credentials: 'include',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });
            const data = await response.json();
            return data.exists ? data.key : null;
        } catch (error) {
            console.error('Error getting key from server:', error);
            return null;
        }
    }

    /**
     * Get all keys from server
     * @returns {Promise<Object>} Keys object
     */
    async getAllFromServer() {
        try {
            const response = await fetch('/api/user-credentials', {
                credentials: 'include',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });
            const data = await response.json();
            return data.keys || {};
        } catch (error) {
            console.error('Error getting keys from server:', error);
            return {};
        }
    }

    /**
     * Delete a key
     * @param {string} provider - Provider name
     */
    async deleteKey(provider) {
        if (this.storageMode === 'server') {
            try {
                await fetch(`/api/user-credentials/${provider}`, {
                    method: 'DELETE',
                    credentials: 'include',
                    headers: {
                        'X-Requested-With': 'XMLHttpRequest'
                    }
                });
            } catch (error) {
                console.error('Error deleting key from server:', error);
            }
        }

        const data = this.getAllLocalData();
        delete data.keys[provider];
        this.getStorage().setItem(this.STORAGE_KEY, JSON.stringify(data));
    }

    /**
     * Clear all keys
     */
    async clearAll() {
        // Clear local storage
        localStorage.removeItem(this.STORAGE_KEY);
        sessionStorage.removeItem(this.STORAGE_KEY);

        // Clear server storage if in server mode
        if (this.storageMode === 'server') {
            try {
                await fetch('/api/user-credentials', {
                    method: 'DELETE',
                    credentials: 'include',
                    headers: {
                        'X-Requested-With': 'XMLHttpRequest'
                    }
                });
            } catch (error) {
                console.error('Error clearing keys from server:', error);
            }
        }
    }

    /**
     * Load keys into the form fields
     */
    async loadKeysToForm() {
        // Set storage mode radio
        const modeRadio = document.querySelector(`input[name="storageMode"][value="${this.storageMode}"]`);
        if (modeRadio) {
            modeRadio.checked = true;
        }

        // Update storage info text
        this.updateStorageInfo();

        // Load keys for each provider
        for (const provider of this.providers) {
            const key = await this.getKey(provider);
            const input = document.getElementById(`key-${provider}`);
            if (input && key) {
                // Show masked key for server mode, actual key for local modes
                if (this.storageMode === 'server') {
                    input.value = key; // Server returns masked key
                    input.dataset.hasServerKey = 'true';
                } else {
                    input.value = key;
                }
                this.updateStatus(provider, 'saved', 'Key saved');
            }
        }
    }

    /**
     * Update the storage info message
     */
    updateStorageInfo() {
        const infoText = document.getElementById('storageInfoText');
        if (!infoText) return;

        const messages = {
            session: 'Your keys are stored only for this browser session and will be cleared when you close the tab.',
            persistent: 'Your keys are stored in this browser and will persist across sessions until manually deleted.',
            server: 'Your keys are encrypted with AES-256 and stored on the server. They will be accessible from any device.'
        };

        infoText.textContent = messages[this.storageMode] || messages.session;
    }

    /**
     * Update status indicator for a provider
     * @param {string} provider - Provider name
     * @param {string} status - 'success' | 'error' | 'testing' | 'saved' | ''
     * @param {string} message - Status message
     */
    updateStatus(provider, status, message = '') {
        const statusEl = document.getElementById(`status-${provider}`);
        if (!statusEl) return;

        statusEl.className = 'status-indicator';

        switch (status) {
            case 'success':
                statusEl.innerHTML = '<i class="fas fa-check-circle text-success"></i>';
                statusEl.title = message || 'Valid';
                break;
            case 'error':
                statusEl.innerHTML = '<i class="fas fa-times-circle text-danger"></i>';
                statusEl.title = message || 'Invalid';
                break;
            case 'testing':
                statusEl.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
                statusEl.title = 'Testing...';
                break;
            case 'saved':
                statusEl.innerHTML = '<i class="fas fa-save text-info"></i>';
                statusEl.title = message || 'Saved';
                break;
            default:
                statusEl.innerHTML = '';
                statusEl.title = '';
        }
    }

    /**
     * Check if user has any configured keys
     * @returns {Promise<boolean>}
     */
    async hasKeys() {
        const keys = await this.getAllKeys();
        return Object.keys(keys).length > 0;
    }

    /**
     * Get keys formatted for API request header
     * @returns {Promise<string|null>} Base64 encoded keys or null
     */
    async getKeysForRequest() {
        const keys = await this.getAllKeys();
        if (Object.keys(keys).length === 0) {
            return null;
        }
        return btoa(JSON.stringify(keys));
    }
}

// Create global instance
window.userCredentials = new UserCredentialsManager();

// DOM Ready - Setup event handlers
document.addEventListener('DOMContentLoaded', () => {
    const manager = window.userCredentials;

    // Storage mode change handlers
    document.querySelectorAll('input[name="storageMode"]').forEach(radio => {
        radio.addEventListener('change', async (e) => {
            await manager.setStorageMode(e.target.value);
            manager.updateStorageInfo();
            NotificationModal.toast('Storage mode changed', 'info');
        });
    });

    // Toggle password visibility
    document.querySelectorAll('.toggle-visibility').forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.dataset.target;
            const input = document.getElementById(targetId);
            const icon = btn.querySelector('i');

            if (input.type === 'password') {
                input.type = 'text';
                icon.classList.remove('fa-eye');
                icon.classList.add('fa-eye-slash');
            } else {
                input.type = 'password';
                icon.classList.remove('fa-eye-slash');
                icon.classList.add('fa-eye');
            }
        });
    });

    // Test individual key
    document.querySelectorAll('.test-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const provider = btn.dataset.provider;
            const input = document.getElementById(`key-${provider}`);
            const key = input.value.trim();

            if (!key) {
                NotificationModal.toast(`Please enter a ${provider} API key first`, 'warning');
                return;
            }

            manager.updateStatus(provider, 'testing');
            btn.disabled = true;

            const result = await manager.testKey(provider, key);

            btn.disabled = false;

            if (result.success) {
                manager.updateStatus(provider, 'success', 'API key is valid');
                NotificationModal.toast(`${provider} API key is valid!`, 'success');
            } else {
                manager.updateStatus(provider, 'error', result.message);
                NotificationModal.toast(`${provider} key invalid: ${result.message}`, 'error');
            }
        });
    });

    // Clear individual key
    document.querySelectorAll('.clear-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const provider = btn.dataset.provider;
            const input = document.getElementById(`key-${provider}`);

            input.value = '';
            await manager.deleteKey(provider);
            manager.updateStatus(provider, '');
            NotificationModal.toast(`${provider} key cleared`, 'info');
        });
    });

    // Save all credentials
    document.getElementById('saveAllCredentials')?.addEventListener('click', async () => {
        const btn = document.getElementById('saveAllCredentials');
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';

        let savedCount = 0;

        for (const provider of manager.providers) {
            const input = document.getElementById(`key-${provider}`);
            const key = input?.value.trim();

            if (key && !input.dataset.hasServerKey) {
                await manager.setKey(provider, key);
                manager.updateStatus(provider, 'saved', 'Key saved');
                savedCount++;
            }
        }

        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-save"></i> Save All';

        if (savedCount > 0) {
            NotificationModal.toast(`Saved ${savedCount} API key(s)`, 'success');
        } else {
            NotificationModal.toast('No new keys to save', 'info');
        }
    });

    // Test all credentials
    document.getElementById('testAllCredentials')?.addEventListener('click', async () => {
        const btn = document.getElementById('testAllCredentials');
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Testing...';

        let validCount = 0;
        let testedCount = 0;

        for (const provider of manager.providers) {
            const input = document.getElementById(`key-${provider}`);
            const key = input?.value.trim();

            if (key) {
                testedCount++;
                manager.updateStatus(provider, 'testing');

                const result = await manager.testKey(provider, key);

                if (result.success) {
                    manager.updateStatus(provider, 'success', 'Valid');
                    validCount++;
                } else {
                    manager.updateStatus(provider, 'error', result.message);
                }
            }
        }

        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-vial"></i> Test All';

        if (testedCount === 0) {
            NotificationModal.toast('No API keys to test', 'info');
        } else {
            NotificationModal.toast(`${validCount}/${testedCount} keys are valid`, validCount === testedCount ? 'success' : 'warning');
        }
    });

    // Clear all credentials
    document.getElementById('clearAllCredentials')?.addEventListener('click', () => {
        NotificationModal.confirm('Clear All Keys', 'Are you sure you want to clear all API keys? This cannot be undone.', async () => {
            await manager.clearAll();

            // Clear form inputs
            for (const provider of manager.providers) {
                const input = document.getElementById(`key-${provider}`);
                if (input) {
                    input.value = '';
                    delete input.dataset.hasServerKey;
                }
                manager.updateStatus(provider, '');
            }

            NotificationModal.toast('All API keys cleared', 'info');
        }, null, { type: 'error', confirmText: 'Clear All' });
    });

    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl, { html: true });
    });
});

