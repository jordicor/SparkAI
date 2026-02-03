/**
 * API Credentials Client
 * Minimal client for loading user API keys in chat and other pages.
 * This is a lightweight version that only handles reading keys.
 */

(function() {
    const STORAGE_KEY = 'spark_user_api_keys';

    // Only initialize if not already defined
    if (window.userCredentials) {
        return;
    }

    window.userCredentials = {
        /**
         * Get the storage mode from preferences
         * @returns {string} 'session' | 'persistent' | 'server'
         */
        getStorageMode: function() {
            const savedMode = localStorage.getItem('spark_credentials_storage_mode');
            if (savedMode) {
                return savedMode;
            }

            // Check if there are keys in localStorage (persistent mode)
            const persistentData = localStorage.getItem(STORAGE_KEY);
            if (persistentData) {
                try {
                    const data = JSON.parse(persistentData);
                    if (data.storageMode) {
                        return data.storageMode;
                    }
                } catch (e) {
                    // Ignore parse errors
                }
            }

            return 'session';
        },

        /**
         * Get the appropriate storage based on mode
         * @returns {Storage} localStorage or sessionStorage
         */
        getStorage: function() {
            return this.getStorageMode() === 'persistent' ? localStorage : sessionStorage;
        },

        /**
         * Get all keys from local storage
         * @returns {Object} Keys object
         */
        getAllKeys: function() {
            const mode = this.getStorageMode();

            // Server mode keys are fetched at save_message time from the database
            // So we don't return them here to avoid double-sending
            if (mode === 'server') {
                return {};
            }

            const storage = this.getStorage();
            const stored = storage.getItem(STORAGE_KEY);

            if (stored) {
                try {
                    const data = JSON.parse(stored);
                    return data.keys || {};
                } catch (e) {
                    console.debug('Error parsing stored API keys:', e);
                }
            }

            return {};
        },

        /**
         * Get keys formatted for API request header
         * @returns {Promise<string|null>} Base64 encoded keys or null
         */
        getKeysForRequest: async function() {
            const keys = this.getAllKeys();

            if (Object.keys(keys).length === 0) {
                return null;
            }

            try {
                return btoa(JSON.stringify(keys));
            } catch (e) {
                console.error('Error encoding API keys:', e);
                return null;
            }
        },

        /**
         * Check if user has any configured keys (local only)
         * @returns {boolean}
         */
        hasKeys: function() {
            const keys = this.getAllKeys();
            return Object.keys(keys).length > 0;
        }
    };
})();
