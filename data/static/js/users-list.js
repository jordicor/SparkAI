/**
 * Users List - Management functionality
 * Handles filtering, sorting, selection and actions for users table
 */

// State management
const UsersListState = {
    users: [],
    sortColumn: 'username',
    sortDirection: 'asc',
    filters: {
        search: '',
        status: '',
        prompt: '',
        llm: '',
        balance: '',
        role: ''
    }
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function() {
    initializeUsersData();
    populateFilterDropdowns();
    setupEventListeners();
    applyFiltersAndSort();
    initUltraAdminState();
});

/**
 * Extract users data from table rows into state
 */
function initializeUsersData() {
    const rows = document.querySelectorAll('#usersTableBody tr[data-username]');
    UsersListState.users = Array.from(rows).map(row => {
        const tokens = parseInt(row.dataset.tokens) || 0;

        // Format tokens cell with abbreviation
        const tokensCell = row.querySelector('.tokens-cell');
        if (tokensCell) {
            tokensCell.textContent = formatNumber(tokens);
        }

        return {
            element: row,
            username: row.dataset.username,
            role: row.dataset.role || '',
            phone: row.dataset.phone || '',
            status: row.dataset.status,
            prompt: row.dataset.prompt,
            llm: row.dataset.llm,
            tokens: tokens,
            cost: parseFloat(row.dataset.cost) || 0,
            balance: parseFloat(row.dataset.balance) || 0,
            chats: parseInt(row.dataset.chats) || 0
        };
    });
}

/**
 * Populate filter dropdowns with unique values from data
 */
function populateFilterDropdowns() {
    const prompts = [...new Set(UsersListState.users.map(u => u.prompt))].sort();
    const llms = [...new Set(UsersListState.users.map(u => u.llm))].sort();

    const promptSelect = document.getElementById('filterPrompt');
    const llmSelect = document.getElementById('filterLLM');

    if (promptSelect) {
        promptSelect.innerHTML = '<option value="">All Prompts</option>';
        prompts.forEach(p => {
            if (p) promptSelect.innerHTML += `<option value="${escapeHtml(p)}">${escapeHtml(p)}</option>`;
        });
    }

    if (llmSelect) {
        llmSelect.innerHTML = '<option value="">All LLMs</option>';
        llms.forEach(l => {
            if (l) llmSelect.innerHTML += `<option value="${escapeHtml(l)}">${escapeHtml(l)}</option>`;
        });
    }
}

/**
 * Setup event listeners for filters and sorting
 */
function setupEventListeners() {
    // Filter inputs
    const searchInput = document.getElementById('filterSearch');
    const statusSelect = document.getElementById('filterStatus');
    const promptSelect = document.getElementById('filterPrompt');
    const llmSelect = document.getElementById('filterLLM');
    const balanceSelect = document.getElementById('filterBalance');
    const roleSelect = document.getElementById('filterRole');

    if (searchInput) searchInput.addEventListener('input', debounce(onFilterChange, 200));
    if (statusSelect) statusSelect.addEventListener('change', onFilterChange);
    if (promptSelect) promptSelect.addEventListener('change', onFilterChange);
    if (llmSelect) llmSelect.addEventListener('change', onFilterChange);
    if (balanceSelect) balanceSelect.addEventListener('change', onFilterChange);
    if (roleSelect) roleSelect.addEventListener('change', onFilterChange);

    // Sortable headers
    document.querySelectorAll('th[data-sort]').forEach(th => {
        th.addEventListener('click', () => onSortClick(th.dataset.sort));
    });

    // Select all checkbox
    const selectAll = document.getElementById('selectAll');
    if (selectAll) selectAll.addEventListener('change', toggleSelectAll);

    // Individual checkboxes
    document.querySelectorAll('.user-checkbox').forEach(cb => {
        cb.addEventListener('change', updateSelectedCount);
    });

    // Delete form confirmation (AJAX-based to handle errors gracefully)
    const usersForm = document.getElementById('usersForm');
    if (usersForm) {
        usersForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const checked = document.querySelectorAll('.user-checkbox:checked');
            const count = checked.length;
            if (count === 0) return;

            NotificationModal.confirm('Delete Users', `Are you sure you want to delete ${count} user(s)? This action cannot be undone.`, async () => {
                try {
                    const formData = new FormData(usersForm);
                    const response = await secureFetch('/admin/delete-users', {
                        method: 'POST',
                        body: formData
                    });
                    if (!response) return;

                    const data = await response.json();
                    if (response.ok) {
                        NotificationModal.success('Deleted', data.message || 'Users deleted successfully.');
                        setTimeout(() => window.location.reload(), 1000);
                    } else {
                        NotificationModal.error('Delete Failed', data.detail || data.error || 'Could not delete users.');
                    }
                } catch (error) {
                    NotificationModal.error('Error', 'An unexpected error occurred.');
                }
            }, null, { type: 'error', confirmText: 'Delete' });
        });
    }
}

/**
 * Handle filter change
 */
function onFilterChange() {
    UsersListState.filters.search = (document.getElementById('filterSearch')?.value || '').toLowerCase();
    UsersListState.filters.status = document.getElementById('filterStatus')?.value || '';
    UsersListState.filters.prompt = document.getElementById('filterPrompt')?.value || '';
    UsersListState.filters.llm = document.getElementById('filterLLM')?.value || '';
    UsersListState.filters.balance = document.getElementById('filterBalance')?.value || '';
    UsersListState.filters.role = document.getElementById('filterRole')?.value || '';

    applyFiltersAndSort();
}

/**
 * Handle sort header click
 */
function onSortClick(column) {
    if (UsersListState.sortColumn === column) {
        UsersListState.sortDirection = UsersListState.sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
        UsersListState.sortColumn = column;
        UsersListState.sortDirection = 'asc';
    }

    updateSortIndicators();
    applyFiltersAndSort();
}

/**
 * Update sort indicators in headers
 */
function updateSortIndicators() {
    document.querySelectorAll('th[data-sort]').forEach(th => {
        th.classList.remove('sort-asc', 'sort-desc');
        if (th.dataset.sort === UsersListState.sortColumn) {
            th.classList.add(UsersListState.sortDirection === 'asc' ? 'sort-asc' : 'sort-desc');
        }
    });
}

/**
 * Apply filters and sorting to table
 */
function applyFiltersAndSort() {
    const { filters, sortColumn, sortDirection } = UsersListState;

    // Filter users
    let filtered = UsersListState.users.filter(user => {
        // Search filter (username or phone)
        if (filters.search) {
            const searchMatch = user.username.toLowerCase().includes(filters.search) ||
                               user.phone.toLowerCase().includes(filters.search);
            if (!searchMatch) return false;
        }

        // Status filter
        if (filters.status && user.status !== filters.status) return false;

        // Prompt filter
        if (filters.prompt && user.prompt !== filters.prompt) return false;

        // LLM filter
        if (filters.llm && user.llm !== filters.llm) return false;

        // Balance filter
        if (filters.balance) {
            switch (filters.balance) {
                case 'positive': if (user.balance <= 0) return false; break;
                case 'zero': if (user.balance !== 0) return false; break;
                case 'negative': if (user.balance >= 0) return false; break;
            }
        }

        // Role filter
        if (filters.role && user.role !== filters.role) return false;

        return true;
    });

    // Sort users
    filtered.sort((a, b) => {
        let valA = a[sortColumn];
        let valB = b[sortColumn];

        // Handle string comparison
        if (typeof valA === 'string') {
            valA = valA.toLowerCase();
            valB = valB.toLowerCase();
        }

        let result = 0;
        if (valA < valB) result = -1;
        else if (valA > valB) result = 1;

        return sortDirection === 'asc' ? result : -result;
    });

    // Update DOM
    const tbody = document.getElementById('usersTableBody');
    if (tbody) {
        // Hide all rows first
        UsersListState.users.forEach(user => {
            user.element.style.display = 'none';
        });

        // Show and reorder filtered rows
        filtered.forEach(user => {
            user.element.style.display = '';
            tbody.appendChild(user.element);
        });
    }

    // Update stats
    updateStats(filtered.length);

    // Update select all state
    updateSelectAllState();
}

/**
 * Update stats display
 */
function updateStats(filteredCount) {
    const total = UsersListState.users.length;
    const active = UsersListState.users.filter(u => u.status === 'active').length;
    const expired = UsersListState.users.filter(u => u.status === 'expired').length;
    const password = UsersListState.users.filter(u => u.status === 'password').length;

    document.getElementById('statTotal').textContent = total;
    document.getElementById('statActive').textContent = active;
    document.getElementById('statExpired').textContent = expired;

    const statPassword = document.getElementById('statPassword');
    if (statPassword) statPassword.textContent = password;

    const filteredStat = document.getElementById('statFiltered');
    const filteredContainer = document.getElementById('statFilteredContainer');

    if (filteredContainer) {
        if (filteredCount !== total) {
            filteredContainer.style.display = 'inline-flex';
            if (filteredStat) filteredStat.textContent = filteredCount;
        } else {
            filteredContainer.style.display = 'none';
        }
    }
}

/**
 * Toggle select all checkbox
 */
function toggleSelectAll() {
    const selectAll = document.getElementById('selectAll');
    const visibleCheckboxes = getVisibleCheckboxes();

    visibleCheckboxes.forEach(cb => {
        cb.checked = selectAll.checked;
    });

    updateSelectedCount();
}

/**
 * Update selected count and button state
 */
function updateSelectedCount() {
    const checkedCount = document.querySelectorAll('.user-checkbox:checked').length;
    const countSpan = document.getElementById('selectedCount');
    const deleteBtn = document.getElementById('deleteBtn');

    if (countSpan) countSpan.textContent = checkedCount;
    if (deleteBtn) deleteBtn.disabled = checkedCount === 0;

    updateSelectAllState();
}

/**
 * Update select all checkbox state based on visible selections
 */
function updateSelectAllState() {
    const selectAll = document.getElementById('selectAll');
    if (!selectAll) return;

    const visibleCheckboxes = getVisibleCheckboxes();
    const checkedVisible = visibleCheckboxes.filter(cb => cb.checked).length;

    selectAll.checked = visibleCheckboxes.length > 0 && checkedVisible === visibleCheckboxes.length;
    selectAll.indeterminate = checkedVisible > 0 && checkedVisible < visibleCheckboxes.length;
}

/**
 * Get visible (not hidden by filter) checkboxes
 */
function getVisibleCheckboxes() {
    return Array.from(document.querySelectorAll('.user-checkbox')).filter(cb => {
        return cb.closest('tr').style.display !== 'none';
    });
}

/**
 * Show magic link for a user
 */
function showMagicLink(magicLink, username, isExpired, isPasswordOnly) {
    const container = document.getElementById('magicLinkContainer');
    const input = document.getElementById('magicLink');
    const actionButton = document.getElementById('actionButton');
    const title = document.getElementById('usernameMagicLink');

    if (!container || !input || !actionButton || !title) return;

    title.innerHTML = '<i class="fas fa-magic me-2"></i>Magic Link for: <strong>' + escapeHtml(username) + '</strong>';
    container.style.display = 'block';

    const copyMessage = document.getElementById('copyMessage');
    if (copyMessage) copyMessage.style.display = 'none';

    if (isPasswordOnly || !magicLink || magicLink === 'None') {
        input.value = 'No magic link (password-only user)';
        input.style.opacity = '0.6';
        actionButton.innerHTML = '<i class="fas fa-magic me-1"></i> Generate';
        actionButton.className = 'btn btn-info';
        actionButton.onclick = function() { renewMagicLink(username); };
    } else if (isExpired) {
        input.value = magicLink;
        input.style.opacity = '1';
        actionButton.innerHTML = '<i class="fas fa-sync me-1"></i> Renew';
        actionButton.className = 'btn btn-warning';
        actionButton.onclick = function() { renewMagicLink(username); };
    } else {
        input.value = magicLink;
        input.style.opacity = '1';
        actionButton.innerHTML = '<i class="fas fa-copy me-1"></i> Copy';
        actionButton.className = 'btn btn-success';
        actionButton.onclick = copyToClipboard;
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/**
 * Close magic link container
 */
function closeMagicLink() {
    const container = document.getElementById('magicLinkContainer');
    if (container) container.style.display = 'none';
}

/**
 * Copy magic link to clipboard
 */
function copyToClipboard() {
    const input = document.getElementById('magicLink');
    if (!input) return;

    input.select();
    document.execCommand('copy');

    const copyMessage = document.getElementById('copyMessage');
    if (copyMessage) copyMessage.style.display = 'block';

    const actionButton = document.getElementById('actionButton');
    if (actionButton) {
        actionButton.innerHTML = '<i class="fas fa-check me-1"></i> Copied!';
        setTimeout(() => {
            actionButton.innerHTML = '<i class="fas fa-copy me-1"></i> Copy';
        }, 2000);
    }
}

/**
 * Renew magic link for a user
 */
async function renewMagicLink(username) {
    const actionButton = document.getElementById('actionButton');
    if (!actionButton) return;

    actionButton.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span> Renewing...';
    actionButton.disabled = true;

    try {
        const response = await secureFetch('/admin/renew-token/' + encodeURIComponent(username), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (!response) {
            actionButton.disabled = false;
            actionButton.innerHTML = '<i class="fas fa-sync me-1"></i> Renew';
            return;
        }

        const data = await response.json();
        actionButton.disabled = false;

        if (data.error) {
            NotificationModal.error('Renewal Failed', data.error);
            actionButton.innerHTML = '<i class="fas fa-sync me-1"></i> Renew';
        } else {
            showMagicLink(data.magic_link, username, false);

            // Update the row status
            const row = document.querySelector(`tr[data-username="${username}"]`);
            if (row) {
                row.dataset.status = 'active';
                const statusCell = row.querySelector('.status-cell');
                if (statusCell) {
                    statusCell.innerHTML = '<span class="status-active" title="Magic link active"><i class="fas fa-check-circle"></i></span>';
                }

                // Update state
                const user = UsersListState.users.find(u => u.username === username);
                if (user) user.status = 'active';

                // Recalculate stats
                updateStats(getVisibleCheckboxes().length);
            }
        }
    } catch (error) {
        console.error('Error:', error);
        actionButton.disabled = false;
        actionButton.innerHTML = '<i class="fas fa-sync me-1"></i> Renew';
        NotificationModal.error('Error', 'An error occurred while renewing the magic link.');
    }
}

/**
 * Reset all filters
 */
function resetFilters() {
    document.getElementById('filterSearch').value = '';
    document.getElementById('filterStatus').value = '';
    document.getElementById('filterPrompt').value = '';
    document.getElementById('filterLLM').value = '';
    document.getElementById('filterBalance').value = '';
    document.getElementById('filterRole').value = '';

    UsersListState.filters = {
        search: '',
        status: '',
        prompt: '',
        llm: '',
        balance: '',
        role: ''
    };

    applyFiltersAndSort();
}

// Utility functions

/**
 * Debounce function for search input
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Copy phone number to clipboard
 */
function copyPhone(element) {
    const phone = element.dataset.phone;
    if (!phone) return;

    navigator.clipboard.writeText(phone).then(() => {
        // Visual feedback
        const icon = element.querySelector('i');
        const originalClass = icon.className;
        icon.className = 'fas fa-check';
        element.classList.add('copied');

        setTimeout(() => {
            icon.className = originalClass;
            element.classList.remove('copied');
        }, 1500);
    }).catch(err => {
        console.error('Failed to copy phone:', err);
    });
}

/**
 * Format number with K, M, B abbreviations
 */
function formatNumber(num) {
    if (num === null || num === undefined || isNaN(num)) return '0';

    const absNum = Math.abs(num);

    if (absNum >= 1000000000) {
        return (num / 1000000000).toFixed(1).replace(/\.0$/, '') + 'B';
    }
    if (absNum >= 1000000) {
        return (num / 1000000).toFixed(1).replace(/\.0$/, '') + 'M';
    }
    if (absNum >= 1000) {
        return (num / 1000).toFixed(1).replace(/\.0$/, '') + 'K';
    }

    return num.toString();
}

// ==========================================================
// Ultra Admin+ — Privilege Elevation
// ==========================================================

let ultraAdminCountdown = null;

/**
 * Initialize Ultra Admin+ state on page load
 */
function initUltraAdminState() {
    const btn = document.getElementById('ultraAdminBtn');
    if (!btn) return;

    const isElevated = btn.dataset.elevated === 'true';
    const ttl = parseInt(btn.dataset.ttl) || 0;

    if (isElevated && ttl > 0) {
        activateUltraAdminUI(ttl);
    }
}

/**
 * Open the Ultra Admin+ modal
 */
function openUltraAdminModal() {
    document.getElementById('ultraAdminStep1').style.display = 'block';
    document.getElementById('ultraAdminStep2').style.display = 'none';
    const codeInput = document.getElementById('ultraAdminCodeInput');
    if (codeInput) codeInput.value = '';
    new bootstrap.Modal(document.getElementById('ultraAdminModal')).show();
}

/**
 * Request elevation code from server
 */
async function requestUltraAdminCode() {
    try {
        const response = await secureFetch('/api/ultra-admin/request-code', { method: 'POST' });
        if (!response) return;

        const data = await response.json();

        if (response.ok) {
            document.getElementById('ultraAdminStep1').style.display = 'none';
            document.getElementById('ultraAdminStep2').style.display = 'block';
            document.getElementById('ultraAdminEmailHint').textContent = data.email_hint || '';
            document.getElementById('ultraAdminCodeInput').focus();
        } else if (response.status === 429) {
            NotificationModal.warning('Cooldown', data.error || 'Please wait before requesting another code.');
        } else if (response.status === 409) {
            NotificationModal.warning('Unavailable', data.error || 'Another admin is currently elevated.');
        } else {
            NotificationModal.error('Error', data.error || 'Could not send code.');
        }
    } catch (error) {
        NotificationModal.error('Error', 'Unexpected error requesting code.');
    }
}

/**
 * Verify the entered code
 */
async function verifyUltraAdminCode() {
    const code = document.getElementById('ultraAdminCodeInput').value.trim();
    if (code.length !== 6 || !/^\d{6}$/.test(code)) {
        NotificationModal.warning('Invalid Code', 'Enter a 6-digit numeric code.');
        return;
    }

    try {
        const response = await secureFetch('/api/ultra-admin/verify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code })
        });
        if (!response) return;

        const data = await response.json();

        if (response.ok && data.status === 'elevated') {
            const modal = bootstrap.Modal.getInstance(document.getElementById('ultraAdminModal'));
            if (modal) modal.hide();
            NotificationModal.success('Ultra Admin+ Active', 'Elevated privileges granted for 30 minutes.');
            activateUltraAdminUI(data.ttl);
        } else {
            NotificationModal.error('Verification Failed', data.error || 'Incorrect code.');
            document.getElementById('ultraAdminCodeInput').value = '';
            document.getElementById('ultraAdminCodeInput').focus();
        }
    } catch (error) {
        NotificationModal.error('Error', 'Unexpected error verifying code.');
    }
}

/**
 * Activate the Ultra Admin+ UI (button changes to active state with countdown)
 */
function activateUltraAdminUI(ttlSeconds) {
    const btn = document.getElementById('ultraAdminBtn');
    if (!btn) return;

    btn.classList.remove('btn-outline-warning', 'btn-sm');
    btn.classList.add('btn-warning', 'btn-sm');
    btn.onclick = null;
    btn.style.cursor = 'default';

    let remaining = ttlSeconds;
    updateUltraAdminTimer(btn, remaining);

    ultraAdminCountdown = setInterval(() => {
        remaining--;
        if (remaining <= 0) {
            deactivateUltraAdminUI();
        } else {
            updateUltraAdminTimer(btn, remaining);
        }
    }, 1000);
}

/**
 * Update the countdown timer display
 */
function updateUltraAdminTimer(btn, seconds) {
    const min = Math.floor(seconds / 60);
    const sec = seconds % 60;
    btn.innerHTML =
        '<i class="fas fa-bolt me-1"></i>' +
        'Ultra Admin+ (' + min + ':' + sec.toString().padStart(2, '0') + ') ' +
        '<i class="fas fa-times-circle ms-1" style="cursor:pointer;" onclick="revokeUltraAdmin(event)" title="Deactivate"></i>';
}

/**
 * Deactivate the Ultra Admin+ UI (return to normal state)
 */
function deactivateUltraAdminUI() {
    clearInterval(ultraAdminCountdown);
    ultraAdminCountdown = null;

    const btn = document.getElementById('ultraAdminBtn');
    if (!btn) return;

    btn.classList.remove('btn-warning');
    btn.classList.add('btn-outline-warning');
    btn.innerHTML = '<i class="fas fa-bolt me-1"></i><span id="ultraAdminLabel">Ultra Admin+</span>';
    btn.onclick = openUltraAdminModal;
    btn.style.cursor = 'pointer';
}

/**
 * Revoke Ultra Admin+ elevation
 */
async function revokeUltraAdmin(event) {
    if (event) event.stopPropagation();
    try {
        await secureFetch('/api/ultra-admin/revoke', { method: 'POST' });
        deactivateUltraAdminUI();
        NotificationModal.info('Ultra Admin+ Deactivated', 'Privileges returned to normal.');
    } catch (error) {
        NotificationModal.error('Error', 'Could not revoke elevation.');
    }
}
