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

    // Delete form confirmation
    const usersForm = document.getElementById('usersForm');
    if (usersForm) {
        let confirmedSubmit = false;
        usersForm.addEventListener('submit', function(e) {
            const count = document.querySelectorAll('.user-checkbox:checked').length;
            if (count === 0) {
                e.preventDefault();
                return;
            }
            if (confirmedSubmit) {
                confirmedSubmit = false;
                return;
            }
            e.preventDefault();
            const form = this;
            NotificationModal.confirm('Delete Users', `Are you sure you want to delete ${count} user(s)? This action cannot be undone.`, () => {
                confirmedSubmit = true;
                form.requestSubmit();
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

    document.getElementById('statTotal').textContent = total;
    document.getElementById('statActive').textContent = active;
    document.getElementById('statExpired').textContent = expired;

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
function showMagicLink(magicLink, username, isExpired) {
    const container = document.getElementById('magicLinkContainer');
    const input = document.getElementById('magicLink');
    const actionButton = document.getElementById('actionButton');
    const title = document.getElementById('usernameMagicLink');

    if (!container || !input || !actionButton || !title) return;

    title.innerHTML = '<i class="fas fa-magic me-2"></i>Magic Link for: <strong>' + escapeHtml(username) + '</strong>';
    input.value = magicLink;
    container.style.display = 'block';

    const copyMessage = document.getElementById('copyMessage');
    if (copyMessage) copyMessage.style.display = 'none';

    if (isExpired) {
        actionButton.innerHTML = '<i class="fas fa-sync me-1"></i> Renew';
        actionButton.className = 'btn btn-warning';
        actionButton.onclick = function() { renewMagicLink(username); };
    } else {
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
                    statusCell.innerHTML = '<span class="status-active"><i class="fas fa-check-circle me-1"></i>Active</span>';
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
