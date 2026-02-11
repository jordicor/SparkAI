/* ============================================================
   PROMPT & PACK EXPLORER - Frontend Logic
   Vanilla JS: fetch, filter, paginate, detail modal
   ============================================================ */

const ExploreState = {
    activeTab: 'prompts',    // 'prompts' | 'packs'
    prompts: [],
    packs: [],
    categories: [],
    activeCategory: null,
    searchQuery: '',
    currentPage: 1,
    totalPages: 1,
    total: 0,
    limit: 24,
    loading: false
};

// Debounce utility
function debounce(fn, ms) {
    let timer;
    return function (...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), ms);
    };
}

// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    loadCategories();
    loadPrompts();
    setupEventListeners();
});

function setupEventListeners() {
    const searchInput = document.getElementById('exploreSearch');
    if (searchInput) {
        searchInput.addEventListener('input', debounce(() => {
            ExploreState.searchQuery = searchInput.value;
            ExploreState.currentPage = 1;
            if (ExploreState.activeTab === 'prompts') {
                loadPrompts();
            } else {
                loadPacks();
            }
        }, 300));
    }

    // Close modal on backdrop click
    const backdrop = document.getElementById('exploreModalBackdrop');
    if (backdrop) {
        backdrop.addEventListener('click', (e) => {
            if (e.target === backdrop) closeModal();
        });
    }

    // Close modal on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });
}

// ============================================================
// TAB SWITCHING
// ============================================================

function switchTab(tab, tabEl) {
    if (tab === ExploreState.activeTab) return;
    ExploreState.activeTab = tab;
    ExploreState.currentPage = 1;
    ExploreState.searchQuery = '';
    ExploreState.activeCategory = null;

    // Reset search input
    const searchInput = document.getElementById('exploreSearch');
    if (searchInput) {
        searchInput.value = '';
        searchInput.placeholder = tab === 'prompts' ? 'Search prompts...' : 'Search packs...';
    }

    // Update tab buttons
    document.querySelectorAll('.explore-tab').forEach(t => t.classList.remove('active'));
    if (tabEl) tabEl.classList.add('active');

    // Show/hide category chips (only for prompts)
    const chips = document.getElementById('categoryChips');
    if (chips) chips.style.display = tab === 'prompts' ? '' : 'none';

    // Load content
    if (tab === 'prompts') {
        loadPrompts();
    } else {
        loadPacks();
    }
}

// ============================================================
// DATA FETCHING
// ============================================================

async function loadCategories() {
    try {
        const res = await fetch('/api/explore/categories');
        if (!res.ok) return;
        const data = await res.json();
        ExploreState.categories = data;
        renderCategories();
    } catch (err) {
        console.error('Failed to load categories:', err);
    }
}

async function loadPrompts() {
    if (ExploreState.loading) return;
    ExploreState.loading = true;
    showLoading(true);

    const params = new URLSearchParams({
        page: ExploreState.currentPage,
        limit: ExploreState.limit
    });

    if (ExploreState.activeCategory) {
        params.set('category', ExploreState.activeCategory);
    }
    if (ExploreState.searchQuery.trim()) {
        params.set('search', ExploreState.searchQuery.trim());
    }

    try {
        const res = await fetch(`/api/explore/prompts?${params}`);
        if (!res.ok) throw new Error('Failed to fetch prompts');
        const data = await res.json();

        ExploreState.prompts = data.prompts;
        ExploreState.total = data.total;
        ExploreState.totalPages = data.total_pages;
        ExploreState.currentPage = data.page;

        renderPrompts();
        renderPagination();
        updateResultsBar();
    } catch (err) {
        console.error('Failed to load prompts:', err);
        showEmptyState('Error loading prompts. Please try again.');
    } finally {
        ExploreState.loading = false;
        showLoading(false);
    }
}

async function loadPacks() {
    if (ExploreState.loading) return;
    ExploreState.loading = true;
    showLoading(true);

    const params = new URLSearchParams({
        page: ExploreState.currentPage,
        limit: ExploreState.limit
    });

    if (ExploreState.searchQuery.trim()) {
        params.set('search', ExploreState.searchQuery.trim());
    }

    try {
        const res = await fetch(`/api/explore/packs?${params}`);
        if (!res.ok) throw new Error('Failed to fetch packs');
        const data = await res.json();

        ExploreState.packs = data.packs;
        ExploreState.total = data.total;
        ExploreState.totalPages = data.pages;
        ExploreState.currentPage = data.page;

        renderPacks();
        renderPagination();
        updateResultsBar();
    } catch (err) {
        console.error('Failed to load packs:', err);
        showEmptyState('Error loading packs. Please try again.');
    } finally {
        ExploreState.loading = false;
        showLoading(false);
    }
}

// ============================================================
// RENDERING - Categories
// ============================================================

function renderCategories() {
    const container = document.getElementById('categoryChips');
    if (!container) return;

    // "All" chip
    let html = `<button class="category-chip active" data-category="" onclick="selectCategory(null, this)">
        <i class="fas fa-globe"></i> All
    </button>`;

    ExploreState.categories.forEach(cat => {
        // Skip age-restricted from main view
        if (cat.is_age_restricted) return;
        html += `<button class="category-chip" data-category="${cat.id}" onclick="selectCategory(${cat.id}, this)">
            <i class="fas ${cat.icon || 'fa-tag'}"></i> ${escapeHtml(cat.name)}
            <span class="chip-count">${cat.count}</span>
        </button>`;
    });

    // Add age-restricted as a separate toggle if exists
    const ageRestricted = ExploreState.categories.filter(c => c.is_age_restricted);
    if (ageRestricted.length > 0) {
        ageRestricted.forEach(cat => {
            html += `<button class="category-chip" data-category="${cat.id}" onclick="selectCategory(${cat.id}, this)" title="Age-restricted content">
                <i class="fas ${cat.icon || 'fa-tag'}"></i> ${escapeHtml(cat.name)}
                <span class="chip-count">${cat.count}</span>
            </button>`;
        });
    }

    container.innerHTML = html;
}

function selectCategory(categoryId, chipEl) {
    ExploreState.activeCategory = categoryId;
    ExploreState.currentPage = 1;

    // Update active chip visual
    document.querySelectorAll('.category-chip').forEach(c => c.classList.remove('active'));
    if (chipEl) chipEl.classList.add('active');

    loadPrompts();
}

// ============================================================
// RENDERING - Prompt Cards
// ============================================================

function renderPrompts() {
    const grid = document.getElementById('exploreGrid');
    if (!grid) return;

    if (ExploreState.prompts.length === 0) {
        showEmptyState('No prompts found matching your criteria.');
        return;
    }

    let html = '';
    ExploreState.prompts.forEach(prompt => {
        const avatarHtml = prompt.image_url
            ? `<img src="${escapeAttr(prompt.image_url)}" alt="" class="card-avatar" loading="lazy" onerror="this.outerHTML=generatePlaceholder('${escapeAttr(prompt.name)}')">`
            : generatePlaceholder(prompt.name);

        const descSnippet = prompt.description
            ? escapeHtml(prompt.description.substring(0, 120))
            : 'No description available.';

        const tagsHtml = (prompt.categories || [])
            .slice(0, 3)
            .map(c => `<span class="card-tag"><i class="fas ${c.icon || 'fa-tag'}"></i> ${escapeHtml(c.name)}</span>`)
            .join('');

        const paidBadge = prompt.is_paid ? '<span class="card-paid-badge">PRO</span>' : '';

        html += `<div class="prompt-card" onclick='openModal(${JSON.stringify(prompt).replace(/'/g, "&#39;")})'>
            ${paidBadge}
            ${avatarHtml}
            <div class="card-name">${escapeHtml(prompt.name)}</div>
            <div class="card-description">${descSnippet}</div>
            <div class="card-tags">${tagsHtml}</div>
        </div>`;
    });

    grid.innerHTML = html;
}

// ============================================================
// RENDERING - Pack Cards
// ============================================================

function renderPacks() {
    const grid = document.getElementById('exploreGrid');
    if (!grid) return;

    if (ExploreState.packs.length === 0) {
        showEmptyState('No packs found matching your criteria.');
        return;
    }

    let html = '';
    ExploreState.packs.forEach(pack => {
        const coverHtml = pack.has_cover_image
            ? `<div class="pack-card-cover"><img src="/api/packs/${pack.id}/cover/512" alt="" loading="lazy"></div>`
            : `<div class="pack-card-cover pack-cover-placeholder"><span>${escapeHtml(pack.name ? pack.name.charAt(0).toUpperCase() : '?')}</span></div>`;

        const priceLabel = pack.is_paid ? `$${Number(pack.price).toFixed(2)}` : 'FREE';
        const itemCount = pack.item_count || 0;
        const creator = pack.created_by_username || 'Unknown';

        const descSnippet = pack.description
            ? escapeHtml(pack.description.substring(0, 100))
            : '';

        html += `<div class="pack-card" onclick='openPackModal(${JSON.stringify(pack).replace(/'/g, "&#39;")})'>
            ${coverHtml}
            <div class="pack-card-body">
                <div class="card-name">${escapeHtml(pack.name)}</div>
                ${descSnippet ? `<div class="card-description">${descSnippet}</div>` : ''}
                <div class="pack-card-meta">
                    <span>${itemCount} prompt${itemCount !== 1 ? 's' : ''}</span>
                    <span>by @${escapeHtml(creator)}</span>
                </div>
                <div class="pack-card-price">${priceLabel}</div>
            </div>
        </div>`;
    });

    grid.innerHTML = html;
}

// ============================================================
// RENDERING - Shared Helpers
// ============================================================

function generatePlaceholder(name) {
    const initial = name ? name.charAt(0).toUpperCase() : '?';
    return `<div class="card-avatar-placeholder">${initial}</div>`;
}

function showEmptyState(message) {
    const grid = document.getElementById('exploreGrid');
    if (!grid) return;
    grid.innerHTML = `<div class="explore-empty" style="grid-column: 1 / -1;">
        <i class="fas fa-search"></i>
        <p>${escapeHtml(message)}</p>
    </div>`;
}

function showLoading(show) {
    const loader = document.getElementById('exploreLoader');
    const grid = document.getElementById('exploreGrid');
    if (loader) loader.style.display = show ? 'flex' : 'none';
    if (grid && show) grid.innerHTML = '';
}

function updateResultsBar() {
    const bar = document.getElementById('resultsInfo');
    if (!bar) return;
    if (ExploreState.total === 0) {
        bar.textContent = '';
        return;
    }
    const start = (ExploreState.currentPage - 1) * ExploreState.limit + 1;
    const end = Math.min(ExploreState.currentPage * ExploreState.limit, ExploreState.total);
    const label = ExploreState.activeTab === 'prompts' ? 'prompts' : 'packs';
    bar.textContent = `Showing ${start}-${end} of ${ExploreState.total} ${label}`;
}

// ============================================================
// RENDERING - Pagination
// ============================================================

function renderPagination() {
    const container = document.getElementById('explorePagination');
    if (!container) return;

    if (ExploreState.totalPages <= 1) {
        container.innerHTML = '';
        return;
    }

    const { currentPage, totalPages } = ExploreState;
    let html = '';

    // Previous button
    html += `<button class="page-btn" ${currentPage <= 1 ? 'disabled' : ''} onclick="goToPage(${currentPage - 1})">
        <i class="fas fa-chevron-left"></i>
    </button>`;

    // Page numbers with ellipsis
    const pages = getVisiblePages(currentPage, totalPages);
    pages.forEach(p => {
        if (p === '...') {
            html += `<span class="page-btn" style="cursor:default;border:none;">...</span>`;
        } else {
            html += `<button class="page-btn ${p === currentPage ? 'active' : ''}" onclick="goToPage(${p})">${p}</button>`;
        }
    });

    // Next button
    html += `<button class="page-btn" ${currentPage >= totalPages ? 'disabled' : ''} onclick="goToPage(${currentPage + 1})">
        <i class="fas fa-chevron-right"></i>
    </button>`;

    container.innerHTML = html;
}

function getVisiblePages(current, total) {
    if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1);

    const pages = [];
    pages.push(1);

    if (current > 3) pages.push('...');

    const start = Math.max(2, current - 1);
    const end = Math.min(total - 1, current + 1);
    for (let i = start; i <= end; i++) pages.push(i);

    if (current < total - 2) pages.push('...');

    pages.push(total);
    return pages;
}

function goToPage(page) {
    if (page < 1 || page > ExploreState.totalPages || page === ExploreState.currentPage) return;
    ExploreState.currentPage = page;
    if (ExploreState.activeTab === 'prompts') {
        loadPrompts();
    } else {
        loadPacks();
    }
    // Scroll to top of grid
    const hero = document.querySelector('.explore-hero');
    if (hero) hero.scrollIntoView({ behavior: 'smooth' });
}

// ============================================================
// DETAIL MODAL - Prompts
// ============================================================

function openModal(prompt) {
    const backdrop = document.getElementById('exploreModalBackdrop');
    if (!backdrop) return;

    const avatarHtml = prompt.image_fullsize_url || prompt.image_url
        ? `<img src="${escapeAttr(prompt.image_fullsize_url || prompt.image_url)}" alt="" class="modal-avatar" onerror="this.outerHTML=generateModalPlaceholder('${escapeAttr(prompt.name)}')">`
        : generateModalPlaceholder(prompt.name);

    const tagsHtml = (prompt.categories || [])
        .map(c => `<span class="modal-tag"><i class="fas ${c.icon || 'fa-tag'}"></i> ${escapeHtml(c.name)}</span>`)
        .join('');

    const description = prompt.description || 'No description available.';
    const creatorName = prompt.creator_name || 'Unknown';

    const landingUrl = prompt.public_id && prompt.slug
        ? `/p/${encodeURIComponent(prompt.public_id)}/${encodeURIComponent(prompt.slug)}/`
        : null;

    const landingBtn = landingUrl
        ? `<a href="${landingUrl}" target="_blank" class="modal-secondary-btn"><i class="fas fa-external-link-alt"></i> Landing Page</a>`
        : '';

    document.getElementById('modalContent').innerHTML = `
        <button class="modal-close-btn" onclick="closeModal()" title="Close"><i class="fas fa-times"></i></button>
        <div class="modal-header-section">
            ${avatarHtml}
            <div class="modal-info">
                <h2 class="modal-prompt-name">${escapeHtml(prompt.name)}</h2>
                <div class="modal-creator">by <span>@${escapeHtml(creatorName)}</span></div>
            </div>
        </div>
        <div class="modal-description">${escapeHtml(description)}</div>
        <div class="modal-tags">${tagsHtml}</div>
        <button class="modal-cta" onclick="chatWithPrompt(${prompt.id}, '${escapeAttr(prompt.name)}')">
            <i class="fas fa-comments"></i> Chat Now
        </button>
        <div class="modal-secondary-actions">
            ${landingBtn}
            <button class="modal-secondary-btn" onclick="sharePrompt('${escapeAttr(prompt.name)}', '${landingUrl || ''}')">
                <i class="fas fa-share-alt"></i> Share
            </button>
        </div>
    `;

    backdrop.classList.add('active');
    document.body.style.overflow = 'hidden';
}

// ============================================================
// DETAIL MODAL - Packs
// ============================================================

function openPackModal(pack) {
    const backdrop = document.getElementById('exploreModalBackdrop');
    if (!backdrop) return;

    const coverHtml = pack.has_cover_image
        ? `<img src="/api/packs/${pack.id}/cover/512" alt="" class="pack-modal-cover">`
        : `<div class="pack-modal-cover-placeholder"><span>${escapeHtml(pack.name ? pack.name.charAt(0).toUpperCase() : '?')}</span></div>`;

    const description = pack.description || '';
    const creator = pack.created_by_username || 'Unknown';
    const itemCount = pack.item_count || 0;
    const priceLabel = pack.is_paid ? `$${Number(pack.price).toFixed(2)}` : 'FREE';
    const slug = pack.slug || '';
    const publicId = pack.public_id || '';

    // Parse tags
    let tags = [];
    if (pack.tags) {
        try {
            tags = typeof pack.tags === 'string' ? JSON.parse(pack.tags) : pack.tags;
        } catch (e) { /* ignore */ }
    }
    const tagsHtml = tags.map(t => `<span class="modal-tag">#${escapeHtml(t)}</span>`).join('');

    const landingUrl = publicId && slug ? `/pack/${encodeURIComponent(publicId)}/${encodeURIComponent(slug)}/` : null;
    const landingBtn = landingUrl
        ? `<a href="${landingUrl}" target="_blank" class="modal-secondary-btn"><i class="fas fa-external-link-alt"></i> View Landing</a>`
        : '';

    document.getElementById('modalContent').innerHTML = `
        <button class="modal-close-btn" onclick="closeModal()" title="Close"><i class="fas fa-times"></i></button>
        <div class="pack-modal-header">
            ${coverHtml}
            <div class="modal-info">
                <h2 class="modal-prompt-name">${escapeHtml(pack.name)}</h2>
                <div class="modal-creator">by <span>@${escapeHtml(creator)}</span> &mdash; ${itemCount} prompt${itemCount !== 1 ? 's' : ''}</div>
                <div class="pack-modal-price">${priceLabel}</div>
            </div>
        </div>
        ${description ? `<div class="modal-description">${escapeHtml(description)}</div>` : ''}
        ${tagsHtml ? `<div class="modal-tags">${tagsHtml}</div>` : ''}
        <div class="pack-modal-prompts" id="packModalPrompts">
            <div class="explore-loading" style="padding:1rem 0"><div class="spinner"></div></div>
        </div>
        ${!pack.is_paid && pack.id ? `
        <button class="modal-cta" style="width:100%;cursor:pointer" onclick="claimFreePack(${pack.id}, '${landingUrl || ''}')">
            <i class="fas fa-rocket"></i> Get This Pack - FREE
        </button>` : pack.is_paid && pack.id ? `
        <button class="modal-cta" style="width:100%;cursor:pointer" id="packPurchaseBtn" onclick="purchasePack(${pack.id}, '${landingUrl || ''}')">
            <i class="fas fa-shopping-cart"></i> Get This Pack - ${priceLabel}
        </button>
        <div id="packPurchaseError" class="modal-error" style="display:none;color:#f04747;padding:0.5rem 0;text-align:center;font-size:0.9rem;"></div>` : landingUrl ? `
        <a href="${landingUrl}" target="_blank" class="modal-cta" style="text-decoration:none;text-align:center;display:block">
            <i class="fas fa-rocket"></i> Get This Pack
        </a>` : ''}
        <div class="modal-secondary-actions">
            ${landingBtn}
            <button class="modal-secondary-btn" onclick="sharePrompt('${escapeAttr(pack.name)}', '${landingUrl || ''}')">
                <i class="fas fa-share-alt"></i> Share
            </button>
        </div>
    `;

    backdrop.classList.add('active');
    document.body.style.overflow = 'hidden';

    // Load pack items asynchronously
    if (pack.id) loadPackModalItems(pack.id);
}

async function loadPackModalItems(packId) {
    const container = document.getElementById('packModalPrompts');
    if (!container) return;

    try {
        const res = await fetch(`/api/explore/packs/${packId}/items`);
        if (!res.ok) {
            container.innerHTML = '';
            return;
        }
        const items = await res.json();

        if (!items.length) {
            container.innerHTML = '';
            return;
        }

        let html = '<h3 class="pack-modal-prompts-title">Included:</h3>';
        items.forEach(item => {
            const initial = item.prompt_name ? item.prompt_name.charAt(0).toUpperCase() : '?';
            const desc = item.prompt_description ? escapeHtml(item.prompt_description.substring(0, 60)) : '';
            html += `<div class="pack-modal-prompt-row">
                <div class="pack-modal-prompt-avatar">${initial}</div>
                <div class="pack-modal-prompt-info">
                    <div class="pack-modal-prompt-name">${escapeHtml(item.prompt_name)}</div>
                    ${desc ? `<div class="pack-modal-prompt-desc">${desc}</div>` : ''}
                </div>
            </div>`;
        });

        container.innerHTML = html;
    } catch (err) {
        console.error('Failed to load pack items:', err);
        container.innerHTML = '';
    }
}

// ============================================================
// MODAL - Shared
// ============================================================

function generateModalPlaceholder(name) {
    const initial = name ? name.charAt(0).toUpperCase() : '?';
    return `<div class="modal-avatar-placeholder">${initial}</div>`;
}

function closeModal() {
    const backdrop = document.getElementById('exploreModalBackdrop');
    if (backdrop) {
        backdrop.classList.remove('active');
        document.body.style.overflow = '';
    }
}

async function chatWithPrompt(promptId, promptName) {
    try {
        const formData = new FormData();
        formData.append('prompt_id', promptId);

        const res = await fetch('/api/select-prompt', {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const err = await res.json();
            NotificationModal.error('Error', err.detail || 'Failed to select prompt');
            return;
        }

        // Redirect to chat and auto-start new conversation
        window.location.href = '/chat?autostart=1';
    } catch (err) {
        console.error('Failed to select prompt:', err);
        NotificationModal.error('Error', 'Connection error. Please try again.');
    }
}

async function claimFreePack(packId, landingUrl) {
    try {
        const res = await fetch(`/api/packs/${packId}/claim-free`, { method: 'POST' });
        if (res.status === 401) {
            // Not logged in: redirect to landing page for registration
            if (landingUrl) {
                window.location.href = landingUrl;
            } else {
                NotificationModal.warning('Login Required', 'Please log in to claim this pack.');
            }
            return;
        }
        if (!res.ok) {
            const err = await res.json();
            NotificationModal.error('Error', err.detail || 'Failed to claim pack');
            return;
        }
        const data = await res.json();
        window.location.href = data.redirect || '/chat';
    } catch (err) {
        console.error('Failed to claim pack:', err);
        NotificationModal.error('Error', 'Connection error. Please try again.');
    }
}

async function purchasePack(packId, landingUrl) {
    const btn = document.getElementById('packPurchaseBtn');
    const errEl = document.getElementById('packPurchaseError');
    if (errEl) errEl.style.display = 'none';

    // Disable button to prevent double-clicks
    let originalBtnHtml = '';
    if (btn) {
        originalBtnHtml = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    }

    try {
        const res = await fetch(`/api/packs/${packId}/purchase`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });

        if (res.status === 401) {
            // Not logged in: redirect to landing page for registration
            if (landingUrl) {
                window.location.href = landingUrl;
            } else {
                NotificationModal.warning('Login Required', 'Please log in to purchase this pack.');
            }
            return;
        }

        const data = await res.json();

        if (res.ok && data.checkout_url) {
            window.location.href = data.checkout_url;
            return;
        }
        if (res.ok && data.free_purchase) {
            window.location.href = data.redirect || '/chat';
            return;
        }
        if (res.ok && data.redirect) {
            window.location.href = data.redirect;
            return;
        }

        // Error: re-enable button so user can retry
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = originalBtnHtml;
        }
        const msg = data.detail || data.message || 'Purchase failed.';
        if (errEl) {
            errEl.textContent = msg;
            errEl.style.display = 'block';
        } else {
            NotificationModal.error('Error', msg);
        }
    } catch (err) {
        // Re-enable button on error so user can retry
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = originalBtnHtml;
        }
        console.error('Failed to purchase pack:', err);
        if (errEl) {
            errEl.textContent = 'Connection error. Please try again.';
            errEl.style.display = 'block';
        } else {
            NotificationModal.error('Error', 'Connection error. Please try again.');
        }
    }
}

async function sharePrompt(name, landingUrl) {
    const shareUrl = landingUrl || window.location.href;
    const shareText = `Check out "${name}" on Spark AI!`;

    if (navigator.share) {
        try {
            await navigator.share({ title: name, text: shareText, url: shareUrl });
        } catch (e) {
            // User cancelled share - no action needed
        }
    } else {
        // Fallback: copy to clipboard
        try {
            await navigator.clipboard.writeText(shareUrl);
            NotificationModal.toast('Link copied to clipboard!', 'success');
        } catch (e) {
            // Double fallback: show URL in a modal so user can select and copy
            const safeUrl = shareUrl.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            NotificationModal.info('Share Link', `<input class="form-control" value="${safeUrl}" readonly onclick="this.select()">`);
        }
    }
}

// ============================================================
// UTILITIES
// ============================================================

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function escapeAttr(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
