/* ============================================================
   PROMPT & PACK EXPLORER - Frontend Logic
   Vanilla JS: fetch, filter, paginate, detail modal
   ============================================================ */

const ExploreState = {
    activeTab: 'prompts',    // 'prompts' | 'packs'
    activeFilter: null,      // 'mine' | 'favorites' | null
    prompts: [],
    packs: [],
    categories: [],
    activeCategory: null,
    searchQuery: '',
    currentPage: 1,
    totalPages: 1,
    total: 0,
    limit: 24,
    loading: false,
    iframeActive: false,
    iframeLandingItems: [],  // items with landings (for left/right nav)
    iframeCurrentIndex: 0
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
        if (ExploreState.iframeActive) {
            if (e.key === 'Escape') closeLandingPreview();
            else if (e.key === 'ArrowLeft') navigatePreview(-1);
            else if (e.key === 'ArrowRight') navigatePreview(1);
            return;
        }
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
    ExploreState.activeFilter = null;

    // Reset search input
    const searchInput = document.getElementById('exploreSearch');
    if (searchInput) {
        searchInput.value = '';
        searchInput.placeholder = tab === 'prompts' ? 'Search prompts...' : 'Search packs...';
    }

    // Update tab buttons
    document.querySelectorAll('.explore-tab').forEach(t => t.classList.remove('active'));
    if (tabEl) tabEl.classList.add('active');

    // Render appropriate chips and load content
    if (tab === 'prompts') {
        renderCategories();
        loadPrompts();
    } else {
        renderPackChips();
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
    if (ExploreState.activeFilter === 'mine') {
        params.set('mine', '1');
    }
    if (ExploreState.activeFilter === 'favorites') {
        params.set('favorites', '1');
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
        buildLandingItemsList();
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

    if (ExploreState.activeFilter === 'mine') {
        params.set('mine', '1');
    }
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
        buildLandingItemsList();
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
    container.style.display = '';

    const noFilter = !ExploreState.activeFilter && !ExploreState.activeCategory;

    // "All" chip
    let html = `<button class="category-chip ${noFilter ? 'active' : ''}" onclick="selectCategory(null, this)">
        <i class="fas fa-globe"></i> All
    </button>`;

    // Special filter chips
    html += `<button class="category-chip ${ExploreState.activeFilter === 'mine' ? 'active' : ''}" onclick="selectFilter('mine', this)">
        <i class="fas fa-user"></i> My Prompts
    </button>`;
    html += `<button class="category-chip ${ExploreState.activeFilter === 'favorites' ? 'active' : ''}" onclick="selectFilter('favorites', this)">
        <i class="fas fa-star"></i> Favorites
    </button>`;

    // Visual divider
    html += '<span class="chip-divider"></span>';

    ExploreState.categories.forEach(cat => {
        if (cat.is_age_restricted) return;
        html += `<button class="category-chip ${ExploreState.activeCategory === cat.id ? 'active' : ''}" data-category="${cat.id}" onclick="selectCategory(${cat.id}, this)">
            <i class="fas ${cat.icon || 'fa-tag'}"></i> ${escapeHtml(cat.name)}
            <span class="chip-count">${cat.count}</span>
        </button>`;
    });

    const ageRestricted = ExploreState.categories.filter(c => c.is_age_restricted);
    if (ageRestricted.length > 0) {
        ageRestricted.forEach(cat => {
            html += `<button class="category-chip ${ExploreState.activeCategory === cat.id ? 'active' : ''}" data-category="${cat.id}" onclick="selectCategory(${cat.id}, this)" title="Age-restricted content">
                <i class="fas ${cat.icon || 'fa-tag'}"></i> ${escapeHtml(cat.name)}
                <span class="chip-count">${cat.count}</span>
            </button>`;
        });
    }

    container.innerHTML = html;
}

function renderPackChips() {
    const container = document.getElementById('categoryChips');
    if (!container) return;
    container.style.display = '';

    const noFilter = !ExploreState.activeFilter;

    let html = `<button class="category-chip ${noFilter ? 'active' : ''}" onclick="selectFilter(null, this)">
        <i class="fas fa-globe"></i> All
    </button>`;
    html += `<button class="category-chip ${ExploreState.activeFilter === 'mine' ? 'active' : ''}" onclick="selectFilter('mine', this)">
        <i class="fas fa-user"></i> My Packs
    </button>`;

    container.innerHTML = html;
}

function selectCategory(categoryId, chipEl) {
    ExploreState.activeCategory = categoryId;
    ExploreState.activeFilter = null;
    ExploreState.currentPage = 1;

    // Update active chip visual
    document.querySelectorAll('.category-chip').forEach(c => c.classList.remove('active'));
    if (chipEl) chipEl.classList.add('active');

    loadPrompts();
}

function selectFilter(filter, chipEl) {
    ExploreState.activeFilter = filter;
    ExploreState.activeCategory = null;
    ExploreState.currentPage = 1;

    // Update active chip visual
    document.querySelectorAll('.category-chip').forEach(c => c.classList.remove('active'));
    if (chipEl) chipEl.classList.add('active');

    if (ExploreState.activeTab === 'prompts') {
        loadPrompts();
    } else {
        loadPacks();
    }
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

        let paidBadge = '';
        if (prompt.purchase_price !== null && prompt.purchase_price !== undefined) {
            paidBadge = prompt.purchase_price > 0
                ? `<span class="card-paid-badge">$${Number(prompt.purchase_price).toFixed(2)}</span>`
                : '<span class="card-paid-badge" style="background:var(--success,#43b581)">FREE</span>';
        } else if (prompt.is_paid) {
            paidBadge = '<span class="card-paid-badge">PRO</span>';
        }

        let visibilityBadge = '';
        if (ExploreState.activeFilter === 'mine') {
            if (!prompt.is_public) {
                visibilityBadge = '<span class="card-visibility-badge private"><i class="fas fa-lock"></i> Private</span>';
            } else if (prompt.is_unlisted) {
                visibilityBadge = '<span class="card-visibility-badge unlisted"><i class="fas fa-eye-slash"></i> Unlisted</span>';
            }
        }

        const favClass = prompt.is_favorite ? 'is-favorite' : '';
        const favIcon = prompt.is_favorite ? 'fas' : 'far';

        html += `<div class="prompt-card" onclick='openExploreItem(${JSON.stringify(prompt).replace(/'/g, "&#39;")}, "prompt")'>
            ${paidBadge}
            ${visibilityBadge}
            <button class="explore-fav-btn ${favClass}" onclick="event.stopPropagation(); toggleExploreFavorite(${prompt.id}, this)" title="${prompt.is_favorite ? 'Remove from favorites' : 'Add to favorites'}">
                <i class="${favIcon} fa-star"></i>
            </button>
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

        let visibilityBadge = '';
        if (ExploreState.activeFilter === 'mine') {
            if (pack.status === 'draft') {
                visibilityBadge = '<span class="card-visibility-badge draft"><i class="fas fa-pencil-alt"></i> Draft</span>';
            } else if (!pack.is_public) {
                visibilityBadge = '<span class="card-visibility-badge private"><i class="fas fa-lock"></i> Private</span>';
            }
        }

        html += `<div class="pack-card" onclick='openExploreItem(${JSON.stringify(pack).replace(/'/g, "&#39;")}, "pack")'>
            ${visibilityBadge}
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

    const modalFavClass = prompt.is_favorite ? 'is-favorite' : '';
    const modalFavIcon = prompt.is_favorite ? 'fas' : 'far';

    let visibilityNotice = '';
    if (ExploreState.activeFilter === 'mine') {
        if (!prompt.is_public) {
            visibilityNotice = '<div class="visibility-notice"><i class="fas fa-lock"></i> This prompt is private — only you can see it</div>';
        } else if (prompt.is_unlisted) {
            visibilityNotice = '<div class="visibility-notice"><i class="fas fa-eye-slash"></i> This prompt is unlisted — only accessible via direct link</div>';
        }
    }

    document.getElementById('modalContent').innerHTML = `
        <button class="modal-close-btn" onclick="closeModal()" title="Close"><i class="fas fa-times"></i></button>
        <button class="modal-fav-btn ${modalFavClass}" onclick="toggleExploreFavorite(${prompt.id}, this)" title="${prompt.is_favorite ? 'Remove from favorites' : 'Add to favorites'}">
            <i class="${modalFavIcon} fa-star"></i>
        </button>
        <div class="modal-header-section">
            ${avatarHtml}
            <div class="modal-info">
                <h2 class="modal-prompt-name">${escapeHtml(prompt.name)}</h2>
                <div class="modal-creator">by <span>@${escapeHtml(creatorName)}</span></div>
            </div>
        </div>
        ${visibilityNotice}
        <div class="modal-description">${escapeHtml(description)}</div>
        <div class="modal-tags">${tagsHtml}</div>
        ${getPromptCTA(prompt)}
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

function getPromptCTA(prompt) {
    // If user already has access or is the owner, show "Chat Now"
    if (prompt.user_has_access || prompt.is_mine) {
        return `<button class="modal-cta" onclick="chatWithPrompt(${prompt.id}, '${escapeAttr(prompt.name)}')">
            <i class="fas fa-comments"></i> Chat Now
        </button>`;
    }
    // If purchase_price is set and > 0, show buy button
    if (prompt.purchase_price !== null && prompt.purchase_price !== undefined && prompt.purchase_price > 0) {
        return `<button class="modal-cta" onclick="purchasePrompt(${prompt.id})">
            <i class="fas fa-shopping-cart"></i> Buy for $${Number(prompt.purchase_price).toFixed(2)}
        </button>`;
    }
    // If purchase_price === 0, show free access button
    if (prompt.purchase_price !== null && prompt.purchase_price !== undefined && prompt.purchase_price === 0) {
        return `<button class="modal-cta" onclick="purchasePrompt(${prompt.id})">
            <i class="fas fa-unlock"></i> Get Access &mdash; Free
        </button>`;
    }
    // Default: Chat Now (public access)
    return `<button class="modal-cta" onclick="chatWithPrompt(${prompt.id}, '${escapeAttr(prompt.name)}')">
        <i class="fas fa-comments"></i> Chat Now
    </button>`;
}

async function purchasePrompt(promptId) {
    try {
        const res = await fetch('/api/prompts/' + promptId + '/purchase', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            credentials: 'include'
        });
        const data = await res.json();
        if (data.checkout_url) {
            window.location = data.checkout_url;
        } else if (data.free_purchase) {
            window.location = '/chat';
        } else if (data.redirect) {
            window.location = data.redirect;
        } else if (data.message) {
            alert(data.message);
        } else if (data.detail) {
            alert(data.detail);
        }
    } catch (err) {
        console.error('Purchase failed:', err);
        alert('Purchase failed. Please try again.');
    }
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

    let packVisibilityNotice = '';
    if (ExploreState.activeFilter === 'mine') {
        if (pack.status === 'draft') {
            packVisibilityNotice = '<div class="visibility-notice"><i class="fas fa-pencil-alt"></i> This pack is a draft — not yet published</div>';
        } else if (!pack.is_public) {
            packVisibilityNotice = '<div class="visibility-notice"><i class="fas fa-lock"></i> This pack is private — only you can see it</div>';
        }
    }

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
        ${packVisibilityNotice}
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
    const shareText = `Check out "${name}" on Aurvek AI!`;

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
// FAVORITES TOGGLE
// ============================================================

async function toggleExploreFavorite(promptId, btnEl) {
    // Optimistic UI update
    const isFav = btnEl.classList.contains('is-favorite');
    const icon = btnEl.querySelector('i');

    btnEl.classList.toggle('is-favorite');
    icon.className = isFav ? 'far fa-star' : 'fas fa-star';
    btnEl.title = isFav ? 'Add to favorites' : 'Remove from favorites';

    try {
        const res = await fetch(`/api/home/favorites/${promptId}`, { method: 'POST' });
        if (!res.ok) throw new Error('Failed to toggle favorite');
        const data = await res.json();

        // Update local state
        const prompt = ExploreState.prompts.find(p => p.id === promptId);
        if (prompt) prompt.is_favorite = data.is_favorite;

        // Sync all buttons for this prompt (card + modal may both be visible)
        document.querySelectorAll(`.explore-fav-btn, .modal-fav-btn`).forEach(btn => {
            // Find buttons that match this promptId by checking onclick
            const onclickAttr = btn.getAttribute('onclick') || '';
            if (onclickAttr.includes(`toggleExploreFavorite(${promptId},`)) {
                const ico = btn.querySelector('i');
                if (data.is_favorite) {
                    btn.classList.add('is-favorite');
                    ico.className = 'fas fa-star';
                    btn.title = 'Remove from favorites';
                } else {
                    btn.classList.remove('is-favorite');
                    ico.className = 'far fa-star';
                    btn.title = 'Add to favorites';
                }
            }
        });

        // Update badge visibility in the card
        renderPrompts();
    } catch (err) {
        console.error('Failed to toggle favorite:', err);
        // Revert optimistic update
        btnEl.classList.toggle('is-favorite');
        icon.className = isFav ? 'fas fa-star' : 'far fa-star';
        btnEl.title = isFav ? 'Remove from favorites' : 'Add to favorites';
    }
}

// ============================================================
// LANDING PREVIEW OVERLAY
// ============================================================

function buildLandingItemsList() {
    // Build list of items with landing pages for iframe navigation
    if (ExploreState.activeTab === 'prompts') {
        ExploreState.iframeLandingItems = ExploreState.prompts
            .filter(p => p.has_landing_page && p.public_id && p.slug)
            .map(p => ({...p, _type: 'prompt'}));
    } else {
        ExploreState.iframeLandingItems = ExploreState.packs
            .filter(p => (p.has_landing_page || p.has_custom_landing) && p.public_id && p.slug)
            .map(p => ({...p, _type: 'pack'}));
    }
}

function openExploreItem(item, type) {
    // Dispatcher: iframe preview for items with landing, modal for others
    if (type === 'prompt') {
        if (item.has_landing_page && item.public_id && item.slug) {
            openLandingPreview(item, type);
        } else {
            openModal(item);
        }
    } else {
        if ((item.has_landing_page || item.has_custom_landing) && item.public_id && item.slug) {
            openLandingPreview(item, type);
        } else {
            openPackModal(item);
        }
    }
}

function openLandingPreview(item, type) {
    const overlay = document.getElementById('landingPreviewOverlay');
    if (!overlay) return;

    // Find index in landing items list
    const idx = ExploreState.iframeLandingItems.findIndex(i => i.id === item.id && i._type === type);
    ExploreState.iframeCurrentIndex = idx >= 0 ? idx : 0;
    ExploreState.iframeActive = true;

    // Build URL — append ?preview=1 to skip custom domain redirects and analytics
    const url = type === 'prompt'
        ? `/p/${encodeURIComponent(item.public_id)}/${encodeURIComponent(item.slug)}/?preview=1`
        : `/pack/${encodeURIComponent(item.public_id)}/${encodeURIComponent(item.slug)}/?preview=1`;

    const iframe = document.getElementById('landingPreviewIframe');
    iframe.src = url;

    updatePreviewBar(item, type);
    updatePreviewNavigation();

    overlay.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeLandingPreview() {
    const overlay = document.getElementById('landingPreviewOverlay');
    if (!overlay) return;

    overlay.classList.remove('active');
    document.body.style.overflow = '';
    ExploreState.iframeActive = false;

    // Clear iframe to stop any running content
    const iframe = document.getElementById('landingPreviewIframe');
    if (iframe) iframe.src = 'about:blank';
}

function navigatePreview(direction) {
    const items = ExploreState.iframeLandingItems;
    if (items.length <= 1) return;

    let newIndex = ExploreState.iframeCurrentIndex + direction;
    // Wrap around
    if (newIndex < 0) newIndex = items.length - 1;
    if (newIndex >= items.length) newIndex = 0;

    ExploreState.iframeCurrentIndex = newIndex;
    const item = items[newIndex];
    const type = item._type;

    const url = type === 'prompt'
        ? `/p/${encodeURIComponent(item.public_id)}/${encodeURIComponent(item.slug)}/?preview=1`
        : `/pack/${encodeURIComponent(item.public_id)}/${encodeURIComponent(item.slug)}/?preview=1`;

    document.getElementById('landingPreviewIframe').src = url;
    updatePreviewBar(item, type);
    updatePreviewNavigation();
}

function updatePreviewBar(item, type) {
    const nameEl = document.getElementById('previewItemName');
    const creatorEl = document.getElementById('previewItemCreator');
    const ctaBtn = document.getElementById('previewCtaBtn');

    if (nameEl) nameEl.textContent = item.name || '';
    if (creatorEl) creatorEl.textContent = '@' + (item.creator_name || item.created_by_username || 'Unknown');

    // CTA button
    if (ctaBtn) {
        if (type === 'prompt') {
            if (item.user_has_access || item.is_mine) {
                ctaBtn.textContent = 'Chat Now';
                ctaBtn.onclick = () => { closeLandingPreview(); chatWithPrompt(item.id, item.name); };
            } else if (item.purchase_price !== null && item.purchase_price !== undefined && item.purchase_price > 0) {
                ctaBtn.textContent = 'Buy $' + Number(item.purchase_price).toFixed(2);
                ctaBtn.onclick = () => { closeLandingPreview(); purchasePrompt(item.id); };
            } else if (item.purchase_price === 0) {
                ctaBtn.textContent = 'Get Free';
                ctaBtn.onclick = () => { closeLandingPreview(); purchasePrompt(item.id); };
            } else {
                ctaBtn.textContent = 'Chat Now';
                ctaBtn.onclick = () => { closeLandingPreview(); chatWithPrompt(item.id, item.name); };
            }
        } else {
            // Pack
            if (item.is_paid) {
                ctaBtn.textContent = 'Buy $' + Number(item.price).toFixed(2);
                ctaBtn.onclick = () => { closeLandingPreview(); purchasePack(item.id, ''); };
            } else {
                ctaBtn.textContent = 'Get Free';
                ctaBtn.onclick = () => { closeLandingPreview(); claimFreePack(item.id, ''); };
            }
        }
    }
}

function updatePreviewNavigation() {
    const items = ExploreState.iframeLandingItems;
    const counter = document.getElementById('previewCounter');
    const prevBtn = document.getElementById('previewPrevBtn');
    const nextBtn = document.getElementById('previewNextBtn');

    if (counter) {
        counter.textContent = items.length > 1
            ? `${ExploreState.iframeCurrentIndex + 1} / ${items.length}`
            : '';
    }

    if (prevBtn) prevBtn.style.display = items.length > 1 ? '' : 'none';
    if (nextBtn) nextBtn.style.display = items.length > 1 ? '' : 'none';
}

function previewCtaAction() {
    // Fallback -- individual CTA buttons set their own onclick
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
