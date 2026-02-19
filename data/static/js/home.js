'use strict';

(function() {
    let homeData = null;
    const canManage = window._homeConfig && window._homeConfig.canManage;

    document.addEventListener('DOMContentLoaded', init);

    async function init() {
        try {
            const resp = await fetch('/api/home');
            if (resp.status === 401) {
                window.location.href = '/login';
                return;
            }
            if (!resp.ok) throw new Error('Failed to load home data');
            homeData = await resp.json();
            render();
        } catch (e) {
            console.error('Home init error:', e);
        }
    }

    function render() {
        applyBranding();
        renderGreeting();
        renderChatInput();
        renderFavorites();
        renderLatestPrompts();
        renderMyLibrary('prompts');
        bindEvents();
    }

    // ==================== Branding ====================
    function applyBranding() {
        var b = homeData.branding;
        if (b && b.brand_color_primary) {
            document.documentElement.style.setProperty('--home-accent', b.brand_color_primary);
        }
    }

    // ==================== Greeting ====================
    function renderGreeting() {
        var titleEl = document.getElementById('home-greeting-title');
        var subtitleEl = document.getElementById('home-greeting-subtitle');
        var hour = new Date().getHours();
        var greeting;
        if (hour < 6) greeting = 'Good night';
        else if (hour < 12) greeting = 'Good morning';
        else if (hour < 18) greeting = 'Good afternoon';
        else greeting = 'Good evening';

        var b = homeData.branding;
        if (b && b.company_name) {
            titleEl.textContent = b.company_name;
            subtitleEl.textContent = greeting + ', ' + homeData.user.username;
        } else {
            titleEl.textContent = greeting + ', ' + homeData.user.username;
            subtitleEl.textContent = 'Ready to start';
        }
    }

    // ==================== Chat Input ====================
    function renderChatInput() {
        var select = document.getElementById('home-prompt-select');
        var packs = homeData.packs || [];
        var prompts = homeData.prompts || [];
        var favIds = homeData.favorites || [];

        // Favorites optgroup
        var favPrompts = prompts.filter(function(p) { return favIds.includes(p.id); });
        if (favPrompts.length) {
            var favGroup = document.createElement('optgroup');
            favGroup.label = '\u2605 Favorites';
            favPrompts.forEach(function(p) {
                var opt = document.createElement('option');
                opt.value = String(p.id);
                opt.textContent = p.name;
                favGroup.appendChild(opt);
            });
            select.appendChild(favGroup);
        }

        // My Prompts optgroup
        var myPrompts = prompts.filter(function(p) { return !!p.is_mine; });
        if (myPrompts.length) {
            var myGroup = document.createElement('optgroup');
            myGroup.label = 'My Prompts';
            myPrompts.forEach(function(p) {
                var opt = document.createElement('option');
                opt.value = String(p.id);
                opt.textContent = p.name;
                myGroup.appendChild(opt);
            });
            select.appendChild(myGroup);
        }

        // All Prompts optgroup
        if (prompts.length) {
            var allGroup = document.createElement('optgroup');
            allGroup.label = 'All Prompts';
            prompts.forEach(function(p) {
                var opt = document.createElement('option');
                opt.value = String(p.id);
                opt.textContent = p.name;
                allGroup.appendChild(opt);
            });
            select.appendChild(allGroup);
        }

        // Packs optgroup
        var activePacks = packs.filter(function(pk) { return pk.prompt_count > 0; });
        if (activePacks.length) {
            var packGroup = document.createElement('optgroup');
            packGroup.label = 'Packs';
            activePacks.forEach(function(pk) {
                var opt = document.createElement('option');
                opt.value = 'pack:' + pk.id;
                opt.textContent = pk.name;
                packGroup.appendChild(opt);
            });
            select.appendChild(packGroup);
        }

        // Auto-select if only one option after the default
        if (select.options.length === 2) {
            select.selectedIndex = 1;
        }
    }

    // ==================== Favorites ====================
    function renderFavorites() {
        var favIds = homeData.favorites || [];
        if (!favIds.length) return;

        var allPrompts = homeData.prompts || [];
        // Also check packs' prompts if needed - for now just loose prompts
        var favPrompts = allPrompts.filter(function(p) { return favIds.includes(p.id); });
        if (!favPrompts.length) return;

        var container = document.getElementById('home-favorites');
        container.style.display = '';
        container.innerHTML = favPrompts.map(function(p) {
            return '<button class="home-fav-chip" data-prompt-id="' + p.id + '" title="' + escapeHtml(p.name) + '">'
                + (p.image_url ? '<img src="' + escapeHtml(p.image_url) + '" alt="" class="home-fav-avatar">' : '<i class="fas fa-star"></i>')
                + '<span>' + escapeHtml(p.name) + '</span>'
                + '</button>';
        }).join('');
    }

    // ==================== Latest Prompts (left column) ====================
    function renderLatestPrompts() {
        var items = homeData.latest_prompts || [];
        if (!items.length) {
            // Hide left column, make right column full width
            document.getElementById('home-columns').classList.add('home-columns--single');
            return;
        }

        var section = document.getElementById('home-latest');
        section.style.display = '';

        var now = new Date();
        var list = document.getElementById('home-latest-list');
        list.innerHTML = items.map(function(p) {
            var isNew = false;
            if (p.created_at) {
                var created = new Date(p.created_at + 'Z');
                isNew = (now - created) < 7 * 24 * 60 * 60 * 1000;
            }
            var badge = isNew ? '<span class="home-item-badge">NEW</span>' : '';
            var actions = '';
            if (p.has_welcome) {
                actions = '<div class="home-item-actions">'
                    + '<button class="home-item-icon" onclick="event.stopPropagation(); navigateToWelcome(\'prompt\', ' + p.id + ')" title="Visit landing page"><i class="fas fa-globe"></i></button>'
                    + '</div>';
            }
            return '<div class="home-item" data-prompt-id="' + p.id + '">'
                + '<a href="/explore" class="home-item-name">' + escapeHtml(p.name) + '</a>'
                + badge
                + actions
                + '</div>';
        }).join('');
    }

    // ==================== My Library (right column) ====================
    function renderMyLibrary(tab) {
        var list = document.getElementById('home-library-list');
        var emptyEl = document.getElementById('home-library-empty');
        var viewAllBtn = document.getElementById('home-view-all-btn');
        var items;
        var maxItems = 5;

        if (tab === 'packs') {
            items = homeData.packs || [];
        } else {
            items = homeData.prompts || [];
        }

        if (!items.length) {
            list.innerHTML = '';
            emptyEl.style.display = '';
            viewAllBtn.style.display = 'none';
            return;
        }

        emptyEl.style.display = 'none';
        var displayItems = items.slice(0, maxItems);

        list.innerHTML = displayItems.map(function(item) {
            var name = escapeHtml(item.name);
            var hasWelcome = item.has_welcome;
            var type = tab === 'packs' ? 'pack' : 'prompt';
            var actions = '';
            if (hasWelcome) {
                actions += '<button class="home-item-icon" onclick="navigateToWelcome(\'' + type + '\', ' + item.id + ')" title="Visit landing page"><i class="fas fa-globe"></i></button>';
            }
            if (tab === 'prompts') {
                actions += '<button class="home-item-icon" onclick="startChatWithPrompt(' + item.id + ')" title="Start chat"><i class="fas fa-comment-dots"></i></button>';
            }
            return '<div class="home-item">'
                + '<span class="home-item-name">' + name + '</span>'
                + '<div class="home-item-actions">' + actions + '</div>'
                + '</div>';
        }).join('');

        viewAllBtn.style.display = items.length > maxItems ? '' : 'none';
    }

    // ==================== View All Modal ====================
    function openViewAllModal(tab) {
        var overlay = document.getElementById('home-modal-overlay');
        overlay.style.display = '';
        renderModalList(tab || 'prompts');
        // Set active tab
        var tabs = document.querySelectorAll('#home-modal-tabs .home-tab');
        tabs.forEach(function(t) {
            t.classList.toggle('active', t.getAttribute('data-tab') === (tab || 'prompts'));
        });
        var title = document.querySelector('.home-modal-title');
        if (title) title.textContent = (tab === 'packs') ? 'All Packs' : 'All Prompts';
    }

    function renderModalList(tab) {
        var list = document.getElementById('home-modal-list');
        var items = (tab === 'packs') ? (homeData.packs || []) : (homeData.prompts || []);

        list.innerHTML = items.map(function(item) {
            var name = escapeHtml(item.name);
            var hasWelcome = item.has_welcome;
            var type = tab === 'packs' ? 'pack' : 'prompt';
            var actions = '';
            if (hasWelcome) {
                actions += '<button class="home-item-icon" onclick="navigateToWelcome(\'' + type + '\', ' + item.id + ')" title="Visit landing page"><i class="fas fa-globe"></i></button>';
            }
            if (tab === 'prompts') {
                actions += '<button class="home-item-icon" onclick="startChatWithPrompt(' + item.id + ')" title="Start chat"><i class="fas fa-comment-dots"></i></button>';
            }
            return '<div class="home-item">'
                + '<span class="home-item-name">' + name + '</span>'
                + '<div class="home-item-actions">' + actions + '</div>'
                + '</div>';
        }).join('');
    }

    function closeModal() {
        document.getElementById('home-modal-overlay').style.display = 'none';
    }

    // ==================== Events ====================
    function bindEvents() {
        // Chat input: Enter key and send button
        var input = document.getElementById('home-input');
        var sendBtn = document.getElementById('home-send-btn');

        function handleSend() {
            var text = input.value.trim();
            var select = document.getElementById('home-prompt-select');
            var selectedValue = select.value;

            if (!selectedValue) {
                select.focus();
                select.classList.add('home-prompt-select--highlight');
                setTimeout(function() { select.classList.remove('home-prompt-select--highlight'); }, 1500);
                return;
            }

            // Determine prompt ID (handle pack:id format)
            var promptId = selectedValue;
            if (selectedValue.startsWith('pack:')) {
                // For packs, we can't start a chat directly - navigate to welcome
                var packId = selectedValue.split(':')[1];
                navigateToWelcome('pack', parseInt(packId));
                return;
            }

            sessionStorage.setItem('home_start_prompt_id', promptId);
            if (text) {
                sessionStorage.setItem('home_start_message', text);
            }
            window.location.href = '/chat';
        }

        input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
            }
        });

        sendBtn.addEventListener('click', handleSend);

        // Favorite chips: click auto-selects in dropdown
        var favsContainer = document.getElementById('home-favorites');
        favsContainer.addEventListener('click', function(e) {
            var chip = e.target.closest('.home-fav-chip');
            if (!chip) return;
            var promptId = chip.getAttribute('data-prompt-id');
            var select = document.getElementById('home-prompt-select');
            // Find and select the matching option
            for (var i = 0; i < select.options.length; i++) {
                if (select.options[i].value === promptId) {
                    select.selectedIndex = i;
                    break;
                }
            }
            // Visual feedback on chips
            favsContainer.querySelectorAll('.home-fav-chip').forEach(function(c) {
                c.classList.remove('active');
            });
            chip.classList.add('active');
            // Focus the input
            document.getElementById('home-input').focus();
        });

        // Tab toggles (library)
        var tabToggle = document.getElementById('home-tab-toggle');
        tabToggle.addEventListener('click', function(e) {
            var btn = e.target.closest('.home-tab');
            if (!btn) return;
            var tab = btn.getAttribute('data-tab');
            tabToggle.querySelectorAll('.home-tab').forEach(function(t) { t.classList.remove('active'); });
            btn.classList.add('active');
            renderMyLibrary(tab);
        });

        // View All button
        document.getElementById('home-view-all-btn').addEventListener('click', function() {
            var activeTab = tabToggle.querySelector('.home-tab.active');
            openViewAllModal(activeTab ? activeTab.getAttribute('data-tab') : 'prompts');
        });

        // Modal close
        document.getElementById('home-modal-close').addEventListener('click', closeModal);
        document.getElementById('home-modal-overlay').addEventListener('click', function(e) {
            if (e.target === this) closeModal();
        });
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') closeModal();
        });

        // Modal tabs
        var modalTabs = document.getElementById('home-modal-tabs');
        modalTabs.addEventListener('click', function(e) {
            var btn = e.target.closest('.home-tab');
            if (!btn) return;
            var tab = btn.getAttribute('data-tab');
            modalTabs.querySelectorAll('.home-tab').forEach(function(t) { t.classList.remove('active'); });
            btn.classList.add('active');
            var title = document.querySelector('.home-modal-title');
            if (title) title.textContent = (tab === 'packs') ? 'All Packs' : 'All Prompts';
            renderModalList(tab);
        });
    }

    // ==================== Navigation ====================

    window.startChatWithPrompt = function(promptId) {
        sessionStorage.setItem('home_start_prompt_id', promptId);
        window.location.href = '/chat';
    };

    window.openChat = function(conversationId) {
        sessionStorage.setItem('home_open_conversation_id', conversationId);
        window.location.href = '/chat';
    };

    window.navigateToWelcome = function(type, id) {
        window.location.href = '/welcome/' + type + '/' + id;
    };

    window.toggleFavorite = async function(promptId, btnEl) {
        var isFav = btnEl.classList.contains('is-favorite');
        btnEl.classList.toggle('is-favorite');
        var icon = btnEl.querySelector('i');
        if (icon) {
            icon.className = isFav ? 'far fa-star' : 'fas fa-star';
        }
        try {
            var resp = await fetch('/api/home/favorites/' + promptId, { method: 'POST' });
            if (!resp.ok) throw new Error('Failed');
            var data = await resp.json();
            if (data.is_favorite) {
                if (!homeData.favorites.includes(promptId)) homeData.favorites.push(promptId);
            } else {
                homeData.favorites = homeData.favorites.filter(function(id) { return id !== promptId; });
            }
        } catch (e) {
            btnEl.classList.toggle('is-favorite');
            if (icon) icon.className = isFav ? 'fas fa-star' : 'far fa-star';
        }
    };

    // ==================== Utility ====================

    function escapeHtml(text) {
        if (!text) return '';
        var div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function formatRelativeDate(dateStr) {
        try {
            var date = new Date(dateStr + 'Z');
            var now = new Date();
            var diff = now - date;
            var mins = Math.floor(diff / 60000);
            if (mins < 1) return 'Just now';
            if (mins < 60) return mins + 'm ago';
            var hours = Math.floor(mins / 60);
            if (hours < 24) return hours + 'h ago';
            var days = Math.floor(hours / 24);
            if (days < 7) return days + 'd ago';
            return date.toLocaleDateString();
        } catch (e) {
            return '';
        }
    }

    function formatNumber(n) {
        if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
        if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
        return String(n);
    }

})();
