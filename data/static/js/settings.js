/**
 * Settings page - Tab management, URL hash deep-linking, lazy initialization
 */
(function() {
    'use strict';

    const TAB_MAP = {
        '#profile': 'profile-tab',
        '#usage': 'usage-tab',
        '#api-keys': 'api-keys-tab'
    };

    const initialized = {
        profile: false,
        usage: false,
        'api-keys': false
    };

    let chartJsLoaded = false;

    // Activate tab from URL hash
    function activateFromHash() {
        const hash = window.location.hash || '#profile';
        const tabId = TAB_MAP[hash];
        if (tabId) {
            const tabEl = document.getElementById(tabId);
            if (tabEl) {
                const tab = new bootstrap.Tab(tabEl);
                tab.show();
            }
        }
    }

    // Update URL hash on tab change
    function setupHashSync() {
        const tabEls = document.querySelectorAll('#settingsTabs button[data-bs-toggle="tab"]');
        tabEls.forEach(tabEl => {
            tabEl.addEventListener('shown.bs.tab', function(event) {
                const target = event.target.getAttribute('data-bs-target');
                history.replaceState(null, '', target);
                initTab(target.replace('#', ''));
            });
        });
    }

    // Lazy init each tab on first view
    function initTab(tabName) {
        if (initialized[tabName]) return;
        initialized[tabName] = true;

        switch(tabName) {
            case 'profile':
                // edit_profile.js initializes on DOMContentLoaded, already fired
                break;
            case 'usage':
                loadUsageTab();
                break;
            case 'api-keys':
                // api-credentials.js initializes on DOMContentLoaded, already fired
                // But if it hasn't been visible yet, we may need to trigger it
                break;
        }
    }

    // --- Usage Tab (adapted from my_usage.html inline JS) ---
    let spendingChart = null;

    function loadUsageTab() {
        // Load Chart.js dynamically if not loaded
        if (!chartJsLoaded) {
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js';
            script.onload = function() {
                chartJsLoaded = true;
                loadUsageData();
            };
            document.head.appendChild(script);
        } else {
            loadUsageData();
        }
    }

    async function loadUsageData() {
        const dateRangeEl = document.getElementById('usageDateRange');
        if (!dateRangeEl) return;
        const days = dateRangeEl.value;
        const params = new URLSearchParams();
        if (days !== 'all') params.append('days', days);

        try {
            const response = await fetch('/api/my-usage?' + params.toString());
            if (!response.ok) throw new Error('Failed to load data');
            const data = await response.json();

            updateBalance(data.balance);
            updateStats(data.stats);
            updateUsageByType(data.by_type);
            updateChart(data.daily);
            updateDailyBreakdown(data.daily);
        } catch (error) {
            console.error('Error loading usage data:', error);
            if (typeof NotificationModal !== 'undefined') {
                NotificationModal.error('Error', 'Failed to load usage data');
            }
        }
    }

    function updateBalance(balance) {
        const el = document.getElementById('usageCurrentBalance');
        if (el) el.textContent = '$' + (balance || 0).toFixed(2);
    }

    function updateStats(stats) {
        const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
        setEl('statOperations', formatNumber(stats.total_operations || 0));
        setEl('statTokens', formatNumber(stats.total_tokens || 0));
        setEl('statTokensBreakdown', formatNumber(stats.tokens_in || 0) + ' in / ' + formatNumber(stats.tokens_out || 0) + ' out');
        setEl('statCost', '$' + (stats.total_cost || 0).toFixed(2));
        setEl('statAvgDaily', '$' + (stats.avg_daily || 0).toFixed(2));
    }

    function updateUsageByType(byType) {
        const container = document.getElementById('usageByType');
        if (!container) return;

        if (!byType || byType.length === 0) {
            container.innerHTML = '<div class="text-center text-muted py-4">No usage data yet</div>';
            return;
        }

        const typeIcons = {
            'ai_tokens': 'fa-robot', 'tts': 'fa-volume-up', 'stt': 'fa-microphone',
            'image': 'fa-image', 'video': 'fa-video', 'domain': 'fa-globe'
        };
        const typeLabels = {
            'ai_tokens': 'AI Conversations', 'tts': 'Text-to-Speech', 'stt': 'Speech-to-Text',
            'image': 'Image Generation', 'video': 'Video Generation', 'domain': 'Custom Domains'
        };

        container.innerHTML = byType.map(t => `
            <div class="usage-item">
                <div class="details">
                    <span class="type-badge ${t.type}">
                        <i class="fas ${typeIcons[t.type] || 'fa-circle'}"></i>
                        ${typeLabels[t.type] || t.type}
                    </span>
                    <span class="ops">${formatNumber(t.operations)} operations</span>
                </div>
                <div class="cost">$${t.total_cost.toFixed(2)}</div>
            </div>
        `).join('');
    }

    function updateChart(daily) {
        const canvas = document.getElementById('usageSpendingChart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        if (spendingChart) spendingChart.destroy();
        if (!daily || daily.length === 0) return;

        const sorted = [...daily].sort((a, b) => a.date.localeCompare(b.date));
        const labels = sorted.map(d => formatDateShort(d.date));
        const costData = sorted.map(d => d.total_cost);

        spendingChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Daily Spending',
                    data: costData,
                    borderColor: 'rgb(250, 166, 26)',
                    backgroundColor: 'rgba(250, 166, 26, 0.15)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: ctx => '$' + ctx.parsed.y.toFixed(2) } }
                },
                scales: {
                    x: {
                        ticks: {
                            color: getComputedStyle(document.documentElement).getPropertyValue('--text-muted').trim() || '#72767d',
                            maxTicksLimit: 8
                        },
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    },
                    y: {
                        ticks: {
                            color: getComputedStyle(document.documentElement).getPropertyValue('--text-muted').trim() || '#72767d',
                            callback: v => '$' + v.toFixed(2)
                        },
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    }
                }
            }
        });
    }

    function updateDailyBreakdown(daily) {
        const container = document.getElementById('usageDailyBreakdown');
        if (!container) return;

        if (!daily || daily.length === 0) {
            container.innerHTML = '<div class="text-center text-muted py-4">No activity yet</div>';
            return;
        }

        const sorted = [...daily].sort((a, b) => b.date.localeCompare(a.date)).slice(0, 14);
        container.innerHTML = sorted.map(d => `
            <div class="daily-item">
                <div class="date">${formatDateLong(d.date)}</div>
                <div class="stats">
                    <span class="stat-val"><strong>${formatNumber(d.operations)}</strong> ops</span>
                    <span class="stat-val"><strong>${formatNumber(d.tokens_in + d.tokens_out)}</strong> tokens</span>
                    <span class="stat-val"><strong>$${d.total_cost.toFixed(2)}</strong></span>
                </div>
            </div>
        `).join('');
    }

    function formatNumber(num) {
        if (num >= 1000000000) return (num / 1000000000).toFixed(1) + 'B';
        if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
        if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
        return num.toString();
    }

    function formatDateShort(dateStr) {
        const date = new Date(dateStr + 'T00:00:00');
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }

    function formatDateLong(dateStr) {
        const date = new Date(dateStr + 'T00:00:00');
        return date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
    }

    // Expose loadUsageData for the date range filter
    window.loadUsageData = loadUsageData;

    // Init
    document.addEventListener('DOMContentLoaded', function() {
        setupHashSync();
        activateFromHash();
        // Always init profile tab (default)
        const hash = window.location.hash || '#profile';
        initTab(hash.replace('#', ''));
    });
})();
