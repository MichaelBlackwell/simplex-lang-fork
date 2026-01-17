// Simplex API Documentation - Dynamic Navigation
// Loads navigation from api-context.json and search from search-index.json

let apiContext = null;
let searchIndex = [];
let searchTimeout = null;

document.addEventListener('DOMContentLoaded', () => {
    loadNavigation();
    loadSearchIndex();
    initSearch();
});

// Determine base path (are we in a subdirectory?)
function getBasePath() {
    const path = window.location.pathname;
    // If in category subdirectory (e.g., /api/std/math.html)
    if (path.match(/\/[^\/]+\/[^\/]+\.html$/)) {
        return '../';
    }
    return '';
}

// Load navigation from api-context.json
function loadNavigation() {
    const basePath = getBasePath();

    fetch(basePath + 'api-context.json')
        .then(response => {
            if (!response.ok) throw new Error('Not found');
            return response.json();
        })
        .then(data => {
            apiContext = data;
            renderNavigation(data);
        })
        .catch(err => {
            console.warn('Could not load navigation:', err.message);
            const navContent = document.getElementById('nav-content');
            if (navContent) {
                navContent.innerHTML = '<p><a href="' + basePath + 'index.html">Back to Index</a></p>';
            }
        });
}

// Render navigation sidebar
function renderNavigation(data) {
    const navContent = document.getElementById('nav-content');
    if (!navContent) return;

    const basePath = getBasePath();
    const currentPath = window.location.pathname;

    let html = '';

    for (const [category, info] of Object.entries(data.categories)) {
        html += '<div class="nav-category">';
        html += '<div class="nav-category-header">' + escapeHtml(category) + '</div>';
        html += '<div class="nav-items">';

        for (const mod of info.modules) {
            const href = basePath + category + '/' + mod + '.html';
            const isActive = currentPath.endsWith('/' + category + '/' + mod + '.html');
            const activeClass = isActive ? ' class="active"' : '';
            html += '<a href="' + href + '"' + activeClass + '>' + escapeHtml(mod) + '</a>';
        }

        html += '</div></div>';
    }

    navContent.innerHTML = html;
}

// Load search index
function loadSearchIndex() {
    const basePath = getBasePath();

    fetch(basePath + 'search-index.json')
        .then(response => {
            if (!response.ok) throw new Error('Not found');
            return response.json();
        })
        .then(data => {
            searchIndex = data;
        })
        .catch(err => {
            console.warn('Search index not available:', err.message);
        });
}

// Initialize search functionality
function initSearch() {
    const input = document.getElementById('search');
    if (!input) return;

    let container = document.getElementById('search-results');
    if (!container) {
        container = document.createElement('div');
        container.id = 'search-results';
        input.parentNode.appendChild(container);
    }

    input.addEventListener('input', (e) => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            performSearch(e.target.value);
        }, 150);
    });

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            hideResults();
            input.blur();
        } else if (e.key === 'ArrowDown') {
            e.preventDefault();
            focusResult(1);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            focusResult(-1);
        } else if (e.key === 'Enter') {
            const focused = container.querySelector('a:focus');
            if (focused) {
                window.location.href = focused.href;
            }
        }
    });

    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-box')) {
            hideResults();
        }
    });

    // Global shortcut: / to focus search
    document.addEventListener('keydown', (e) => {
        if (e.key === '/' && document.activeElement !== input) {
            e.preventDefault();
            input.focus();
        }
    });
}

function performSearch(query) {
    if (!query || query.length < 2) {
        hideResults();
        return;
    }

    const q = query.toLowerCase().trim();
    const scored = searchIndex
        .map(item => ({ item, score: scoreMatch(item, q) }))
        .filter(r => r.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, 15);

    if (scored.length === 0) {
        showNoResults(query);
    } else {
        showResults(scored.map(r => r.item));
    }
}

function scoreMatch(item, query) {
    let score = 0;
    const name = item.name.toLowerCase();
    const doc = (item.doc || '').toLowerCase();
    const category = item.category.toLowerCase();
    const module = item.module.toLowerCase();

    if (name === query) score += 100;
    else if (name.startsWith(query)) score += 50;
    else if (name.includes(query)) score += 25;
    else if (module.includes(query)) score += 15;
    else if (category.includes(query)) score += 10;
    else if (doc.includes(query)) score += 5;

    if (item.kind === 'fn' && score > 0) score += 5;
    return score;
}

function showResults(results) {
    const container = document.getElementById('search-results');
    if (!container) return;

    const basePath = getBasePath();
    container.innerHTML = results.map(r => `
        <a href="${basePath}${r.url}" tabindex="0">
            <strong>${escapeHtml(r.name)}</strong>
            <small>${escapeHtml(r.category)} / ${escapeHtml(r.module)} - ${escapeHtml(r.kind)}</small>
        </a>
    `).join('');
    container.style.display = 'block';
}

function showNoResults(query) {
    const container = document.getElementById('search-results');
    if (!container) return;
    container.innerHTML = `<div style="padding:12px;color:#666;text-align:center;">No results for "${escapeHtml(query)}"</div>`;
    container.style.display = 'block';
}

function hideResults() {
    const container = document.getElementById('search-results');
    if (container) {
        container.style.display = 'none';
        container.innerHTML = '';
    }
}

function focusResult(direction) {
    const container = document.getElementById('search-results');
    if (!container) return;

    const links = container.querySelectorAll('a');
    if (links.length === 0) return;

    const focused = container.querySelector('a:focus');
    let index = Array.from(links).indexOf(focused);

    if (index === -1) {
        index = direction === 1 ? 0 : links.length - 1;
    } else {
        index += direction;
        if (index < 0) index = links.length - 1;
        if (index >= links.length) index = 0;
    }

    links[index].focus();
}

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
