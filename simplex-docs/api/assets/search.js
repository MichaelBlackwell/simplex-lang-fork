// Simplex API Documentation - Client-side Search
// Loads search-index.json and provides instant search functionality

let searchIndex = [];
let searchTimeout = null;

// Load search index on page load
document.addEventListener('DOMContentLoaded', () => {
    loadSearchIndex();
    initSearch();
});

// Determine base path (are we in a subdirectory?)
function getBasePath() {
    const path = window.location.pathname;
    // If in category subdirectory (e.g., /api/std/math.html)
    // Match pattern: /something/something.html where second part has a slash before it
    if (path.match(/\/[^\/]+\/[^\/]+\.html$/)) {
        return '../';
    }
    return '';
}

function loadSearchIndex() {
    const basePath = getBasePath();

    fetch(basePath + 'search-index.json')
        .then(response => {
            if (!response.ok) {
                throw new Error('Search index not found');
            }
            return response.json();
        })
        .then(data => {
            searchIndex = data;
            console.log(`Loaded ${searchIndex.length} items for search`);
        })
        .catch(err => {
            console.warn('Search index not available:', err.message);
        });
}

function initSearch() {
    const input = document.getElementById('search');
    if (!input) return;

    // Create results container if it doesn't exist
    let container = document.getElementById('search-results');
    if (!container) {
        container = document.createElement('div');
        container.id = 'search-results';
        input.parentNode.appendChild(container);
    }

    // Handle input events with debounce
    input.addEventListener('input', (e) => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            performSearch(e.target.value);
        }, 150);
    });

    // Handle keyboard navigation
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

    // Hide results when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-box')) {
            hideResults();
        }
    });

    // Global keyboard shortcut: / to focus search
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

    // Score and filter results
    const scored = searchIndex
        .map(item => ({
            item,
            score: scoreMatch(item, q)
        }))
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

    // Exact name match
    if (name === query) {
        score += 100;
    }
    // Name starts with query
    else if (name.startsWith(query)) {
        score += 50;
    }
    // Name contains query
    else if (name.includes(query)) {
        score += 25;
    }
    // Module name match
    else if (module.includes(query)) {
        score += 15;
    }
    // Category match
    else if (category.includes(query)) {
        score += 10;
    }
    // Doc contains query
    else if (doc.includes(query)) {
        score += 5;
    }

    // Boost functions over other types
    if (item.kind === 'fn' && score > 0) {
        score += 5;
    }

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

    container.innerHTML = `
        <div style="padding: 12px; color: #666; text-align: center;">
            No results for "${escapeHtml(query)}"
        </div>
    `;
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
