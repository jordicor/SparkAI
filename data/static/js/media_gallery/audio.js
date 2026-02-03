// Global variables
let currentAudio = null;
let currentlyPlaying = null;

// Combined audio utilities object
const audioUtils = {
    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
    },

    resetTimeDisplay(container) {
        const timeDisplay = container.querySelector('.time-display');
        const currentTime = container.querySelector('.current-time');
        const timeSeparator = container.querySelector('.time-separator');
        const totalDuration = container.querySelector('.total-duration');
        
        currentTime.textContent = '--:--';
        totalDuration.textContent = '--:--';
        timeSeparator.classList.add('hidden');
        timeDisplay.classList.remove('active');
    },

    updateProgressBar(container, audio) {
        const progressBar = container.querySelector('.progress-filled');
        const progress = (audio.currentTime / audio.duration) * 100;
        progressBar.style.width = `${progress}%`;
    },

    validatePath(path) {
        return path && typeof path === 'string' && path.toLowerCase().endsWith('.mp3');
    },

    getFileName(path) {
        return path.split('/').pop().split('\\').pop();
    }
};

function togglePlay(path, overlayElement) {
    const container = overlayElement.closest('.mp3-container');
    const audioElement = overlayElement.nextElementSibling;
    const playIcon = overlayElement.querySelector('.play-icon');
    const timeDisplay = container.querySelector('.time-display');
    const currentTime = container.querySelector('.current-time');
    const timeSeparator = container.querySelector('.time-separator');
    const totalDuration = container.querySelector('.total-duration');

    if (currentlyPlaying && currentlyPlaying !== audioElement) {
        currentlyPlaying.pause();
        currentlyPlaying.currentTime = 0;
        const oldContainer = currentlyPlaying.closest('.mp3-container');
        audioUtils.resetTimeDisplay(oldContainer);
        audioUtils.updateProgressBar(oldContainer, currentlyPlaying);
        resetPlayIcon(currentlyPlaying);
    }

    if (audioElement.paused) {
        if (!audioElement.src) {
            const baseUrl = window.cdnFilesUrl || window.location.origin;
            const url = new URL(path, baseUrl);
            if (window.mp3Token) {
                url.searchParams.append('token', window.mp3Token);
            }
            audioElement.src = url.toString();
        }
        audioElement.play()
            .then(() => {
                playIcon.className = 'play-icon fas fa-pause-circle';
                currentAudio = audioElement;
                currentlyPlaying = audioElement;
                timeDisplay.classList.add('active');
                timeSeparator.classList.remove('hidden');
            })
            .catch(error => {
                console.error('Error playing audio:', error);
                NotificationModal.error('Playback Error', 'Error playing audio file');
            });
    } else {
        audioElement.pause();
        playIcon.className = 'play-icon fas fa-play-circle';
        currentAudio = null;
        currentlyPlaying = null;
        audioUtils.resetTimeDisplay(container);
    }

    // Event listeners for current audio
    audioElement.ontimeupdate = () => {
        if (!audioElement.paused) {
            currentTime.textContent = audioUtils.formatTime(audioElement.currentTime);
            totalDuration.textContent = audioUtils.formatTime(audioElement.duration);
            audioUtils.updateProgressBar(container, audioElement);
        }
    };

    audioElement.onended = () => {
        playIcon.className = 'play-icon fas fa-redo-alt';
        currentAudio = null;
        currentlyPlaying = null;
        audioUtils.resetTimeDisplay(container);
        audioUtils.updateProgressBar(container, audioElement);
    };
}

function resetPlayIcon(audioElement) {
    const overlay = audioElement.previousElementSibling;
    const playIcon = overlay.querySelector('.play-icon');
    if (audioElement.currentTime === audioElement.duration) {
        playIcon.className = 'play-icon fas fa-redo-alt';
    } else {
        playIcon.className = 'play-icon fas fa-play-circle';
    }
}

function downloadMp3(path) {
    const baseUrl = window.cdnFilesUrl || window.location.origin;
    const url = new URL(path, baseUrl);
    if (window.mp3Token) {
        url.searchParams.append('token', window.mp3Token);
        url.searchParams.append('download', 'true');
    }
    window.open(url.toString(), '_blank');
}

function deleteMp3(path) {
    if (confirm('Are you sure you want to delete this MP3?')) {
        fetch('/delete-mp3', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ mp3_path: path })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            NotificationModal.info('MP3 Deleted', data.message);
            location.reload();
        })
        .catch(error => {
            console.error('Error:', error);
            NotificationModal.error('Delete Error', 'An error occurred while deleting the MP3.');
        });
    }
}

function deleteSelectedMp3s() {
    const selectedMp3s = document.querySelectorAll('.mp3-checkbox:checked');
    if (selectedMp3s.length === 0) {
        NotificationModal.warning('Selection Required', 'Please select at least one MP3 to delete');
        return;
    }

    if (confirm(`Are you sure you want to delete ${selectedMp3s.length} selected MP3s?`)) {
        const mp3Paths = Array.from(selectedMp3s).map(checkbox => decodeURIComponent(checkbox.dataset.path));
        
        fetch('/delete-mp3s', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ mp3_paths: mp3Paths })
        })
        .then(response => response.json())
        .then(data => {
            NotificationModal.info('MP3s Deleted', data.message);
            location.reload();
        })
        .catch(error => {
            console.error('Error:', error);
            NotificationModal.error('Delete Error', 'An error occurred while deleting the MP3s');
        });
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    const mp3Tab = document.getElementById('mp3-tab');
    mp3Tab.addEventListener('shown.bs.tab', function (e) {
        if (!mp3Tab.dataset.loaded) {
            loadMP3s();
            mp3Tab.dataset.loaded = true;
        }
    })
    // Initialize time displays
    document.querySelectorAll('.mp3-container').forEach(container => {
        audioUtils.resetTimeDisplay(container);
    });

    // Configure progress bar
    document.querySelectorAll('.mp3-progress-bar').forEach(progressBar => {
        progressBar.addEventListener('click', (e) => {
            const container = progressBar.closest('.mp3-container');
            const audio = container.querySelector('.audio-player');
            
            if (audio === currentlyPlaying) {
                const rect = progressBar.getBoundingClientRect();
                const clickPosition = (e.clientX - rect.left) / rect.width;
                audio.currentTime = clickPosition * audio.duration;
                audioUtils.updateProgressBar(container, audio);
            }
        });
    });

    // Configure multiple deletion button
    const deleteSelectedMp3sButton = document.getElementById('deleteSelectedMp3s');
    if (deleteSelectedMp3sButton) {
        deleteSelectedMp3sButton.addEventListener('click', deleteSelectedMp3s);
    }

    // Handle tab changes
    document.querySelectorAll('button[data-bs-toggle="tab"]').forEach(tab => {
        tab.addEventListener('shown.bs.tab', function () {
            if (currentlyPlaying) {
                currentlyPlaying.pause();
                currentlyPlaying.currentTime = 0;
                resetPlayIcon(currentlyPlaying);
                const container = currentlyPlaying.closest('.mp3-container');
                audioUtils.resetTimeDisplay(container);
                currentlyPlaying = null;
                currentAudio = null;
            }
        });
    });

    // Handle audio errors
    document.querySelectorAll('.audio-player').forEach(audio => {
        audio.addEventListener('error', function(e) {
            console.error('Audio Error:', e);
            NotificationModal.error('Audio Error', 'Error loading audio file');
            resetPlayIcon(this);
        });
    });
});

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        togglePlay,
        downloadMp3,
        deleteMp3,
        audioUtils
    };
}

let allMp3s = [];
let currentMp3Page = 1;

function loadMP3s() {
    fetch('/get-mp3s')
        .then(response => response.json())
        .then(data => {
            allMp3s = data.mp3s;
            mp3Token = data.mp3_token;
            renderMp3Page(1);
        })
        .catch(error => {
            console.error('Error loading MP3s:', error);
            NotificationModal.error('Load Error', 'Error loading MP3s');
        });
}

function renderMp3Page(page) {
    currentMp3Page = page;
    const pageData = paginationUtils.getCurrentPageData(allMp3s, page, ITEMS_PER_PAGE);
    const container = document.getElementById('mp3Container');

    container.innerHTML = pageData.map(mp3 => `
        <div class="mp3-container">
            <div class="mp3-wrapper">
                <input type="checkbox" class="mp3-checkbox" data-path="${encodeURIComponent(mp3.path)}" title="Select for bulk delete">
                <div class="mp3-player-wrapper">
                    <div class="mp3-overlay" onclick="togglePlay('${mp3.nginx_path}', this)" title="Click to play/pause">
                        <i class="play-icon fas fa-play-circle"></i>
                    </div>
                    <audio class="audio-player" data-src="${mp3.nginx_path}"></audio>
                </div>
            </div>
            <div class="mp3-info">
                <div class="mp3-name" title="${mp3.name}">${mp3.name}</div>
                <div class="time-display">
                    <span class="current-time">--:--</span>
                    <span class="time-separator hidden"> / </span>
                    <span class="total-duration">--:--</span>
                </div>
            </div>
            <div class="mp3-progress-bar">
                <div class="progress-filled"></div>
            </div>
            <div class="mp3-controls">
                <button onclick="downloadMp3('${mp3.nginx_path}')" class="btn btn-sm btn-primary" title="Download MP3">
                    <i class="fas fa-download"></i> Download
                </button>
                <button onclick="deleteMp3('${encodeURIComponent(mp3.path)}')" class="btn btn-sm btn-danger" title="Delete MP3">
                    <i class="fas fa-trash"></i> Delete
                </button>
            </div>
        </div>
    `).join('');

    // Reinitialize time displays
    document.querySelectorAll('.mp3-container').forEach(container => {
        audioUtils.resetTimeDisplay(container);
    });

    // Update pagination controls
    const paginationElement = document.getElementById('pagination-mp3s');
    paginationElement.innerHTML = paginationUtils.createPaginationControls(
        allMp3s.length,
        page,
        ITEMS_PER_PAGE,
        renderMp3Page
    );

    // Add event listeners to pagination controls
    paginationElement.querySelectorAll('.page-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const newPage = parseInt(e.target.dataset.page);
            if (!isNaN(newPage) && newPage > 0 && newPage <= Math.ceil(allMp3s.length / ITEMS_PER_PAGE)) {
                renderMp3Page(newPage);
            }
        });
    });
}