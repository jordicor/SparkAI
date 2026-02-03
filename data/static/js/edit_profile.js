// edit_profile.js

let phoneInputJS;
let audioPlayer = null;
let alterEgoModal;

document.addEventListener('DOMContentLoaded', function() {
    initPhoneInput();
    initializeAlterEgoState();
    loadVoices();
    loadAlterEgos(currentAlterEgoId);
    setupEventListeners();
    initializeProfileHandlers();
    setupUsernameValidation();

    alterEgoModal = new bootstrap.Modal(document.getElementById('alterEgoModal'));

    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });

    const deleteProfilePictureButton = document.getElementById('deleteProfilePictureButton');
    if (deleteProfilePictureButton) {
        deleteProfilePictureButton.addEventListener('click', deleteProfilePicture);
    }

    initializeAlterEgoHandlers();
});

// Initialize when DOM is ready

function initPhoneInput() {
    const phoneInputField = document.getElementById('phone');
    phoneInputJS = window.intlTelInput(phoneInputField, {
        initialCountry: "auto",
        separateDialCode: true,
        utilsScript: "https://cdnjs.cloudflare.com/ajax/libs/intl-tel-input/17.0.13/js/utils.js",
        geoIpLookup: function(success, failure) {
            secureFetch('/api/get-ip-info')
                .then(function(response) {
                    if (!response) return null;
                    if (response.ok) return response.json();
                    throw new Error('Failed to fetch IP info');
                })
                .then(function(ipinfo) {
                    if (!ipinfo) return;
                    success(ipinfo.country);
                })
                .catch(function() {
                    success("us");
                });
        },
    });
}

function initializeAlterEgoState() {
    const useRealProfile = document.getElementById('useRealProfile');
    const useAlterEgo = document.getElementById('useAlterEgo');
    const alterEgoSelection = document.getElementById('alterEgoSelection');

    if (currentAlterEgoId && currentAlterEgoId !== "0") {
        useAlterEgo.checked = true;
        alterEgoSelection.style.display = 'block';
    } else {
        useRealProfile.checked = true;
        alterEgoSelection.style.display = 'none';
    }

    useRealProfile.addEventListener('change', toggleAlterEgoSelection);
    useAlterEgo.addEventListener('change', toggleAlterEgoSelection);
}

function toggleAlterEgoSelection() {
    const useAlterEgo = document.getElementById('useAlterEgo');
    const alterEgoSelection = document.getElementById('alterEgoSelection');
    alterEgoSelection.style.display = useAlterEgo.checked ? 'block' : 'none';
}

function loadVoices() {
    const voiceSelect = document.getElementById('voice');
    secureFetch('/api/voices')
        .then(response => {
            if (!response) return null;
            return response.json();
        })
        .then(voices => {
            if (!voices) return;
            voiceSelect.innerHTML = '<option value="">Default Voice</option>';
            voices.forEach(voice => {
                const option = document.createElement('option');
                option.value = voice.id;
                option.textContent = voice.name;
                voiceSelect.appendChild(option);
            });
            if (currentUserVoiceId && currentUserVoiceId !== "None") {
                voiceSelect.value = currentUserVoiceId;
            }
            addPlayButton();
        })
        .catch(error => console.error('Error loading voices:', error));
}

function addPlayButton() {
    const playButtonContainer = document.getElementById('playButtonContainer');
    playButtonContainer.innerHTML = '';

    const categorySelect = document.createElement('select');
    categorySelect.className = 'form-select form-select-sm me-2';
    categorySelect.id = 'sampleCategory';
    categorySelect.style.display = 'inline-block';
    categorySelect.style.width = 'auto';

    const categories = [
        "Children and Basic Education",
        "Finance and Business",
        "Relaxation and Meditation",
        "Casual Conversation",
        "Drama and Emotional Narration",
        "Storytelling",
        "Advertising and Announcements",
        "Science and Technology",
        "Education and Advanced Training",
        "Corporate Environments",
        "Mystery and Suspense",
        "Sports and Energy"
    ];

    categories.forEach((category, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = category;
        categorySelect.appendChild(option);
    });

    const playButton = document.createElement('button');
    playButton.textContent = '▶️ Play Sample';
    playButton.className = 'btn btn-sm btn-outline-secondary play-voice';
    playButton.id = 'playVoiceButton';
    playButton.style.display = document.getElementById('voice').value ? 'inline-block' : 'none';
    playButton.addEventListener('click', function(e) {
        e.preventDefault();
        playVoiceSample(document.getElementById('voice').value, categorySelect.value);
    });

    playButtonContainer.appendChild(categorySelect);
    playButtonContainer.appendChild(playButton);

    document.getElementById('voice').addEventListener('change', function() {
        playButton.style.display = this.value ? 'inline-block' : 'none';
        categorySelect.style.display = this.value ? 'inline-block' : 'none';
    });
}

function playVoiceSample(voiceId, categoryId) {
    if (!voiceId) return;

    if (audioPlayer) {
        audioPlayer.pause();
        audioPlayer.currentTime = 0;
        audioPlayer = null;
        const playButton = document.getElementById('playVoiceButton');
        playButton.textContent = '▶️ Play Sample';
        return;
    }

    const playButton = document.getElementById('playVoiceButton');
    playButton.textContent = '🔄 Loading...';
    playButton.disabled = true;

    secureFetch(`/api/voice-sample/${voiceId}?category=${categoryId}`)
        .then(response => {
            if (!response) return null;
            return response.blob();
        })
        .then(blob => {
            if (!blob) return;
            const url = URL.createObjectURL(blob);
            audioPlayer = new Audio(url);

            audioPlayer.onended = function() {
                playButton.textContent = '▶️ Play Sample';
                playButton.disabled = false;
                audioPlayer = null;
            };

            audioPlayer.play();
            playButton.textContent = '⏹️ Stop Sample';
            playButton.disabled = false;

            playButton.onclick = function() {
                if (audioPlayer) {
                    audioPlayer.pause();
                    audioPlayer.currentTime = 0;
                    audioPlayer = null;
                    playButton.textContent = '▶️ Play Sample';
                }
            };
        })
        .catch(error => {
            console.error('Error playing voice sample:', error);
            playButton.textContent = '▶️ Play Sample';
            playButton.disabled = false;
        });
}

function setupEventListeners() {
    const phoneInput = document.getElementById('phone');
    const sendCodeButton = document.getElementById('sendCodeButton');
    const verificationCodeContainer = document.getElementById('verificationCodeContainer');
    const verificationCodeInput = document.getElementById('verificationCode');
    const editProfileForm = document.getElementById('editProfileForm');

    const verifyCodeButton = document.createElement('button');
    verifyCodeButton.textContent = 'Verify Code';
    verifyCodeButton.className = 'btn btn-primary mt-2';
    verifyCodeButton.style.display = 'none';
    verificationCodeContainer.appendChild(verifyCodeButton);

    verificationCodeInput.addEventListener('input', function() {
        verifyCodeButton.style.display = this.value ? 'block' : 'none';
    });

    verifyCodeButton.addEventListener('click', async function(e) {
        e.preventDefault();
        const phoneNumber = phoneInputJS.getNumber(intlTelInputUtils.numberFormat.E164);
        const code = verificationCodeInput.value;

        try {
            const response = await secureFetch('/api/verify-code', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ phone: phoneNumber, code: code }),
            });
            if (!response) return; // Session expired
            const result = await response.json();
            if (result.status === 'approved') {
                showGenericModal('Success', 'Phone number verified successfully!', { cancelText: 'Close' });
                verifyCodeButton.style.display = 'none';
                verificationCodeInput.disabled = true;
                phoneInput.dataset.verified = 'true';
            } else {
                showGenericModal('Error', 'Verification failed. Please check the code and try again.', { cancelText: 'Close' });
                verificationCodeInput.value = '';
            }
        } catch (error) {
            console.error('Error:', error);
            showGenericModal('Error', 'An error occurred while verifying the code. Please try again.', { cancelText: 'Close' });
        }
    });

    phoneInput.addEventListener('input', function() {
        if (this.value) {
            sendCodeButton.style.display = 'block';
        } else {
            sendCodeButton.style.display = 'none';
            verificationCodeContainer.style.display = 'none';
        }
    });

    sendCodeButton.addEventListener('click', async function() {
        const phoneNumber = phoneInputJS.getNumber(intlTelInputUtils.numberFormat.E164);

        try {
            const checkResponse = await secureFetch('/api/check-phone-number', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ phone: phoneNumber, user_id: currentUserId }),
            });
            if (!checkResponse) return; // Session expired
            const checkResult = await checkResponse.json();

            if (checkResult.exists) {
                showGenericModal('Error', 'This phone number is already in use. Please use a different number.', { cancelText: 'Close' });
                return;
            }

            const response = await secureFetch('/api/send-verification-code', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ phone: phoneNumber }),
            });
            if (!response) return; // Session expired
            const result = await response.json();
            if (response.ok) {
                if (result.status === 'pending') {
                    verificationCodeContainer.style.display = 'block';
                    showGenericModal('Success', 'Verification code sent successfully!', { cancelText: 'Close' });
                } else {
                    showGenericModal('Error', `Unexpected status: ${result.status}`, { cancelText: 'Close' });
                }
            } else {
                showGenericModal('Error', `Error sending verification code: ${result.detail}`, { cancelText: 'Close' });
            }
        } catch (error) {
            console.error('Error:', error);
            showGenericModal('Error', 'An unexpected error occurred. Please try again.', { cancelText: 'Close' });
        }
    });

    editProfileForm.addEventListener('submit', handleFormSubmit);
}

async function handleFormSubmit(event) {
    event.preventDefault();
    const formData = new FormData(event.target);
    
    const fullPhoneNumber = getFullPhoneNumber();
    formData.set('phone_number', fullPhoneNumber);

    if (fullPhoneNumber !== originalPhoneNumber) {
        try {
            const checkResponse = await secureFetch('/api/check-phone-number', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ phone: fullPhoneNumber, user_id: currentUserId }),
            });
            if (!checkResponse) return; // Session expired
            const checkResult = await checkResponse.json();

            if (checkResult.exists) {
                showGenericModal('Error', 'This phone number is already in use. Please use a different number.', {
                    cancelText: 'Close'
                });
                return;
            }

            const phoneInput = document.getElementById('phone');
            if (phoneInput.dataset.verified !== 'true') {
                showGenericModal('Error', 'Please verify your new phone number before submitting.', {
                    cancelText: 'Close'
                });
                return;
            }
        } catch (error) {
            console.error('Error:', error);
            showGenericModal('Error', 'An error occurred while checking the phone number.', {
                cancelText: 'Close'
            });
            return;
        }
    }

    const useAlterEgo = document.getElementById('useAlterEgo').checked;
    const alterEgoSelect = document.getElementById('alterEgo');
    if (useAlterEgo && alterEgoSelect.value !== "0") {
        formData.set('alter_ego_id', alterEgoSelect.value);
    } else {
        formData.set('alter_ego_id', "0");
    }

    secureFetch('/api/edit-profile', {
        method: 'POST',
        body: formData,
        headers: {
            'X-Requested-With': 'XMLHttpRequest'
        }
    })
    .then(response => {
        if (!response) return null;
        return response.json();
    })
    .then(data => {
        if (!data) return;
        if (data.success) {
            showGenericModal('Success', 'Profile updated successfully', {
                cancelText: 'Close'
            });
            currentUserVoiceId = formData.get('sample_voice_id');
            originalPhoneNumber = fullPhoneNumber;
            currentAlterEgoId = formData.get('alter_ego_id');
        } else {
            showGenericModal('Error', 'Error updating profile: ' + data.message, {
                cancelText: 'Close'
            });
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showGenericModal('Error', 'An error occurred while updating the profile.', {
            cancelText: 'Close'
        });
    });
}

function getFullPhoneNumber() {
    return phoneInputJS.getNumber(intlTelInputUtils.numberFormat.E164);
}

function showGenericModal(title, message, options = {}) {
    const modalElement = document.getElementById('genericModal');
    const modal = new bootstrap.Modal(modalElement);
    const confirmBtn = document.getElementById('genericModalConfirmBtn');
    const cancelBtn = document.getElementById('genericModalCancelBtn');
    const modalLabel = document.getElementById('genericModalLabel');
    const modalBody = document.getElementById('genericModalBody');

    modalLabel.textContent = title;
    modalBody.textContent = message;

    if (options.confirmText) {
        confirmBtn.textContent = options.confirmText;
        confirmBtn.className = `btn ${options.confirmClass || 'btn-primary'}`;
        confirmBtn.style.display = '';
        confirmBtn.onclick = () => {
            if (options.onConfirm) {
                options.onConfirm(modal);
            }
            if (options.hideOnConfirm !== false) {
                modal.hide();
            }
        };
    } else {
        confirmBtn.style.display = 'none';
    }

    if (options.cancelText) {
        cancelBtn.textContent = options.cancelText;
        cancelBtn.style.display = '';
        cancelBtn.onclick = () => {
            if (options.onCancel) {
                options.onCancel();
            }
            modal.hide();
        };
    } else {
        cancelBtn.textContent = 'Close';
        cancelBtn.onclick = () => modal.hide();
    }

    modalElement.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && options.confirmText) {
            event.preventDefault();
            confirmBtn.click();
        } else if (event.key === 'Escape') {
            event.preventDefault();
            cancelBtn.click();
        }
    }, { once: true });

    modal.show();

    modalElement.addEventListener('hidden.bs.modal', function () {
        const backdrop = document.querySelector('.modal-backdrop');
        if (backdrop) {
            backdrop.remove();
        }
        document.body.classList.remove('modal-open');
        document.body.style.removeProperty('padding-right');
        document.body.style.removeProperty('overflow');
    }, { once: true });
}

// Functions related to alter-ego management
function loadAlterEgos(currentAlterEgoId) {
    secureFetch('/api/get-alter-egos', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        }
    }).then(response => {
        if (!response) return null;
        return response.json();
    })
    .then(data => {
        if (!data) return;
        if (data.success) {
            const alterEgoSelect = document.getElementById('alterEgo');
            alterEgoSelect.innerHTML = '<option value="0">Select an alter-ego</option>';
            data.alterEgos.forEach(alterEgo => {
                let option = document.createElement('option');
                option.value = alterEgo.id;
                option.text = alterEgo.name;
                alterEgoSelect.appendChild(option);
            });

            if (currentAlterEgoId && currentAlterEgoId !== "0") {
                alterEgoSelect.value = currentAlterEgoId;
                showAlterEgoDetails(currentAlterEgoId);
            } else {
                document.getElementById('alterEgoDetails').innerHTML = '';
            }
        } else {
            console.error('Error loading alter-egos:', data.message);
        }
    })
    .catch(error => {
        console.error('Error fetching alter-egos:', error);
    });
}

function showAlterEgoDetails(alterEgoId) {
    const alterEgoDetails = document.getElementById('alterEgoDetails');

    if (!alterEgoId || alterEgoId === "0" || alterEgoId === "") {
        alterEgoDetails.innerHTML = '';
        return;
    }

    secureFetch(`/api/get-alter-ego-details/${alterEgoId}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        }
    }).then(response => {
        if (!response) return null;
        return response.json();
    })
    .then(data => {
        if (!data) return;
        if (data.success) {
            const profilePicture = data.alterEgo.profilePicture
                ? `<img src="${data.alterEgo.profilePicture}" alt="Alter-Ego Profile Picture" style="width: 100px; height: 100px; object-fit: cover; border-radius: 50%;">`
                : `<div style="width: 100px; height: 100px; background-color: #f0f0f0; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-size: 2em;">${data.alterEgo.name.charAt(0).toUpperCase()}</div>`;

            alterEgoDetails.innerHTML = `
                <div class="alter-ego-container">
                    <div class="alter-ego-header">
                        ${profilePicture}
                        <div>
                            <h3>${data.alterEgo.name}</h3>
                            <p>${data.alterEgo.description || 'No description available'}</p>
                        </div>
                    </div>
                    <div class="btn-group">
                        <button type="button" class="btn btn-primary" id="editAlterEgoButton">Edit</button>
                        <button type="button" class="btn btn-danger" id="deleteAlterEgoButton">Delete</button>
                    </div>
                </div>
            `;

            document.getElementById('editAlterEgoButton').addEventListener('click', function() {
                editAlterEgo(alterEgoId);
            });
            document.getElementById('deleteAlterEgoButton').addEventListener('click', function() {
                deleteAlterEgo(alterEgoId);
            });
        } else {
            console.error('Error loading alter-ego details:', data);
            alterEgoDetails.innerHTML = '<p>Error loading alter-ego details</p>';
        }
    })
    .catch(error => {
        console.error('Error fetching alter-ego details:', error);
        alterEgoDetails.innerHTML = '<p>An error occurred while loading alter-ego details</p>';
    });
}

function editAlterEgo(alterEgoId) {
    secureFetch(`/api/get-alter-ego-details/${alterEgoId}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        }
    }).then(response => {
        if (!response) return null;
        return response.json();
    })
    .then(data => {
        if (!data) return;
        if (data.success) {
            document.getElementById('alterEgoId').value = alterEgoId;
            document.getElementById('alterEgoName').value = data.alterEgo.name;
            document.getElementById('alterEgoDescription').value = data.alterEgo.description || '';

            const alterEgoPictureContainer = document.getElementById('alterEgoPictureContainer');
            if (data.alterEgo.profilePicture) {
                alterEgoPictureContainer.innerHTML = `
                    <img src="${data.alterEgo.profilePicture}" alt="Alter-Ego Profile Picture" id="currentAlterEgoPicture">
                    <div class="avatar-icons">
                        <span class="avatar-icon edit"><i class="fas fa-pencil-alt"></i></span>
                        <span class="avatar-icon delete"><i class="fas fa-trash"></i></span>
                    </div>
                `;
            } else {
                const initial = data.alterEgo.name.charAt(0).toUpperCase();
                alterEgoPictureContainer.innerHTML = `
                    <span class="avatar-initial" id="defaultAlterEgoInitial">${initial}</span>
                    <div class="avatar-icons">
                        <span class="avatar-icon edit"><i class="fas fa-pencil-alt"></i></span>
                    </div>
                `;
            }

            document.getElementById('previewAlterContainer').classList.add('hidden');
            alterEgoPictureContainer.classList.remove('hidden');

            alterEgoModal.show();
        } else {
            console.error('Error loading alter-ego details:', data);
            showGenericModal('Error', 'Error loading alter-ego details for editing', {
                cancelText: 'Close'
            });
        }
    })
    .catch(error => {
        console.error('Error fetching alter-ego details:', error);
        showGenericModal('Error', 'An error occurred while loading alter-ego details', {
            cancelText: 'Close'
        });
    });
}

function deleteAlterEgo(alterEgoId) {
    showGenericModal('Confirm Deletion', 'Are you sure you want to delete this alter-ego?', {
        confirmText: 'Delete',
        cancelText: 'Cancel',
        confirmClass: 'btn-danger',
        onConfirm: () => {
            secureFetch(`/api/delete-alter-ego/${alterEgoId}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                }
            }).then(response => {
                if (!response) return null;
                return response.json();
            })
            .then(data => {
                if (!data) return;
                if (data.success) {
                    showGenericModal('Success', 'Alter-ego deleted successfully', {
                        cancelText: 'Close'
                    });
                    loadAlterEgos();
                    const alterEgoDetails = document.getElementById('alterEgoDetails');
                    if (alterEgoDetails) {
                        alterEgoDetails.innerHTML = '';
                    }
                    const alterEgoSelect = document.getElementById('alterEgo');
                    if (alterEgoSelect) {
                        alterEgoSelect.value = '';
                    }
                } else {
                    throw new Error(data.message || 'Error deleting alter-ego');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showGenericModal('Error', 'An error occurred while deleting the alter-ego', {
                    cancelText: 'Close'
                });
            });
        }
    });
}

function deleteAlterEgoPicture() {
    showGenericModal(
        'Confirm Deletion',
        'Are you sure you want to delete this alter-ego picture?',
        {
            confirmText: 'Delete',
            cancelText: 'Cancel',
            confirmClass: 'btn-danger',
            onConfirm: () => {
                const alterEgoId = document.getElementById('alterEgoId').value;
                if (alterEgoId) {
                    secureFetch(`/api/delete-alter-ego-picture/${alterEgoId}`, {
                        method: 'DELETE'
                    })
                    .then(response => {
                        if (!response) return null;
                        return response.json();
                    })
                    .then(data => {
                        if (!data) return;
                        if (data.success) {
                            updateAlterEgoPictureUI(alterEgoId);
                            showGenericModal('Success', 'Alter-ego picture deleted successfully', {
                                cancelText: 'Close'
                            });
                        } else {
                            showGenericModal('Error', 'Error deleting alter-ego picture', {
                                cancelText: 'Close'
                            });
                        }
                    })
                    .catch(error => {
                        console.error('Error deleting alter-ego picture:', error);
                        showGenericModal('Error', 'An error occurred while deleting the alter-ego picture', {
                            cancelText: 'Close'
                        });
                    });
                } else {
                    updateAlterEgoPictureUI();
                }
            }
        }
    );
}

function updateAlterEgoPictureUI(alterEgoId) {
    const alterEgoPictureContainer = document.getElementById('alterEgoPictureContainer');
    const alterEgoName = document.getElementById('alterEgoName').value;
    alterEgoPictureContainer.innerHTML = `
        <span class="avatar-initial" id="defaultAlterEgoInitial">${alterEgoName.charAt(0).toUpperCase()}</span>
        <div class="avatar-icons">
            <span class="avatar-icon edit"><i class="fas fa-pencil-alt"></i></span>
        </div>
    `;

    if (alterEgoId) {
        const alterEgoDetails = document.getElementById('alterEgoDetails');
        if (alterEgoDetails) {
            const profilePicture = `<div style="width: 100px; height: 100px; background-color: #f0f0f0; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-size: 2em;">${alterEgoName.charAt(0).toUpperCase()}</div>`;
            const headerElement = alterEgoDetails.querySelector('.alter-ego-header');
            if (headerElement) {
                headerElement.innerHTML = `
                    ${profilePicture}
                    <div>
                        <h3>${alterEgoName}</h3>
                        <p>${document.getElementById('alterEgoDescription').value || 'No description available'}</p>
                    </div>
                `;
            }
        }
    }

    const fileInput = document.getElementById('alterEgoProfilePicture');
    if (fileInput) {
        fileInput.value = '';
    }
}

function saveAlterEgo() {
    const formData = new FormData(document.getElementById('alterEgoForm'));
    const alterEgoId = formData.get('id');
    const url = alterEgoId ? `/api/update-alter-ego/${alterEgoId}` : '/api/create-alter-ego';
    const method = alterEgoId ? 'PUT' : 'POST';

    const fileInput = document.getElementById('alterEgoProfilePicture');
    const previewImage = document.getElementById('previewAlterEgoImage');
    
    if (fileInput.files.length > 0) {
        formData.set('profile_picture', fileInput.files[0]);
    } else if (previewImage.src && !previewImage.src.includes('data:image')) {
        formData.set('keep_existing_image', 'true');
    } else {
        formData.delete('profile_picture');
    }

    sendSaveRequest(url, method, formData);
}

function sendSaveRequest(url, method, formData) {
    secureFetch(url, {
        method: method,
        body: formData
    })
    .then(response => {
        if (!response) return null;
        if (!response.ok) {
            return response.json().then(err => { throw err; });
        }
        return response.json();
    })
    .then(data => {
        if (!data) return;
        if (data.success) {
            showGenericModal('Success', 'Alter-ego saved successfully', {
                cancelText: 'Close'
            });
            alterEgoModal.hide();
            loadAlterEgos();
        } else {
            throw new Error(data.message || 'Error saving alter-ego');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showGenericModal('Error', error.message || 'An error occurred while saving the alter-ego', {
            cancelText: 'Close'
        });
    });
}

function deleteProfilePicture() {
    showGenericModal(
        'Confirm Deletion',
        'Are you sure you want to delete your profile picture?',
        {
            confirmText: 'Delete',
            cancelText: 'Cancel',
            confirmClass: 'btn-danger',
            onConfirm: () => {
                secureFetch('/api/delete-profile-picture', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ user_id: currentUserId })
                }).then(response => {
                    if (!response) return null;
                    return response.json();
                })
                .then(data => {
                    if (!data) return;
                    if (data.success) {
                        const profilePictureContainer = document.getElementById('profilePictureContainer');
                        profilePictureContainer.innerHTML = `
                            <span class="avatar-initial" id="defaultProfileInitial">${document.getElementById('username').value.charAt(0).toUpperCase()}</span>
                            <div class="avatar-icons">
                                <span class="avatar-icon edit"><i class="fas fa-pencil-alt"></i></span>
                            </div>
                        `;
                        showGenericModal('Success', 'Profile picture deleted successfully', {
                            cancelText: 'Close'
                        });
                    } else {
                        showGenericModal('Error', 'Error deleting profile picture', {
                            cancelText: 'Close'
                        });
                    }
                });
            }
        }
    );
}

function cancelAlterEgoPictureChange() {
    const previewContainer = document.getElementById('previewAlterContainer');
    const alterEgoPictureContainer = document.getElementById('alterEgoPictureContainer');

    previewContainer.classList.add('hidden');
    alterEgoPictureContainer.classList.remove('hidden');

    document.getElementById('alterEgoProfilePicture').value = '';
}

function initializeProfileHandlers() {
    const profilePictureContainer = document.getElementById('profilePictureContainer');
    const previewContainer = document.getElementById('previewContainer');
    const profilePictureInput = document.getElementById('profilePicture');

    if (profilePictureContainer) {
        const editIcon = profilePictureContainer.querySelector('.avatar-icon.edit');
        const deleteIcon = profilePictureContainer.querySelector('.avatar-icon.delete');

        if (editIcon) {
            editIcon.addEventListener('click', function() {
                profilePictureInput.click();
            });
        }

        if (deleteIcon) {
            deleteIcon.addEventListener('click', deleteProfilePicture);
        }
    }

    if (profilePictureInput) {
        profilePictureInput.addEventListener('change', function(e) {
            if (e.target.files && e.target.files[0]) {
                let reader = new FileReader();
                reader.onload = function(event) {
                    previewContainer.classList.remove('hidden');
                    const previewImage = previewContainer.querySelector('#previewImage');
                    if (previewImage) {
                        previewImage.src = event.target.result;
                    }
                    profilePictureContainer.classList.add('hidden');
                }
                reader.readAsDataURL(e.target.files[0]);
            }
        });
    }

    if (previewContainer) {
        const editIcon = previewContainer.querySelector('.avatar-icon.edit');
        const cancelIcon = previewContainer.querySelector('.avatar-icon.cancel');

        if (editIcon) {
            editIcon.addEventListener('click', function() {
                profilePictureInput.click();
            });
        }

        if (cancelIcon) {
            cancelIcon.addEventListener('click', function() {
                previewContainer.classList.add('hidden');
                profilePictureContainer.classList.remove('hidden');
                profilePictureInput.value = '';
            });
        }
    }
}

function initializeAlterEgoHandlers() {
    const createAlterEgoButton = document.getElementById('createAlterEgoButton');
    const alterEgoSelect = document.getElementById('alterEgo');
    const saveAlterEgoButton = document.getElementById('saveAlterEgo');
    const alterEgoPictureContainer = document.getElementById('alterEgoPictureContainer');
    const alterEgoPictureInput = document.getElementById('alterEgoProfilePicture');
    const previewContainer = document.getElementById('previewAlterContainer');

    if (createAlterEgoButton) {
        createAlterEgoButton.addEventListener('click', function() {
            document.getElementById('alterEgoId').value = '';
            document.getElementById('alterEgoName').value = '';
            document.getElementById('alterEgoDescription').value = '';
            if (alterEgoPictureInput) alterEgoPictureInput.value = '';

            alterEgoPictureContainer.innerHTML = `
                <span class="avatar-initial" id="defaultAlterEgoInitial">P</span>
                <div class="avatar-icons">
                    <span class="avatar-icon edit"><i class="fas fa-pencil-alt"></i></span>
                </div>
            `;

            previewContainer.classList.add('hidden');
            alterEgoPictureContainer.classList.remove('hidden');

            alterEgoModal.show();
        });
    }

    if (alterEgoSelect) {
        alterEgoSelect.addEventListener('change', function() {
            const selectedAlterEgoId = this.value;
            if (selectedAlterEgoId && selectedAlterEgoId !== "0") {
                showAlterEgoDetails(selectedAlterEgoId);
            } else {
                const alterEgoDetails = document.getElementById('alterEgoDetails');
                if (alterEgoDetails) alterEgoDetails.innerHTML = '';
            }
        });
    }

    if (saveAlterEgoButton) {
        saveAlterEgoButton.addEventListener('click', saveAlterEgo);
    }

    if (alterEgoPictureContainer) {
        alterEgoPictureContainer.addEventListener('click', function(e) {
            if (e.target.closest('.avatar-icon.edit')) {
                alterEgoPictureInput.click();
            } else if (e.target.closest('.avatar-icon.delete')) {
                deleteAlterEgoPicture();
            }
        });
    }

    if (alterEgoPictureInput) {
        alterEgoPictureInput.addEventListener('change', function(e) {
            if (e.target.files && e.target.files[0]) {
                let reader = new FileReader();
                reader.onload = function(event) {
                    previewContainer.classList.remove('hidden');
                    document.getElementById('previewAlterEgoImage').src = event.target.result;
                    alterEgoPictureContainer.classList.add('hidden');
                };
                reader.readAsDataURL(e.target.files[0]);
            }
        });
    }

    if (previewContainer) {
        previewContainer.addEventListener('click', function(e) {
            if (e.target.closest('.avatar-icon.edit')) {
                alterEgoPictureInput.click();
            } else if (e.target.closest('.avatar-icon.cancel')) {
                cancelAlterEgoPictureChange();
            }
        });
    }
}

// Add this line at the end of the DOMContentLoaded event listener
document.getElementById('alterEgo').addEventListener('change', function() {
    const selectedAlterEgoId = this.value;
    if (selectedAlterEgoId && selectedAlterEgoId !== "0") {
        showAlterEgoDetails(selectedAlterEgoId);
    } else {
        document.getElementById('alterEgoDetails').innerHTML = '';
    }
});

document.getElementById('deleteAccountBtn').addEventListener('click', async (e) => {
    e.preventDefault();
    
    // First warning
    const genericModal = new bootstrap.Modal(document.getElementById('genericModal'));
    const modalTitle = document.getElementById('genericModalLabel');
    const modalBody = document.getElementById('genericModalBody');
    const confirmBtn = document.getElementById('genericModalConfirmBtn');
    const cancelBtn = document.getElementById('genericModalCancelBtn');

    modalTitle.textContent = 'Warning';
    modalBody.innerHTML = `
        <p class="text-danger"><strong>You are about to delete your account.</strong></p>
        <p>This action will:</p>
        <ul>
            <li>Delete all your personal information</li>
            <li>Remove all your prompts and configurations</li>
            <li>Cancel any active subscriptions</li>
            <li>This action cannot be undone</li>
        </ul>
        <p>Are you sure you want to continue?</p>
    `;

    confirmBtn.textContent = 'Continue';
    confirmBtn.className = 'btn btn-danger';

    const handleFirstConfirmation = () => {
        // Second confirmation
        modalBody.innerHTML = `
            <p class="text-danger"><strong>Final confirmation required</strong></p>
            <p>Please type "DELETE" below to confirm you want to permanently delete your account:</p>
            <input type="text" class="form-control" id="deleteConfirmInput" placeholder="Type DELETE">
        `;
        
        confirmBtn.textContent = 'Delete Account';
        
        const handleFinalConfirmation = async () => {
            const confirmInput = document.getElementById('deleteConfirmInput');
            if (confirmInput.value === 'DELETE') {
                try {
                    const response = await secureFetch('/api/delete-account', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    });

                    if (!response) return; // Session expired
                    if (response.ok) {
                        window.location.href = '/logout';
                    } else {
                        const data = await response.json();
                        throw new Error(data.detail || 'Error deleting account');
                    }
                } catch (error) {
                    modalBody.innerHTML = `<p class="text-danger">Error: ${error.message}</p>`;
                    confirmBtn.style.display = 'none';
                }
            } else {
                modalBody.innerHTML += `<p class="text-danger mt-2">Please type "DELETE" correctly to confirm.</p>`;
            }
        };
        
        confirmBtn.removeEventListener('click', handleFirstConfirmation);
        confirmBtn.addEventListener('click', handleFinalConfirmation);
    };
    
    confirmBtn.addEventListener('click', handleFirstConfirmation);
    genericModal.show();
});

function setupUsernameValidation() {
    const usernameInput = document.getElementById('username');
    const originalUsername = usernameInput.value;
    let timeoutId;

    usernameInput.addEventListener('input', function() {
        clearTimeout(timeoutId);
        const username = this.value.trim();

        if (username.toLowerCase() === originalUsername.toLowerCase()) {
            this.setCustomValidity('');
            return;
        }

        timeoutId = setTimeout(async () => {
            try {
                const response = await secureFetch('/api/check-username', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ username })
                });

                if (!response) return; // Session expired
                const data = await response.json();

                if (data.exists) {
                    this.setCustomValidity('This username is already taken');
                    showUsernameError('This username is already taken');
                } else {
                    this.setCustomValidity('');
                    hideUsernameError();
                }
            } catch (error) {
                console.error('Error checking username:', error);
            }
        }, 500);
    });
}

function showUsernameError(message) {
    let errorDiv = document.querySelector('.username-error');
    if (!errorDiv) {
        errorDiv = document.createElement('div');
        errorDiv.className = 'username-error text-danger mt-1';
        const usernameInput = document.getElementById('username');
        usernameInput.parentNode.appendChild(errorDiv);
    }
    errorDiv.textContent = message;
}

function hideUsernameError() {
    const errorDiv = document.querySelector('.username-error');
    if (errorDiv) {
        errorDiv.remove();
    }
}