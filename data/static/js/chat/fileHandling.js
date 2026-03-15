const MAX_PDF_SIZE_MB = 25;

function isAcceptedFileType(file) {
    return file.type.startsWith('image/') || file.type === 'application/pdf';
}

function validateFileSize(file) {
    if (file.type === 'application/pdf' && file.size > MAX_PDF_SIZE_MB * 1024 * 1024) {
        NotificationModal.warning('File too large', `PDF files must be under ${MAX_PDF_SIZE_MB}MB`);
        return false;
    }
    return true;
}

function handlePasteEvent(event) {

    if (!Config.have_vision) return;
    var items = (event.clipboardData || event.originalEvent.clipboardData).items;
    for (var index in items) {
        var item = items[index];
        if (item.kind === 'file' && item.type.startsWith('image/')) {
            var blob = item.getAsFile();
            attachedFiles.push(blob);
            var reader = new FileReader();
            reader.onload = function(event){
                var img = document.createElement('img');
                img.src = event.target.result;
                img.className = 'preview-image';
                document.getElementById('image-previews').appendChild(img);
                document.getElementById('image-previews').classList.remove('hidden');
            };
            reader.readAsDataURL(blob);
        }
    }
    setTimeout(() => {
        document.getElementById('message-text').focus();
    }, 0);
}

function processFiles(files, formData, imagePreviews) {
    for (var i = 0; i < files.length; i++) {
        if (!isAcceptedFileType(files[i]) || !validateFileSize(files[i])) {
            if (!isAcceptedFileType(files[i])) {
                NotificationModal.warning('Invalid File', 'Only image and PDF files are allowed.');
            }
            continue;
        }

        formData.append('file', files[i]);

        if (files[i].type === 'application/pdf') {
            // Show PDF attachment in chat (mirrors the image pattern)
            var userMessageElement = document.createElement('div');
            userMessageElement.classList.add('message', 'user');
            var pdfAttachment = document.createElement('div');
            pdfAttachment.className = 'chat-pdf-attachment';
            var badge = document.createElement('span');
            badge.className = 'pdf-badge';
            badge.textContent = 'PDF';
            var label = document.createElement('span');
            label.textContent = ' ' + files[i].name;  // textContent — XSS safe
            pdfAttachment.appendChild(badge);
            pdfAttachment.appendChild(label);
            userMessageElement.appendChild(pdfAttachment);

            var chatMessagesContainer = document.getElementById('chat-messages-container');
            chatMessagesContainer.appendChild(userMessageElement);
            var chatWindow = document.getElementById('chat-window');
            chatWindow.scrollTop = chatWindow.scrollHeight;

            imagePreviews.innerHTML = '';
            imagePreviews.classList.add('hidden');
            document.getElementById('chat-files').value = '';
        } else {
            // Existing image preview logic
            var reader = new FileReader();
            reader.onload = function (e) {
                var img = document.createElement('img');
                img.src = e.target.result;
                img.className = 'preview-image';
                imagePreviews.appendChild(img);

                var userMessageElement = document.createElement('div');
                userMessageElement.classList.add('message', 'user');

                var userMessageImage = document.createElement('img');
                userMessageImage.src = e.target.result;
                userMessageImage.style.maxWidth = '100%';
                userMessageImage.style.height = 'auto';
                userMessageImage.style.display = 'block';
                userMessageImage.style.margin = '0 auto';

                userMessageElement.appendChild(userMessageImage);

                var chatMessagesContainer = document.getElementById('chat-messages-container');
                chatMessagesContainer.appendChild(userMessageElement);

                var chatWindow = document.getElementById('chat-window');
                chatWindow.scrollTop = chatWindow.scrollHeight;

                imagePreviews.innerHTML = '';
                imagePreviews.classList.add('hidden');
                document.getElementById('chat-files').value = '';
            };
            reader.onerror = function (e) {
                console.error('Error reading file:', e);
            };
            reader.readAsDataURL(files[i]);
        }
    }
    if (imagePreviews.children.length > 0) {
        imagePreviews.classList.remove('hidden');
    }
}


if (Config.have_vision) {
    document.addEventListener('paste', handlePasteEvent);
    document.getElementById('chat-files').addEventListener('change', handleFileSelect);
}

function handleFileSelect(event) {
    const files = event.target.files;
    const imagePreviews = document.getElementById('image-previews');

    for (const file of files) {
        if (!isAcceptedFileType(file) || !validateFileSize(file)) {
            if (!isAcceptedFileType(file)) {
                NotificationModal.warning('Invalid File', 'Only image and PDF files are allowed.');
            }
            continue;
        }

        if (file.type === 'application/pdf') {
            // PDF preview
            const pdfPreview = document.createElement('div');
            pdfPreview.className = 'pdf-preview-item';
            const iconSpan = document.createElement('span');
            iconSpan.className = 'pdf-icon';
            iconSpan.textContent = 'PDF';
            const nameSpan = document.createElement('span');
            nameSpan.className = 'pdf-name';
            nameSpan.textContent = file.name;
            pdfPreview.appendChild(iconSpan);
            pdfPreview.appendChild(nameSpan);
            imagePreviews.appendChild(pdfPreview);
        } else {
            // Existing image preview logic
            const reader = new FileReader();
            reader.onload = function (e) {
                const img = document.createElement('img');
                img.src = e.target.result;
                img.className = 'preview-image';
                imagePreviews.appendChild(img);
            };
            reader.onerror = function (e) {
                console.error('Error reading file:', e);
            };
            reader.readAsDataURL(file);
        }
        attachedFiles.push(file);
    }

    if (attachedFiles.length > 0) {
        imagePreviews.classList.remove('hidden');
    }
    document.getElementById('message-text').focus();
}
