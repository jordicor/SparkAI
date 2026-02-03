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
        if (files[i].type.startsWith('image/')) {
            formData.append('file', files[i]);

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

                // Here we ensure it's added to the messages container
                var chatMessagesContainer = document.getElementById('chat-messages-container');
                chatMessagesContainer.appendChild(userMessageElement);
                
                var chatWindow = document.getElementById('chat-window');
                chatWindow.scrollTop = chatWindow.scrollHeight;

                imagePreviews.innerHTML = '';
                imagePreviews.classList.add('hidden');
                document.getElementById('image-files').value = '';
            };
            reader.onerror = function (e) {
                console.error('Error reading file:', e);
            };
            reader.readAsDataURL(files[i]);
        } else {
            NotificationModal.warning('Invalid File', 'Only image files are allowed.');
        }
    }
}


if (Config.have_vision) {
    document.addEventListener('paste', handlePasteEvent);
    document.getElementById('image-files').addEventListener('change', handleFileSelect);
}

function handleFileSelect(event) {
	const files = event.target.files;
	const imagePreviews = document.getElementById('image-previews');

	for (const file of files) {
		if (file.type.startsWith('image/')) {
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
			attachedFiles.push(file);
		} else {
			NotificationModal.warning('Invalid File', 'Only image files are allowed.');
		}
	}

	if (attachedFiles.length > 0) {
		imagePreviews.classList.remove('hidden');
	}
    document.getElementById('message-text').focus();
}