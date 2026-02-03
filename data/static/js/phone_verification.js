function getFullPhoneNumber() {
    return phoneInputJS.getNumber(intlTelInputUtils.numberFormat.E164);
}
document.addEventListener('DOMContentLoaded', function() {
    initPhoneInput();

	function initPhoneInput() {
		const phoneInputField = document.getElementById('phone');
		phoneInputJS = window.intlTelInput(phoneInputField, {
			initialCountry: "auto",
			separateDialCode: true,
			utilsScript: "https://cdnjs.cloudflare.com/ajax/libs/intl-tel-input/17.0.13/js/utils.js",
			geoIpLookup: function(success, failure) {
				fetch('/api/get-ip-info')
					.then(function(response) {
						if (response.ok) return response.json();
						throw new Error('Failed to fetch IP info');
					})
					.then(function(ipinfo) {
						success(ipinfo.country);
					})
					.catch(function() {
						success("us");
					});
			},
		});
	}

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
            const response = await fetch('/api/verify-code', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ phone: phoneNumber, code: code }),
            });
            const result = await response.json();
            if (result.status === 'approved') {
                NotificationModal.success('Verified', 'Phone number verified successfully!');
                verifyCodeButton.style.display = 'none';
                verificationCodeInput.disabled = true;
                // Add flag to indicate the number has been verified
                phoneInput.dataset.verified = 'true';
            } else {
                NotificationModal.error('Verification Failed', 'Please check the code and try again.');
                verificationCodeInput.value = ''; // Clear the input
            }
        } catch (error) {
            console.error('Error:', error);
            NotificationModal.error('Error', 'An error occurred while verifying the code. Please try again.');
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
            const checkResponse = await fetch('/api/check-phone-number', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ phone: phoneNumber, user_id: currentUserId }), // Make sure currentUserId is available
            });
            const checkResult = await checkResponse.json();
            
            if (checkResult.exists) {
                NotificationModal.error('Phone In Use', 'This phone number is already in use. Please use a different number.');
                return;
            }

            // If number doesn't exist, proceed with code sending
            const response = await fetch('/api/send-verification-code', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ phone: phoneNumber }),
            });
            const result = await response.json();
            if (response.ok) {
                if (result.status === 'pending') {
                    verificationCodeContainer.style.display = 'block';
                    NotificationModal.success('Code Sent', 'Verification code sent successfully!');
                } else {
                    NotificationModal.warning('Unexpected Status', `Received unexpected status: ${result.status}`);
                }
            } else {
                NotificationModal.error('Error', `Error sending verification code: ${result.detail}`);
            }
        } catch (error) {
            console.error('Error:', error);
            NotificationModal.error('Unexpected Error', 'An unexpected error occurred. Please try again.');
        }
    });
});