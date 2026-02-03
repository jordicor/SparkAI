function startCountdown(targetElementId, durationInSeconds) {
    let endTime = localStorage.getItem("countdownEndTime");
    let now = new Date().getTime();

    if (!endTime || now > endTime) {
        endTime = now + durationInSeconds * 1000;
        localStorage.setItem("countdownEndTime", endTime);
    }

    let countdownFunction = setInterval(function() {
        now = new Date().getTime();
        let timeleft = endTime - now;
        
        let hours = Math.floor((timeleft % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
        let minutes = Math.floor((timeleft % (1000 * 60 * 60)) / (1000 * 60));
        let seconds = Math.floor((timeleft % (1000 * 60)) / 1000);
        
        document.getElementById(targetElementId).innerHTML = hours + "h " + minutes + "m " + seconds + "s ";
        
        if (timeleft < 0) {
            clearInterval(countdownFunction);
            document.getElementById(targetElementId).innerHTML = "EXPIRED";
            localStorage.removeItem("countdownEndTime");
        }
    }, 1000);
}
