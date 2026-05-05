const rtspUrl = document.getElementById("rtspUrl");
const btnConnect = document.getElementById("btnConnect");
const btnDisconnect = document.getElementById("btnDisconnect");
const btnCountStart = document.getElementById("btnCountStart");
const btnCountStop = document.getElementById("btnCountStop");
const feedImg = document.getElementById("feedImg");
const feedPlaceholder = document.getElementById("feedPlaceholder");
const statusDot = document.getElementById("statusDot");
const statusLabel = document.getElementById("statusLabel");
const wingCount = document.getElementById("wingCount");
const fpsVal = document.getElementById("fpsVal");
const roiSlider = document.getElementById("roiSlider");
const roiValue = document.getElementById("roiValue");
const confSlider = document.getElementById("confSlider");
const confValue = document.getElementById("confValue");

let pollInterval = null;

fetch("/api/config").then(r => r.json()).then(cfg => {
    if (cfg.rtsp_url) rtspUrl.value = cfg.rtsp_url;
    roiSlider.value = cfg.roi_position;
    roiValue.textContent = cfg.roi_position.toFixed(2);
    confSlider.value = cfg.confidence;
    confValue.textContent = cfg.confidence.toFixed(2);
});

btnConnect.addEventListener("click", async () => {
    const url = rtspUrl.value.trim();
    const body = url ? { url } : {};
    try {
        const resp = await fetch("/api/stream/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        if (!resp.ok) {
            const err = await resp.json();
            alert(err.detail || "Failed to connect");
            return;
        }
        feedImg.src = `/api/stream/feed?t=${Date.now()}`;
        feedImg.style.display = "block";
        feedPlaceholder.style.display = "none";
        btnConnect.disabled = true;
        btnDisconnect.disabled = false;
        btnCountStart.disabled = false;
        setStatus(true, "Connected");
        startPolling();
    } catch (err) {
        console.error(err);
    }
});

btnDisconnect.addEventListener("click", async () => {
    await fetch("/api/stream/stop", { method: "POST" });
    feedImg.style.display = "none";
    feedPlaceholder.style.display = "block";
    btnConnect.disabled = false;
    btnDisconnect.disabled = true;
    btnCountStart.disabled = true;
    btnCountStop.disabled = true;
    setStatus(false, "Disconnected");
    stopPolling();
});

btnCountStart.addEventListener("click", async () => {
    await fetch("/api/stream/counting/start", { method: "POST" });
    btnCountStart.disabled = true;
    btnCountStop.disabled = false;
});

btnCountStop.addEventListener("click", async () => {
    await fetch("/api/stream/counting/stop", { method: "POST" });
    btnCountStart.disabled = false;
    btnCountStop.disabled = true;
});

roiSlider.addEventListener("input", () => {
    roiValue.textContent = parseFloat(roiSlider.value).toFixed(2);
});
roiSlider.addEventListener("change", () => {
    fetch("/api/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ roi_position: parseFloat(roiSlider.value) }),
    });
});

confSlider.addEventListener("input", () => {
    confValue.textContent = parseFloat(confSlider.value).toFixed(2);
});
confSlider.addEventListener("change", () => {
    fetch("/api/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confidence: parseFloat(confSlider.value) }),
    });
});

function setStatus(active, text) {
    statusDot.className = `status-dot ${active ? "active" : "inactive"}`;
    statusLabel.textContent = text;
}

function startPolling() {
    pollInterval = setInterval(async () => {
        try {
            const resp = await fetch("/api/stream/status");
            const s = await resp.json();
            wingCount.textContent = s.wing_count;
            fpsVal.textContent = s.fps;
            if (!s.is_connected) {
                setStatus(false, s.error || "Disconnected");
                btnConnect.disabled = false;
                btnDisconnect.disabled = true;
                stopPolling();
            }
        } catch (_) {}
    }, 500);
}

function stopPolling() {
    if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
}
