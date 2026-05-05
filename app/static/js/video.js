const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
const uploadCard = document.getElementById("uploadCard");
const playerSection = document.getElementById("playerSection");

const btnPlay = document.getElementById("btnPlay");
const btnStop = document.getElementById("btnStop");
const btnCountStart = document.getElementById("btnCountStart");
const btnCountStop = document.getElementById("btnCountStop");
const feedImg = document.getElementById("feedImg");
const feedPlaceholder = document.getElementById("feedPlaceholder");
const statusDot = document.getElementById("statusDot");
const statusLabel = document.getElementById("statusLabel");
const wingCount = document.getElementById("wingCount");
const frameNum = document.getElementById("frameNum");
const fpsVal = document.getElementById("fpsVal");
const downloadBtn = document.getElementById("downloadBtn");

let sessionId = null;
let pollInterval = null;

uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("dragover"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleUpload(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files.length) handleUpload(fileInput.files[0]); });

async function handleUpload(file) {
    uploadZone.querySelector(".label").textContent = "Uploading...";
    const formData = new FormData();
    formData.append("file", file);

    try {
        const resp = await fetch("/api/video/upload", { method: "POST", body: formData });
        const data = await resp.json();
        sessionId = data.session_id;
        uploadCard.style.display = "none";
        playerSection.style.display = "block";
    } catch (err) {
        uploadZone.querySelector(".label").textContent = "Upload failed - try again";
        console.error(err);
    }
}

btnPlay.addEventListener("click", async () => {
    await fetch(`/api/video/${sessionId}/start`, { method: "POST" });
    feedImg.src = `/api/video/${sessionId}/feed?t=${Date.now()}`;
    feedImg.style.display = "block";
    feedPlaceholder.style.display = "none";
    btnPlay.disabled = true;
    btnStop.disabled = false;
    btnCountStart.disabled = false;
    setStatus(true, "Playing");
    startPolling();
});

btnStop.addEventListener("click", async () => {
    await fetch(`/api/video/${sessionId}/stop`, { method: "POST" });
    btnPlay.disabled = false;
    btnStop.disabled = true;
    btnCountStart.disabled = true;
    btnCountStop.disabled = true;
    setStatus(false, "Stopped");
    stopPolling();
    showDownload();
});

btnCountStart.addEventListener("click", async () => {
    await fetch(`/api/video/${sessionId}/counting/start`, { method: "POST" });
    btnCountStart.disabled = true;
    btnCountStop.disabled = false;
});

btnCountStop.addEventListener("click", async () => {
    await fetch(`/api/video/${sessionId}/counting/stop`, { method: "POST" });
    btnCountStart.disabled = false;
    btnCountStop.disabled = true;
});

function setStatus(active, text) {
    statusDot.className = `status-dot ${active ? "active" : "inactive"}`;
    statusLabel.textContent = text;
}

function startPolling() {
    pollInterval = setInterval(async () => {
        try {
            const resp = await fetch(`/api/video/${sessionId}/status`);
            const s = resp.ok ? await resp.json() : null;
            if (!s) return;

            wingCount.textContent = s.wing_count;
            frameNum.textContent = s.total_frames > 0
                ? `${s.frame_num}/${s.total_frames}`
                : s.frame_num;
            fpsVal.textContent = s.fps;

            if (s.is_complete) {
                setStatus(false, "Complete");
                btnPlay.disabled = false;
                btnStop.disabled = true;
                btnCountStart.disabled = true;
                btnCountStop.disabled = true;
                stopPolling();
                showDownload();
            }
        } catch (_) {}
    }, 500);
}

function stopPolling() {
    if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
}

function showDownload() {
    downloadBtn.href = `/api/video/${sessionId}/download`;
    downloadBtn.style.display = "inline-flex";
}
