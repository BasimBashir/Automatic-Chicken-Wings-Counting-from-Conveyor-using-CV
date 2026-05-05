const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
const results = document.getElementById("results");
const wingCount = document.getElementById("wingCount");
const originalImg = document.getElementById("originalImg");
const annotatedImg = document.getElementById("annotatedImg");
const downloadBtn = document.getElementById("downloadBtn");

uploadZone.addEventListener("click", () => fileInput.click());

uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
});

uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("dragover");
});

uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length) {
        handleFile(fileInput.files[0]);
    }
});

async function handleFile(file) {
    originalImg.src = URL.createObjectURL(file);

    uploadZone.querySelector(".label").textContent = "Processing...";
    const formData = new FormData();
    formData.append("file", file);

    try {
        const resp = await fetch("/api/image/detect", { method: "POST", body: formData });
        const count = resp.headers.get("X-Wing-Count") || "0";
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);

        annotatedImg.src = url;
        downloadBtn.href = url;
        wingCount.textContent = count;
        results.style.display = "block";
        uploadZone.querySelector(".label").textContent = "Drop another image or click to browse";
    } catch (err) {
        uploadZone.querySelector(".label").textContent = "Error - try again";
        console.error(err);
    }
}
