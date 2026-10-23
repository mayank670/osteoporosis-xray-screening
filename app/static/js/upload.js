// Upload page: drag-and-drop, preview, client-side size check, loading state.
// The server re-validates everything — these checks only give faster feedback.
(function () {
  "use strict";
  const form = document.getElementById("upload-form");
  if (!form) return;

  const input = document.getElementById("image");
  const dropzone = document.getElementById("dropzone");
  const preview = document.getElementById("preview");
  const prompt = document.getElementById("dropzone-prompt");
  const fileName = document.getElementById("file-name");
  const errorBox = document.getElementById("client-error");
  const button = document.getElementById("submit-button");
  const maxBytes = Number(document.getElementById("upload-script").dataset.maxBytes);
  let previewUrl = null;

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = !message;
  }

  function setLoading(loading) {
    button.disabled = loading;
    button.querySelector(".spinner").hidden = !loading;
    button.querySelector(".label").textContent = loading ? "Analysing… (a few seconds)" : "Analyse X-ray";
    form.setAttribute("aria-busy", String(loading));
  }

  function showFile(file) {
    showError("");
    if (previewUrl) URL.revokeObjectURL(previewUrl);  // free the previous preview's memory
    fileName.textContent = file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB` : "";
    if (file && file.size > maxBytes) {
      showError(`This file is larger than the ${Math.round(maxBytes / 1024 / 1024)} MB limit.`);
    }
    // Browsers cannot preview TIFF; the server still accepts it.
    if (file && /^image\/(png|jpeg|bmp)$/.test(file.type)) {
      previewUrl = URL.createObjectURL(file);
      preview.src = previewUrl;
      preview.hidden = false;
      prompt.hidden = true;
    } else {
      preview.hidden = true;
      prompt.hidden = false;
    }
  }

  input.addEventListener("change", () => showFile(input.files[0]));

  // When the user comes back with the browser's Back button, the browser can
  // restore the previously chosen file inside the <input> while our preview is
  // gone — the page then *looked* empty but silently re-sent the old file.
  // Fix: whenever the page is shown, redraw the preview from what the input
  // really contains, so what you see is exactly what will be sent.
  window.addEventListener("pageshow", () => {
    showFile(input.files.length ? input.files[0] : null);
    syncClinical();
    setLoading(false);
  });

  // Show the patient-details fields only when the checkbox is ticked.
  const clinicalToggle = document.getElementById("use_clinical");
  const clinicalFields = document.getElementById("clinical-fields");
  function syncClinical() {
    if (clinicalToggle && clinicalFields) clinicalFields.hidden = !clinicalToggle.checked;
  }
  if (clinicalToggle) clinicalToggle.addEventListener("change", syncClinical);

  ["dragenter", "dragover"].forEach((type) =>
    dropzone.addEventListener(type, (event) => { event.preventDefault(); dropzone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((type) =>
    dropzone.addEventListener(type, () => dropzone.classList.remove("dragover")));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    if (event.dataTransfer.files.length) {
      input.files = event.dataTransfer.files;
      showFile(input.files[0]);
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = input.files[0];
    if (!file) return showError("Choose an image to upload.");
    if (file.size > maxBytes) return showError("This file is too large.");

    setLoading(true);
    try {
      const response = await fetch(form.action, {
        method: "POST",
        body: new FormData(form),  // includes the CSRF token field
        headers: { Accept: "application/json" },
        credentials: "same-origin",
      });
      const data = await response.json().catch(() => ({}));
      if (response.ok && data.redirect) {
        window.location.assign(data.redirect);
        return;
      }
      showError(data.error || `Upload failed (HTTP ${response.status}).`);
    } catch (err) {
      showError("Network error — please check your connection and try again.");
    }
    setLoading(false);
  });
})();
