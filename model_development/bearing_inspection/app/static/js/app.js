/**
 * Casting Defect Detector — Client Application Logic
 * Pure Vanilla JavaScript (No external frameworks)
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropzone = document.getElementById("dropzone");
  const imageInput = document.getElementById("image-input");
  const dropzonePrompt = document.getElementById("dropzone-prompt");
  const previewWrapper = document.getElementById("preview-wrapper");
  const imagePreview = document.getElementById("image-preview");
  const previewFilename = document.getElementById("preview-filename");
  const previewFilesize = document.getElementById("preview-filesize");
  const btnReplace = document.getElementById("btn-replace");
  const btnRemove = document.getElementById("btn-remove");
  const btnSubmit = document.getElementById("btn-submit");
  const btnText = document.getElementById("btn-text");
  const btnSpinner = document.getElementById("btn-spinner");
  const clientErrorBox = document.getElementById("client-error-box");
  const clientErrorMessage = document.getElementById("client-error-message");

  // Defect Guidance Elements
  const previewViewSwitcher = document.getElementById("preview-view-switcher");
  const btnShowDefects = document.getElementById("btn-show-defects");
  const btnShowOriginal = document.getElementById("btn-show-original");
  const defectGuidanceSection = document.getElementById("defect-guidance-section");
  const defectCountBadge = document.getElementById("defect-count-badge");
  const defectLocationsContainer = document.getElementById("defect-locations-container");
  const defectSvgOverlay = document.getElementById("defect-svg-overlay");

  let originalImageSrc = null;
  let annotatedImageSrc = null;

  // Result Elements
  const resultStatusBadge = document.getElementById("result-status-badge");
  const resultEmptyState = document.getElementById("result-empty-state");
  const resultLoadingState = document.getElementById("result-loading-state");
  const resultContent = document.getElementById("result-content");
  const verdictBanner = document.getElementById("verdict-banner");
  const verdictLabel = document.getElementById("verdict-label");
  const verdictConfidence = document.getElementById("verdict-confidence");
  const progressBarOk = document.getElementById("progress-bar-ok");
  const progressBarDefective = document.getElementById("progress-bar-defective");
  const scoreValOk = document.getElementById("score-val-ok");
  const scoreValDefective = document.getElementById("score-val-defective");
  const metaInspectionId = document.getElementById("meta-inspection-id");
  const metaModelVersion = document.getElementById("meta-model-version");
  const metaProcessingTime = document.getElementById("meta-processing-time");
  const metaDimensions = document.getElementById("meta-dimensions");
  const metaBrightness = document.getElementById("meta-brightness");
  const metaBlur = document.getElementById("meta-blur");
  const reviewReasonsBox = document.getElementById("review-reasons-box");
  const reviewReasonsList = document.getElementById("review-reasons-list");

  // Operator Review Elements
  const operatorReviewForm = document.getElementById("operator-review-form");
  const operatorStatusSelect = document.getElementById("operator-status-select");
  const operatorNoteInput = document.getElementById("operator-note-input");
  const reviewSaveMsg = document.getElementById("review-save-msg");

  // History Elements
  const historyTableBody = document.getElementById("history-table-body");
  const btnRefreshHistory = document.getElementById("btn-refresh-history");

  let currentSelectedFile = null;
  let currentInspectionId = null;

  // Maximum file size in bytes (10 MB)
  const MAX_FILE_SIZE = 10 * 1024 * 1024;
  const ALLOWED_EXTS = ["jpg", "jpeg", "png"];

  // ==========================================================================
  // File Selection & Drag-and-Drop Handlers
  // ==========================================================================

  dropzone.addEventListener("click", (e) => {
    // Only trigger file picker if user did not click action buttons inside preview
    if (!e.target.closest("button")) {
      imageInput.click();
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  imageInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  btnReplace.addEventListener("click", (e) => {
    e.stopPropagation();
    imageInput.click();
  });

  btnRemove.addEventListener("click", (e) => {
    e.stopPropagation();
    resetUploadState();
    resetResultPanel();
  });

  btnShowDefects.addEventListener("click", (e) => {
    e.stopPropagation();
    btnShowDefects.classList.add("active");
    btnShowOriginal.classList.remove("active");
    if (annotatedImageSrc) {
      imagePreview.src = annotatedImageSrc;
    }
  });

  btnShowOriginal.addEventListener("click", (e) => {
    e.stopPropagation();
    btnShowOriginal.classList.add("active");
    btnShowDefects.classList.remove("active");
    if (originalImageSrc) {
      imagePreview.src = originalImageSrc;
    }
  });

  function showError(msg) {
    clientErrorMessage.textContent = msg;
    clientErrorBox.style.display = "block";
  }

  function clearError() {
    clientErrorBox.style.display = "none";
    clientErrorMessage.textContent = "";
  }

  function formatBytes(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  function handleFile(file) {
    clearError();

    // Check extension
    const ext = file.name.split(".").pop().toLowerCase();
    if (!ALLOWED_EXTS.includes(ext)) {
      showError(`Format file '${ext}' tidak didukung. Harap pilih file gambar JPG, JPEG, atau PNG.`);
      resetUploadState();
      return;
    }

    // Check size
    if (file.size > MAX_FILE_SIZE) {
      showError(`Ukuran file (${formatBytes(file.size)}) melebihi batas maksimum 10 MB.`);
      resetUploadState();
      return;
    }

    if (file.size === 0) {
      showError("File gambar yang dipilih kosong (0 bytes).");
      resetUploadState();
      return;
    }

    currentSelectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      originalImageSrc = e.target.result;
      annotatedImageSrc = null;
      imagePreview.src = originalImageSrc;
      previewViewSwitcher.style.display = "none";
      defectGuidanceSection.style.display = "none";
      previewFilename.textContent = file.name;
      previewFilesize.textContent = formatBytes(file.size);
      dropzonePrompt.style.display = "none";
      previewWrapper.style.display = "flex";
      btnSubmit.disabled = false;
    };
    reader.onerror = () => {
      showError("Gagal membaca file gambar lokal.");
      resetUploadState();
    };
    reader.readAsDataURL(file);
  }

  function resetUploadState() {
    currentSelectedFile = null;
    imageInput.value = "";
    imagePreview.src = "#";
    previewWrapper.style.display = "none";
    dropzonePrompt.style.display = "block";
    btnSubmit.disabled = true;
  }

  function resetResultPanel() {
    resultStatusBadge.className = "badge badge-neutral";
    resultStatusBadge.textContent = "Menunggu Citra";
    resultEmptyState.style.display = "block";
    resultLoadingState.style.display = "none";
    resultContent.style.display = "none";
    previewViewSwitcher.style.display = "none";
    defectGuidanceSection.style.display = "none";
    if (originalImageSrc) {
      imagePreview.src = originalImageSrc;
    }
    currentInspectionId = null;
  }

  // ==========================================================================
  // Form Submission & Analysis
  // ==========================================================================

  const inspectionForm = document.getElementById("inspection-form");
  inspectionForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!currentSelectedFile) return;

    clearError();
    // Set UI to loading state
    btnSubmit.disabled = true;
    btnText.textContent = "Menganalisis...";
    btnSpinner.style.display = "inline-block";

    resultEmptyState.style.display = "none";
    resultContent.style.display = "none";
    resultLoadingState.style.display = "block";
    resultStatusBadge.className = "badge badge-neutral";
    resultStatusBadge.textContent = "Sedang Memproses...";

    const formData = new FormData();
    formData.append("image", currentSelectedFile);

    try {
      const response = await fetch("/api/inspect/casting", {
        method: "POST",
        body: formData
      });

      const data = await response.json();
      renderInspectionResult(data);
    } catch (err) {
      showError("Gagal terhubung ke server inspeksi: " + err.message);
      resetResultPanel();
    } finally {
      btnSubmit.disabled = false;
      btnText.textContent = "Mulai Analisis Inspeksi";
      btnSpinner.style.display = "none";
      resultLoadingState.style.display = "none";
      loadHistory();
    }
  });

  // ==========================================================================
  // Render Inspection Results
  // ==========================================================================

  function renderInspectionResult(data) {
    if (data.status === "invalid_input" || data.status === "error") {
      showError(data.error_message || "Validasi atau inspeksi gagal.");
      resetResultPanel();
      return;
    }

    if (data.status === "model_unavailable") {
      showError("Model Belum Tersedia: " + (data.error_message || "Bobot model belum dikonfigurasi."));
      resetResultPanel();
      resultStatusBadge.className = "badge badge-warning";
      resultStatusBadge.textContent = "Model Belum Tersedia";
      return;
    }

    currentInspectionId = data.inspection_id;
    resultEmptyState.style.display = "none";
    resultLoadingState.style.display = "none";
    resultContent.style.display = "block";

    // Verdict styling
    verdictBanner.className = "verdict-banner";
    const label = data.predicted_label;

    if (label === "OK") {
      verdictBanner.classList.add("verdict-ok");
      verdictLabel.textContent = "OK (Lolos Mutu)";
      resultStatusBadge.className = "badge badge-success";
      resultStatusBadge.textContent = "Lolos (OK)";
    } else if (label === "Defective") {
      verdictBanner.classList.add("verdict-defective");
      verdictLabel.textContent = "DEFECTIVE (Cacat)";
      resultStatusBadge.className = "badge badge-danger";
      resultStatusBadge.textContent = "Cacat (Defective)";
    } else {
      verdictBanner.classList.add("verdict-review");
      verdictLabel.textContent = "PERLU REVIEW";
      resultStatusBadge.className = "badge badge-warning";
      resultStatusBadge.textContent = "Perlu Review";
    }

    // Scores
    const scoreVal = data.confidence_score !== null ? (data.confidence_score * 100).toFixed(1) + "%" : "-";
    verdictConfidence.textContent = scoreVal;

    const okScore = (data.scores && data.scores.OK !== undefined) ? (data.scores.OK * 100) : 0;
    const defScore = (data.scores && data.scores.Defective !== undefined) ? (data.scores.Defective * 100) : 0;

    scoreValOk.textContent = okScore.toFixed(1) + "%";
    progressBarOk.style.width = okScore + "%";

    scoreValDefective.textContent = defScore.toFixed(1) + "%";
    progressBarDefective.style.width = defScore + "%";

    // Metadata
    metaInspectionId.textContent = data.inspection_id ? data.inspection_id.substring(0, 8) + "..." : "-";
    metaModelVersion.textContent = data.model_version || "-";
    metaProcessingTime.textContent = data.processing_ms ? data.processing_ms + " ms" : "-";

    if (data.quality_details && data.quality_details.dimensions) {
      metaDimensions.textContent = `${data.quality_details.dimensions[0]} x ${data.quality_details.dimensions[1]} px`;
      metaBrightness.textContent = `${data.quality_details.mean_brightness} / 255`;
      metaBlur.textContent = `${data.quality_details.blur_score}`;
    } else {
      metaDimensions.textContent = "-";
      metaBrightness.textContent = "-";
      metaBlur.textContent = "-";
    }

    // Review Reasons
    if (data.review_reasons && data.review_reasons.length > 0) {
      reviewReasonsBox.style.display = "block";
      reviewReasonsList.innerHTML = "";
      data.review_reasons.forEach(r => {
        const li = document.createElement("li");
        li.textContent = r;
        reviewReasonsList.appendChild(li);
      });
    } else {
      reviewReasonsBox.style.display = "none";
    }

    // Defect Localization (Bounding Boxes & Marker Dots)
    if (data.defect_localization && data.defect_localization.has_defect) {
      annotatedImageSrc = data.defect_localization.annotated_image;
      previewViewSwitcher.style.display = "flex";
      btnShowDefects.classList.add("active");
      btnShowOriginal.classList.remove("active");

      // Switch image preview to show annotated defect boxes & dots
      if (annotatedImageSrc) {
        imagePreview.src = annotatedImageSrc;
      }

      defectGuidanceSection.style.display = "block";
      const count = data.defect_localization.defect_count || data.defect_localization.boxes.length;
      defectCountBadge.textContent = `${count} Area Terdeteksi`;

      defectLocationsContainer.innerHTML = "";
      data.defect_localization.boxes.forEach(box => {
        const chip = document.createElement("div");
        chip.className = "defect-chip";
        chip.innerHTML = `
          <div class="defect-chip-title">
            <span>🔴 ${box.label}</span>
            <span>Intensitas: ${box.intensity}</span>
          </div>
          <div class="defect-chip-coord">
            Koordinat: X=${box.x}px, Y=${box.y}px<br>
            Ukuran: ${box.width} x ${box.height} px
          </div>
        `;
        defectLocationsContainer.appendChild(chip);
      });
    } else {
      previewViewSwitcher.style.display = "none";
      defectGuidanceSection.style.display = "none";
      if (originalImageSrc) {
        imagePreview.src = originalImageSrc;
      }
    }

    // Reset operator review fields
    operatorStatusSelect.value = "";
    operatorNoteInput.value = "";
    reviewSaveMsg.textContent = "";
  }

  // ==========================================================================
  // Operator Manual Review Form (FR-CAST-016)
  // ==========================================================================

  operatorReviewForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!currentInspectionId) return;

    const opReview = operatorStatusSelect.value;
    const opNote = operatorNoteInput.value;

    try {
      const res = await fetch(`/api/inspections/${currentInspectionId}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          operator_review: opReview,
          operator_note: opNote
        })
      });

      const resData = await res.json();
      if (res.ok) {
        reviewSaveMsg.textContent = "✓ Verifikasi berhasil disimpan";
        reviewSaveMsg.className = "review-save-msg success";
        loadHistory();
      } else {
        reviewSaveMsg.textContent = "Gagal: " + resData.error_message;
        reviewSaveMsg.className = "review-save-msg text-danger";
      }
    } catch (err) {
      reviewSaveMsg.textContent = "Gagal menyimpan: " + err.message;
      reviewSaveMsg.className = "review-save-msg text-danger";
    }
  });

  // ==========================================================================
  // History Log Loading (FR-CAST-015)
  // ==========================================================================

  btnRefreshHistory.addEventListener("click", () => {
    loadHistory();
  });

  async function loadHistory() {
    try {
      const res = await fetch("/api/history?limit=15");
      const data = await res.json();

      if (!data.inspections || data.inspections.length === 0) {
        historyTableBody.innerHTML = `
          <tr><td colspan="8" class="text-center text-muted">Belum ada catatan riwayat inspeksi.</td></tr>
        `;
        return;
      }

      historyTableBody.innerHTML = "";
      data.inspections.forEach(item => {
        const tr = document.createElement("tr");

        let badgeClass = "badge-neutral";
        if (item.predicted_label === "OK") badgeClass = "badge-success";
        else if (item.predicted_label === "Defective") badgeClass = "badge-danger";
        else if (item.predicted_label === "Perlu Review") badgeClass = "badge-warning";

        const shortId = item.inspection_id ? item.inspection_id.substring(0, 8) : "-";
        const scoreStr = item.confidence_score !== null ? (item.confidence_score * 100).toFixed(1) + "%" : "-";
        const opReviewStr = item.operator_review || '<span class="text-muted">Belum diverifikasi</span>';
        const noteStr = item.operator_note || "-";

        tr.innerHTML = `
          <td class="font-mono">${item.created_at || "-"}</td>
          <td class="font-mono"><code>${shortId}</code></td>
          <td><span class="badge ${badgeClass}">${item.status}</span></td>
          <td><strong>${item.predicted_label || "-"}</strong></td>
          <td class="font-mono">${scoreStr}</td>
          <td class="font-mono">${item.processing_ms || "-"}</td>
          <td>${opReviewStr}</td>
          <td>${noteStr}</td>
        `;
        historyTableBody.appendChild(tr);
      });
    } catch (err) {
      console.error("Gagal memuat riwayat inspeksi:", err);
    }
  }

  // Initial load
  loadHistory();
});

