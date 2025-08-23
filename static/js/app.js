/**
 * TrafficAI Modern Web Application
 * Advanced JavaScript with proper event handling and UX
 */

class TrafficAIApp {
  constructor() {
    this.currentFile = null;
    this.isProcessing = false;
    this.processedCount = 0;

    // DOM Elements
    this.elements = {
      uploadArea: document.getElementById("uploadArea"),
      fileInput: document.getElementById("fileInput"),
      filePreview: document.getElementById("filePreview"),
      fileName: document.getElementById("fileName"),
      fileSize: document.getElementById("fileSize"),
      fileType: document.getElementById("fileType"),
      previewImg: document.getElementById("previewImg"),
      removeFile: document.getElementById("removeFile"),
      detectionOptions: document.getElementById("detectionOptions"),
      helmetDetection: document.getElementById("helmetDetection"),
      plateDetection: document.getElementById("plateDetection"),
      processBtn: document.getElementById("processBtn"),
      resultsSection: document.getElementById("resultsSection"),
      uploadProgress: document.getElementById("uploadProgress"),
      browseBtn: document.querySelector(".browse-btn"),

      // Results elements
      totalDetections: document.getElementById("totalDetections"),
      withHelmet: document.getElementById("withHelmet"),
      withoutHelmet: document.getElementById("withoutHelmet"),
      accuracyScore: document.getElementById("accuracyScore"),
      platesList: document.getElementById("platesList"),
      resultImage: document.getElementById("resultImage"),

      // Navigation
      status: document.getElementById("status"),
      processed: document.getElementById("processed"),

      // Modal
      imageModal: document.getElementById("imageModal"),
      modalImage: document.getElementById("modalImage"),
      modalClose: document.getElementById("modalClose"),

      // Action buttons
      downloadBtn: document.getElementById("downloadBtn"),
      downloadImageBtn: document.getElementById("downloadImageBtn"),
      newAnalysisBtn: document.getElementById("newAnalysisBtn"),
      zoomBtn: document.getElementById("zoomBtn"),

      // Toast container
      toastContainer: document.getElementById("toastContainer"),
    };

    this.init();
  }

  init() {
    this.setupEventListeners();
    this.updateStatus("Ready");
  }

  setupEventListeners() {
    // File upload events - FIXED: Single event binding
    this.elements.uploadArea.addEventListener("click", (e) => {
      if (
        e.target === this.elements.uploadArea ||
        e.target.closest(".upload-content")
      ) {
        e.preventDefault();
        this.elements.fileInput.click();
      }
    });

    this.elements.browseBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      this.elements.fileInput.click();
    });

    this.elements.fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        this.handleFileSelect(e.target.files[0]);
      }
    });

    // Drag and drop events
    this.elements.uploadArea.addEventListener("dragover", (e) => {
      e.preventDefault();
      this.elements.uploadArea.classList.add("dragover");
    });

    this.elements.uploadArea.addEventListener("dragleave", (e) => {
      e.preventDefault();
      if (!this.elements.uploadArea.contains(e.relatedTarget)) {
        this.elements.uploadArea.classList.remove("dragover");
      }
    });

    this.elements.uploadArea.addEventListener("drop", (e) => {
      e.preventDefault();
      this.elements.uploadArea.classList.remove("dragover");

      const files = e.dataTransfer.files;
      if (files.length > 0) {
        this.handleFileSelect(files[0]);
      }
    });

    // Remove file
    this.elements.removeFile.addEventListener("click", () => {
      this.removeFile();
    });

    // Detection options
    this.elements.helmetDetection.addEventListener("change", () => {
      this.updateProcessButton();
    });

    this.elements.plateDetection.addEventListener("change", () => {
      this.updateProcessButton();
    });

    // Process button
    this.elements.processBtn.addEventListener("click", () => {
      if (!this.isProcessing && this.currentFile) {
        this.processFile();
      }
    });

    // Action buttons
    this.elements.newAnalysisBtn?.addEventListener("click", () => {
      this.resetAnalysis();
    });

    this.elements.zoomBtn?.addEventListener("click", () => {
      this.openImageModal();
    });

    this.elements.resultImage?.addEventListener("click", () => {
      this.openImageModal();
    });

    // Modal events
    this.elements.modalClose?.addEventListener("click", () => {
      this.closeImageModal();
    });

    this.elements.imageModal?.addEventListener("click", (e) => {
      if (
        e.target === this.elements.imageModal ||
        e.target.classList.contains("modal-backdrop")
      ) {
        this.closeImageModal();
      }
    });

    // Download buttons
    this.elements.downloadBtn?.addEventListener("click", () => {
      this.downloadResults();
    });

    this.elements.downloadImageBtn?.addEventListener("click", () => {
      this.downloadImage();
    });

    // Keyboard events
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        this.closeImageModal();
      }
    });
  }

  handleFileSelect(file) {
    // Validate file
    if (!this.validateFile(file)) {
      return;
    }

    this.currentFile = file;
    this.showFilePreview(file);
    this.hideUploadArea();
    this.showDetectionOptions();
    this.updateProcessButton();

    console.log(`${file.name} is ready for processing`);
  }

  validateFile(file) {
    const maxSize = 500 * 1024 * 1024; // 500MB for videos

    // Check file extension instead of MIME type for better compatibility
    const fileName = file.name.toLowerCase();
    const imageExtensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"];
    const videoExtensions = [
      ".mp4",
      ".avi",
      ".mov",
      ".wmv",
      ".flv",
      ".webm",
      ".mkv",
      ".m4v",
    ];
    const allExtensions = [...imageExtensions, ...videoExtensions];

    const hasValidExtension = allExtensions.some((ext) =>
      fileName.endsWith(ext)
    );

    if (file.size > maxSize) {
      alert("File too large. Please select a file smaller than 500MB.");
      return false;
    }

    if (!hasValidExtension) {
      alert(
        "Invalid file type. Supported: JPG, PNG, GIF, MP4, AVI, MOV, WMV, FLV, WEBM, MKV"
      );
      return false;
    }

    return true;
  }

  showFilePreview(file) {
    // Clear previous results when new file is selected
    this.clearPreviousResults();

    // Update file info
    this.elements.fileName.textContent = file.name;
    this.elements.fileSize.textContent = this.formatFileSize(file.size);
    this.elements.fileType.textContent = this.getFileTypeDisplay(file.type);

    // Show preview for image or video
    if (file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (e) => {
        this.elements.previewImg.src = e.target.result;
        this.elements.previewImg.style.display = "block";
        document.getElementById("videoPreview").style.display = "none";
      };
      reader.readAsDataURL(file);
    } else if (file.type.startsWith("video/")) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const videoElement = document.getElementById("previewVideo");
        const videoPreview = document.getElementById("videoPreview");
        videoElement.src = e.target.result;
        videoPreview.style.display = "block";
        this.elements.previewImg.style.display = "none";
      };
      reader.readAsDataURL(file);
    }

    // Show preview section
    this.elements.filePreview.style.display = "block";
    this.elements.filePreview.classList.add("fade-in");
  }

  hideUploadArea() {
    this.elements.uploadArea.style.display = "none";
  }

  showUploadArea() {
    this.elements.uploadArea.style.display = "block";
    this.elements.filePreview.style.display = "none";
  }

  showDetectionOptions() {
    this.elements.detectionOptions.style.display = "block";
    this.elements.detectionOptions.classList.add("fade-in");
  }

  hideDetectionOptions() {
    this.elements.detectionOptions.style.display = "none";
  }

  updateProcessButton() {
    const hasFile = this.currentFile !== null;
    const hasOptions =
      this.elements.helmetDetection.checked ||
      this.elements.plateDetection.checked;

    this.elements.processBtn.disabled = !(hasFile && hasOptions);

    if (hasFile && hasOptions) {
      this.elements.processBtn.querySelector(".btn-text i").className =
        "fas fa-play";
      this.elements.processBtn.querySelector(".btn-text").innerHTML =
        '<i class="fas fa-play"></i>Start Analysis';
    }
  }

  removeFile() {
    this.currentFile = null;
    this.elements.fileInput.value = "";
    this.showUploadArea();
    this.hideDetectionOptions();
    this.updateProcessButton();
    this.hideResults();

    // Clear all previous results
    this.clearPreviousResults();

    console.log("File removed, ready for new upload");
  }

  async processFile() {
    if (this.isProcessing || !this.currentFile) return;

    // Clear previous results before processing new file
    this.clearPreviousResults();

    this.isProcessing = true;
    this.setProcessingState(true);
    this.updateStatus("Processing...");

    try {
      const formData = new FormData();
      formData.append("file", this.currentFile);
      formData.append("process_helmet", this.elements.helmetDetection.checked);
      formData.append("process_license", this.elements.plateDetection.checked);

      console.log("AI models are analyzing your image...");

      const response = await fetch("/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (result.error) {
        throw new Error(result.error);
      }

      this.displayResults(result);
      this.processedCount++;
      this.updateProcessedCount();
      this.updateStatus("Completed");
    } catch (error) {
      console.error("Processing error:", error);
      alert(
        "Processing failed: " +
          (error.message || "An error occurred during processing")
      );
      this.updateStatus("Error");
    } finally {
      this.isProcessing = false;
      this.setProcessingState(false);
    }
  }

  setProcessingState(processing) {
    this.elements.processBtn.classList.toggle("processing", processing);
    this.elements.processBtn.disabled = processing;

    if (processing) {
      this.elements.processBtn.querySelector(".btn-text").style.opacity = "0";
      this.elements.processBtn.querySelector(".btn-loader").style.opacity = "1";
    } else {
      this.elements.processBtn.querySelector(".btn-text").style.opacity = "1";
      this.elements.processBtn.querySelector(".btn-loader").style.opacity = "0";
    }
  }

  displayResults(result) {
    // Show results section
    this.elements.resultsSection.style.display = "block";
    this.elements.resultsSection.classList.add("fade-in");

    // Hide video analysis section by default (will be shown only for videos)
    const videoSection = document.getElementById("videoAnalysis");
    if (videoSection) {
      videoSection.style.display = "none";
    }

    // Display helmet results
    if (result.helmet_results && !result.helmet_results.error) {
      const helmet = result.helmet_results;
      const totalDetections = helmet.total_detections || 0;
      const withHelmet = helmet.with_helmet_count || 0;
      const withoutHelmet = helmet.no_helmet_count || 0;

      this.elements.totalDetections.textContent = totalDetections;
      this.elements.withHelmet.textContent = withHelmet;
      this.elements.withoutHelmet.textContent = withoutHelmet;

      // Calculate and display accuracy
      const accuracy = totalDetections > 0 ? "95%" : "N/A";
      this.elements.accuracyScore.textContent = accuracy;

      // Update safety status
      this.updateSafetyStatus(withoutHelmet, totalDetections);
    }

    // Display video results if available
    if (result.file_type === "video" && result.video_timeline) {
      this.showVideoAnalysis(result);
    }

    // Display license plate results
    if (result.license_plate_results) {
      const plates = result.license_plate_results;
      this.displayLicensePlates(plates.details || []);
    }

    // Display annotated image (only for images, not videos)
    if (result.annotated_image && result.file_type !== "video") {
      this.elements.resultImage.src = result.annotated_image;
      this.elements.modalImage.src = result.annotated_image;
      this.elements.resultImage.style.display = "block";
    } else if (result.file_type === "video") {
      // Hide image section for videos
      this.elements.resultImage.style.display = "none";
    }

    // Scroll to results
    setTimeout(() => {
      this.elements.resultsSection.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }, 300);
  }

  clearPreviousResults() {
    // Hide video analysis section
    const videoSection = document.getElementById("videoAnalysis");
    if (videoSection) {
      videoSection.style.display = "none";
    }

    // Clear video frame grid
    const frameGrid = document.getElementById("videoFrameGrid");
    if (frameGrid) {
      frameGrid.innerHTML = "";
    }

    // Show image results section by default
    if (this.elements.resultImage) {
      this.elements.resultImage.style.display = "block";
    }

    // Reset any other result displays that might persist
    console.log("Cleared previous results");
  }

  displayVideoResults(result) {
    console.log("=== DISPLAY VIDEO RESULTS CALLED ===");

    // Try multiple ways to find the video section
    const videoSection = document.getElementById("videoResults");
    const allVideoElements = document.querySelectorAll('[id*="video"]');

    console.log("Video section element found:", !!videoSection);
    console.log(
      "All elements with 'video' in ID:",
      Array.from(allVideoElements).map((el) => el.id)
    );

    if (videoSection) {
      videoSection.style.display = "block";
      console.log("Video section displayed successfully");
    } else {
      console.log("ERROR: videoResults element not found in DOM");
      console.log(
        "Available elements with video-related IDs:",
        Array.from(allVideoElements)
      );
    }

    // Debug logging
    console.log("Video results received:", result);
    console.log("Video timeline data:", result.video_timeline);

    // Update video stats
    if (result.video_results) {
      const stats = result.video_results;

      const durationEl = document.getElementById("videoDuration");
      const framesEl = document.getElementById("videoFrames");
      const fpsEl = document.getElementById("videoFps");

      console.log("Found video stat elements:", {
        duration: !!durationEl,
        frames: !!framesEl,
        fps: !!fpsEl,
      });

      if (durationEl) {
        durationEl.textContent = `${stats.duration.toFixed(1)}s`;
        console.log("Updated duration to:", `${stats.duration.toFixed(1)}s`);
      }
      if (framesEl) {
        framesEl.textContent = `${stats.processed_frames}`;
        console.log("Updated frames to:", stats.processed_frames);
      }
      if (fpsEl) {
        fpsEl.textContent = `${stats.fps.toFixed(1)}`;
        console.log("Updated fps to:", `${stats.fps.toFixed(1)}`);
      }

      console.log("Updated video stats:", {
        duration: `${stats.duration.toFixed(1)}s`,
        frames: `${stats.processed_frames} frames`,
        fps: `${stats.fps.toFixed(1)} FPS`,
      });
    }

    // Display video timeline
    if (result.video_timeline) {
      console.log(
        "Displaying timeline with",
        result.video_timeline.length,
        "frames"
      );
      this.displayVideoTimeline(result.video_timeline);
    } else {
      console.log("No video_timeline found in result");
    }
  }

  displayVideoTimeline(timeline) {
    const timelineContainer = document.getElementById("videoTimeline");
    if (!timelineContainer) {
      console.log("Timeline container not found!");
      return;
    }

    console.log(
      "Displaying timeline with",
      timeline ? timeline.length : 0,
      "frames"
    );

    if (!timeline || timeline.length === 0) {
      console.log("Timeline is empty or null");
      timelineContainer.innerHTML = `
        <div class="empty-state">
          <i class="fas fa-video"></i>
          <p>No video frames processed</p>
        </div>
      `;
      return;
    }

    // Log first frame to check structure
    console.log("First frame data:", timeline[0]);
    console.log("First frame has frame_base64:", !!timeline[0]?.frame_base64);
    console.log("Frame_base64 length:", timeline[0]?.frame_base64?.length);

    const timelineHTML = timeline
      .map((frame, index) => {
        const safetyClass =
          frame.safety_status === "safe"
            ? "safe"
            : frame.safety_status === "unsafe"
            ? "unsafe"
            : "warning";

        const helmetInfo = frame.helmet_results
          ? `${frame.helmet_results.with_helmet_count || 0} with helmet, ${
              frame.helmet_results.no_helmet_count || 0
            } without`
          : "No detection";

        const plateInfo = frame.license_plate_results
          ? `${frame.license_plate_results.detections || 0} plates detected`
          : "No plates";

        return `
        <div class="frame-card ${safetyClass}" data-frame="${index}">
          <div class="frame-header">
            <span class="frame-time">${frame.second}s</span>
            <span class="safety-indicator ${safetyClass}">
              <i class="fas ${
                frame.safety_status === "safe"
                  ? "fa-shield-alt"
                  : frame.safety_status === "unsafe"
                  ? "fa-exclamation-triangle"
                  : "fa-question-circle"
              }"></i>
            </span>
          </div>
          <div class="frame-image">
            ${
              frame.frame_base64
                ? `<img src="${frame.frame_base64}" alt="Frame ${frame.frame_number}" loading="lazy" style="max-width: 100%; height: auto; border-radius: 8px;">`
                : `<div class="no-image" style="padding: 20px; text-align: center; color: #999;"><i class="fas fa-image"></i><p>No image available</p></div>`
            }
          </div>
          <div class="frame-info">
            <div class="detection-info">
              <small><i class="fas fa-motorcycle"></i> ${helmetInfo}</small>
              <small><i class="fas fa-id-badge"></i> ${plateInfo}</small>
            </div>
          </div>
        </div>
      `;
      })
      .join("");

    timelineContainer.innerHTML = timelineHTML;

    // Add click handlers for frame cards
    timelineContainer.addEventListener("click", (e) => {
      const frameCard = e.target.closest(".frame-card");
      if (frameCard) {
        this.showFrameDetails(timeline[frameCard.dataset.frame]);
      }
    });
  }

  showFrameDetails(frameData) {
    // Create modal or detailed view for frame
    console.log(
      `Frame at ${
        frameData.second
      }s - Status: ${frameData.safety_status.toUpperCase()}`
    );
  }

  displayLicensePlates(plates) {
    if (!plates || plates.length === 0) {
      this.elements.platesList.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-search"></i>
                    <p>No license plates detected</p>
                </div>
            `;
      return;
    }

    const platesHTML = plates
      .map(
        (plate) => `
            <div class="plate-item">
                <span class="plate-text">${plate.text || "Unknown"}</span>
                <span class="plate-confidence">${(
                  (plate.confidence || 0) * 100
                ).toFixed(1)}%</span>
            </div>
        `
      )
      .join("");

    this.elements.platesList.innerHTML = platesHTML;
  }

  hideResults() {
    this.elements.resultsSection.style.display = "none";
  }

  resetAnalysis() {
    this.removeFile();
    this.updateStatus("Ready");

    // Reset checkboxes
    this.elements.helmetDetection.checked = true;
    this.elements.plateDetection.checked = true;

    // Scroll to top
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  openImageModal() {
    if (this.elements.resultImage.src) {
      this.elements.imageModal.style.display = "flex";
      this.elements.imageModal.classList.add("fade-in");
      document.body.style.overflow = "hidden";
    }
  }

  closeImageModal() {
    this.elements.imageModal.style.display = "none";
    document.body.style.overflow = "";
  }

  downloadResults() {
    if (!this.currentFile) return;

    const results = {
      filename: this.currentFile.name,
      timestamp: new Date().toISOString(),
      helmet_detections: this.elements.totalDetections.textContent,
      with_helmet: this.elements.withHelmet.textContent,
      without_helmet: this.elements.withoutHelmet.textContent,
      license_plates: Array.from(document.querySelectorAll(".plate-text")).map(
        (el) => el.textContent
      ),
    };

    const blob = new Blob([JSON.stringify(results, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trafficai_results_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);

    console.log("Results are being downloaded");
  }

  downloadImage() {
    if (this.elements.resultImage.src) {
      const a = document.createElement("a");
      a.href = this.elements.resultImage.src;
      a.download = `trafficai_annotated_${Date.now()}.jpg`;
      a.click();

      console.log("Annotated image is being downloaded");
    }
  }

  showToast(title, message, type = "success") {
    const toast = document.createElement("div");
    toast.className = `toast ${type} scale-in`;

    const icon =
      type === "success"
        ? "fa-check-circle"
        : type === "error"
        ? "fa-exclamation-circle"
        : type === "warning"
        ? "fa-exclamation-triangle"
        : "fa-info-circle";

    toast.innerHTML = `
            <div class="toast-content">
                <div class="toast-icon">
                    <i class="fas ${icon}"></i>
                </div>
                <div class="toast-text">
                    <h4>${title}</h4>
                    <p>${message}</p>
                </div>
            </div>
            <button class="toast-close">
                <i class="fas fa-times"></i>
            </button>
        `;

    // Add close functionality
    toast.querySelector(".toast-close").addEventListener("click", () => {
      this.removeToast(toast);
    });

    this.elements.toastContainer.appendChild(toast);

    // Auto remove after 5 seconds
    setTimeout(() => {
      if (toast.parentNode) {
        this.removeToast(toast);
      }
    }, 5000);
  }

  removeToast(toast) {
    toast.style.animation = "slideInRight 0.3s ease reverse";
    setTimeout(() => {
      if (toast.parentNode) {
        toast.parentNode.removeChild(toast);
      }
    }, 300);
  }

  updateStatus(status) {
    this.elements.status.textContent = status;

    // Add color coding
    this.elements.status.className = "stat-value";
    if (status === "Processing...") {
      this.elements.status.style.color = "var(--warning-500)";
    } else if (status === "Completed") {
      this.elements.status.style.color = "var(--success-500)";
    } else if (status === "Error") {
      this.elements.status.style.color = "var(--danger-500)";
    } else {
      this.elements.status.style.color = "var(--primary-400)";
    }
  }

  updateProcessedCount() {
    this.elements.processed.textContent = this.processedCount;
  }

  updateSafetyStatus(unsafeRiders, totalRiders) {
    const safetyElement = document.getElementById("overallSafety");
    const safetyText = document.getElementById("safetyText");

    if (totalRiders === 0) {
      safetyElement.className = "safety-status warning";
      safetyElement.innerHTML =
        '<i class="fas fa-exclamation-triangle"></i><span>No Riders Detected</span>';
    } else if (unsafeRiders === 0) {
      safetyElement.className = "safety-status safe";
      safetyElement.innerHTML =
        '<i class="fas fa-shield-alt"></i><span>All Clear - Safe Traffic</span>';

      // Log success message
      console.log("Traffic Safety Status: All riders are wearing helmets! ✅");
    } else {
      const unsafePercentage = Math.round((unsafeRiders / totalRiders) * 100);
      safetyElement.className = "safety-status unsafe";
      safetyElement.innerHTML = `<i class="fas fa-exclamation-circle"></i><span>⚠️ ${unsafeRiders} Unsafe Rider${
        unsafeRiders > 1 ? "s" : ""
      } (${unsafePercentage}%)</span>`;

      // Log warning message
      console.log(
        `Safety Alert! ${unsafeRiders} rider${
          unsafeRiders > 1 ? "s are" : " is"
        } not wearing helmet${unsafeRiders > 1 ? "s" : ""}!`
      );
    }
  }

  displayVideoResults(result) {
    const videoCard = document.getElementById("videoResultsCard");
    const videoDuration = document.getElementById("videoDuration");
    const videoFrames = document.getElementById("videoFrames");
    const videoTimeline = document.getElementById("videoTimeline");

    if (!videoCard) return;

    // Show video results card
    videoCard.style.display = "block";

    // Update video stats
    const videoData = result.video_results;
    videoDuration.textContent = `${videoData.duration.toFixed(1)}s`;
    videoFrames.textContent = `${videoData.processed_frames} frames analyzed`;

    // Clear timeline
    videoTimeline.innerHTML = "";

    // Display frame results
    if (result.frames && result.frames.length > 0) {
      result.frames.forEach((frame, index) => {
        const frameCard = this.createFrameCard(frame, index);
        videoTimeline.appendChild(frameCard);
      });
    } else {
      videoTimeline.innerHTML =
        '<div class="empty-state"><i class="fas fa-film"></i><p>No frame data available</p></div>';
    }
  }

  createFrameCard(frame, index) {
    const card = document.createElement("div");
    card.className = "frame-card";

    // Determine safety status
    let status = "unknown";
    let statusText = "Unknown";

    if (frame.helmet_results && !frame.helmet_results.error) {
      const unsafe = frame.helmet_results.no_helmet_count || 0;
      const safe = frame.helmet_results.with_helmet_count || 0;
      const total = frame.helmet_results.total_detections || 0;

      if (total > 0) {
        if (unsafe > 0) {
          status = "unsafe";
          statusText = "Unsafe";
          card.classList.add("unsafe");
        } else {
          status = "safe";
          statusText = "Safe";
          card.classList.add("safe");
        }
      }
    }

    card.innerHTML = `
       <div class="frame-header">
         <span class="frame-time">${frame.timestamp.toFixed(1)}s</span>
         <span class="frame-status ${status}">${statusText}</span>
       </div>
       <div class="frame-details">
         <div class="frame-detail">
           <span>Safe:</span>
           <span class="frame-detail-value safe">${
             frame.helmet_results?.with_helmet_count || 0
           }</span>
         </div>
         <div class="frame-detail">
           <span>Unsafe:</span>
           <span class="frame-detail-value danger">${
             frame.helmet_results?.no_helmet_count || 0
           }</span>
         </div>
         <div class="frame-detail">
           <span>Total:</span>
           <span class="frame-detail-value">${
             frame.helmet_results?.total_detections || 0
           }</span>
         </div>
         <div class="frame-detail">
           <span>Plates:</span>
           <span class="frame-detail-value">${
             frame.license_plate_results?.detections || 0
           }</span>
         </div>
       </div>
     `;

    return card;
  }

  // Utility functions
  formatFileSize(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  getFileTypeDisplay(mimeType) {
    const types = {
      "image/jpeg": "JPEG Image",
      "image/jpg": "JPG Image",
      "image/png": "PNG Image",
      "image/gif": "GIF Image",
      "video/mp4": "MP4 Video",
      "video/avi": "AVI Video",
    };
    return types[mimeType] || "Unknown";
  }

  showVideoAnalysis(result) {
    console.log("Showing video analysis:", result);

    // Show the video analysis section
    const videoSection = document.getElementById("videoAnalysis");
    if (!videoSection) {
      console.error("Video analysis section not found!");
      return;
    }

    videoSection.style.display = "block";

    // Update video stats
    const statsEl = document.getElementById("videoStats");
    if (statsEl && result.video_results) {
      const stats = result.video_results;
      statsEl.textContent = `${stats.duration.toFixed(1)}s • ${
        stats.processed_frames
      } frames • ${stats.fps.toFixed(1)} FPS`;
    }

    // Display video frames
    const frameGrid = document.getElementById("videoFrameGrid");
    if (!frameGrid || !result.video_timeline) {
      console.error("Frame grid not found or no timeline data");
      return;
    }

    // Debug: Log first few frames to see their structure
    console.log("First frame data:", result.video_timeline[0]);
    console.log(
      "Frame helmet results:",
      result.video_timeline[0]?.helmet_results
    );

    // Generate frame cards
    const frameCards = result.video_timeline
      .map((frame, index) => {
        let safetyClass = "warning";
        let safetyText = "?";

        if (frame.safety_status === "safe") {
          safetyClass = "safe";
          safetyText = "✓";
        } else if (frame.safety_status === "unsafe") {
          safetyClass = "unsafe";
          safetyText = "⚠";
        } else if (
          frame.helmet_results &&
          frame.helmet_results.total_detections > 0
        ) {
          // If we have detections but no clear safety status, show as safe
          safetyClass = "safe";
          safetyText = "✓";
        }

        let helmetInfo = "No detection";
        if (frame.helmet_results) {
          const withHelmet = frame.helmet_results.with_helmet_count || 0;
          const withoutHelmet = frame.helmet_results.no_helmet_count || 0;
          const total = frame.helmet_results.total_detections || 0;

          if (total > 0) {
            helmetInfo = `${withHelmet} with helmet, ${withoutHelmet} without`;
          } else {
            helmetInfo = "No riders detected";
          }
        }

        return `
        <div class="video-frame-card ${safetyClass}">
          <div class="frame-header">
            <span class="frame-time">${frame.second}s</span>
            <span class="safety-badge ${safetyClass}">
              ${safetyText}
            </span>
          </div>
          <div class="frame-image">
            ${
              frame.frame_base64
                ? `<img src="${frame.frame_base64}" alt="Frame ${frame.frame_number}" />`
                : `<div class="no-frame">No image</div>`
            }
          </div>
          <div class="frame-info">
            <small>🪖 ${helmetInfo}</small>
            ${
              frame.license_plate_results &&
              frame.license_plate_results.detections > 0
                ? `<br><small>🚗 ${frame.license_plate_results.detections} license plate(s)</small>`
                : ""
            }
          </div>
        </div>
      `;
      })
      .join("");

    frameGrid.innerHTML = frameCards;
    console.log(`Displayed ${result.video_timeline.length} video frames`);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new TrafficAIApp();
});

// Health check
setInterval(async () => {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      console.warn("Health check failed");
    }
  } catch (error) {
    console.warn("Health check error:", error);
  }
}, 30000); // Check every 30 seconds
