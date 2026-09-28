/* ==========================================================================
   CoNLL-2003 NER SVM Web Application Logic
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  // Preset Scenarios
  const PRESET_SAMPLES = {
    world: "German Foreign Minister Klaus Kinkel met with British officials in Brussels to discuss the European Union trade accord.",
    sports: "Manchester United secured a thrilling 2-1 victory over Juventus in Turin yesterday with two goals by David Beckham.",
    business: "Apple CEO Tim Cook announced a new research facility in Tokyo during the Asian Technology Summit on Wednesday.",
    conll: "Peter Blackburn reported from Brussels that Germany rejected French proposals on agricultural subsidies."
  };

  // Tab Switching
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      tabButtons.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add("active");
      }
    });
  });

  // Sample Presets
  const samplePills = document.querySelectorAll(".pill");
  const inputText = document.getElementById("input-text");

  samplePills.forEach(pill => {
    pill.addEventListener("click", () => {
      const sampleKey = pill.getAttribute("data-sample");
      if (PRESET_SAMPLES[sampleKey]) {
        inputText.value = PRESET_SAMPLES[sampleKey];
        analyzeText();
      }
    });
  });

  // Buttons
  const btnAnalyze = document.getElementById("btn-analyze");
  const btnClear = document.getElementById("btn-clear");
  const spinner = document.getElementById("loading-spinner");
  const annotatedOutput = document.getElementById("annotated-output");
  const entityCountBadge = document.getElementById("entity-count-badge");
  const entityChipsContainer = document.getElementById("entity-chips");
  const toggleTokenTable = document.getElementById("toggle-token-table");
  const tokenTableWrapper = document.getElementById("token-table-wrapper");
  const tokenTableBody = document.getElementById("token-table-body");

  btnClear.addEventListener("click", () => {
    inputText.value = "";
    annotatedOutput.innerHTML = '<p class="placeholder-text">Enter text above and click "Analyze Entities" to see highlighted named entities.</p>';
    entityCountBadge.textContent = "0 Entities Detected";
    entityChipsContainer.innerHTML = "";
    tokenTableBody.innerHTML = "";
    tokenTableWrapper.classList.add("hidden");
  });

  toggleTokenTable.addEventListener("click", () => {
    tokenTableWrapper.classList.toggle("hidden");
    const isHidden = tokenTableWrapper.classList.contains("hidden");
    toggleTokenTable.querySelector("span").textContent = isHidden
      ? "▼ Show Token-Level Feature & Decision Breakdown"
      : "▲ Hide Token-Level Feature & Decision Breakdown";
  });

  btnAnalyze.addEventListener("click", analyzeText);

  // Live Analysis Function
  async function analyzeText() {
    const text = inputText.value.trim();
    if (!text) {
      alert("Please enter text or select a preset sample.");
      return;
    }

    spinner.classList.remove("hidden");
    btnAnalyze.disabled = true;

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Prediction request failed.");
      }

      const data = await response.json();
      renderPredictions(data);
    } catch (err) {
      alert(`Error during analysis: ${err.message}`);
    } finally {
      spinner.classList.add("hidden");
      btnAnalyze.disabled = false;
    }
  }

  function renderPredictions(data) {
    // 1. Render annotated highlighted text
    annotatedOutput.innerHTML = data.formatted_html;

    // 2. Render entity summary badges
    const entities = data.detected_entities || [];
    entityCountBadge.textContent = `${entities.length} ${entities.length === 1 ? 'Entity' : 'Entities'} Detected`;

    entityChipsContainer.innerHTML = "";
    if (entities.length > 0) {
      entities.forEach(ent => {
        const chip = document.createElement("div");
        chip.className = `entity-chip tag-${ent.type.toLowerCase()}`;
        chip.innerHTML = `<strong>${ent.text}</strong> <span class="badge">${ent.type}</span>`;
        entityChipsContainer.appendChild(chip);
      });
    }

    // 3. Render token breakdown table
    tokenTableBody.innerHTML = "";
    data.predictions.forEach((pred, i) => {
      const tr = document.createElement("tr");

      const shapeSignals = [];
      if (pred.token[0] === pred.token[0].toUpperCase() && pred.token[0] !== pred.token[0].toLowerCase()) shapeSignals.push("TitleCase");
      if (pred.token === pred.token.toUpperCase() && pred.token.length > 1) shapeSignals.push("ALL_CAPS");
      if (/\d/.test(pred.token)) shapeSignals.push("HasDigits");
      if (/-/.test(pred.token)) shapeSignals.push("Hyphen");

      tr.innerHTML = `
        <td>${i + 1}</td>
        <td><strong class="code-mono">${escapeHtml(pred.token)}</strong></td>
        <td><span class="code-mono">${escapeHtml(pred.pos)}</span></td>
        <td><span class="code-mono tag-${pred.entity_type.toLowerCase()}">${escapeHtml(pred.predicted_tag)}</span></td>
        <td>${pred.entity_type !== 'O' ? `<span class="badge tag-${pred.entity_type.toLowerCase()}">${pred.entity_type}</span>` : '<span style="color:var(--text-dim)">None</span>'}</td>
        <td>${pred.token.length}</td>
        <td><span style="font-size:0.75rem; color:var(--text-muted)">${shapeSignals.join(", ") || "Standard"}</span></td>
      `;
      tokenTableBody.appendChild(tr);
    });
  }

  function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // Figure Zoom Modal
  const imageModal = document.getElementById("image-modal");
  const modalImg = document.getElementById("modal-img");
  const modalClose = document.getElementById("modal-close");
  const modalBackdrop = document.getElementById("modal-backdrop");

  document.querySelectorAll(".figure-card").forEach(card => {
    card.addEventListener("click", () => {
      const src = card.getAttribute("data-img");
      if (src) {
        modalImg.src = src;
        imageModal.classList.remove("hidden");
      }
    });
  });

  const closeModal = () => imageModal.classList.add("hidden");
  modalClose.addEventListener("click", closeModal);
  modalBackdrop.addEventListener("click", closeModal);

  // Test Set Explorer Logic
  let testSamples = [];
  let currentSampleIdx = 0;
  const sentenceBox = document.getElementById("explorer-sentence-display");
  const tokensGrid = document.getElementById("explorer-tokens-grid");
  const sampleLabel = document.getElementById("sample-id-label");
  const btnNextSample = document.getElementById("btn-next-sample");

  async function loadTestSamples() {
    try {
      const res = await fetch("/api/test-samples");
      if (res.ok) {
        testSamples = await res.json();
        if (testSamples.length > 0) {
          showTestSample(0);
        }
      }
    } catch (e) {
      console.warn("Test samples could not be loaded:", e);
    }
  }

  async function showTestSample(index) {
    if (!testSamples || testSamples.length === 0) return;
    const sample = testSamples[index];
    sampleLabel.textContent = `Sentence #${index + 1} (CoNLL ID: ${sample.id})`;

    const text = sample.tokens.join(" ");
    sentenceBox.innerHTML = `<strong>Context:</strong> "${escapeHtml(text)}"`;

    // Fetch model prediction for this sentence
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });
      const data = await res.json();

      tokensGrid.innerHTML = "";
      sample.tokens.forEach((tok, i) => {
        const gtTag = sample.ner_tags[i] || "O";
        const predTag = data.predictions[i] ? data.predictions[i].predicted_tag : "O";
        const isMatch = (gtTag === predTag);

        const card = document.createElement("div");
        card.className = `token-compare-card ${isMatch ? 'match' : 'mismatch'}`;
        card.innerHTML = `
          <div class="token-name">${escapeHtml(tok)}</div>
          <div style="font-size:0.75rem; color:var(--text-muted)">True: <strong class="tag-${gtTag.replace(/^[BI]-/, '').toLowerCase()}">${gtTag}</strong></div>
          <div style="font-size:0.75rem; color:var(--text-muted)">Pred: <strong class="tag-${predTag.replace(/^[BI]-/, '').toLowerCase()}">${predTag}</strong></div>
          <div style="margin-top:0.25rem; font-size:0.7rem; font-weight:700; color:${isMatch ? '#34d399' : '#f87171'}">${isMatch ? 'MATCH' : 'ERROR'}</div>
        `;
        tokensGrid.appendChild(card);
      });
    } catch (e) {
      console.error(e);
    }
  }

  btnNextSample.addEventListener("click", () => {
    if (testSamples.length > 0) {
      currentSampleIdx = (currentSampleIdx + 1) % testSamples.length;
      showTestSample(currentSampleIdx);
    }
  });

  loadTestSamples();

  // Load initial demo
  inputText.value = PRESET_SAMPLES.world;
  analyzeText();
});
