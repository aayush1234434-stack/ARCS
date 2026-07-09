(() => {
  const $ = (id) => document.getElementById(id);

  const queryEl = $("query");
  const submitBtn = $("submit");
  const loadingEl = $("loading");
  const resultEl = $("result");
  const errorEl = $("error");
  const errorText = $("error-text");
  const answerEl = $("answer");
  const metaDomain = $("meta-domain");
  const metaVerdict = $("meta-verdict");
  const metaTiming = $("meta-timing");
  const btnPositive = $("btn-positive");
  const btnNegative = $("btn-negative");
  const feedbackMsg = $("feedback-msg");
  const domainPicker = $("domain-picker");

  let currentQueryId = null;
  let feedbackSent = false;

  function hideAll() {
    loadingEl.classList.add("hidden");
    resultEl.classList.add("hidden");
    errorEl.classList.add("hidden");
  }

  function showError(message) {
    hideAll();
    errorText.textContent = message;
    errorEl.classList.remove("hidden");
  }

  function formatConfidence(value) {
    if (value == null || Number.isNaN(value)) return "";
    return `${Math.round(value * 100)}%`;
  }

  function setFeedbackButtons(enabled) {
    btnPositive.disabled = !enabled;
    btnNegative.disabled = !enabled;
  }

  function hideDomainPicker() {
    domainPicker.classList.add("hidden");
  }

  function showDomainPicker() {
    domainPicker.classList.remove("hidden");
  }

  function formatErrorDetail(data) {
    if (!data || data.detail == null) return "Request failed";
    const detail = data.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
    }
    return String(detail);
  }

  async function ask() {
    const query = queryEl.value.trim();
    if (!query) {
      showError("Please enter a question.");
      return;
    }

    currentQueryId = null;
    feedbackSent = false;
    feedbackMsg.classList.add("hidden");
    hideDomainPicker();
    setFeedbackButtons(false);
    submitBtn.disabled = true;
    hideAll();
    loadingEl.classList.remove("hidden");

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(formatErrorDetail(data));
      }

      currentQueryId = data.query_id;
      answerEl.textContent = data.answer || "(No answer text)";

      const domain = data.domain || "UNKNOWN";
      const conf = formatConfidence(data.confidence);
      metaDomain.textContent = conf
        ? `${domain} · ${conf} confidence`
        : domain;

      const verdict = data.verdict || data.status || "—";
      metaVerdict.textContent = `Verdict: ${verdict}`;
      metaVerdict.className = "pill " + (verdict === "PASS" ? "pass" : verdict === "FAIL" ? "fail" : "");

      metaTiming.textContent = data.timing_ms
        ? `${(data.timing_ms / 1000).toFixed(1)}s`
        : "";

      hideAll();
      resultEl.classList.remove("hidden");
      setFeedbackButtons(true);
    } catch (err) {
      showError(err.message || "Something went wrong.");
    } finally {
      submitBtn.disabled = false;
    }
  }

  async function sendFeedback(signal, correctDomain) {
    if (!currentQueryId || feedbackSent) return;

    setFeedbackButtons(false);
    hideDomainPicker();
    feedbackMsg.classList.add("hidden");

    const payload = { query_id: currentQueryId, signal };
    if (signal === "NEGATIVE" && correctDomain) {
      payload.correct_domain = correctDomain;
    }

    try {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(formatErrorDetail(data));
      }

      feedbackSent = true;
      feedbackMsg.textContent = data.message || "Feedback recorded.";
      feedbackMsg.classList.remove("hidden");
      if (signal === "NEGATIVE") {
        feedbackMsg.classList.add("negative");
      } else {
        feedbackMsg.classList.remove("negative");
      }
    } catch (err) {
      setFeedbackButtons(true);
      feedbackMsg.textContent = err.message || "Could not send feedback.";
      feedbackMsg.classList.remove("hidden", "negative");
    }
  }

  function onNegativeClick() {
    if (!currentQueryId || feedbackSent) return;
    setFeedbackButtons(false);
    feedbackMsg.classList.add("hidden");
    showDomainPicker();
  }

  submitBtn.addEventListener("click", ask);
  queryEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask();
  });
  btnPositive.addEventListener("click", () => sendFeedback("POSITIVE"));
  btnNegative.addEventListener("click", onNegativeClick);

  domainPicker.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-domain]");
    if (!btn) return;
    const domain = btn.getAttribute("data-domain") || null;
    sendFeedback("NEGATIVE", domain || undefined);
  });
})();
