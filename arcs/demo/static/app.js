(() => {
  const $ = (id) => document.getElementById(id);

  const transcript = $("transcript");
  const emptyState = $("empty-state");
  const queryEl = $("query");
  const submitBtn = $("submit");
  const composer = $("composer");
  const publicBanner = $("public-banner");

  let busy = false;
  let activeFeedback = null; // { queryId, root, sent }
  let thinkingTimer = null;
  let thinkingStep = 0;

  const THINKING_STEPS = [
    "Routing",
    "Generating",
    "Verifying",
  ];

  async function loadPublicBanner() {
    if (!publicBanner) return;
    try {
      const res = await fetch("/health");
      if (!res.ok) return;
      const data = await res.json();
      if (data.public_demo && data.disclaimer) {
        publicBanner.textContent = data.disclaimer;
        publicBanner.classList.remove("hidden");
      }
    } catch (_) {
      /* optional */
    }
  }

  function hideEmpty() {
    if (emptyState) emptyState.classList.add("hidden");
  }

  function scrollToBottom() {
    requestAnimationFrame(() => {
      transcript.scrollTop = transcript.scrollHeight;
    });
  }

  function autoResize() {
    queryEl.style.height = "auto";
    queryEl.style.height = `${Math.min(queryEl.scrollHeight, 140)}px`;
  }

  function formatConfidence(value) {
    if (value == null || Number.isNaN(Number(value))) return null;
    return `${Math.round(Number(value) * 100)}%`;
  }

  function formatErrorDetail(data, status) {
    if (status === 429) {
      return (data && typeof data.detail === "string" && data.detail)
        || "Rate limit exceeded. Please wait and try again.";
    }
    if (status === 503) {
      return (data && typeof data.detail === "string" && data.detail)
        || "Service temporarily unavailable. Please try again shortly.";
    }
    if (!data || data.detail == null) return "Request failed";
    const detail = data.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
    }
    return String(detail);
  }

  function appendTurn() {
    hideEmpty();
    const turn = document.createElement("div");
    turn.className = "turn";
    transcript.appendChild(turn);
    return turn;
  }

  function appendUserBubble(turn, text) {
    const bubble = document.createElement("div");
    bubble.className = "bubble user";
    bubble.textContent = text;
    turn.appendChild(bubble);
  }

  function appendThinking(turn) {
    const el = document.createElement("div");
    el.className = "thinking";
    el.setAttribute("aria-live", "polite");
    el.innerHTML = `
      <div class="thinking-dots" aria-hidden="true">
        <span></span><span></span><span></span>
      </div>
      <div class="thinking-label"><strong>${THINKING_STEPS[0]}</strong>…</div>
    `;
    turn.appendChild(el);

    thinkingStep = 0;
    clearInterval(thinkingTimer);
    thinkingTimer = setInterval(() => {
      thinkingStep = (thinkingStep + 1) % THINKING_STEPS.length;
      const label = el.querySelector(".thinking-label");
      if (label) {
        label.innerHTML = `<strong>${THINKING_STEPS[thinkingStep]}</strong>…`;
      }
    }, 2200);

    scrollToBottom();
    return el;
  }

  function clearThinking(el) {
    clearInterval(thinkingTimer);
    thinkingTimer = null;
    if (el && el.parentNode) el.remove();
  }

  function appendAssistantMessage(turn, data) {
    const bubble = document.createElement("div");
    bubble.className = "bubble assistant";
    bubble.textContent = data.answer || "(No answer text)";
    turn.appendChild(bubble);

    const meta = document.createElement("div");
    meta.className = "meta-row";

    const domain = data.domain || "UNKNOWN";
    const conf = formatConfidence(data.confidence);
    const domainChip = document.createElement("span");
    domainChip.className = "chip domain";
    domainChip.textContent = conf ? `${domain} · ${conf}` : domain;
    meta.appendChild(domainChip);

    const verdict = data.verdict || data.status || "—";
    const verdictChip = document.createElement("span");
    verdictChip.className = "chip " + (verdict === "PASS" ? "pass" : verdict === "FAIL" ? "fail" : "");
    verdictChip.textContent = verdict;
    meta.appendChild(verdictChip);

    if (data.timing_ms) {
      const timingChip = document.createElement("span");
      timingChip.className = "chip timing";
      timingChip.textContent = `${(data.timing_ms / 1000).toFixed(1)}s`;
      meta.appendChild(timingChip);
    }

    turn.appendChild(meta);

    const feedback = buildFeedbackBlock(data.query_id);
    turn.appendChild(feedback);
    activeFeedback = {
      queryId: data.query_id,
      root: feedback,
      sent: false,
    };

    scrollToBottom();
  }

  function appendErrorBubble(turn, message) {
    const bubble = document.createElement("div");
    bubble.className = "bubble error";
    bubble.textContent = message;
    turn.appendChild(bubble);
    scrollToBottom();
  }

  function buildFeedbackBlock(queryId) {
    const root = document.createElement("div");
    root.className = "feedback-block";
    root.dataset.queryId = queryId;

    root.innerHTML = `
      <div class="feedback-actions">
        <button type="button" class="fb-btn" data-fb="POSITIVE" title="Helpful">👍</button>
        <button type="button" class="fb-btn" data-fb="NEGATIVE" title="Not helpful">👎</button>
      </div>
      <div class="domain-picker hidden" role="group" aria-label="Correct domain">
        <p class="domain-picker-lead">
          <strong>Which domain should this have been routed to?</strong>
          Pick one if routing was wrong — this label feeds router retrain.
        </p>
        <div class="domain-buttons">
          <button type="button" class="domain-btn" data-domain="CODING">CODING</button>
          <button type="button" class="domain-btn" data-domain="MEDICAL">MEDICAL</button>
          <button type="button" class="domain-btn" data-domain="LEGAL">LEGAL</button>
          <button type="button" class="domain-btn" data-domain="GENERAL">GENERAL</button>
          <button type="button" class="domain-btn skip" data-domain="">Skip</button>
        </div>
        <p class="domain-picker-hint">Skip only if the domain was correct but the answer was bad.</p>
      </div>
      <p class="feedback-msg hidden"></p>
    `;

    const actions = root.querySelector(".feedback-actions");
    const picker = root.querySelector(".domain-picker");
    const msg = root.querySelector(".feedback-msg");

    actions.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-fb]");
      if (!btn || !activeFeedback || activeFeedback.root !== root || activeFeedback.sent) return;
      const signal = btn.getAttribute("data-fb");
      if (signal === "POSITIVE") {
        sendFeedback(root, "POSITIVE");
      } else {
        actions.querySelectorAll(".fb-btn").forEach((b) => {
          b.disabled = true;
        });
        picker.classList.remove("hidden");
        msg.classList.add("hidden");
        scrollToBottom();
      }
    });

    picker.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-domain]");
      if (!btn || !activeFeedback || activeFeedback.root !== root || activeFeedback.sent) return;
      const domain = btn.getAttribute("data-domain") || null;
      if (!domain && !window.confirm(
        "Skip domain label? Only skip if routing was correct but the answer was bad. "
        + "If the wrong specialist ran, pick the correct domain instead."
      )) {
        return;
      }
      sendFeedback(root, "NEGATIVE", domain || undefined);
    });

    return root;
  }

  function setFeedbackBusy(root, disabled) {
    root.querySelectorAll(".fb-btn, .domain-btn").forEach((el) => {
      el.disabled = disabled;
    });
  }

  async function sendFeedback(root, signal, correctDomain) {
    if (!activeFeedback || activeFeedback.root !== root || activeFeedback.sent) return;

    const picker = root.querySelector(".domain-picker");
    const msg = root.querySelector(".feedback-msg");
    const actions = root.querySelector(".feedback-actions");

    setFeedbackBusy(root, true);
    picker.classList.add("hidden");
    msg.classList.add("hidden");

    const payload = { query_id: activeFeedback.queryId, signal };
    if (signal === "NEGATIVE" && correctDomain) {
      payload.correct_domain = correctDomain;
    }

    try {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(formatErrorDetail(data, res.status));
      }

      activeFeedback.sent = true;
      actions.querySelectorAll(".fb-btn").forEach((b) => {
        b.disabled = true;
        if (b.getAttribute("data-fb") === signal) {
          b.classList.add(signal === "POSITIVE" ? "active-pos" : "active-neg");
        }
      });

      msg.textContent = data.message || "Feedback recorded.";
      msg.classList.remove("hidden");
      if (signal === "NEGATIVE") {
        msg.classList.add("negative");
        if (!correctDomain) {
          msg.textContent +=
            " Tip: pick a domain next time if routing was wrong — it feeds router retrain.";
        }
      } else {
        msg.classList.remove("negative");
      }
    } catch (err) {
      setFeedbackBusy(root, false);
      msg.textContent = err.message || "Could not send feedback.";
      msg.classList.remove("hidden", "negative");
      if (signal === "NEGATIVE") {
        picker.classList.remove("hidden");
      }
    }
    scrollToBottom();
  }

  async function ask() {
    const query = queryEl.value.trim();
    if (!query || busy) return;

    busy = true;
    submitBtn.disabled = true;
    queryEl.value = "";
    autoResize();

    const turn = appendTurn();
    appendUserBubble(turn, query);
    const thinking = appendThinking(turn);
    scrollToBottom();

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const data = await res.json().catch(() => ({}));
      clearThinking(thinking);
      if (!res.ok) {
        throw new Error(formatErrorDetail(data, res.status));
      }
      appendAssistantMessage(turn, data);
    } catch (err) {
      clearThinking(thinking);
      appendErrorBubble(turn, err.message || "Something went wrong.");
    } finally {
      busy = false;
      submitBtn.disabled = false;
      queryEl.focus();
    }
  }

  composer.addEventListener("submit", (e) => {
    e.preventDefault();
    ask();
  });

  queryEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask();
    }
  });

  queryEl.addEventListener("input", autoResize);

  loadPublicBanner();
  autoResize();
  queryEl.focus();
})();
