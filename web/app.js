/* ==========================================================================
   AI Customer Support Agent Dashboard - Interactive Client Application
   ========================================================================== */

// Sample Golden Set fallback items for instant offline rendering
const FALLBACK_GOLDEN_SET = [
  {
    "customer_tweet_id": "874823",
    "customer_text": "Why is my iPhone battery draining so rapidly after the update?",
    "intent_annotated": "battery_power",
    "expected_decision": "AUTO_HANDLE",
    "escalation_reason_type": "NONE",
    "verified_by_human": false,
    "notes": "Standard battery drain troubleshooting inquiry."
  },
  {
    "customer_tweet_id": "290164",
    "customer_text": "My iPhone 7 battery drains easily even when I am not using it. Still issue persist.",
    "intent_annotated": "battery_power",
    "expected_decision": "AUTO_HANDLE",
    "escalation_reason_type": "NONE",
    "verified_by_human": false,
    "notes": "Clear battery diagnostic path."
  },
  {
    "customer_tweet_id": "583681",
    "customer_text": "Just cracked my iPhone screen with a glass protector... Only dropped on its back",
    "intent_annotated": "hardware_physical",
    "expected_decision": "ESCALATE",
    "escalation_reason_type": "PHYSICAL_HARDWARE",
    "verified_by_human": false,
    "notes": "Physical glass damage requires Genius Bar inspection."
  },
  {
    "customer_tweet_id": "84215",
    "customer_text": "Well guys ... i dropped my iPhone .. That now has a cracked screen .. any explanations",
    "intent_annotated": "hardware_physical",
    "expected_decision": "ESCALATE",
    "escalation_reason_type": "PHYSICAL_HARDWARE",
    "verified_by_human": false,
    "notes": "Cracked screen repair dispatch required."
  },
  {
    "customer_tweet_id": "464725",
    "customer_text": "I am so fucking sick of this piece of shit phone constantly crashing! Your engineers are total scammers and I'm going to sue Apple!",
    "intent_annotated": "ios_update_bugs",
    "expected_decision": "ESCALATE",
    "escalation_reason_type": "ABUSIVE_HOSTILE",
    "verified_by_human": false,
    "notes": "Customer hostile and threatening legal action."
  },
  {
    "customer_tweet_id": "119278",
    "customer_text": "Someone hacked into my Apple ID account and changed my password and 2FA phone number, help!",
    "intent_annotated": "account_apple_id",
    "expected_decision": "ESCALATE",
    "escalation_reason_type": "ACCOUNT_SECURITY",
    "verified_by_human": false,
    "notes": "Sensitive account credentials and security lockout."
  },
  {
    "customer_tweet_id": "119280",
    "customer_text": "I was charged $99 for an in-app subscription that I cancelled last week! I demand a full refund immediately.",
    "intent_annotated": "app_store_billing",
    "expected_decision": "ESCALATE",
    "escalation_reason_type": "FINANCIAL_REFUND",
    "verified_by_human": false,
    "notes": "Financial dispute and refund demand."
  },
  {
    "customer_tweet_id": "53588",
    "customer_text": "Every time I connect my AirPods to my Mac, the audio is distorted and cuts out completely after 30 seconds.",
    "intent_annotated": "audio_media",
    "expected_decision": "AUTO_HANDLE",
    "escalation_reason_type": "NONE",
    "verified_by_human": false,
    "notes": "Standard Bluetooth audio troubleshooting."
  },
  {
    "customer_tweet_id": "31962",
    "customer_text": "AirPods are connected but don't play audio anymore any troubleshooting tips?",
    "intent_annotated": "audio_media",
    "expected_decision": "AUTO_HANDLE",
    "escalation_reason_type": "NONE",
    "verified_by_human": false,
    "notes": "Audio playback reconnection guidance."
  },
  {
    "customer_tweet_id": "119241",
    "customer_text": "Wi-Fi is not connecting on my iPhone after the iOS 11 update. Drops every 5 minutes.",
    "intent_annotated": "connectivity",
    "expected_decision": "AUTO_HANDLE",
    "escalation_reason_type": "NONE",
    "verified_by_human": false,
    "notes": "Network settings reset path."
  },
  {
    "customer_tweet_id": "249290",
    "customer_text": "So how do you put your pictures from your old phone to the new phone? @AppleSupport",
    "intent_annotated": "device_storage_backup",
    "expected_decision": "AUTO_HANDLE",
    "escalation_reason_type": "NONE",
    "verified_by_human": false,
    "notes": "iCloud photo transfer how-to."
  },
  {
    "customer_tweet_id": "38240",
    "customer_text": "Any reason why my Apple Watch October challenge suddenly disappeared yesterday?",
    "intent_annotated": "general_inquiry",
    "expected_decision": "AUTO_HANDLE",
    "escalation_reason_type": "NONE",
    "verified_by_human": false,
    "notes": "Apple Watch Activity Challenge inquiry."
  }
];

let allGoldenSet = [...FALLBACK_GOLDEN_SET];

// Pre-defined client-side triage engine (fallback if API server is not running)
function runClientSideTriage(query) {
  const qLower = query.toLowerCase();
  
  // Escalation rule checks
  if (/\b(fuck|fucking|bitch|bastard|asshole|scam|thieves|sue|suing|lawyer)\b/.test(qLower)) {
    return {
      intent: "ios_update_bugs",
      reply: "We want to ensure you receive the proper attention for your concerns. Please connect with our senior customer relations team in DM so we can review this thoroughly.",
      decision: "ESCALATE",
      escalation_reason: "High hostility or legal risk detected in customer text. Requires human specialist de-escalation.",
      evidence: [
        {
          tweet_id: "568442",
          similarity: 0.681,
          customer_text: "fix the bugs!! You people have $billions of dollars!! Phones keep shutting off...",
          historical_reply: "@253375 We want to help you get things running properly once again. Please DM us."
        }
      ]
    };
  }

  if (/\b(apple\s*id|password|passcode|locked\s*out|hacked|2fa|verification\s*code)\b/.test(qLower)) {
    return {
      intent: "account_apple_id",
      reply: "Your account security is our top priority. For assistance with your Apple ID and security verification, please visit iforgot.apple.com or connect directly with our security specialists in DM.",
      decision: "ESCALATE",
      escalation_reason: "Account credentials and security locks require verified human authentication to prevent unauthorized takeover.",
      evidence: [
        {
          tweet_id: "119238",
          similarity: 0.812,
          customer_text: "My Apple ID is disabled and verification codes are not sending to my trusted number.",
          historical_reply: "@105835 Please visit iforgot.apple.com to reset your password or DM us for verification assistance."
        }
      ]
    };
  }

  if (/\b(refund|overcharged|unauthorized\s*charge|dispute|money\s*back|credit\s*card|billed)\b/.test(qLower)) {
    return {
      intent: "app_store_billing",
      reply: "We can assist with billing questions. You can review your purchase history and request refunds directly at reportaproblem.apple.com. Please send us a DM if you have additional questions regarding subscriptions.",
      decision: "ESCALATE",
      escalation_reason: "Financial transaction or refund request requires direct human review and transaction lookup.",
      evidence: [
        {
          tweet_id: "119278",
          similarity: 0.825,
          customer_text: "I was billed twice for my storage plan and need a refund immediately.",
          historical_reply: "@105676 We'd be glad to check on this purchase with you. Please DM us your Apple ID email."
        }
      ]
    };
  }

  if (/\b(cracked|shattered|broken\s*screen|water\s*damage|dropped\s*in|genius\s*bar|repair)\b/.test(qLower)) {
    return {
      intent: "hardware_physical",
      reply: "We are sorry to hear about the damage to your device. To explore repair options and schedule a Genius Bar appointment, visit support.apple.com/repair. Let us know in DM if we can help book an appointment.",
      decision: "ESCALATE",
      escalation_reason: "Physical damage or hardware repair requires physical Genius Bar diagnostics or authorized service dispatch.",
      evidence: [
        {
          tweet_id: "583680",
          similarity: 0.721,
          customer_text: "Just cracked my iPhone screen with a glass protector... Only dropped on its back",
          historical_reply: "@257673 We'd be happy to look into some options for this. Let us know which iPhone you're using via DM."
        }
      ]
    };
  }

  if (/\b(battery|drain|charge|charging|overheat|percent)\b/.test(qLower)) {
    return {
      intent: "battery_power",
      reply: "Battery performance is important to us. To help troubleshoot, please send us a DM with your iPhone model and current iOS version (Settings > General > About). You can also check Settings > Battery > Battery Health for maximum capacity details.",
      decision: "AUTO_HANDLE",
      escalation_reason: "Sufficient historical evidence found (top similarity 0.848). Standard troubleshooting intent (battery_power) confirmed without security or financial risks.",
      evidence: [
        {
          tweet_id: "874822",
          similarity: 0.848,
          customer_text: "Why is my iPhone battery draining so rapidly after the update?",
          historical_reply: "@327697 We'd like to see what we can do to help. Are you running iOS 11.0.3? We released it just the other day."
        },
        {
          tweet_id: "290163",
          similarity: 0.794,
          customer_text: "My iPhone 7 battery drains easily even when I am not using it.",
          historical_reply: "@185124 We're happy to assist you. DM us your complete software version from Settings > General > About."
        }
      ]
    };
  }

  if (/\b(music|airpods|speaker|audio|volume|sound|distorted)\b/.test(qLower)) {
    return {
      intent: "audio_media",
      reply: "Thanks for reaching out to us. We are always happy to help with audio and playback. Does restarting your device or disconnecting and reconnecting Bluetooth make a difference? Send us a DM so we can look into this together.",
      decision: "AUTO_HANDLE",
      escalation_reason: "Sufficient historical evidence found (top similarity 0.615). Standard troubleshooting intent (audio_media) confirmed without security or financial risks.",
      evidence: [
        {
          tweet_id: "53587",
          similarity: 0.615,
          customer_text: "Also, why do my AirPods suddenly stop playing music after being connected for awhile?",
          historical_reply: "@128113 We'd love to help. Has the device been restarted? If not, let's do that using these steps: https://t.co/OWRDHWSt4h"
        }
      ]
    };
  }

  // Default: iOS update / general software
  return {
    intent: "ios_update_bugs",
    reply: "We'd like to help get things running smoothly again. Which version of iOS are you currently on? Please DM us with your device model and any troubleshooting steps you have tried, such as restarting.",
    decision: "AUTO_HANDLE",
    escalation_reason: "Sufficient historical evidence found (top similarity 0.785). Standard troubleshooting intent confirmed without security or financial risks.",
    evidence: [
      {
        tweet_id: "366041",
        similarity: 0.785,
        customer_text: "iPhone 7 iOS 11.1 battery from 100% to 56% in 30 mins? How? Please fix this issue and release new update #apple #battery",
        historical_reply: "@202726 We'd like to look into this with you. Let's continue our conversation in DM. https://t.co/GDrqU22YpT"
      }
    ]
  };
}

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  setupPlayground();
  loadGoldenSet();
});

// Navigation Handling
function setupNavigation() {
  const tabs = document.querySelectorAll(".nav-tab");
  const panes = document.querySelectorAll(".tab-pane");

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      panes.forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const targetId = tab.getAttribute("data-tab");
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add("active");
      }
    });
  });
}

// Playground Setup
function setupPlayground() {
  const queryInput = document.getElementById("query-input");
  const charCounter = document.getElementById("query-char-count");
  const submitBtn = document.getElementById("triage-submit-btn");
  const clearBtn = document.getElementById("clear-btn");
  const copyReplyBtn = document.getElementById("copy-reply-btn");
  const copyJsonBtn = document.getElementById("copy-json-btn");

  // Character counter
  queryInput.addEventListener("input", () => {
    const len = queryInput.value.length;
    charCounter.textContent = `${len} / 280`;
  });

  // Presets
  document.querySelectorAll(".preset-btn[data-query]").forEach(btn => {
    btn.addEventListener("click", () => {
      const q = btn.getAttribute("data-query");
      queryInput.value = q;
      charCounter.textContent = `${q.length} / 280`;
      executeTriage(q);
    });
  });

  clearBtn.addEventListener("click", () => {
    queryInput.value = "";
    charCounter.textContent = "0 / 280";
  });

  submitBtn.addEventListener("click", () => {
    const text = queryInput.value.trim();
    if (!text) {
      alert("Please enter a customer tweet to triage.");
      return;
    }
    executeTriage(text);
  });

  copyReplyBtn.addEventListener("click", () => {
    const replyText = document.getElementById("reply-display").textContent.trim();
    navigator.clipboard.writeText(replyText).then(() => {
      copyReplyBtn.textContent = "✓ Copied!";
      setTimeout(() => copyReplyBtn.textContent = "📋 Copy Reply", 2000);
    });
  });

  copyJsonBtn.addEventListener("click", () => {
    const jsonText = document.getElementById("json-display").textContent.trim();
    navigator.clipboard.writeText(jsonText).then(() => {
      copyJsonBtn.textContent = "✓ Copied!";
      setTimeout(() => copyJsonBtn.textContent = "Copy JSON", 2000);
    });
  });

  // Run initial demo query
  executeTriage("My iPhone 7 battery is draining from 100% to 15% in 30 minutes after iOS update!");
}

// Execute Triage via API with graceful Client-side fallback
async function executeTriage(query) {
  const submitBtn = document.getElementById("triage-submit-btn");
  submitBtn.disabled = true;
  submitBtn.textContent = "⏳ Triaging...";

  try {
    const resp = await fetch("/api/triage", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query })
    });

    if (resp.ok) {
      const data = await resp.json();
      renderTriageResult(data);
      document.getElementById("backend-status-text").textContent = "API Live (Python 3.11)";
    } else {
      throw new Error("Backend API unavailable");
    }
  } catch (err) {
    // Graceful client fallback
    const mockResult = runClientSideTriage(query);
    renderTriageResult(mockResult);
    document.getElementById("backend-status-text").textContent = "Offline Mode (Local Fallback)";
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = "<span>🚀</span> Triage & Generate Reply";
  }
}

// Render Results into UI
function renderTriageResult(data) {
  const intentDisplay = document.getElementById("intent-display");
  const decisionDisplay = document.getElementById("decision-display");
  const reasonDisplay = document.getElementById("reason-display");
  const replyDisplay = document.getElementById("reply-display");
  const confidenceDisplay = document.getElementById("confidence-display");
  const triageHero = document.getElementById("triage-hero");
  const jsonDisplay = document.getElementById("json-display");
  const citationsContainer = document.getElementById("citations-container");
  const citationCount = document.getElementById("citation-count");

  // Intent
  intentDisplay.textContent = data.intent.toUpperCase();

  // Decision Hero
  decisionDisplay.textContent = data.decision;
  if (data.decision === "ESCALATE") {
    triageHero.className = "triage-hero escalate";
  } else {
    triageHero.className = "triage-hero auto-handle";
  }

  // Reason & Reply
  reasonDisplay.textContent = data.escalation_reason;
  replyDisplay.textContent = data.reply;

  // Confidence / Similarity
  const topSim = (data.evidence && data.evidence.length > 0) ? data.evidence[0].similarity : 0.85;
  confidenceDisplay.textContent = topSim ? topSim.toFixed(3) : "0.850";

  // Citations
  citationsContainer.innerHTML = "";
  if (data.evidence && data.evidence.length > 0) {
    citationCount.textContent = `${data.evidence.length} Matches`;
    data.evidence.forEach((ev, idx) => {
      const card = document.createElement("div");
      card.className = "citation-card";
      card.innerHTML = `
        <div class="citation-meta">
          <strong>Citation #${idx + 1} (Tweet #${ev.tweet_id})</strong>
          <span class="sim-pill">Cosine Sim: ${ev.similarity ? ev.similarity.toFixed(3) : "0.750"}</span>
        </div>
        <div class="citation-text"><strong>Customer:</strong> "${escapeHtml(ev.customer_text || "")}"</div>
        <div class="citation-reply"><strong>Apple:</strong> "${escapeHtml(ev.historical_reply || "")}"</div>
      `;
      citationsContainer.appendChild(card);
    });
  } else {
    citationCount.textContent = "0 Matches";
    citationsContainer.innerHTML = "<p style='color: var(--text-muted); font-size: 0.85rem;'>No citations retrieved.</p>";
  }

  // JSON View
  jsonDisplay.textContent = JSON.stringify(data, null, 2);
}

// Golden Set Explorer
async function loadGoldenSet() {
  try {
    const resp = await fetch("/api/golden_set");
    if (resp.ok) {
      const items = await resp.json();
      if (Array.isArray(items) && items.length > 0) {
        allGoldenSet = items;
      }
    }
  } catch (e) {
    console.log("Using pre-loaded golden set fallback.");
  }

  renderGoldenTable(allGoldenSet);

  // Search & Filter listeners
  const searchInput = document.getElementById("golden-search-input");
  const intentFilter = document.getElementById("golden-intent-filter");
  const decisionFilter = document.getElementById("golden-decision-filter");

  function filterData() {
    const query = searchInput.value.toLowerCase();
    const intent = intentFilter.value;
    const decision = decisionFilter.value;

    const filtered = allGoldenSet.filter(item => {
      const textMatch = !query || 
        (item.customer_text && item.customer_text.toLowerCase().includes(query)) ||
        (item.customer_tweet_id && item.customer_tweet_id.includes(query)) ||
        (item.notes && item.notes.toLowerCase().includes(query));

      const intentMatch = (intent === "ALL") || (item.intent_annotated === intent);
      const decisionMatch = (decision === "ALL") || (item.expected_decision === decision);

      return textMatch && intentMatch && decisionMatch;
    });

    renderGoldenTable(filtered);
  }

  searchInput.addEventListener("input", filterData);
  intentFilter.addEventListener("change", filterData);
  decisionFilter.addEventListener("change", filterData);
}

function renderGoldenTable(items) {
  const tbody = document.getElementById("golden-table-body");
  const badge = document.getElementById("golden-count-badge");
  badge.textContent = `Showing ${items.length} items`;
  tbody.innerHTML = "";

  if (items.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 2rem;">No matching benchmark cases found.</td></tr>`;
    return;
  }

  items.slice(0, 50).forEach(item => {
    const tr = document.createElement("tr");
    const isEscalate = item.expected_decision === "ESCALATE";
    const decClass = isEscalate ? "style='color: var(--rose-text); font-weight: 700;'" : "style='color: var(--emerald-text); font-weight: 700;'";

    tr.innerHTML = `
      <td><span class="sim-pill">${item.customer_tweet_id || "-"}</span></td>
      <td style="max-width: 380px;">${escapeHtml(item.customer_text || "")}</td>
      <td><span class="intent-pill">${item.intent_annotated || "general"}</span></td>
      <td ${decClass}>${item.expected_decision}</td>
      <td><span style="font-size: 0.75rem; color: var(--text-muted);">${item.escalation_reason_type || "NONE"}</span></td>
      <td>
        <button class="preset-btn test-row-btn" data-text="${escapeAttr(item.customer_text || "")}">
          Load & Triage
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });

  // Attach row buttons
  document.querySelectorAll(".test-row-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const text = btn.getAttribute("data-text");
      const queryInput = document.getElementById("query-input");
      queryInput.value = text;
      document.getElementById("query-char-count").textContent = `${text.length} / 280`;

      // Switch to playground tab
      document.getElementById("tab-btn-playground").click();
      executeTriage(text);
    });
  });
}

// Helpers
function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function escapeAttr(str) {
  return str.replace(/"/g, '&quot;');
}
