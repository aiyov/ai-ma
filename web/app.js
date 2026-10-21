const state = {
  mode: "text",
  sessionId: sessionStorage.getItem("chatSessionId") || "",
  file: null,
};

const queryInput = document.querySelector("#query");
const fileInput = document.querySelector("#file");
const preview = document.querySelector("#preview");
const dropCopy = document.querySelector("#drop-copy");
const results = document.querySelector("#results");
const resultMeta = document.querySelector("#result-meta");
const thread = document.querySelector("#thread");
const status = document.querySelector("#status");
const apiKeyInput = document.querySelector("#api-key");

apiKeyInput.value = sessionStorage.getItem("apiKey") || "";
apiKeyInput.addEventListener("change", () => {
  sessionStorage.setItem("apiKey", apiKeyInput.value.trim());
});

document.querySelectorAll(".mode").forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

document.querySelector("#search-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  await search();
});

document.querySelector("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  await sendChat();
});

fileInput.addEventListener("change", () => useFile(fileInput.files[0]));

const drop = document.querySelector("#drop");
drop.addEventListener("dragover", (event) => event.preventDefault());
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  useFile(event.dataTransfer.files[0]);
});

boot();

async function boot() {
  try {
    const health = await request("/health");
    status.textContent = `商品库 ${health.count} 条`;
  } catch (error) {
    status.textContent = "商品库未连接";
    showError("search-error", error.message);
  }
}

const modeCopy = {
  text: {
    hint: "按介绍、材质和规格查找。",
    placeholder: "例如：grenadilla 唇板的中级长笛",
  },
  visual: {
    hint: "按外观描述查找，例如颜色和造型。",
    placeholder: "例如：银色长笛",
  },
  image: {
    hint: "上传一张图，找外观相近的商品。",
    placeholder: "",
  },
};

function setMode(mode) {
  state.mode = mode;
  document.body.dataset.mode = mode;
  document.querySelectorAll(".mode").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.mode === mode);
  });
  const copy = modeCopy[mode] || modeCopy.text;
  document.querySelector("#mode-hint").textContent = copy.hint;
  queryInput.placeholder = copy.placeholder;
}

function useFile(file) {
  if (!file) return;
  state.file = file;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  dropCopy.hidden = true;
}

async function search() {
  clearError("search-error");
  resultMeta.textContent = "搜索中…";
  try {
    const payload =
      state.mode === "image" ? await searchByImage() : await searchByText();
    renderProducts(payload.products);
    resultMeta.textContent = `搜索到 ${payload.products.length} 件`;
  } catch (error) {
    resultMeta.textContent = "";
    showError("search-error", error.message);
  }
}

function searchByText() {
  const query = queryInput.value.trim();
  if (!query) throw new Error("先写一句要找的内容");
  return request("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      mode: state.mode,
      limit: 8,
      filters: currentFilters(),
    }),
  });
}

function searchByImage() {
  if (!state.file) throw new Error("先选择一张图片");
  const filters = currentFilters();
  const body = new FormData();
  body.set("file", state.file);
  body.set("limit", "8");
  if (filters.brand) body.set("brand", filters.brand);
  if (filters.max_price != null) body.set("max_price", String(filters.max_price));
  if (filters.in_stock != null) body.set("in_stock", String(filters.in_stock));
  return request("/search/image", { method: "POST", body });
}

async function sendChat() {
  const input = document.querySelector("#message");
  const message = input.value.trim();
  if (!message) return;
  clearError("chat-error");
  input.value = "";
  appendBubble("user", message);
  try {
    const result = await request("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        session_id: state.sessionId || null,
        limit: 4,
        filters: currentFilters(),
      }),
    });
    if (result.session_id) {
      state.sessionId = result.session_id;
      sessionStorage.setItem("chatSessionId", result.session_id);
    }
    appendBubble("assistant", result.answer);
    appendChatProducts(result.products);
    appendChunks(result.chunks);
  } catch (error) {
    showError("chat-error", error.message);
  }
}

function currentFilters() {
  const brand = document.querySelector("#brand").value.trim();
  const maxPrice = document.querySelector("#max-price").value;
  const inStock = document.querySelector("#in-stock").checked;
  return {
    brand: brand || null,
    category: null,
    condition: null,
    in_stock: inStock ? true : null,
    min_price: null,
    max_price: maxPrice === "" ? null : Number(maxPrice),
  };
}

function renderProducts(products) {
  results.replaceChildren();
  results.classList.toggle("is-empty", !products.length);
  if (!products.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "没有匹配的商品。可以换个描述，或放宽品牌和价格。";
    results.append(empty);
    return;
  }
  for (const product of products) {
    const card = document.createElement("article");
    card.className = "card";
    const image = document.createElement("img");
    image.alt = product.title;
    image.src = product.image_url || product.thumb_url;
    const body = document.createElement("div");
    body.className = "card-body";
    body.innerHTML = `
      <p class="brand">${escapeHtml(product.brand || "未标注品牌")}</p>
      <h3>${escapeHtml(product.title)}</h3>
      <p class="price">${formatPrice(product.sale_price)} · ${product.in_stock ? "有货" : "无货"}</p>
      <p class="meta">${escapeHtml([product.part_number, product.category].filter(Boolean).join(" · "))}</p>
      ${product.score == null ? "" : `<p class="meta">相似度 ${product.score}</p>`}
    `;
    if (product.sku_url) {
      const link = document.createElement("a");
      link.href = productUrl(product.sku_url);
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = "商品页";
      body.append(link);
    }
    card.append(image, body);
    results.append(card);
  }
}

function appendChatProducts(products) {
  if (!products || !products.length) return;
  const box = document.createElement("div");
  box.className = "chat-products";
  for (const product of products) {
    const item = document.createElement(product.sku_url ? "a" : "div");
    item.className = "chat-product";
    if (product.sku_url) {
      item.href = productUrl(product.sku_url);
      item.target = "_blank";
      item.rel = "noreferrer";
    }
    const image = document.createElement("img");
    image.alt = "";
    image.src = product.thumb_url || product.image_url;
    const body = document.createElement("div");
    body.innerHTML = `
      <p class="brand">${escapeHtml(product.brand || "未标注品牌")}</p>
      <h3>${escapeHtml(product.title)}</h3>
      <p class="meta">${escapeHtml([product.part_number, formatPrice(product.sale_price), product.in_stock ? "有货" : "无货"].filter(Boolean).join(" · "))}</p>
    `;
    item.append(image, body);
    box.append(item);
  }
  thread.append(box);
  thread.scrollTop = thread.scrollHeight;
}

function appendChunks(chunks) {
  if (!chunks || !chunks.length) return;
  const box = document.createElement("div");
  box.className = "chunks";
  for (const chunk of chunks) {
    const item = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `${chunk.title} · 第 ${chunk.chunk_index + 1} 段`;
    const text = document.createElement("p");
    text.textContent = chunk.text;
    item.append(summary, text);
    box.append(item);
  }
  thread.append(box);
  thread.scrollTop = thread.scrollHeight;
}

function appendBubble(role, text) {
  const hint = thread.querySelector(".hint");
  if (hint) hint.remove();
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  bubble.textContent = text;
  thread.append(bubble);
  thread.scrollTop = thread.scrollHeight;
}

function productUrl(path) {
  if (path.startsWith("http")) return path;
  return `https://www.musicarts.com${path.startsWith("/") ? path : `/${path}`}`;
}

function formatPrice(value) {
  if (value == null) return "价格未知";
  return `$${Number(value).toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

async function request(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const key = apiKeyInput.value.trim();
  if (key) headers.set("X-API-Key", key);
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      /* keep status text */
    }
    throw new Error(typeof detail === "string" ? detail : "请求失败");
  }
  return response.json();
}

function showError(id, message) {
  const node = document.querySelector(`#${id}`);
  node.hidden = false;
  node.textContent = message;
}

function clearError(id) {
  const node = document.querySelector(`#${id}`);
  node.hidden = true;
  node.textContent = "";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
