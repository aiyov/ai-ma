const state = {
  mode: "text",
  history: [],
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

function setMode(mode) {
  state.mode = mode;
  document.body.dataset.mode = mode;
  document.querySelectorAll(".mode").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.mode === mode);
  });
  queryInput.placeholder =
    mode === "visual" ? "例如：银色长笛" : "例如：grenadilla 唇板的中级长笛";
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
    resultMeta.textContent = `${payload.products.length} 件商品`;
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
        history: state.history,
        limit: 4,
        filters: currentFilters(),
      }),
    });
    state.history.push({ role: "user", content: message });
    state.history.push({ role: "assistant", content: result.answer });
    appendBubble("assistant", result.answer);
    if (result.products.length) {
      renderProducts(result.products);
      resultMeta.textContent = `对话匹配 ${result.products.length} 件商品`;
    }
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
  if (!products.length) {
    results.textContent = "没有匹配的商品。";
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
      <p class="meta">${escapeHtml([product.category, product.style].filter(Boolean).join(" · "))}</p>
      <p class="meta">相似度 ${product.score ?? "—"}</p>
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

async function request(path, options) {
  const response = await fetch(path, options);
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
