// S&P 500 Financial Assistant Chatbot
const CONFIG = {
  GROQ_API_KEY: "GROQ_API",
  // Usamos newsapi.org por ser más fiable para este caso de uso.
  NEWSAPI_KEY: "NEW_API", // Asegúrate de que esta clave sea de newsapi.org
  NEWSAPI_PROVIDER: "newsapi",
  NEWSAPI_BASE: "https://newsapi.org/v2/everything",
  GROQ_API_URL: "https://api.groq.com/openai/v1/chat/completions",
  // path relative to index.html (index.html is in the project root)
  SP500_CSV_PATH: "../data/sp500_data.csv",
}

let conversationHistory = []
let isProcessing = false
let sp500DataCache = null
// store chart instances so we can update them dynamically
window.__analysisCharts = window.__analysisCharts || {}

const messagesContainer = document.getElementById("messagesContainer")
const chatForm = document.getElementById("chatForm")
const chatInput = document.getElementById("chatInput")
const sendBtn = document.getElementById("sendBtn")
const clearChatBtn = document.getElementById("clearChat")
const activityLog = document.getElementById("activityLog")
const activitySteps = document.getElementById("activitySteps")

document.addEventListener("DOMContentLoaded", async () => {
  initializeEventListeners()
  await loadSP500Data()
})

async function loadSP500Data() {
  try {
    const response = await fetch(CONFIG.SP500_CSV_PATH)
    if (!response.ok) {
      const txt = await response.text()
      throw new Error(`Failed to load CSV: ${response.status} ${response.statusText} - ${txt}`)
    }
    const csvText = await response.text()
    sp500DataCache = parseCSV2(csvText)
    console.log("[v0] SP500 data loaded:", sp500DataCache.length, "records")
    addMessage(`✅ Datos del S&P 500 cargados: ${sp500DataCache.length} registros.`, "bot")
  } catch (error) {
    console.error("[v0] Error loading SP500 data:", error)
    addMessage("⚠️ No se pudo cargar los datos del S&P 500. Algunas funciones estarán limitadas.", "bot")
  }
}

function parseCSV2(text) {
  const lines = text.trim().split("\n")
  const headers = lines[0].split(",")
  return lines.slice(1).map((line) => {
    const values = line.split(",")
    const obj = {}
    headers.forEach((header, index) => {
      obj[header.trim()] = values[index]?.trim()
    })
    return obj
  })
}

function initializeEventListeners() {
  chatForm.addEventListener("submit", handleSubmit)
  clearChatBtn.addEventListener("click", clearChat)
  document.addEventListener("click", (e) => {
    if (e.target.closest(".quick-action-btn")) {
      const action = e.target.closest(".quick-action-btn").dataset.action
      chatInput.value = action
      handleSubmit(new Event("submit"))
    }
  })
  document.addEventListener("click", (e) => {
    if (e.target.closest(".news-btn[data-action='relevant']")) {
      handleNewsRelevance(e.target.closest(".news-btn"))
    }
  })
}

async function handleSubmit(e) {
  e.preventDefault()
  if (isProcessing || !chatInput.value.trim()) return
  const userMessage = chatInput.value.trim()
  chatInput.value = ""
  addMessage(userMessage, "user")
  conversationHistory.push({ role: "user", content: userMessage })
  await processUserMessage(userMessage)
}

function addMessage(content, type = "bot") {
  const messageDiv = document.createElement("div")
  messageDiv.className = `message ${type}-message`
  const avatar = document.createElement("div")
  avatar.className = "message-avatar"
  avatar.innerHTML = type === "bot" ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>'
  const contentDiv = document.createElement("div")
  contentDiv.className = "message-content"
  const textDiv = document.createElement("div")
  textDiv.className = "message-text"
  if (typeof content === "string") {
    textDiv.innerHTML = formatMessageContent(content)
  } else {
    textDiv.appendChild(content)
  }
  contentDiv.appendChild(textDiv)
  const timeDiv = document.createElement("div")
  timeDiv.className = "message-time"
  timeDiv.textContent = new Date().toLocaleTimeString("es-ES", { hour: "2-digit", minute: "2-digit" })
  contentDiv.appendChild(timeDiv)
  messageDiv.appendChild(avatar)
  messageDiv.appendChild(contentDiv)
  messagesContainer.appendChild(messageDiv)
  scrollToBottom()
  return messageDiv
}

function formatMessageContent(text) {
  text = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
  text = text.replace(/\*(.*?)\*/g, "<em>$1</em>")
  text = text.replace(/\n/g, "<br>")
  return text
}

function addTypingIndicator() {
  const indicator = document.createElement("div")
  indicator.className = "message bot-message typing-message"
  indicator.innerHTML = `<div class="message-avatar"><i class="fas fa-robot"></i></div><div class="message-content"><div class="message-text"><div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div></div></div>`
  messagesContainer.appendChild(indicator)
  scrollToBottom()
  return indicator
}

function removeTypingIndicator() {
  const indicator = document.querySelector(".typing-message")
  if (indicator) indicator.remove()
}

function showActivityLog(show = true) {
  activityLog.style.display = show ? "block" : "none"
  if (!show) activitySteps.innerHTML = ""
}

function addActivityStep(text, status = "active") {
  const step = document.createElement("div")
  step.className = `activity-step ${status}`
  const icon = status === "completed" ? "fa-check-circle" : status === "active" ? "fa-spinner fa-spin" : "fa-circle"
  step.innerHTML = `<i class="fas ${icon}"></i><span>${text}</span>`
  activitySteps.appendChild(step)
  scrollToBottom()
  return step
}

function updateActivityStep(step, text, status = "completed") {
  const icon = status === "completed" ? "fa-check-circle" : status === "error" ? "fa-exclamation-circle" : "fa-circle"
  step.className = `activity-step ${status}`
  step.innerHTML = `<i class="fas ${icon}"></i><span>${text}</span>`
}

async function processUserMessage(userMessage) {
  isProcessing = true
  sendBtn.disabled = true
  const typingIndicator = addTypingIndicator()
  showActivityLog(true)
  try {
  const intentStep = addActivityStep("Analizando tu pregunta...")

  // Prepare containers for data that may be used during processing
  let sp500Data = null, newsData = null, chartData = null

  // If the user asked about a specific date (YYYY-MM-DD), pre-generate a chart from CSV data
    // This gives immediate visualization without waiting for function-calling to request it.
    let prerequestedDate = null
    const dateMatch = userMessage.match(/\b(\d{4}-\d{2}-\d{2})\b/)
    if (dateMatch) {
      prerequestedDate = dateMatch[1]
      try {
        // include 3 days after by default so the chart shows short-term movement after the date
        const preSp500 = getSP500DataByDate(prerequestedDate, 6, 3)
        if (preSp500 && preSp500.length > 0) {
          // create or update a dynamic chart canvas for in-chat display
          chartData = await createOrUpdateChartCanvas(preSp500, prerequestedDate)
          // provide a small activity step showing chart was prepared
          addActivityStep(`Gráfica preparada para ${prerequestedDate}`, "completed")
        } else {
          addActivityStep(`No hay datos del S&P 500 para ${prerequestedDate}`, "error")
        }
      } catch (err) {
        console.warn('[v0] Error preparando gráfica previa:', err)
      }
    }

    const intentResponse = await analyzeWithFunctionCalling(userMessage)
    updateActivityStep(intentStep, "Pregunta analizada ✓", "completed")
      
    if (intentResponse.toolCalls && intentResponse.toolCalls.length > 0) {
      // Ordenamos las llamadas para asegurar que los datos se obtienen antes de generar el gráfico
      const toolCalls = intentResponse.toolCalls.sort((a,b) => {
          const order = { get_sp500_data_by_date: 1, search_news_by_date: 2, generate_chart: 3 };
          const funcA = a.function?.name || a.name || '';
          const funcB = b.function?.name || b.name || '';
          return (order[funcA] || 99) - (order[funcB] || 99);
      });
      
      for (const toolCall of toolCalls) {
        let functionName = null
        let argsRaw = null
        if (toolCall.function) {
          functionName = toolCall.function.name
          argsRaw = toolCall.function.arguments
        } else if (toolCall.name) {
          functionName = toolCall.name
          argsRaw = toolCall.arguments
        } else if (toolCall.function_call && toolCall.function_call.name) {
          functionName = toolCall.function_call.name
          argsRaw = toolCall.function_call.arguments
        } else {
          console.warn('[v0] Unknown toolCall shape:', toolCall)
          continue
        }

        let args = {}
        if (argsRaw) {
          try {
            args = typeof argsRaw === 'string' ? JSON.parse(argsRaw) : argsRaw
          } catch (err) {
            console.warn('[v0] Failed to parse function arguments:', argsRaw, err)
            args = {}
          }
        }

        if (functionName === "get_sp500_data_by_date") {
          const dataStep = addActivityStep(`Consultando datos del ${args.date}...`)
          sp500Data = getSP500DataByDate(args.date, args.days_before || 6, args.days_after || 3) // Agregamos 3 días después por defecto para mostrar movimiento posterior
          updateActivityStep(dataStep, `Datos obtenidos (${sp500Data.length} días) ✓`, "completed")
        }
        if (functionName === "search_news_by_date") {
          const newsStep = addActivityStep(`Buscando noticias del ${args.date}...`)
          newsData = await searchNewsByDate(args.date)
          updateActivityStep(newsStep, `${newsData.length} noticias encontradas ✓`, "completed")
        }
        if (functionName === "generate_chart") {
          // Asegurarnos de que tenemos datos antes de intentar generar la gráfica
          if(!sp500Data) {
              sp500Data = getSP500DataByDate(args.date, 7, 1);
          }
          const chartStep = addActivityStep("Generando gráfica...")
          chartData = await createOrUpdateChartCanvas(sp500Data, args.date)
          updateActivityStep(chartStep, "Gráfica generada ✓", "completed")
        }
      }
    }
    const responseStep = addActivityStep("Preparando explicación...")
    const finalResponse = await generateFinalResponse(userMessage, sp500Data, newsData)
    updateActivityStep(responseStep, "Respuesta lista ✓", "completed")
    removeTypingIndicator()
    showActivityLog(false)
    displayBotResponse(finalResponse, newsData, chartData)
  } catch (error) {
    console.error("[v0] Error:", error)
    removeTypingIndicator()
    showActivityLog(false)
    addMessage("Error al procesar. Verifica las API keys.", "bot")
  } finally {
    isProcessing = false
    sendBtn.disabled = false
  }
}

// ============== FUNCIÓN MODIFICADA CLAVE ==============
async function analyzeWithFunctionCalling(userMessage) {
  const tools = [
    {
      type: "function",
      function: {
        name: "get_sp500_data_by_date",
        description: "Obtiene los datos NUMÉRICOS del S&P 500 (cierre, apertura, etc.) para una fecha específica.",
        parameters: {
          type: "object",
          properties: { date: { type: "string" }, days_before: { type: "integer", default: 6 }, days_after: { type: "integer", default: 1 } },
          required: ["date"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "search_news_by_date",
        description: "Busca titulares de noticias financieras relevantes para una fecha específica.",
        parameters: { type: "object", properties: { date: { type: "string" } }, required: ["date"] },
      },
    },
    {
      type: "function",
      function: {
        name: "generate_chart",
        description: "Genera una GRÁFICA LINEAL para visualizar el rendimiento del S&P 500 en torno a una fecha. Es la mejor forma de mostrar visualmente qué pasó.",
        parameters: {
          type: "object",
          properties: { date: { type: "string" } },
          required: ["date"],
        },
      },
    },
  ];
  
  // ============== INSTRUCCIÓN DE SISTEMA MEJORADA ==============
  const systemPrompt = `Eres un asistente financiero experto. Tu objetivo es dar la respuesta más completa posible.
  - Si el usuario pregunta por una fecha específica (ej: '¿qué pasó el 2024-10-25?', 'analiza esta fecha', 'dime sobre el 25 de octubre'), SIEMPRE debes usar las tres herramientas para dar una respuesta integral:
  1. \`get_sp500_data_by_date\`: Para obtener los datos duros del mercado.
  2. \`search_news_by_date\`: Para encontrar el contexto de lo que ocurría.
  3. \`generate_chart\`: Para mostrar VISUALMENTE la información.
  - No pidas permiso, simplemente ejecuta las tres herramientas.`;

  try {
    const functionsPayload = tools.map((t) => (t.function ? t.function : t))

    const response = await fetch(CONFIG.GROQ_API_URL, {
      method: "POST",
      headers: { Authorization: `Bearer ${CONFIG.GROQ_API_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "llama-3.1-8b-instant",
        messages: [
          {
            role: "system",
            content: systemPrompt,
          },
          ...conversationHistory.slice(-4),
          { role: "user", content: userMessage },
        ],
        functions: functionsPayload,
        function_call: "auto",
        temperature: 0.1, // Hacemos que sea muy determinístico en su elección de herramientas
      }),
    })
    if (!response.ok) {
      const txt = await response.text()
      console.error(`[v0] analyzeWithFunctionCalling failed: ${response.status} ${response.statusText} - ${txt}`)
      return { content: null, toolCalls: [] }
    }
    const data = await response.json()
    const message = data.choices?.[0]?.message || {}
  const toolCalls = []
  if (message.tool_calls) toolCalls.push(...message.tool_calls)
  else if (message.function_call) toolCalls.push(message.function_call)
  return { content: message.content || null, toolCalls }
  } catch (error) {
    console.error("[v0] analyzeWithFunctionCalling exception:", error)
    return { content: null, toolCalls: [] }
  }
}

function getSP500DataByDate(targetDate, daysBefore = 6, daysAfter = 0) {
  if (!sp500DataCache) return []
  const target = new Date(targetDate + "T00:00:00")
  const startDate = new Date(target)
  startDate.setDate(startDate.getDate() - daysBefore)
  const endDate = new Date(target)
  endDate.setDate(endDate.getDate() + daysAfter)
  return sp500DataCache
    .filter((row) => {
      const rowDate = new Date(row.Date)
      return rowDate >= startDate && rowDate <= endDate
    })
    .sort((a, b) => new Date(a.Date) - new Date(b.Date))
}

async function searchNewsByDate(date) {
    const targetDateISO = new Date(date + "T00:00:00").toISOString().split('T')[0];
    
    const queries = ['"S&P 500"', 'mercado de acciones', 'Wall Street', 'índices bursátiles'];
    let allArticles = [];
    let seenUnauthorized = false;
    let hasNotifiedError = false; // Flag para notificar solo una vez

    for (const query of queries) {
        try {
            const params = new URLSearchParams({
                q: query,
                from: targetDateISO,
                to: targetDateISO,
                language: 'es,en',
                sortBy: 'relevancy',
                pageSize: '10',
                apiKey: CONFIG.NEWSAPI_KEY,
            });

            const url = `${CONFIG.NEWSAPI_BASE}?${params.toString()}`;
            const response = await fetch(url);
            
            if (!response.ok) {
                const errorBody = await response.json();
                console.error(`[v0] NewsAPI.org request failed: ${response.status}`, errorBody);

                if (response.status === 401 && !seenUnauthorized) {
                    addMessage("⚠️ No se pudieron obtener noticias: la API key de NewsAPI es inválida o no autorizada.", "bot");
                    seenUnauthorized = true;
                } else if (errorBody.code === 'parameterInvalid' && !hasNotifiedError) {
                    addMessage("⚠️ La fecha de la noticia no es válida o es demasiado antigua (más de 1 mes).", "bot");
                    hasNotifiedError = true;
                }
                if (errorBody.code === 'parameterInvalid') break;
                continue;
            }

            const data = await response.json();
            allArticles = allArticles.concat(data.articles || []);

        } catch (error) {
            console.error("[v0] News processing error:", error);
        }
    }

    const seenUrls = new Set();
    const uniqueArticles = allArticles.filter(article => {
        if (!article.url || seenUrls.has(article.url)) {
            return false;
        }
        seenUrls.add(article.url);
        return true;
    });

    return uniqueArticles
        .map(article => {
            let score = 0;
            const text = `${article.title} ${article.description || ''}`.toLowerCase();
            if (text.includes("s&p 500")) score += 5;
            if (text.includes("mercado de acciones") || text.includes("stock market")) score += 3;
            if (text.includes("wall street")) score += 2;

            return {
                title: article.title,
                source: article.source.name,
                url: article.url,
                relevance_score: Math.min(score, 10),
            };
        })
        .filter(a => a.relevance_score > 0)
        .sort((a, b) => b.relevance_score - a.relevance_score)
        .slice(0, 5);
}

function generateChartFromData(data, targetDate) {
  if (!data || data.length === 0) return null
  const canvas = document.createElement("canvas")
  // Do NOT set canvas.width/height attributes so Chart.js can manage responsiveness.
  canvas.style.maxWidth = "100%"
  const dates = data.map((d) => new Date(d.Date).toLocaleDateString("es-ES", { month: "short", day: "numeric" }))
  const prices = data.map((d) => Number.parseFloat(d.Close))
  const targetIndex = data.findIndex((d) => d.Date.startsWith(targetDate))
  const chart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: {
      labels: dates,
      datasets: [
        {
          label: "Cierre S&P 500",
          data: prices,
          borderColor: "#0066ff",
          backgroundColor: "rgba(0,102,255,0.1)",
          fill: true,
          borderWidth: 2,
          tension: 0.1,
          pointRadius: dates.map((_, i) => (i === targetIndex ? 6 : 3)),
          pointBackgroundColor: dates.map((_, i) => (i === targetIndex ? "#ff3d00" : "#0066ff")),
        },
      ],
    },
    options: {
      responsive: true,
      // Keep a consistent aspect ratio so height scales with width of the grid cell
      maintainAspectRatio: true,
      aspectRatio: 16 / 9,
      plugins: { legend: { display: true }, title: { display: true, text: `Evolución del S&P 500 cerca de ${targetDate}` } },
      scales: { y: { beginAtZero: false, ticks: { callback: (v) => "$" + v.toLocaleString() } } },
    },
  })
  // Register and return the created chart and canvas so callers can update it dynamically
  const id = `chart_${Date.now()}_${Math.floor(Math.random() * 10000)}`
  window.__analysisCharts[id] = chart
  return { canvas, chart, id }
}

function createChartFromData(data, targetDate) {
  if (!data || data.length === 0) return null
  const canvas = document.createElement("canvas")
  canvas.style.maxWidth = "100%"
  const dates = data.map((d) => new Date(d.Date).toLocaleDateString("es-ES", { month: "short", day: "numeric" }))
  const prices = data.map((d) => Number.parseFloat(d.Close))
  const targetIndex = data.findIndex((d) => d.Date.startsWith(targetDate))
  const ctx = canvas.getContext("2d")
  const chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: dates,
      datasets: [
        {
          label: "Cierre S&P 500",
          data: prices,
          borderColor: "#0066ff",
          backgroundColor: "rgba(0,102,255,0.08)",
          fill: true,
          borderWidth: 3,
          tension: 0.1,
          pointRadius: 4,
          pointBackgroundColor: "#0066ff",
          pointHoverRadius: 6,
        },
        {
          label: "Fecha objetivo",
          data: prices.map((v, i) => (i === targetIndex ? v : null)),
          borderColor: "#ff3d00",
          backgroundColor: "#ff3d00",
          pointStyle: 'circle',
          pointRadius: dates.map((_, i) => (i === targetIndex ? 12 : 0)),
          pointBackgroundColor: dates.map((_, i) => (i === targetIndex ? "#ff3d00" : "transparent")),
          pointBorderColor: dates.map((_, i) => (i === targetIndex ? "#fff" : "transparent")),
          pointBorderWidth: 3,
          showLine: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 2.2,
      plugins: { 
        legend: { 
          display: true,
          position: 'top',
          labels: {
            padding: 20,
            font: {
              size: 13,
              weight: 500
            },
            usePointStyle: true,
            pointStyle: 'circle'
          }
        }, 
        title: { 
          display: true, 
          text: `Evolución del S&P 500 cerca de ${targetDate}`,
          font: {
            size: 16,
            weight: 'bold'
          },
          padding: {
            top: 15,
            bottom: 25
          }
        },
        tooltip: {
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          padding: 12,
          titleFont: {
            size: 14,
            weight: 'bold'
          },
          bodyFont: {
            size: 13
          },
          callbacks: {
            label: function(context) {
              let label = context.dataset.label || '';
              if (label) {
                label += ': ';
              }
              label += '$' + context.parsed.y.toLocaleString('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
              });
              return label;
            }
          }
        }
      },
      scales: { 
        y: { 
          beginAtZero: false, 
          ticks: { 
            callback: (v) => "$" + v.toLocaleString(),
            padding: 10,
            font: {
              size: 12
            }
          },
          grid: {
            color: 'rgba(0, 0, 0, 0.06)',
            drawBorder: false
          }
        },
        x: {
          ticks: {
            padding: 10,
            font: {
              size: 11
            }
          },
          grid: {
            color: 'rgba(0, 0, 0, 0.04)',
            drawBorder: false
          }
        }
      },
      layout: {
        padding: {
          left: 15,
          right: 15,
          top: 15,
          bottom: 15
        }
      }
    },
  })
  
  // register chart instance
  const id = `chart_${Date.now()}_${Math.floor(Math.random() * 10000)}`
  window.__analysisCharts[id] = chart
  return { canvas, chart, id }
}

// Helper: create or update a chart and return the canvas (for in-chat embedding)
async function createOrUpdateChartCanvas(data, targetDate) {
  if (!data || data.length === 0) return null
  // look for an existing chat message panel with this date
  const existingMsg = document.querySelector(`.message.bot-message [data-panel-date="${targetDate}"]`) || document.querySelector(`[data-panel-date="${targetDate}"]`)
  if (existingMsg) {
    // find associated chart id
    const panel = existingMsg.closest('[data-panel-date]') || existingMsg
    const chartId = panel?.dataset?.chartId
    if (chartId && window.__analysisCharts[chartId]) {
      updateChartInstance(window.__analysisCharts[chartId], data, targetDate)
      // return the canvas element if available
      const canvas = panel.querySelector('canvas')
      return canvas || null
    }
  }

  // create a new chart instance (but don't attach to dashboard)
  const { canvas, chart, id } = createChartFromData(data, targetDate)
  // store metadata on the canvas for potential future updates
  canvas.dataset.chartId = id
  canvas.dataset.panelDate = targetDate
  return canvas
}

// Helper: update an existing Chart instance with new data
function updateChartInstance(chart, data, targetDate) {
  if (!chart || !data || data.length === 0) return
  const labels = data.map((d) => new Date(d.Date).toLocaleDateString("es-ES", { month: "short", day: "numeric" }))
  const prices = data.map((d) => Number.parseFloat(d.Close))
  const targetIndex = data.findIndex((d) => d.Date.startsWith(targetDate))
  chart.data.labels = labels
    if (chart.data.datasets && chart.data.datasets.length) {
    // main dataset
    if (chart.data.datasets[0]) {
      chart.data.datasets[0].data = prices
      chart.data.datasets[0].pointRadius = 3
      chart.data.datasets[0].pointBackgroundColor = "#0066ff"
    }
    // highlight dataset (second)
    if (chart.data.datasets[1]) {
      chart.data.datasets[1].data = prices.map((v, i) => (i === targetIndex ? v : null))
      chart.data.datasets[1].pointRadius = labels.map((_, i) => (i === targetIndex ? 10 : 0))
      chart.data.datasets[1].pointBackgroundColor = labels.map((_, i) => (i === targetIndex ? "#ff3d00" : "transparent"))
      chart.data.datasets[1].pointBorderColor = labels.map((_, i) => (i === targetIndex ? "#fff" : "transparent"))
      chart.data.datasets[1].borderColor = "#ff3d00"
      chart.data.datasets[1].backgroundColor = "#ff3d00"
    }
  }
  if (chart.options && chart.options.plugins && chart.options.plugins.title) {
    chart.options.plugins.title.text = `Evolución del S&P 500 cerca de ${targetDate}`
  }
  try {
    chart.update()
  } catch (err) {
    console.warn('[v0] chart.update() failed, attempting resize + update', err)
    try { chart.resize(); chart.update() } catch (e) { console.error(e) }
  }
}

// Place (create or update) a chart inside the dashboard grid for a given date
async function placeChartInDashboardWithData(data, targetDate) {
  if (!data || data.length === 0) return
  const grid = document.getElementById("analysisGrid")
  if (!grid) {
    console.warn('[v0] analysisGrid not found — chart will not be placed')
    return
  }
  // Try to find existing panel for this date
  const existing = grid.querySelector(`[data-panel-date="${targetDate}"]`)
  if (existing && existing.dataset.chartId) {
    const chartId = existing.dataset.chartId
    const chart = window.__analysisCharts[chartId]
    if (chart) {
      updateChartInstance(chart, data, targetDate)
      return { updated: true, chartId }
    }
  }

  // create new chart
  const { canvas, chart, id } = createChartFromData(data, targetDate)
  // Create panel card
  const panel = document.createElement("section")
  panel.className = "analysis-panel"
  panel.setAttribute('data-panel-date', targetDate)
  panel.dataset.chartId = id

  const header = document.createElement("div")
  header.className = "analysis-panel-header"
  const title = document.createElement("h4")
  title.innerHTML = `<i class="fas fa-chart-line"></i> Análisis ${targetDate}`
  const ts = document.createElement("div")
  ts.className = "analysis-ts"
  ts.textContent = new Date().toLocaleString()
  header.appendChild(title)
  header.appendChild(ts)

  const body = document.createElement("div")
  body.className = "analysis-panel-body"
  body.appendChild(canvas)

  const footer = document.createElement("div")
  footer.className = "analysis-panel-footer"
  footer.innerHTML = `<small style="color:var(--text-secondary)"><i class="fas fa-info-circle"></i> Generado automáticamente</small>`

  panel.appendChild(header)
  panel.appendChild(body)
  panel.appendChild(footer)

  // Insert at top so newest charts appear first and centered by grid
  grid.insertAdjacentElement('afterbegin', panel)
  return { created: true, chartId: id }
}

async function generateFinalResponse(userMessage, sp500Data, newsData) {
  let contextForLLM = "Analiza el siguiente contexto para responder la pregunta del usuario:\n\n";
  let hasContext = false;

  if (sp500Data && sp500Data.length > 0) {
    hasContext = true;
    contextForLLM += "**Datos del S&P 500:**\n";
    sp500Data.forEach(day => {
      contextForLLM += `- ${day.Date}: Cierre en $${Number.parseFloat(day.Close).toFixed(2)}\n`;
    });
    contextForLLM += "\n";
  }

  if (newsData && newsData.length > 0) {
    hasContext = true;
    contextForLLM += "**Noticias Relevantes:**\n";
    newsData.forEach(news => {
      contextForLLM += `- Fuente: ${news.source} | Título: ${news.title}\n`;
    });
    contextForLLM += "\n";
  }

  if (!hasContext) {
    contextForLLM = "No se encontró información específica (ni datos del S&P 500 ni noticias) para la fecha solicitada. Responde al usuario amablemente, indicando que no tienes datos para esa fecha y que por favor intente con otra.";
  }

  const finalUserMessage = `${contextForLLM}\n\n**Pregunta del usuario:** ${userMessage}`;
  
  const systemPrompt = "Eres un asistente financiero experto en el S&P 500. Tu tarea es responder a la pregunta del usuario basándote **única y exclusivamente** en el contexto proporcionado (datos del S&P 500). Sintetiza la información para explicar qué pasó en la fecha consultada. No menciones que eres un LLM ni hables de tu fecha de corte de conocimiento.";

  const messagesForAPI = [
    { role: "system", content: systemPrompt },
    { role: "user", content: finalUserMessage }
  ];

  try {
    const response = await fetch(CONFIG.GROQ_API_URL, {
      method: "POST",
      headers: { Authorization: `Bearer ${CONFIG.GROQ_API_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "llama-3.1-8b-instant",
        messages: messagesForAPI,
        temperature: 0.5,
        max_tokens: 2000,
      }),
    })
    if (!response.ok) {
      const txt = await response.text()
      console.error(`[v0] generateFinalResponse failed: ${response.status} ${response.statusText} - ${txt}`)
      return "Error generando respuesta. (ver consola para detalles)"
    }
    const data = await response.json()
    const msg = data.choices?.[0]?.message?.content || ""
    conversationHistory.push({ role: "assistant", content: msg })
    return msg
  } catch (error) {
    console.error("[v0] generateFinalResponse exception:", error);
    return "Error generando respuesta."
  }
}

function displayBotResponse(text, newsData, chartData) {
  addMessage(text, "bot")
  const lastMessage = messagesContainer.lastElementChild
  const messageText = lastMessage.querySelector(".message-text")
  
  if (chartData) {
    // Insertar preview pequeño en el chat
    const chartDiv = document.createElement("div")
    chartDiv.className = "message-chart"
    
    // Clonar canvas para preview
    const previewCanvas = chartData.cloneNode(true)
    previewCanvas.style.cursor = 'pointer'
    previewCanvas.style.maxHeight = '200px'
    chartDiv.appendChild(previewCanvas)
    
    const caption = document.createElement("p")
    caption.style.cssText = "font-size:0.85rem;color:var(--text-secondary);margin-top:8px;display:flex;align-items:center;gap:0.5rem;"
    caption.innerHTML = `<i class="fas fa-chart-line"></i> Gráfica del S&P 500 <button class="quick-action-btn" style="margin-left: auto;" onclick="openChartModal(this.closest('.message-chart').querySelector('canvas'), 'Análisis del S&P 500')"><i class="fas fa-expand"></i> Ver en grande</button>`
    chartDiv.appendChild(caption)
    
    messageText.appendChild(chartDiv)
    
    // Abrir modal automáticamente
    setTimeout(() => {
      openChartModal(chartData, 'Análisis del S&P 500')
    }, 500)
  }
  
  if (newsData && newsData.length > 0) {
    const newsDiv = document.createElement("div")
    newsDiv.className = "message-news"
    let newsHTML = '<h4 style="margin-bottom:12px;"><i class="fas fa-newspaper"></i> Noticias Relevantes</h4>'
    newsData.forEach((news, index) => {
      newsHTML += `<div class="news-item" data-news-index="${index}"><div class="news-item-content"><div class="news-title">${news.title}</div><div class="news-source">${news.source} • ${news.relevance_score}/10</div></div><div class="news-actions"><button class="news-btn" data-action="relevant"><i class="fas fa-check"></i></button><a href="${news.url}" target="_blank" class="news-btn"><i class="fas fa-external-link-alt"></i></a></div></div>`
    })
    newsDiv.innerHTML = newsHTML
    messageText.appendChild(newsDiv)
  }
}
function placeChartInDashboard(chartCanvas, captionText) {
  const grid = document.getElementById("analysisGrid")
  // If no dashboard grid exists, fallback to inserting into the chat message area
  if (!grid) {
    console.warn('[v0] analysisGrid not found — falling back to chat display')
    const chartDiv = document.createElement("div")
    chartDiv.className = "message-chart"
    chartDiv.appendChild(chartCanvas)
    const caption = document.createElement("p")
    caption.style.cssText = "font-size:0.85rem;color:var(--text-secondary);margin-top:8px;"
    caption.innerHTML = `<i class="fas fa-chart-line"></i> Gráfica del S&P 500`;
    chartDiv.appendChild(caption)
    const lastMessage = messagesContainer.lastElementChild
    if (lastMessage) lastMessage.querySelector(".message-text").appendChild(chartDiv)
    return
  }

  // Crear una card para el dashboard
  const panel = document.createElement("section")
  // Create a centered analysis panel inside the dashboard grid
  panel.className = "analysis-panel"
  // Título/encabezado con tiempo y breve caption
  const header = document.createElement("div")
  header.className = "analysis-panel-header"
  const title = document.createElement("h4")
  title.innerHTML = `<i class="fas fa-chart-line"></i> Análisis del S&P 500`
  const ts = document.createElement("div")
  ts.className = "analysis-ts"
  ts.textContent = new Date().toLocaleString()
  header.appendChild(title)
  header.appendChild(ts)

  const body = document.createElement("div")
  body.className = "analysis-panel-body"
  chartCanvas.style.maxWidth = "100%"
  body.appendChild(chartCanvas)

  const footer = document.createElement("div")
  footer.className = "analysis-panel-footer"
  footer.innerHTML = `<small style="color:var(--text-secondary)"><i class="fas fa-info-circle"></i> Generado automáticamente</small>`

  panel.appendChild(header)
  panel.appendChild(body)
  panel.appendChild(footer)

  // Insert at top so newest charts appear first
  grid.insertAdjacentElement('afterbegin', panel)
}

function handleNewsRelevance(button) {
  const newsItem = button.closest(".news-item")
  newsItem.style.background = "rgba(0,200,83,0.1)"
  newsItem.style.borderLeft = "3px solid var(--success)"
  button.disabled = true
  button.innerHTML = '<i class="fas fa-check"></i> Relevante'
  addMessage("Noticia marcada como relevante.", "bot")
}

function clearChat() {
  conversationHistory = []
  messagesContainer.innerHTML =
    '<div class="message bot-message welcome-message"><div class="message-avatar"><i class="fas fa-robot"></i></div><div class="message-content"><div class="message-text"><p>👋 ¡Conversación reiniciada!</p><ul><li><i class="fas fa-calendar"></i> Analizar fechas del S&P 500</li><li><i class="fas fa-newspaper"></i> Buscar noticias</li><li><i class="fas fa-chart-line"></i> Generar visualizaciones</li></ul></div><div class="quick-actions"><button class="quick-action-btn" data-action="¿Qué pasó el 2024-11-07?"><i class="fas fa-calendar-day"></i> Analizar fecha</button><button class="quick-action-btn" data-action="Muestra noticias del 2024-01-15"><i class="fas fa-newspaper"></i> Buscar noticias</button><button class="quick-action-btn" data-action="¿Cómo funciona el S&P 500?"><i class="fas fa-question-circle"></i> Aprender</button></div></div></div>'
}

function scrollToBottom() {
  messagesContainer.scrollTop = messagesContainer.scrollHeight
}


// ============== FUNCIONES PARA MODAL FLOTANTE ==============

function openChartModal(canvas, title) {
  const modal = document.getElementById('chartModal')
  const modalCanvas = document.getElementById('chartModalCanvas')
  const modalTitle = document.getElementById('chartModalTitle')
  
  // Limpiar contenido previo
  modalCanvas.innerHTML = ''
  
  // Clonar el canvas para el modal
  const clonedCanvas = canvas.cloneNode(true)
  clonedCanvas.style.width = '100%'
  clonedCanvas.style.height = 'auto'
  modalCanvas.appendChild(clonedCanvas)
  
  // Actualizar título
  modalTitle.textContent = title || 'Análisis del S&P 500'
  
  // Mostrar modal
  modal.classList.add('active')
  document.body.style.overflow = 'hidden'
  
  // Recrear la gráfica en el canvas clonado
  const originalChart = Object.values(window.__analysisCharts).find(chart => 
    chart.canvas === canvas
  )
  
  if (originalChart) {
    const newCtx = clonedCanvas.getContext('2d')
    new Chart(newCtx, {
      type: originalChart.config.type,
      data: originalChart.config.data,
      options: originalChart.config.options
    })
  }
}

function closeChartModal() {
  const modal = document.getElementById('chartModal')
  modal.classList.remove('active')
  document.body.style.overflow = 'auto'
}

function downloadChart() {
  const canvas = document.querySelector('#chartModalCanvas canvas')
  if (canvas) {
    const link = document.createElement('a')
    link.download = `sp500-analysis-${new Date().toISOString().split('T')[0]}.png`
    link.href = canvas.toDataURL('image/png')
    link.click()
  }
}

// Event listeners para el modal
document.addEventListener('DOMContentLoaded', () => {
  const modal = document.getElementById('chartModal')
  const closeBtn = document.getElementById('chartModalClose')
  const closeBtn2 = document.getElementById('closeChartModalBtn')
  const downloadBtn = document.getElementById('downloadChartBtn')
  
  // Cerrar con botones
  if (closeBtn) closeBtn.addEventListener('click', closeChartModal)
  if (closeBtn2) closeBtn2.addEventListener('click', closeChartModal)
  
  // Descargar gráfica
  if (downloadBtn) downloadBtn.addEventListener('click', downloadChart)
  
  // Cerrar al hacer click fuera del modal
  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        closeChartModal()
      }
    })
  }
  
  // Cerrar con ESC
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.classList.contains('active')) {
      closeChartModal()
    }
  })
})