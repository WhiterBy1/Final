// ==========================================
// CONFIGURACIÓN
// ==========================================
const API_BASE_URL = 'http://localhost:8000';
// Usamos el path del archivo más reciente que enviaste
const CSV_FILE_PATH = '../data/sp500_data.csv';

// ==========================================
// VARIABLES GLOBALES
// ==========================================
let sp500Chart = null;
let historicalData = [];
let forecastData = null; // <- Esta variable la usa 'loadAndDisplayForecast'

// ==========================================
// INICIALIZACIÓN (PUNTO DE ENTRADA)
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();

    // Cargar datos después de un pequeño delay
    setTimeout(() => {
        loadHistoricalData();
        loadTodayPredictions();
    }, 500);
});

function initializeApp() {
    const today = new Date();
    const startOfYear = new Date(today.getFullYear(), 0, 1);

    document.getElementById('startDate').valueAsDate = startOfYear;
    document.getElementById('endDate').valueAsDate = today;
}


function renderSentimentTrend(sentimentData) {
    const container = document.getElementById('sentimentTrend');

    if (!sentimentData || !sentimentData.weekly_data) {
        container.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No hay datos de sentimiento disponibles</p>';
        return;
    }

    const trendIcon = sentimentData.trend === 'improving' ? 'fa-arrow-trend-up' :
        sentimentData.trend === 'declining' ? 'fa-arrow-trend-down' :
            'fa-arrows-left-right';
    const trendClass = sentimentData.trend === 'improving' ? 'improving' :
        sentimentData.trend === 'declining' ? 'declining' :
            'stable';
    const trendText = sentimentData.trend === 'improving' ? 'Mejorando' :
        sentimentData.trend === 'declining' ? 'Empeorando' :
            'Estable';

    container.innerHTML = `
        <div class="sentiment-trend-indicator">
            <i class="fas ${trendIcon} trend-icon ${trendClass}"></i>
            <div>
                <div style="font-weight: 600;">${trendText}</div>
                <div style="font-size: 0.75rem; color: var(--text-secondary);">
                    Cambio: ${sentimentData.sentiment_change > 0 ? '+' : ''}${sentimentData.sentiment_change.toFixed(3)}
                </div>
            </div>
        </div>
        
        <div class="sentiment-bars">
            ${sentimentData.weekly_data.reverse().map(week => {
        const sentiment = week.combined_sentiment;
        const barClass = sentiment > 0.1 ? 'positive' : sentiment < -0.1 ? 'negative' : 'neutral';
        const width = Math.min(Math.abs(sentiment) * 100, 100);

        return `
                    <div class="sentiment-bar-item">
                        <span class="sentiment-bar-label">${week.week}</span>
                        <div class="sentiment-bar-wrapper">
                            <div class="sentiment-bar-fill ${barClass}" style="width: ${width}%"></div>
                        </div>
                    </div>
                `;
    }).join('')}
        </div>
    `;
}

function setupEventListeners() {
    document.getElementById('loadDataBtn').addEventListener('click', loadHistoricalData);
    document.getElementById('refreshPredictionsBtn').addEventListener('click', loadTodayPredictions);
    
    // ✅ NUEVO: Cambiar horizonte de predicción en tiempo real
    const horizonSelector = document.getElementById('forecastHorizon');
    if (horizonSelector) {
        horizonSelector.addEventListener('change', async (e) => {
            const horizonDays = parseInt(e.target.value);
            console.log(`🔄 Cambiando horizonte a ${horizonDays} días...`);
            
            if (historicalData && historicalData.length > 0) {
                const dailyForecast = await loadDailyForecast(horizonDays);
                if (dailyForecast) {
                    renderChartWithDailyForecast(historicalData, dailyForecast);
                }
            }
        });
    }
    
    document.querySelector('.close').addEventListener('click', closeModal);

    window.addEventListener('click', (e) => {
        const modal = document.getElementById('analysisModal');
        if (e.target === modal) {
            closeModal();
        }
    });

    document.getElementById('searchNewsBtn').addEventListener('click', searchRelatedNews);
}
// ==========================================
// ORQUESTADORES PRINCIPALES (CARGA DE DATOS)
// ==========================================

/**
 * Carga los datos históricos del CSV, los procesa y lanza el renderizado
 * de la gráfica y las predicciones diarias (versión robusta).
 */
async function loadHistoricalData() {
    try {
        showLoading();

        const startDate = document.getElementById('startDate').value;
        const endDate = document.getElementById('endDate').value;

        console.log('📥 Cargando datos del CSV...');

        const response = await fetch(CSV_FILE_PATH);
        const csvText = await response.text();

        historicalData = parseCSV(csvText, startDate, endDate);

        console.log(`✅ Cargados ${historicalData.length} registros históricos`);

        // Renderizar gráfica simple primero
        renderChart(historicalData);

        // ✅ Puedes cambiar el horizonte aquí (7, 15, 30, 60 días)
        // Obtener del selector si existe
        const horizonSelector = document.getElementById('forecastHorizon');
        const horizonDays = horizonSelector ? parseInt(horizonSelector.value) : 15;
        
        console.log(`🔮 Cargando predicciones para ${horizonDays} días...`);

        // Cargar predicciones contextual_llm
        const dailyForecast = await loadDailyForecast(horizonDays);
        if (dailyForecast) {
            renderChartWithDailyForecast(historicalData, dailyForecast);
        }

        hideLoading();

    } catch (error) {
        console.error('❌ Error cargando datos:', error);
        alert('Error cargando datos históricos. Verifica que el archivo CSV esté en: ' + CSV_FILE_PATH);
        hideLoading();
    }
}

/**
 * Carga todas las predicciones y análisis para el dashboard "Hoy".
 * Utiliza el endpoint robusto de secuencia diaria.
 */
async function loadTodayPredictions() {
    try {
        showPredictionsLoading();
        
        console.log('📊 Cargando análisis de hoy y patrones similares...');
        
        // 1. CARGAR ANÁLISIS DE HOY (para la sección izquierda)
        const today = new Date().toISOString().split('T')[0];
        const analysisResponse = await axios.get(`${API_BASE_URL}/api/analysis/date`, {
            params: {
                date: today,
                user_level: 'principiante'
            }
        });
        
        console.log('✅ Datos de /api/analysis/date:', analysisResponse.data);
        
        // 2. CARGAR PATRONES SIMILARES (para la sección derecha)
        const patternsResponse = await axios.get(`${API_BASE_URL}/api/prediction/advanced`, {
            params: {
                horizon_days: 30,
                include_similar_patterns: true,
                user_level: 'principiante'
            }
        });
        
        console.log('✅ Datos de /api/prediction/advanced:', patternsResponse.data);
        
        // 3. RENDERIZAR CADA SECCIÓN CON SU ENDPOINT CORRESPONDIENTE
        renderMainPrediction(analysisResponse.data);
        renderSimilarPatternsAdvanced(patternsResponse.data);
        
    } catch (error) {
        console.error('❌ Error cargando predicciones de hoy:', error);
        showPredictionsError();
    }
}

function renderSimilarPatternsAdvanced(data) {
    const container = document.getElementById('similarPatterns');
    
    if (!data.similar_patterns || data.similar_patterns.length === 0) {
        container.innerHTML = `
            <div class="no-patterns">
                <i class="fas fa-search"></i>
                <p>No se encontraron patrones similares</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = `
        <div class="patterns-container">
            ${data.similar_patterns.map(pattern => renderPatternCard(pattern)).join('')}
        </div>
    `;
}

/**
 * Renderiza una tarjeta individual de patrón histórico
 */
function renderPatternCard(pattern) {
    const behavior = pattern.post_pattern_behavior || {};
    const behaviorClass = behavior.direction === 'bullish' ? 'bullish' : 
                         behavior.direction === 'bearish' ? 'bearish' : 'neutral';
    const behaviorIcon = behavior.direction === 'bullish' ? 'fa-arrow-up' : 
                        behavior.direction === 'bearish' ? 'fa-arrow-down' : 'fa-arrows-left-right';
    const behaviorText = behavior.direction === 'bullish' ? 'Alcista' : 
                        behavior.direction === 'bearish' ? 'Bajista' : 'Neutral';
    
    // Calcular similitud combinada
    const similarity = pattern.similarity_score || 
                      (pattern.technical_similarity * 0.7 + (pattern.news_similarity || 0) * 0.3);
    
    // Topics de noticias
    const topicsHTML = pattern.key_news_topics && pattern.key_news_topics.length > 0 
        ? `
        <div class="pattern-topics">
            <i class="fas fa-tags"></i>
            <span>${pattern.key_news_topics.slice(0, 3).join(', ')}</span>
        </div>
    ` : '';
    
    return `
        <div class="pattern-card">
            <div class="pattern-card-header">
                <div class="pattern-period">
                    <div class="pattern-dates">
                        <strong>${formatDate(pattern.pattern_start)}</strong>
                        <span>→</span>
                        <strong>${formatDate(pattern.pattern_end)}</strong>
                    </div>
                </div>
                <div class="similarity-badge">
                    ${(similarity * 100).toFixed(0)}%
                </div>
            </div>
            
            ${topicsHTML}
            
            <div class="pattern-behavior-section">
                <div class="behavior-label">Comportamiento posterior:</div>
                <div class="behavior-result ${behaviorClass}">
                    <i class="fas ${behaviorIcon}"></i>
                    <span class="behavior-direction">${behaviorText}</span>
                    <span class="behavior-change">
                        ${behavior.total_return_pct > 0 ? '+' : ''}${behavior.total_return_pct ? behavior.total_return_pct.toFixed(2) : '0.00'}%
                    </span>
                    <span class="behavior-duration">
                        en ${behavior.duration_days || 0} días
                    </span>
                </div>
            </div>
            
            <div class="pattern-metrics">
                <div class="metric-item">
                    <span class="metric-label">Similitud técnica:</span>
                    <span class="metric-value">${(pattern.technical_similarity * 100).toFixed(0)}%</span>
                </div>
                ${pattern.news_similarity !== undefined ? `
                <div class="metric-item">
                    <span class="metric-label">Similitud noticias:</span>
                    <span class="metric-value">${(pattern.news_similarity * 100).toFixed(0)}%</span>
                </div>
                ` : ''}
            </div>
        </div>
    `;
}

function formatFactorText(factor) {
    if (factor.length > 80) {
        return factor.substring(0, 80) + '...';
    }
    return factor;
}

function renderNewsItemMini(news) {
    const sentimentIcon = news.sentiment_score > 15 ? 'fa-face-smile' : 
                         news.sentiment_score > 10 ? 'fa-face-meh' : 'fa-face-frown';
    const sentimentClass = news.sentiment_score > 15 ? 'positive' : 
                          news.sentiment_score > 10 ? 'neutral' : 'negative';
    
    return `
        <div class="news-item-mini" onclick="openNewsModal('${news.url}')">
            <div class="news-header-mini">
                <span class="news-source">${news.source}</span>
                <span class="news-sentiment ${sentimentClass}">
                    <i class="fas ${sentimentIcon}"></i>
                </span>
            </div>
            <p class="news-title-mini">${formatNewsTitle(news.title)}</p>
            <div class="news-footer-mini">
                <span class="news-impact">Impacto: ${news.impact_score}/10</span>
            </div>
        </div>
    `;
}

function formatNewsTitle(title) {
    if (title.length > 70) {
        return title.substring(0, 70) + '...';
    }
    return title;
}

function getRSIClass(rsi) {
    if (rsi > 70) return 'overbought';
    if (rsi < 30) return 'oversold';
    return 'neutral';
}

function formatTrend(trend) {
    const trends = {
        'alcista': 'Alcista',
        'bullish': 'Alcista',
        'bearish': 'Bajista',
        'bajista': 'Bajista',
        'neutral': 'Neutral',
        'fuerte_alcista': 'Muy Alcista',
        'fuerte_bajista': 'Muy Bajista'
    };
    return trends[trend] || trend;
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('es-ES', { 
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

function openNewsModal(url) {
    window.open(url, '_blank');
}


/**
 * Obtiene el análisis detallado para una fecha específica (al hacer clic en la gráfica).
 */
async function fetchDateAnalysis(date) {
    try {
        openModal(); // <-- Función completada
        showModalLoading(); // <-- Función completada

        console.log(`📊 Obteniendo análisis para: ${date}`);
        date = date.split(' ')[0]; // Limpia la fecha si viene con hora
        console.log(`🔍 Fecha limpia: ${date}`)
        ;
        const response = await axios.get(`${API_BASE_URL}/api/analysis/date`, {
            params: {
                date: date,
                user_level: 'principiante'
            }
        });

        console.log('✅ Análisis recibido');

        renderAnalysis(response.data); // <-- Función completada
        hideModalLoading(); // <-- Función completada

    } catch (error) {
        console.error('❌ Error obteniendo análisis:', error);
        alert('Error obteniendo análisis. Verifica que el servidor esté activo en ' + API_BASE_URL);
        closeModal(); // <-- Función completada
    }
}


// ==========================================
// HELPERS DE API Y PROCESAMIENTO DE DATOS
// ==========================================

/**
 * Parsea el texto CSV y filtra por rango de fechas.
 */
function parseCSV(csvText, startDate, endDate) {
    const lines = csvText.split('\n');
    const data = [];

    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');

        if (values.length < 5) continue;

        const date = values[0].trim();
        const close = parseFloat(values[4]);

        if (date >= startDate && date <= endDate && !isNaN(close)) {
            data.push({
                date: date,
                close: close
            });
        }
    }

    return data;
}

/**
 * Obtiene la predicción robusta de secuencia diaria (la principal).
 */
async function loadDailyForecast(horizonDays = 30) {
    try {
        if (!historicalData || historicalData.length === 0) {
            console.warn('⚠️ Esperando datos históricos (loadDailyForecast)...');
            return null;
        }

        console.log(`📊 Cargando predicciones contextual_llm (${horizonDays} días)...`);
        console.log('⏳ Esto puede tomar unos segundos...');

        // ✅ USAR EL NUEVO ENDPOINT CON contextual_llm
        const response = await axios.get(`${API_BASE_URL}/api/prediction/daily-forecast`, {
            params: {
                days: horizonDays,
                method: 'contextual_llm'
            }
        });

        const dailyForecast = response.data;

        console.log('✅ Predicciones contextual_llm cargadas:', dailyForecast.predictions.length);
        console.log(`📈 Precio actual: $${dailyForecast.last_real_price.toFixed(2)}`);
        console.log(`🎯 Predicción día ${horizonDays}: $${dailyForecast.summary.final_predicted_price.toFixed(2)} (${dailyForecast.summary.total_change_pct > 0 ? '+' : ''}${dailyForecast.summary.total_change_pct.toFixed(2)}%)`);

        return dailyForecast;

    } catch (error) {
        console.error('❌ Error cargando predicciones contextual_llm:', error);
        return null;
    }
}

/**
 * (Método alternativo/antiguo) Carga un forecast simple.
 */
async function loadAndDisplayForecast(horizonDays = 15) {
    try {
        if (!historicalData || historicalData.length === 0) {
            console.warn('⚠️ Esperando datos históricos (loadAndDisplayForecast)...');
            return;
        }

        console.log(`🔮 Cargando predicciones (método simple) para ${horizonDays} días...`);

        const response = await axios.get(`${API_BASE_URL}/api/prediction/forecast-data`, {
            params: {
                horizon_days: horizonDays,
                method: 'ensemble'
            }
        });

        forecastData = response.data;

        console.log('✅ Predicciones (método simple) cargadas');

        // Re-renderizar gráfica con predicciones
        renderChartWithForecast(historicalData, forecastData);

    } catch (error) {
        console.error('❌ Error cargando predicciones (método simple):', error);
    }
}

// ==========================================
// RENDERIZADO DE GRÁFICAS
// ==========================================

/**
 * Renderiza la gráfica principal solo con datos históricos.
 */
function renderChart(data) {
    const ctx = document.getElementById('sp500Chart').getContext('2d');

    if (sp500Chart) {
        sp500Chart.destroy();
    }

    sp500Chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => d.date),
            datasets: [{
                label: 'S&P 500',
                data: data.map(d => d.close),
                borderColor: '#2563eb',
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 6,
                pointHoverBackgroundColor: '#2563eb',
                pointHoverBorderColor: '#fff',
                pointHoverBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: false
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
                        title: function (context) {
                            const date = new Date(context[0].label);
                            return date.toLocaleDateString('es-ES', {
                                year: 'numeric',
                                month: 'long',
                                day: 'numeric'
                            });
                        },
                        label: function (context) {
                            return '$' + context.parsed.y.toFixed(2);
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxTicksLimit: 12
                    }
                },
                y: {
                    display: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        callback: function (value) {
                            return '$' + value.toFixed(0);
                        }
                    }
                }
            },
            onClick: (event, elements) => {
                if (elements.length > 0) {
                    const index = elements[0].index;
                    const clickedDate = data[index].date;
                    fetchDateAnalysis(clickedDate);
                }
            }
        }
    });
}

/**
 * Renderiza la gráfica con datos históricos + la predicción de secuencia diaria (Robusta).
 */
function renderChartWithDailyForecast(historical, dailyForecast) {
    const ctx = document.getElementById('sp500Chart').getContext('2d');

    if (sp500Chart) {
        sp500Chart.destroy();
    }

    const datasets = [];

    // 1. Datos históricos
    datasets.push({
        label: 'S&P 500 (real)',
        data: historical.map(d => ({ x: d.date, y: d.close })),
        borderColor: '#2563eb',
        backgroundColor: 'rgba(37, 99, 235, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 6,
        pointHoverBackgroundColor: '#2563eb',
        pointHoverBorderColor: '#fff',
        pointHoverBorderWidth: 2
    });

    // 2. Predicciones contextual_llm (USANDO NUEVOS CAMPOS)
    if (dailyForecast && dailyForecast.predictions && dailyForecast.predictions.length > 0) {
        const lastHistorical = historical[historical.length - 1];
        
        // ✅ Mapear predicciones con los nuevos campos
        const predictionPoints = dailyForecast.predictions.map(p => ({
            x: p.date,
            y: p.predicted_price
        }));
        
        const connectionPoint = { x: lastHistorical.date, y: lastHistorical.close };

        datasets.push({
            label: `Predicción ${dailyForecast.total_days} días (LLM)`,
            data: [connectionPoint, ...predictionPoints],
            borderColor: '#ef4444',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            borderWidth: 2,
            borderDash: [5, 5],
            fill: false,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 6,
            pointHoverBackgroundColor: '#ef4444',
            pointHoverBorderColor: '#fff',
            pointHoverBorderWidth: 2
        });

        // 3. Bandas de confianza (USANDO confidence_lower y confidence_upper)
        const upperBandPoints = dailyForecast.predictions.map(p => ({
            x: p.date,
            y: p.confidence_upper
        }));
        
        const lowerBandPoints = dailyForecast.predictions.map(p => ({
            x: p.date,
            y: p.confidence_lower
        }));

        datasets.push({
            label: 'Banda superior (95%)',
            data: upperBandPoints,
            borderColor: 'rgba(239, 68, 68, 0.3)',
            borderWidth: 1,
            borderDash: [2, 2],
            fill: false,
            pointRadius: 0,
            tension: 0.4
        });

        datasets.push({
            label: 'Banda inferior (95%)',
            data: lowerBandPoints,
            borderColor: 'rgba(239, 68, 68, 0.3)',
            borderWidth: 1,
            borderDash: [2, 2],
            fill: '-1',
            backgroundColor: 'rgba(239, 68, 68, 0.05)',
            pointRadius: 0,
            tension: 0.4
        });
    }

    sp500Chart = new Chart(ctx, {
        type: 'line',
        data: { datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        filter: function (item) {
                            return !item.text.includes('Banda');
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    callbacks: {
                        title: function (context) {
                            const date = new Date(context[0].parsed.x);
                            return date.toLocaleDateString('es-ES', {
                                year: 'numeric',
                                month: 'long',
                                day: 'numeric'
                            });
                        },
                        label: function (context) {
                            const datasetLabel = context.dataset.label || '';
                            const value = '$' + context.parsed.y.toFixed(2);

                            // ✅ MOSTRAR INFO CORRECTA CON NUEVOS CAMPOS
                            if (datasetLabel.includes('Predicción')) {
                                const predIndex = context.dataIndex - 1; // -1 por el punto de conexión
                                
                                if (predIndex >= 0 && dailyForecast && dailyForecast.predictions[predIndex]) {
                                    const pred = dailyForecast.predictions[predIndex];
                                    return [
                                        `Día ${pred.day}: ${value}`,
                                        `Retorno día: ${pred.predicted_return > 0 ? '+' : ''}${pred.predicted_return.toFixed(2)}%`,
                                        `Cambio total: ${pred.change_from_current > 0 ? '+' : ''}${pred.change_from_current.toFixed(2)}%`,
                                        `Rango: $${pred.confidence_lower.toFixed(2)} - $${pred.confidence_upper.toFixed(2)}`
                                    ];
                                }
                                return `${datasetLabel}: ${value}`;
                            } else if (datasetLabel.includes('real')) {
                                return `Precio real: ${value}`;
                            } else if (datasetLabel.includes('Banda')) {
                                return `${datasetLabel}: ${value}`;
                            }
                            return `${datasetLabel}: ${value}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        displayFormats: { day: 'MMM dd' }
                    },
                    grid: { display: false },
                    ticks: { maxTicksLimit: 12 }
                },
                y: {
                    display: true,
                    grid: { color: 'rgba(0, 0, 0, 0.05)' },
                    ticks: {
                        callback: function (value) {
                            return '$' + value.toFixed(0);
                        }
                    }
                }
            },
            onClick: (event, elements) => {
                if (elements.length > 0) {
                    const element = elements[0];
                    const datasetIndex = element.datasetIndex;
                    const dataIndex = element.index;

                    // Solo permitir click en datos históricos (dataset 0)
                    if (datasetIndex === 0 && historical[dataIndex]) {
                        const clickedDate = historical[dataIndex].date;
                        fetchDateAnalysis(clickedDate);
                    }
                }
            }
        }
    });
}
/**
 * Renderiza la gráfica con el método de forecast alternativo/antiguo.
 */
function renderChartWithForecast(historical, forecast) {
    const ctx = document.getElementById('sp500Chart').getContext('2d');

    if (sp500Chart) {
        sp500Chart.destroy();
    }

    const datasets = [];

    // 1. Datos históricos
    datasets.push({
        label: 'S&P 500 (real)',
        data: historical.map(d => ({ x: d.date, y: d.close })),
        borderColor: '#2563eb',
        backgroundColor: 'rgba(37, 99, 235, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 0
    });

    // 2. Predicciones (método simple)
    if (forecast && forecast.forecast && historical.length > 0) {
        const lastHistorical = historical[historical.length - 1];
        const predictionPoints = [
            { x: lastHistorical.date, y: lastHistorical.close },
            ...forecast.forecast.map(f => ({ x: f.date, y: f.predicted_price }))
        ];

        datasets.push({
            label: `Predicción ${forecast.horizon_days} días`,
            data: predictionPoints,
            borderColor: '#ef4444',
            borderWidth: 2,
            borderDash: [5, 5],
            fill: false,
            tension: 0.4,
            pointRadius: 3,
            pointBackgroundColor: '#ef4444'
        });

        // 3. Bandas de confianza
        datasets.push({
            label: 'Banda superior',
            data: forecast.forecast.map(f => ({ x: f.date, y: f.confidence_upper })),
            borderColor: 'rgba(239, 68, 68, 0.3)',
            borderWidth: 1,
            borderDash: [2, 2],
            fill: false,
            pointRadius: 0,
            tension: 0.4
        });
        datasets.push({
            label: 'Banda inferior',
            data: forecast.forecast.map(f => ({ x: f.date, y: f.confidence_lower })),
            borderColor: 'rgba(239, 68, 68, 0.3)',
            borderWidth: 1,
            borderDash: [2, 2],
            fill: '-1',
            backgroundColor: 'rgba(239, 68, 68, 0.05)',
            pointRadius: 0,
            tension: 0.4
        });
    }

    sp500Chart = new Chart(ctx, {
        type: 'line',
        data: { datasets: datasets },
        options: {
            // ... (Opciones de la gráfica simple, adaptadas para 'time') ...
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        displayFormats: { day: 'MMM dd' }
                    },
                    grid: { display: false },
                    ticks: { maxTicksLimit: 12 }
                },
                y: {
                    grid: { color: 'rgba(0, 0, 0, 0.05)' },
                    ticks: {
                        callback: function (value) {
                            return '$' + value.toFixed(0);
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        filter: (item) => !item.text.includes('Banda')
                    }
                },
                tooltip: {
                    // ... (Callbacks de tooltip para mostrar rangos, etc.) ...
                }
            },
            onClick: (event, elements) => {
                if (elements.length > 0) {
                    const element = elements[0];
                    if (element.datasetIndex === 0 && historical[element.index]) {
                        fetchDateAnalysis(historical[element.index].date);
                    }
                }
            }
        }
    });
}


// ==========================================
// RENDERIZADO DE COMPONENTES (DASHBOARD)
// (Funciones completadas del segundo archivo)
// ==========================================
/**
 * Renderiza el cuadro principal de predicción con datos directos del backend.
 */
async function loadMainPrediction() {
    try {
        const today = new Date().toISOString().split('T')[0]; // Formato YYYY-MM-DD

        const response = await axios.get(`${API_BASE_URL}/api/analysis/date?date=${today}&user_level=principiante`);
        const data = response.data;
        
        // ============================================
        // 🔍 DEBUG: Mostrar datos de /api/analysis/date
        // ============================================
        console.log('✅ DATOS DE /api/analysis/date (HOY):');
        console.log('='.repeat(50));
        console.log(data);
        console.log('='.repeat(50));
        console.log('Claves:', Object.keys(data));
        
        renderMainPrediction(data);
    } catch (error) {
        console.error('Error cargando predicción principal:', error);
        document.getElementById('mainPrediction').innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle"></i>
                <p>Error cargando análisis del día</p>
            </div>
        `;
    }
}



function renderMainPrediction(data) {
    const container = document.getElementById('mainPrediction');
    
    // Determinar icono y clase según market_direction
    const directionConfig = {
        'alcista': { icon: 'fa-arrow-trend-up', class: 'bullish', text: 'Alcista' },
        'bullish': { icon: 'fa-arrow-trend-up', class: 'bullish', text: 'Alcista' },
        'bajista': { icon: 'fa-arrow-trend-down', class: 'bearish', text: 'Bajista' },
        'bearish': { icon: 'fa-arrow-trend-down', class: 'bearish', text: 'Bajista' },
        'neutral': { icon: 'fa-arrows-left-right', class: 'neutral', text: 'Neutral' }
    };
    
    const direction = directionConfig[data.market_direction] || directionConfig['neutral'];
    
    // Formatear volumen
    const volumeFormatted = (data.sp500_volume / 1000000).toFixed(1) + 'M';
    
    // Procesar explicación causal
    const explanationHTML = formatCausalityExplanation(data.causality_explanation);
    
    // Crear lista de factores clave
    const factorsHTML = data.key_factors && data.key_factors.length > 0 
        ? `
        <div class="factors-mini">
            <h5 class="prediction-subtitle">Factores Clave</h5>
            <ul class="factors-list-mini">
                ${data.key_factors.map(factor => `<li>${formatFactorText(factor)}</li>`).join('')}
            </ul>
        </div>
    ` : '';

    // Crear lista de noticias destacadas
    const newsHTML = data.news && data.news.length > 0 
        ? `
        <div class="news-mini">
            <h5 class="prediction-subtitle">
                <i class="fas fa-newspaper"></i> Noticias Destacadas
            </h5>
            <div class="news-list-mini">
                ${data.news.slice(0, 3).map(news => renderNewsItemMini(news)).join('')}
            </div>
        </div>
    ` : '';

    container.innerHTML = `
        <div class="prediction-header">
            <div class="market-direction-indicator ${direction.class}">
                <i class="fas ${direction.icon}"></i>
                <span class="direction-text">${direction.text}</span>
            </div>
        </div>
        
        <div class="prediction-details">
            <div class="detail-row">
                <span class="detail-label">Precio Actual:</span>
                <span class="detail-value">$${data.sp500_close.toFixed(2)}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Volumen:</span>
                <span class="detail-value">${volumeFormatted}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Confianza:</span>
                <span class="detail-value">${data.confidence_score}%</span>
            </div>
        </div>
        
        <div class="prediction-explanation">
            <h5 class="prediction-subtitle">¿Qué está pasando hoy?</h5>
            <div class="explanation-text">${explanationHTML}</div>
        </div>
        
        ${factorsHTML}
        ${newsHTML}
        
        <div class="technical-mini">
            <h5 class="prediction-subtitle">
                <i class="fas fa-chart-line"></i> Indicadores Técnicos
            </h5>
            <div class="technical-grid-mini">
                <div class="tech-indicator">
                    <span class="tech-label">RSI</span>
                    <span class="tech-value ${getRSIClass(data.technical.rsi)}">${data.technical.rsi.toFixed(1)}</span>
                </div>
                <div class="tech-indicator">
                    <span class="tech-label">Tendencia</span>
                    <span class="tech-value trend-${data.technical.trend}">${formatTrend(data.technical.trend)}</span>
                </div>
                <div class="tech-indicator">
                    <span class="tech-label">MACD</span>
                    <span class="tech-value ${data.technical.macd.value > data.technical.macd.signal ? 'positive' : 'negative'}">
                        ${data.technical.macd.value.toFixed(1)}
                    </span>
                </div>
            </div>
        </div>
    `;
}

function formatCausalityExplanation(explanation) {
    if (!explanation) return '<p>No hay explicación disponible</p>';
    
    // Convertir markdown básico a HTML
    let html = explanation
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // Negritas
        .replace(/\*(.*?)\*/g, '<em>$1</em>')              // Cursivas
        .replace(/\n\n/g, '</p><p>')                       // Párrafos
        .replace(/\n/g, '<br>');                           // Saltos de línea
    
    // Envolver en párrafos si no están ya envueltos
    if (!html.startsWith('<p>')) {
        html = '<p>' + html + '</p>';
    }
    
    return html;
}

// Función auxiliar para formatear texto de factores
function formatFactorText(factor) {
    // Limitar longitud y agregar ellipsis si es muy largo
    if (factor.length > 80) {
        return factor.substring(0, 80) + '...';
    }
    return factor;
}

// Función para formatear explicación mini
function formatExplanationMini(explanation) {
    // Extraer la parte más importante de la explicación
    const lines = explanation.split('\n');
    const mainExplanation = lines.find(line => line.includes('movimiento del S&P 500') || line.includes('Causas principales'));
    
    if (mainExplanation) {
        // Limpiar markdown y limitar longitud
        const cleanText = mainExplanation.replace(/\*\*/g, '').replace(/\*/g, '');
        if (cleanText.length > 150) {
            return cleanText.substring(0, 150) + '...';
        }
        return cleanText;
    }
    
    // Fallback: tomar primeros 120 caracteres
    const cleanText = explanation.replace(/\*\*/g, '').replace(/\*/g, '');
    if (cleanText.length > 120) {
        return cleanText.substring(0, 120) + '...';
    }
    return cleanText;
}

// Función para renderizar item de noticia mini
function renderNewsItemMini(news) {
    const sentimentIcon = news.sentiment_score > 15 ? 'fa-face-smile' : 
                         news.sentiment_score > 10 ? 'fa-face-meh' : 'fa-face-frown';
    const sentimentClass = news.sentiment_score > 15 ? 'positive' : 
                          news.sentiment_score > 10 ? 'neutral' : 'negative';
    
    return `
        <div class="news-item-mini" onclick="openNewsModal('${news.url}')">
            <div class="news-header-mini">
                <span class="news-source">${news.source}</span>
                <span class="news-sentiment ${sentimentClass}">
                    <i class="fas ${sentimentIcon}"></i>
                </span>
            </div>
            <p class="news-title-mini">${formatNewsTitle(news.title)}</p>
            <div class="news-footer-mini">
                <span class="news-impact">Impacto: ${news.impact_score}/10</span>
            </div>
        </div>
    `;
}

// Función para formatear título de noticia
function formatNewsTitle(title) {
    if (title.length > 70) {
        return title.substring(0, 70) + '...';
    }
    return title;
}

// Función para obtener clase CSS del RSI
function getRSIClass(rsi) {
    if (rsi > 70) return 'overbought';
    if (rsi < 30) return 'oversold';
    return 'neutral';
}

// Función para formatear tendencia
function formatTrend(trend) {
    const trends = {
        'alcista': 'Alcista',
        'bearish': 'Bajista', 
        'neutral': 'Neutral'
    };
    return trends[trend] || trend;
}

// Modal para noticias (nuevo)
function openNewsModal(url) {
    window.open(url, '_blank');
}

// Función para cargar patrones similares (mejorada para usar más espacio)
async function loadSimilarPatterns() {
    try {
        const response = await axios.get(`${API_BASE_URL}/api/prediction/advanced?horizon_days=30&include_similar_patterns=true&user_level=experto`);
        const data = response.data;
        console.log('Patrones similares cargados:', data.similar_patterns.length);
        console.log('DATA = ', data )
        renderSimilarPatterns(data.similar_patterns || []);
    } catch (error) {
        console.error('Error cargando patrones similares:', error);
        document.getElementById('similarPatterns').innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle"></i>
                <p>Error cargando patrones históricos</p>
            </div>
        `;
    }
}

// Función para renderizar patrones similares (mejorada)
function renderSimilarPatterns(patterns) {
    const container = document.getElementById('similarPatterns');
    
    if (!patterns || patterns.length === 0) {
        container.innerHTML = `
            <div class="no-patterns">
                <i class="fas fa-search"></i>
                <p>No se encontraron patrones similares</p>
            </div>
        `;
        return;
    }

    container.innerHTML = `
        <div class="patterns-header">
            <span>Período Histórico</span>
            <span>Similitud</span>
        </div>
        <div class="patterns-list">
            ${patterns.map(pattern => renderPatternItem(pattern)).join('')}
        </div>
    `;
}

// Función para renderizar item de patrón (mejorada)
function renderPatternItem(pattern) {
    const behavior = pattern.post_pattern_behavior;
    const behaviorClass = behavior && behavior.direction ? behavior.direction : 'neutral';
    const behaviorText = behavior && behavior.direction ? 
        (behavior.direction === 'bullish' ? 'Alcista' : 
         behavior.direction === 'bearish' ? 'Bajista' : 'Neutral') : 'Sin datos';
    
    const behaviorChange = behavior && behavior.avg_change ? 
        `${behavior.avg_change > 0 ? '+' : ''}${behavior.avg_change.toFixed(1)}%` : 'N/A';

    // Calcular similitud combinada si existe
    const similarity = pattern.similarity_score || 
                     (pattern.technical_similarity * 0.7 + (pattern.news_similarity || 0) * 0.3);
    
    return `
        <div class="pattern-item expanded" onclick="showPatternDetails('${pattern.pattern_start}')">
            <div class="pattern-period">
                <div class="pattern-dates">
                    <strong>${formatDate(pattern.pattern_start)}</strong>
                    <span>a</span>
                    <strong>${formatDate(pattern.pattern_end)}</strong>
                </div>
                ${pattern.key_news_topics && pattern.key_news_topics.length > 0 ? `
                    <div class="pattern-topics">
                        <i class="fas fa-tags"></i>
                        ${pattern.key_news_topics.slice(0, 2).join(', ')}
                    </div>
                ` : ''}
            </div>
            
            <div class="pattern-metrics">
                <div class="similarity-score">
                    <span class="score-value">${(similarity * 100).toFixed(0)}%</span>
                    <div class="score-bar">
                        <div class="score-fill" style="width: ${similarity * 100}%"></div>
                    </div>
                </div>
                
                <div class="pattern-behavior">
                    <span class="behavior-indicator ${behaviorClass}">
                        <i class="fas ${behaviorClass === 'bullish' ? 'fa-arrow-up' : 
                                      behaviorClass === 'bearish' ? 'fa-arrow-down' : 'fa-arrows-alt-h'}"></i>
                        ${behaviorText}
                    </span>
                    <span class="behavior-change">${behaviorChange}</span>
                </div>
            </div>
        </div>
    `;
}

// Función para formatear fecha
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('es-ES', { 
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

// Inicializar cuando se carga la página
document.addEventListener('DOMContentLoaded', function() {
    loadMainPrediction();
    loadSimilarPatterns();
    
    // Botón de actualizar
    document.getElementById('refreshPredictionsBtn').addEventListener('click', function() {
        loadMainPrediction();
        loadSimilarPatterns();
        
        // Efecto visual de actualización
        this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Actualizando...';
        setTimeout(() => {
            this.innerHTML = '<i class="fas fa-sync-alt"></i> Actualizar';
        }, 2000);
    });
});


/**
 * 
 * 
 * 
 * 
 * */ 

function renderSimilarPatterns(patterns) {
    const container = document.getElementById('similarPatterns');

    if (!patterns || patterns.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No se encontraron patrones similares</p>';
        return;
    }

    container.innerHTML = patterns.map(pattern => {
        const behavior = pattern.post_pattern_behavior;
        const behaviorClass = behavior.direction === 'bullish' ? 'bullish' :
            behavior.direction === 'bearish' ? 'bearish' :
                'neutral';
        const behaviorIcon = behavior.direction === 'bullish' ? 'fa-arrow-up' :
            behavior.direction === 'bearish' ? 'fa-arrow-down' :
                'fa-arrows-left-right';

        return `
            <div class="pattern-item" onclick="showPatternDetails('${pattern.pattern_start}')">
                <div class="pattern-header">
                    <span class="pattern-date">${pattern.pattern_start}</span>
                    <span class="similarity-badge">${(pattern.similarity_score * 100).toFixed(0)}%</span>
                </div>
                
                <div class="pattern-behavior">
                    Después de este patrón:
                    <div class="behavior-indicator ${behaviorClass}">
                        <i class="fas ${behaviorIcon}"></i>
                        ${behavior.total_return_pct > 0 ? '+' : ''}${behavior.total_return_pct}% en ${behavior.duration_days} días
                    </div>
                </div>
                
                <div class="news-placeholder" style="margin-top: 0.5rem; padding: 0.5rem; font-size: 0.75rem;">
                    <i class="fas fa-info-circle"></i>
                    Análisis de noticias: Próximamente
                </div>
            </div>
        `;
    }).join('');
}

function showPatternDetails(patternDate) {
    console.log('Mostrando detalles del patrón:', patternDate);
    alert(`Detalles del patrón del ${patternDate}\n\nEsta funcionalidad estará disponible próximamente.`);
}

// ==========================================
// RENDERIZADO DE COMPONENTES (MODAL ANÁLISIS)
// (Funciones completadas del segundo archivo)
// ==========================================

function renderAnalysis(data) {
    // Fecha
    const dateObj = new Date(data.date);
    const dateFormatted = dateObj.toLocaleDateString('es-ES', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
    document.getElementById('modalDate').textContent = `Análisis del ${dateFormatted}`;

    // Dirección del mercado
    const directionEl = document.getElementById('marketDirection');
    directionEl.innerHTML = getMarketDirectionHTML(data.market_direction);
    directionEl.className = `market-direction ${data.market_direction}`;

    // Precio de cierre
    document.getElementById('closePrice').textContent = `$${data.sp500_close.toFixed(2)}`;

    // Cambio porcentual
    const changeEl = document.getElementById('changePercent');
    const changeValue = data.sp500_change_pct;
    changeEl.textContent = `${changeValue >= 0 ? '+' : ''}${changeValue.toFixed(2)}%`;
    changeEl.className = `change ${changeValue >= 0 ? 'positive' : 'negative'}`;

    // Volumen
    document.getElementById('volume').textContent = formatVolume(data.sp500_volume);

    // Confianza
    document.getElementById('confidence').textContent = `${data.confidence_score.toFixed(0)}%`;
    const confidenceFill = document.getElementById('confidenceFill');
    confidenceFill.style.width = `${data.confidence_score}%`;

    // Explicación
    document.getElementById('explanationText').innerHTML = formatExplanation(data.causality_explanation);

    // Factores clave
    renderKeyFactors(data.key_factors);

    // Indicadores técnicos
    renderTechnicalIndicators(data.technical);

    // Sentimiento social
    renderSocialSentiment(data.social);
}

function getMarketDirectionHTML(direction) {
    const icons = {
        'alcista': '<i class="fas fa-arrow-up"></i> Alcista',
        'bajista': '<i class="fas fa-arrow-down"></i> Bajista',
        'neutral': '<i class="fas fa-arrows-alt-h"></i> Neutral'
    };
    return icons[direction] || '<i class="fas fa-question"></i> Desconocido';
}

function formatVolume(volume) {
    if (volume >= 1e9) {
        return (volume / 1e9).toFixed(1) + 'B';
    } else if (volume >= 1e6) {
        return (volume / 1e6).toFixed(1) + 'M';
    }
    return volume.toFixed(0);
}

function formatExplanation(text) {
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\n/g, '<br>');
    return text;
}

function renderKeyFactors(factors) {
    const container = document.getElementById('keyFactors');
    container.innerHTML = '';

    factors.forEach(factor => {
        const factorEl = document.createElement('div');
        factorEl.className = 'factor-item';
        factorEl.innerHTML = `
            <span class="factor-icon">${getFactorIcon(factor)}</span>
            <span>${factor}</span>
        `;
        container.appendChild(factorEl);
    });
}

function getFactorIcon(factor) {
    const match = factor.match(/^(📊|📈|📉|💰|⚡|🌐|📰)/);
    return match ? match[0] : '📌';
}

function renderTechnicalIndicators(technical) {
    const container = document.getElementById('technicalIndicators');
    container.innerHTML = '';

    const indicators = [
        {
            name: 'RSI (Índice de Fuerza Relativa)',
            value: technical.rsi.toFixed(2),
            description: interpretRSI(technical.rsi)
        },
        {
            name: 'MACD',
            value: technical.macd.value.toFixed(2),
            description: interpretMACD(technical.macd)
        },
        {
            name: 'Media Móvil 50 días',
            value: `$${technical.sma_50.toFixed(2)}`,
            description: 'Promedio de precio de los últimos 50 días'
        },
        {
            name: 'Tendencia',
            value: technical.trend,
            description: interpretTrend(technical.trend)
        }
    ];

    indicators.forEach(indicator => {
        const card = document.createElement('div');
        card.className = 'indicator-card';
        card.innerHTML = `
            <div class="indicator-name">${indicator.name}</div>
            <div class="indicator-value">${indicator.value}</div>
            <div class="indicator-description">${indicator.description}</div>
        `;
        container.appendChild(card);
    });
}

function interpretRSI(rsi) {
    if (rsi > 70) return 'El mercado podría estar "sobrecomprado" (muchos comprando)';
    if (rsi < 30) return 'El mercado podría estar "sobrevendido" (muchos vendiendo)';
    return 'El mercado está en equilibrio';
}

function interpretMACD(macd) {
    if (macd.value > macd.signal) return 'Señal positiva, podría subir';
    if (macd.value < macd.signal) return 'Señal negativa, podría bajar';
    return 'Sin señal clara';
}

function interpretTrend(trend) {
    const trends = {
        'fuerte_alcista': 'Subida fuerte y consistente',
        'alcista': 'Subida moderada',
        'bajista': 'Bajada moderada',
        'fuerte_bajista': 'Bajada fuerte y consistente',
        'neutral': 'Sin tendencia clara, movimiento lateral'
    };
    return trends[trend] || 'Tendencia no identificada';
}

function renderSocialSentiment(social) {
    const container = document.getElementById('socialSentiment');
    container.innerHTML = '';

    if (!social || social.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No hay datos de redes sociales disponibles</p>';
        return;
    }

    social.forEach(platform => {
        const card = document.createElement('div');
        card.className = 'social-card';

        const sentimentClass = platform.sentiment_average > 0.1 ? 'positive' :
            platform.sentiment_average < -0.1 ? 'negative' : 'neutral';

        card.innerHTML = `
            <div class="platform-name">${platform.platform}</div>
            <div class="sentiment-value ${sentimentClass}">${platform.sentiment_average.toFixed(2)}</div>
            <div class="mentions">${platform.mentions_count} menciones</div>
        `;
        container.appendChild(card);
    });
}

// ==========================================
// BUSCAR NOTICIAS RELACIONADAS
// (Funciones completadas del segundo archivo)
// ==========================================
function searchRelatedNews() {
    const factors = Array.from(document.querySelectorAll('.factor-item span:last-child'))
        .map(el => el.textContent);

    const keywords = extractKeywords(factors);
    const searchQuery = keywords.join(' ') + ' S&P 500';
    const url = `https://news.google.com/search?q=${encodeURIComponent(searchQuery)}`;

    window.open(url, '_blank');
}

function extractKeywords(factors) {
    const keywords = [];

    factors.forEach(factor => {
        const cleaned = factor.replace(/[\u{1F300}-\u{1F9FF}]/gu, '').trim();

        if (cleaned.includes('RSI')) keywords.push('RSI');
        if (cleaned.includes('MACD')) keywords.push('MACD');
        if (cleaned.includes('volatilidad')) keywords.push('volatility');
        if (cleaned.includes('volumen')) keywords.push('volume');
        if (cleaned.includes('tendencia')) keywords.push('trend');
        if (cleaned.includes('alcista')) keywords.push('bullish');
        if (cleaned.includes('bajista')) keywords.push('bearish');
    });

    return keywords.length > 0 ? keywords : ['market analysis'];
}

// ==========================================
// UTILIDADES DE UI (MODAL Y LOADERS)
// (Funciones completadas del segundo archivo)
// ==========================================
function openModal() {
    document.getElementById('analysisModal').style.display = 'block';
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    document.getElementById('analysisModal').style.display = 'none';
    document.body.style.overflow = 'auto';
}

function showModalLoading() {
    document.getElementById('modalLoading').style.display = 'block';
    document.getElementById('modalBody').style.display = 'none';
}

function hideModalLoading() {
    document.getElementById('modalLoading').style.display = 'none';
    document.getElementById('modalBody').style.display = 'block';
}

// Loaders generales
function showLoading() {
    // Implementación simple (del segundo archivo)
    console.log('⏳ Cargando...');
    // (Puedes reemplazar esto con un loader visual si lo deseas)
}

function hideLoading() {
    // Implementación simple (del segundo archivo)
    console.log('✅ Carga completada');
}


function renderSimilarPatterns(patterns) {
    const container = document.getElementById('similarPatterns');

    if (!patterns || patterns.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No se encontraron patrones similares</p>';
        return;
    }

    container.innerHTML = patterns.map(pattern => {
        const behavior = pattern.post_pattern_behavior;
        const behaviorClass = behavior.direction === 'bullish' ? 'bullish' :
            behavior.direction === 'bearish' ? 'bearish' :
                'neutral';
        const behaviorIcon = behavior.direction === 'bullish' ? 'fa-arrow-up' :
            behavior.direction === 'bearish' ? 'fa-arrow-down' :
                'fa-arrows-left-right';

        return `
            <div class="pattern-item" onclick="showPatternDetails('${pattern.pattern_start}')">
                <div class="pattern-header">
                    <span class="pattern-date">${pattern.pattern_start}</span>
                    <span class="similarity-badge">${(pattern.similarity_score * 100).toFixed(0)}%</span>
                </div>
                
                <div class="pattern-behavior">
                    Después de este patrón:
                    <div class="behavior-indicator ${behaviorClass}">
                        <i class="fas ${behaviorIcon}"></i>
                        ${behavior.total_return_pct > 0 ? '+' : ''}${behavior.total_return_pct.toFixed(2)}% en ${behavior.duration_days} días
                    </div>
                </div>
                
                <div class="news-placeholder" style="margin-top: 0.5rem; padding: 0.5rem; font-size: 0.75rem;">
                    <i class="fas fa-info-circle"></i>
                    Análisis de noticias: Próximamente
                </div>
            </div>
        `;
    }).join('');
}

function showPatternDetails(patternDate) {
    console.log('Mostrando detalles del patrón:', patternDate);
    alert(`Detalles del patrón del ${patternDate}\n\nEsta funcionalidad estará disponible próximamente.`);
}

// Loaders específicos del Dashboard
function showPredictionsLoading() {
    // (Esta es la versión más descriptiva del primer archivo)
    const loadingHTML = `
        <div class="loading-small">
            <i class="fas fa-spinner fa-spin"></i>
            <p>Generando predicciones robustas...</p>
            <p style="font-size: 0.8rem; color: var(--text-secondary);">
                Analizando sentimiento, patrones y técnicos
            </p>
        </div>
    `;
    document.getElementById('mainPrediction').innerHTML = loadingHTML;
    document.getElementById('sentimentTrend').innerHTML = loadingHTML;
    document.getElementById('similarPatterns').innerHTML = loadingHTML;
}

function showPredictionsError() {
    const errorHTML = '<p style="color: var(--danger-color); text-align: center;">Error cargando datos</p>';
    document.getElementById('mainPrediction').innerHTML = errorHTML;
    document.getElementById('sentimentTrend').innerHTML = errorHTML;
    document.getElementById('similarPatterns').innerHTML = errorHTML;
}

async function searchRelatedNews() {
    try {
        const modalDate = document.getElementById('modalDate').textContent;
        const dateMatch = modalDate.match(/(\d{1,2}) de (\w+) de (\d{4})/);
        
        if (dateMatch) {
            const [, day, month, year] = dateMatch;
            const months = {
                'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
                'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
                'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'
            };
            
            const dateStr = `${year}-${months[month.toLowerCase()]}-${day.padStart(2, '0')}`;
            
            // Obtener noticias reales del backend
            const response = await axios.get(`${API_BASE_URL}/api/news/date/${dateStr}`);
            const newsData = response.data;
            
            if (newsData.articles && newsData.articles.length > 0) {
                showNewsModal(newsData.articles);
            } else {
                // Fallback a búsqueda en Google
                fallbackNewsSearch();
            }
        } else {
            fallbackNewsSearch();
        }
    } catch (error) {
        console.error('Error obteniendo noticias:', error);
        fallbackNewsSearch();
    }
}

function showNewsModal(articles) {
    const modalHTML = `
        <div id="newsModal" class="modal" style="display: block;">
            <div class="modal-content" style="max-width: 800px;">
                <span class="close" onclick="closeNewsModal()">&times;</span>
                <h2>📰 Noticias Relevantes</h2>
                <div class="news-list">
                    ${articles.map(article => `
                        <div class="news-item">
                            <h3>${article.title}</h3>
                            <p class="news-summary">${article.summary}</p>
                            <div class="news-meta">
                                <span class="news-source">Fuente: ${article.source}</span>
                                <span class="news-relevance">Relevancia: ${article.relevance_score}/10</span>
                            </div>
                            <a href="${article.url}" target="_blank" class="news-link">
                                Leer noticia completa →
                            </a>
                        </div>
                    `).join('')}
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHTML);
}

function closeNewsModal() {
    const modal = document.getElementById('newsModal');
    if (modal) {
        modal.remove();
    }
}

function fallbackNewsSearch() {
    const factors = Array.from(document.querySelectorAll('.factor-item span:last-child'))
        .map(el => el.textContent);
    const keywords = extractKeywords(factors);
    const searchQuery = keywords.join(' ') + ' S&P 500';
    const url = `https://news.google.com/search?q=${encodeURIComponent(searchQuery)}`;
    window.open(url, '_blank');
}

// Añadir esta función al final del archivo
function hideLoading() {
    const loadingElement = document.getElementById('loadingIndicator');
    if (loadingElement) {
        loadingElement.style.display = 'none';
    }
}

function showLoading() {
    const loadingElement = document.getElementById('loadingIndicator');
    if (loadingElement) {
        loadingElement.style.display = 'block';
    }
}

// Añadir estas funciones de carga para predicciones
function showPredictionsLoading() {
    const elements = ['mainPrediction', 'sentimentTrend', 'similarPatterns'];
    const loadingHTML = `
        <div class="loading-small">
            <i class="fas fa-spinner fa-spin"></i>
            <p>Generando predicciones...</p>
        </div>
    `;
    
    elements.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.innerHTML = loadingHTML;
        }
    });
}

function showPredictionsError() {
    const errorHTML = '<p style="color: var(--danger-color); text-align: center;">Error cargando datos</p>';
    const elements = ['mainPrediction', 'sentimentTrend', 'similarPatterns'];
    
    elements.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.innerHTML = errorHTML;
        }
    });
}