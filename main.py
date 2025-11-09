"""
S&P 500 Intelligence Platform - Backend Completo
Sistema de análisis multi-agente con predicción y explicación contextual
"""

import os
import json
from pathlib import Path
import time
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import logging

# Data & Analysis
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Web & APIs
import requests
import aiohttp
from bs4 import BeautifulSoup
from news_service import NewsService

# NLP & Sentiment
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Technical Analysis
import ta

# FastAPI
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Cache
import redis
import pickle

# LLM
# from groq import Groq  
from openai import OpenAI  

# Configuración de logging

logger = logging.getLogger("cache")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
# ==========================================
# CONFIGURACIÓN Y VARIABLES DE ENTORNO
# ==========================================

class Config:
    """Configuración centralizada del sistema"""   
    # APIs Keys (usar variables de entorno en producción)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "6074c877b3164bf88f88b6e2d53d63a8")
    ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "LQO3KFQJTUIXKBP2")
    FRED_API_KEY = os.getenv("FRED_API_KEY", "468040c926e8e51ad2e32b5666377eef")
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "rJjpudhvNsqvozzwyLP9Zw")
    REDDIT_SECRET = os.getenv("REDDIT_SECRET", "ljou4Urpifj97vcYh2BFtVrSQfRXzw")

    # Redis
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    
    # Cache TTL (segundos)
    CACHE_TTL_SHORT = 300  # 5 minutos
    CACHE_TTL_MEDIUM = 3600  # 1 hora
    CACHE_TTL_LONG = 86400  # 24 horas
    
    # modelos disponibles en Groq
    LLM_MODEL = "llama-3.1-8b-instant"  # o "mixtral-8x7b-32768", "gemma-7b-it"
    
    # Rate limiting
    RATE_LIMIT_CALLS = 100
    RATE_LIMIT_PERIOD = 3600  # 1 hora

config = Config()

# ==========================================
# MODELOS DE DATOS (Pydantic)
# ==========================================

class UserLevel(str, Enum):
    """Niveles de conocimiento del usuario"""
    BEGINNER = "principiante"
    INTERMEDIATE = "intermedio"
    EXPERT = "experto"

class MarketDirection(str, Enum):
    """Dirección del mercado"""
    BULLISH = "alcista"
    BEARISH = "bajista"
    NEUTRAL = "neutral"

class NewsItem(BaseModel):
    """Modelo para noticias individuales"""
    title: str
    source: str
    url: str
    published_at: Optional[datetime] = None
    sentiment_score: float
    sentiment_label: str
    impact_score: int = Field(ge=0, le=10)
    summary: Optional[str] = None

class TechnicalIndicators(BaseModel):
    """Indicadores técnicos del mercado"""
    rsi: float
    macd: Dict[str, float]
    sma_50: float
    #sma_200: float
    volume: float
    vix: Optional[float]
    support_levels: List[float]
    resistance_levels: List[float]
    trend: str

class SocialSentiment(BaseModel):
    """Análisis de sentimiento en redes sociales"""
    platform: str
    mentions_count: int
    sentiment_average: float
    trending_topics: List[str]
    influencer_posts: List[Dict[str, Any]]
    volume_change_pct: float

class MarketPrediction(BaseModel):
    """Predicción del mercado"""
    direction: MarketDirection
    change_pct: float
    confidence: float
    price_current: float
    price_predicted: float
    range_upper: float
    range_lower: float
    horizon_days: int
    factors: Dict[str, float]
    explanation: str
    risks: List[str]
    similar_patterns: List[Dict[str, Any]]

class DateAnalysis(BaseModel):
    """Análisis completo de una fecha específica"""
    date: datetime
    sp500_close: float
    sp500_change_pct: float
    sp500_volume: float
    market_direction: MarketDirection
    news: List[NewsItem]
    technical: TechnicalIndicators
    social: List[SocialSentiment]
    causality_explanation: str
    confidence_score: float
    key_factors: List[str]

# ==========================================
# SISTEMA DE CACHE
# ==========================================

class CacheManager:
    """Gestor de caché mejorado con TTL inteligente"""

    def __init__(self):
        try:
            # 🔧 Hardcodeado según tu entorno local
            self.redis_client = redis.Redis(
                host="127.0.0.1",
                port=6379,
                db=6,
                password="123456.+",
                decode_responses=False
            )
            self.redis_client.ping()
            logger.info("✅ Redis conectado correctamente en localhost:6379 (DB 6)")
        except Exception as e:
            logger.warning(f"⚠️ Redis no disponible, usando caché en memoria: {e}")
            self.redis_client = None
            self.memory_cache = {}
            self.memory_cache_timestamps = {}

    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del caché con logging"""
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    logger.info(f"✅ Cache HIT: {key}")
                    return pickle.loads(value)
                else:
                    logger.info(f"❌ Cache MISS: {key}")
            else:
                # Verificar si está en memoria y no ha expirado
                if key in self.memory_cache:
                    timestamp = self.memory_cache_timestamps.get(key, 0)
                    if time.time() - timestamp < 3600:  # 1 hora por defecto
                        logger.info(f"✅ Memory Cache HIT: {key}")
                        return self.memory_cache[key]
                    else:
                        del self.memory_cache[key]
                        del self.memory_cache_timestamps[key]
                        logger.info(f"⏰ Cache EXPIRED: {key}")
                else:
                    logger.info(f"❌ Memory Cache MISS: {key}")
        except Exception as e:
            logger.error(f"Error obteniendo caché: {e}")
        return None

    def set(self, key: str, value: Any, ttl: int = 3600):
        """Guardar valor en caché con TTL específico"""
        try:
            if self.redis_client:
                self.redis_client.setex(key, ttl, pickle.dumps(value))
                logger.info(f"💾 Cache SET (Redis): {key} [TTL: {ttl}s]")
            else:
                self.memory_cache[key] = value
                self.memory_cache_timestamps[key] = time.time()
                logger.info(f"💾 Cache SET (Memory): {key} [TTL: {ttl}s]")
        except Exception as e:
            logger.error(f"Error guardando caché: {e}")

    def get_or_compute(self, key: str, compute_func, ttl: int = 3600, *args, **kwargs):
        """Obtener del caché o computar y guardar"""
        cached = self.get(key)
        if cached is not None:
            return cached

        logger.info(f"🔄 Computing: {key}")
        result = compute_func(*args, **kwargs)
        self.set(key, result, ttl)
        return result

    def invalidate(self, pattern: str = "*"):
        """Invalidar caché por patrón"""
        try:
            if self.redis_client:
                for key in self.redis_client.scan_iter(match=pattern):
                    self.redis_client.delete(key)
                logger.info(f"🗑️ Cache invalidated (Redis): {pattern}")
            else:
                keys_to_delete = [k for k in self.memory_cache.keys() if pattern in k or pattern == "*"]
                for key in keys_to_delete:
                    del self.memory_cache[key]
                    if key in self.memory_cache_timestamps:
                        del self.memory_cache_timestamps[key]
                logger.info(f"🗑️ Memory Cache invalidated: {pattern}")
        except Exception as e:
            logger.error(f"Error invalidando caché: {e}")

    def delete_pattern(self, pattern: str) -> int:
        """Elimina todas las keys que coincidan con el patrón"""
        try:
            if hasattr(self.redis, 'keys'):
                keys = self.redis.keys(pattern)
                if keys:
                    return self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error deleting pattern {pattern}: {e}")
            return 0

# ✅ Instancia global lista para usar
cache_manager = CacheManager()


class CacheTTL:
    """TTL específicos para diferentes tipos de datos"""
    MARKET_DATA_HISTORICAL = 864000000000 # 24 horas (datos históricos no cambian)
    MARKET_DATA_TODAY = 300000000000  # 5 minutos (datos del día actual)
    TECHNICAL_INDICATORS = 360000000000  # 1 hora
    NEWS_ANALYSIS = 8000000000  # 30 minutos
    SOCIAL_SENTIMENT = 90000000000  # 15 minutos
    PREDICTIONS_SHORT = 18000000  # 30 minutos (predicciones corto plazo)
    PREDICTIONS_LONG = 7200000  # 2 horas (predicciones largo plazo)
    SIMILAR_PATTERNS = 864000000  # 24 horas (patrones históricos)
    SENTIMENT_TREND = 360000000  # 1 hora
    FORECAST_DATA = 180000000  # 30 minutos


# ==========================================
# AGENTE 1: NEWS SCRAPER & AGGREGATOR
# ==========================================

class NewsScraperAgent:
    """Agente para recopilar noticias de múltiples fuentes"""
    
    def __init__(self):
        self.sources = {
            'newsapi': 'https://newsapi.org/v2/everything',
            'yahoo': 'https://finance.yahoo.com/news',
            'bloomberg': 'https://www.bloomberg.com/markets',
            'cnbc': 'https://www.cnbc.com/markets/',
            'reuters': 'https://www.reuters.com/markets/'
        }
    
    async def fetch_news_for_date(self, date: datetime, keywords: List[str] = None) -> List[Dict]:
        """Obtener noticias para una fecha específica"""
        
        if keywords is None:
            keywords = ["S&P 500", "stock market", "Federal Reserve", "inflation", "economy"]
        
        all_news = []
        
        # NewsAPI
        try:
            newsapi_results = await self._fetch_newsapi(date, keywords)
            all_news.extend(newsapi_results)
        except Exception as e:
            logger.error(f"Error en NewsAPI: {e}")
        
        # Web Scraping de otras fuentes
        for source_name, source_url in self.sources.items():
            if source_name != 'newsapi':
                try:
                    scraped = await self._scrape_source(source_url, date)
                    all_news.extend(scraped)
                except Exception as e:
                    logger.error(f"Error scraping {source_name}: {e}")
        
        return self._deduplicate_news(all_news)
    
    async def _fetch_newsapi(self, date: datetime, keywords: List[str]) -> List[Dict]:
        """Obtener noticias de NewsAPI"""
        news_list = []
        
        from_date = date.strftime('%Y-%m-%d')
        to_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        for keyword in keywords:
            params = {
                'q': keyword,
                'from': from_date,
                'to': to_date,
                'sortBy': 'relevancy',
                'apiKey': config.NEWS_API_KEY,
                'language': 'en'
            }
            
            try:
                response = requests.get(self.sources['newsapi'], params=params)
                if response.status_code == 200:
                    data = response.json()
                    for article in data.get('articles', [])[:10]:
                        news_list.append({
                            'title': article.get('title', ''),
                            'source': article.get('source', {}).get('name', 'Unknown'),
                            'url': article.get('url', ''),
                            'published_at': article.get('publishedAt', ''),
                            'description': article.get('description', ''),
                            'content': article.get('content', '')
                        })
            except Exception as e:
                logger.error(f"Error fetching from NewsAPI: {e}")
        
        return news_list
    
    async def _scrape_source(self, url: str, date: datetime) -> List[Dict]:
        """Scraping básico de una fuente de noticias"""
        news_list = []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Buscar artículos (simplificado, ajustar según cada fuente)
                        articles = soup.find_all(['article', 'div'], class_=lambda x: x and 'article' in x.lower())[:5]
                        
                        for article in articles:
                            title_elem = article.find(['h1', 'h2', 'h3', 'a'])
                            if title_elem:
                                news_list.append({
                                    'title': title_elem.get_text(strip=True),
                                    'source': url.split('/')[2],
                                    'url': url,
                                    'published_at': date.isoformat(),
                                    'description': '',
                                    'content': article.get_text(strip=True)[:500]
                                })
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
        
        return news_list
    
    def _deduplicate_news(self, news_list: List[Dict]) -> List[Dict]:
        """Eliminar noticias duplicadas basándose en títulos similares"""
        seen = set()
        unique_news = []
        
        for news in news_list:
            # Crear hash del título para deduplicación
            title_hash = hashlib.md5(news['title'].lower().encode()).hexdigest()[:8]
            if title_hash not in seen:
                seen.add(title_hash)
                unique_news.append(news)
        
        return unique_news

# ==========================================
# AGENTE 2: SENTIMENT ANALYZER
# ==========================================

class SentimentAnalyzerAgent:
    """Agente para análisis de sentimiento de noticias"""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
    def analyze_news_sentiment(self, news_list: List[Dict]) -> List[Dict]:
        """Analizar sentimiento de una lista de noticias"""
        analyzed_news = []
        
        for news in news_list:
            text_to_analyze = f"{news.get('title', '')} {news.get('description', '')} {news.get('content', '')[:500]}"
            
            # VADER sentiment
            vader_scores = self.vader.polarity_scores(text_to_analyze)
            
            # TextBlob sentiment
            try:
                blob = TextBlob(text_to_analyze)
                textblob_sentiment = blob.sentiment.polarity
            except:
                textblob_sentiment = 0
            
            # Combinar ambos análisis
            final_sentiment = (vader_scores['compound'] + textblob_sentiment) / 2
            
            # Clasificar sentimiento
            if final_sentiment >= 0.1:
                sentiment_label = "POSITIVO"
            elif final_sentiment <= -0.1:
                sentiment_label = "NEGATIVO"
            else:
                sentiment_label = "NEUTRAL"
            
            news['sentiment_score'] = round(final_sentiment, 3)
            news['sentiment_label'] = sentiment_label
            news['sentiment_details'] = {
                'vader': vader_scores,
                'textblob': textblob_sentiment
            }
            
            analyzed_news.append(news)
        
        return analyzed_news
    
    def calculate_aggregate_sentiment(self, analyzed_news: List[Dict]) -> Dict:
        """Calcular sentimiento agregado de todas las noticias"""
        if not analyzed_news:
            return {'average': 0, 'positive_ratio': 0, 'negative_ratio': 0, 'neutral_ratio': 0}
        
        sentiments = [news['sentiment_score'] for news in analyzed_news]
        labels = [news['sentiment_label'] for news in analyzed_news]
        
        total = len(labels)
        
        return {
            'average': round(np.mean(sentiments), 3),
            'std_dev': round(np.std(sentiments), 3),
            'positive_ratio': round(labels.count('POSITIVO') / total, 2),
            'negative_ratio': round(labels.count('NEGATIVO') / total, 2),
            'neutral_ratio': round(labels.count('NEUTRAL') / total, 2),
            'total_news': total
        }

# ==========================================
# AGENTE 3: IMPACT SCORER
# ==========================================

class ImpactScorerAgent:
    """Agente para calcular el impacto de las noticias en el mercado"""
    
    def __init__(self):
        self.high_impact_keywords = {
            'fed': 5, 'federal reserve': 5, 'interest rate': 5, 'inflation': 4,
            'gdp': 4, 'unemployment': 4, 'earnings': 3, 'revenue': 3,
            'war': 5, 'crisis': 5, 'recession': 5, 'crash': 5,
            'rally': 4, 'surge': 4, 'plunge': 4, 'soar': 4,
            'apple': 3, 'microsoft': 3, 'google': 3, 'amazon': 3,
            'oil': 3, 'gold': 2, 'bitcoin': 2, 'dollar': 3,
            'china': 4, 'europe': 3, 'brexit': 3, 'trade': 3
        }
    
    def score_news_impact(self, news_list: List[Dict]) -> List[Dict]:
        """Calcular score de impacto para cada noticia"""
        scored_news = []
        
        for news in news_list:
            text = f"{news.get('title', '')} {news.get('description', '')}".lower()
            
            # Calcular score basado en keywords
            impact_score = 1  # Score base
            keyword_matches = []
            
            for keyword, score in self.high_impact_keywords.items():
                if keyword in text:
                    impact_score = max(impact_score, score)
                    keyword_matches.append(keyword)
            
            # Ajustar por sentimiento extremo
            sentiment = abs(news.get('sentiment_score', 0))
            if sentiment > 0.5:
                impact_score = min(5, impact_score + 1)
            
            # Ajustar por fuente
            trusted_sources = ['bloomberg', 'reuters', 'cnbc', 'wsj', 'ft']
            if any(source in news.get('source', '').lower() for source in trusted_sources):
                impact_score = min(5, impact_score + 1)
            
            news['impact_score'] = impact_score
            news['impact_keywords'] = keyword_matches
            
            scored_news.append(news)
        
        # Ordenar por impacto
        return sorted(scored_news, key=lambda x: x['impact_score'], reverse=True)

# ==========================================
# AGENTE 4: TECHNICAL INDICATORS FETCHER
# ==========================================

class TechnicalIndicatorsFetcher:
    """Agente para obtener indicadores técnicos"""
    
    def fetch_technical_indicators(self, symbol: str, date: datetime) -> Dict:
        """Obtener indicadores técnicos desde datos locales - CORREGIDO"""
        
        cache_key = f"technical_{symbol}_{date.strftime('%Y%m%d')}"
        cached = cache_manager.get(cache_key)
        if cached:
            return cached
        
        try:
            end_date = date + timedelta(days=1)
            start_date = date - timedelta(days=200)
            
            df = data_manager.get_technical_data_range(symbol, start_date, end_date)
            
            if df.empty:
                logger.warning(f"No hay datos locales para {symbol}, intentando descargar...")
                data_manager.download_initial_data()
                df = data_manager.get_technical_data_range(symbol, start_date, end_date)
                
                if df.empty:
                    logger.error(f"No hay datos después de descargar para {symbol}")
                    return self._get_default_indicators()
            
            # Calcular indicadores técnicos
            df = self._calculate_technical_indicators(df)
            
            # Buscar la fecha específica
            date_utc = date.astimezone(timezone.utc) if date.tzinfo else date.replace(tzinfo=timezone.utc)
            date_str = date_utc.strftime('%Y-%m-%d')
            
            index_as_str = df.index.strftime('%Y-%m-%d')
            matches = np.where(index_as_str == date_str)[0]
            
            if len(matches) > 0:
                idx = matches[0]
                row = df.iloc[idx]
                logger.info(f"✅ Fecha técnica encontrada: {df.index[idx]}")
            else:
                previous_dates = df[df.index <= date_utc]
                if len(previous_dates) > 0:
                    closest_date = previous_dates.index[-1]
                    row = df.loc[closest_date]
                    logger.info(f"Usando fecha técnica más cercana: {closest_date.strftime('%Y-%m-%d')}")
                else:
                    logger.error("No hay fechas anteriores disponibles")
                    return self._get_default_indicators()
            
            # FUNCIÓN AUXILIAR para convertir valores seguros
            def safe_float(value, default=0.0):
                try:
                    val = float(value)
                    if np.isnan(val) or np.isinf(val):
                        return default
                    return val
                except:
                    return default
            
            indicators = {
                'rsi': safe_float(row.get('RSI'), 50.0),
                'macd': {
                    'value': safe_float(row.get('MACD'), 0.0),
                    'signal': safe_float(row.get('MACD_signal'), 0.0),
                    'histogram': safe_float(row.get('MACD_diff'), 0.0)
                },
                'sma_50': safe_float(row.get('SMA_50'), 0.0),
                'volume': safe_float(row.get('Volume'), 0.0),
                'vix': self._fetch_vix(date),
                'support_levels': self._calculate_support_resistance(df).get('supports', []),
                'resistance_levels': self._calculate_support_resistance(df).get('resistances', []),
                'trend': self._detect_trend(df, closest_date if 'closest_date' in locals() else date),
                'close': safe_float(row.get('Close'), 0.0),

                
                'bb_width': safe_float(row.get('BB_width'), 0.0),
                'volume_ratio': safe_float(row.get('Volume_Ratio'), 1.0)
                
            }
            
            # Guardar en caché
            cache_manager.set(cache_key, indicators, config.CACHE_TTL_MEDIUM)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculando indicadores técnicos: {e}")
            import traceback
            logger.error(f"🔍 DEBUG - Traceback: {traceback.format_exc()}")
            return self._get_default_indicators()

    def _detect_trend(self, df: pd.DataFrame, date: datetime) -> str:
        """Detectar tendencia del mercado - SIN MA200"""
        try:
            date_str = date.strftime('%Y-%m-%d')
            df_dates_str = df.index.strftime('%Y-%m-%d')

            # Buscar coincidencia exacta
            matching_indices = np.where(df_dates_str == date_str)[0]

            if len(matching_indices) > 0:
                date_index = matching_indices[0]
                logger.info(f"✅ Fecha exacta: {df.index[date_index]}")
            else:
                # Buscar fecha anterior más cercana
                date_target = pd.to_datetime(date_str)
                df_index_normalized = pd.to_datetime(df_dates_str)
                previous_mask = df_index_normalized <= date_target

                if previous_mask.sum() == 0:
                    logger.warning(f"Sin datos previos a {date_str}")
                    return "neutral"

                previous_indices = np.where(previous_mask)[0]
                date_index = previous_indices[-1]
                logger.info(f"📅 Fecha cercana: {df.index[date_index]}")

            current_row = df.iloc[date_index]

            # Función auxiliar para valores seguros
            def safe_float(value, default):
                try:
                    val = float(value)
                    if np.isnan(val) or np.isinf(val):
                        return default
                    return val
                except:
                    return default

            current_close = safe_float(current_row['Close'], 0)
            sma_50 = safe_float(current_row.get('SMA_50', current_close), current_close)
            # ❌ QUITAMOS sma_200

            # Validar datos
            if current_close == 0 or sma_50 == 0:
                logger.warning(f"Datos inválidos - Close: {current_close}, SMA50: {sma_50}")
                return "neutral"

            logger.info(f"📊 Análisis - Close: ${current_close:.2f}, SMA50: ${sma_50:.2f}")

            # ✅ NUEVA LÓGICA: Solo comparar Close vs SMA50
            if current_close > sma_50 * 1.02:  # 2% por encima
                return "fuerte_alcista"
            elif current_close > sma_50:
                return "alcista"
            elif current_close < sma_50 * 0.98:  # 2% por debajo
                return "fuerte_bajista"
            elif current_close < sma_50:
                return "bajista"
            else:
                return "neutral"

        except Exception as e:
            logger.error(f"❌ Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return "neutral"
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Calcular niveles de soporte y resistencia"""
        try:
            highs = df['High'].rolling(window=20, center=True).max()
            lows = df['Low'].rolling(window=20, center=True).min()

            recent_highs = df[df['High'] == highs]['High'].unique()[-3:]
            recent_lows = df[df['Low'] == lows]['Low'].unique()[-3:]

            # FILTRAR valores NaN e infinitos
            def clean_values(arr):
                return [float(x) for x in arr if not (np.isnan(x) or np.isinf(x))]

            return {
                'supports': sorted(clean_values(recent_lows)),
                'resistances': sorted(clean_values(recent_highs), reverse=True)
            }
        except:
            return {'supports': [], 'resistances': []}
    
    def _fetch_vix(self, date: datetime) -> Optional[float]:
        """Obtener valor del VIX para una fecha"""
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(start=date, end=date + timedelta(days=1))
            if not hist.empty:
                return float(hist.iloc[0]['Close'])
        except:
            pass
        return None
    
    def _get_default_indicators(self) -> Dict:
        """Indicadores por defecto cuando no hay datos"""
        return {
            'rsi': 50.0,
            'macd': {'value': 0.0, 'signal': 0.0, 'histogram': 0.0},
            'sma_50': 0.0,
            #'sma_200': 0.0,
            'volume': 0.0,
            'vix': None,
            'support_levels': [],
            'resistance_levels': [],
            'trend': 'neutral',
            'close': 0.0
        }
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular todos los indicadores técnicos para el DataFrame"""
        try:
            logger.info(f"🔧 Calculando indicadores técnicos para {len(df)} registros")

            # Hacer una copia para no modificar el original
            df_indicators = df.copy()

            # 1. RSI (Relative Strength Index)
            logger.info("📊 Calculando RSI...")
            try:
                rsi_indicator = ta.momentum.RSIIndicator(df_indicators['Close'], window=14)
                df_indicators['RSI'] = rsi_indicator.rsi()
                logger.info(f"✅ RSI calculado - valores: {df_indicators['RSI'].notna().sum()}")
            except Exception as e:
                logger.error(f"Error calculando RSI: {e}")
                df_indicators['RSI'] = 50.0  # Valor neutral por defecto

            # 2. MACD (Moving Average Convergence Divergence)
            logger.info("📈 Calculando MACD...")
            try:
                macd_indicator = ta.trend.MACD(df_indicators['Close'])
                df_indicators['MACD'] = macd_indicator.macd()
                df_indicators['MACD_signal'] = macd_indicator.macd_signal()
                df_indicators['MACD_diff'] = macd_indicator.macd_diff()
                logger.info(f"✅ MACD calculado")
            except Exception as e:
                logger.error(f"Error calculando MACD: {e}")
                df_indicators['MACD'] = 0.0
                df_indicators['MACD_signal'] = 0.0
                df_indicators['MACD_diff'] = 0.0

            # 3. Medias Móviles Simples
            logger.info("📉 Calculando medias móviles...")
            try:
                # SMA 20
                sma_20 = ta.trend.SMAIndicator(df_indicators['Close'], window=20)
                df_indicators['SMA_20'] = sma_20.sma_indicator()

                # SMA 50
                sma_50 = ta.trend.SMAIndicator(df_indicators['Close'], window=50)
                df_indicators['SMA_50'] = sma_50.sma_indicator()

                # SMA 200
                #sma_200 = ta.trend.SMAIndicator(df_indicators['Close'], window=200)
                #df_indicators['SMA_200'] = sma_200.sma_indicator()

                logger.info(f"✅ Medias móviles calculadas")
            except Exception as e:
                logger.error(f"Error calculando medias móviles: {e}")
                df_indicators['SMA_20'] = df_indicators['Close']
                df_indicators['SMA_50'] = df_indicators['Close']
                #df_indicators['SMA_200'] = df_indicators['Close']

            # 4. Bollinger Bands
            logger.info("📊 Calculando Bollinger Bands...")
            try:
                bb_indicator = ta.volatility.BollingerBands(df_indicators['Close'], window=20, window_dev=2)
                df_indicators['BB_upper'] = bb_indicator.bollinger_hband()
                df_indicators['BB_middle'] = bb_indicator.bollinger_mavg()
                df_indicators['BB_lower'] = bb_indicator.bollinger_lband()
                df_indicators['BB_width'] = bb_indicator.bollinger_wband()
                logger.info(f"✅ Bollinger Bands calculadas")
            except Exception as e:
                logger.error(f"Error calculando Bollinger Bands: {e}")
                df_indicators['BB_upper'] = df_indicators['Close']
                df_indicators['BB_middle'] = df_indicators['Close']
                df_indicators['BB_lower'] = df_indicators['Close']
                df_indicators['BB_width'] = 0.0

            # 5. Estocástico
            logger.info("📈 Calculando Estocástico...")
            try:
                stoch_indicator = ta.momentum.StochasticOscillator(
                    high=df_indicators['High'],
                    low=df_indicators['Low'], 
                    close=df_indicators['Close'],
                    window=14,
                    smooth_window=3
                )
                df_indicators['Stoch_K'] = stoch_indicator.stoch()
                df_indicators['Stoch_D'] = stoch_indicator.stoch_signal()
                logger.info(f"✅ Estocástico calculado")
            except Exception as e:
                logger.error(f"Error calculando Estocástico: {e}")
                df_indicators['Stoch_K'] = 50.0
                df_indicators['Stoch_D'] = 50.0

            # 6. Volumen Promedio
            logger.info("💰 Calculando indicadores de volumen...")
            try:
                df_indicators['Volume_SMA_20'] = df_indicators['Volume'].rolling(window=20).mean()
                df_indicators['Volume_Ratio'] = df_indicators['Volume'] / df_indicators['Volume_SMA_20']
                logger.info(f"✅ Indicadores de volumen calculados")
            except Exception as e:
                logger.error(f"Error calculando indicadores de volumen: {e}")
                df_indicators['Volume_SMA_20'] = df_indicators['Volume']
                df_indicators['Volume_Ratio'] = 1.0

            # 7. ATR (Average True Range) - Volatilidad
            logger.info("⚡ Calculando ATR...")
            try:
                atr_indicator = ta.volatility.AverageTrueRange(
                    high=df_indicators['High'],
                    low=df_indicators['Low'],
                    close=df_indicators['Close'],
                    window=14
                )
                df_indicators['ATR'] = atr_indicator.average_true_range()
                logger.info(f"✅ ATR calculado")
            except Exception as e:
                logger.error(f"Error calculando ATR: {e}")
                df_indicators['ATR'] = 0.0

            # 8. Retornos y volatilidad
            logger.info("📊 Calculando retornos y volatilidad...")
            try:
                df_indicators['Daily_Return'] = df_indicators['Close'].pct_change()
                df_indicators['Volatility_20d'] = df_indicators['Daily_Return'].rolling(window=20).std()
                logger.info(f"✅ Retornos y volatilidad calculados")
            except Exception as e:
                logger.error(f"Error calculando retornos: {e}")
                df_indicators['Daily_Return'] = 0.0
                df_indicators['Volatility_20d'] = 0.0

            # 9. Ichimoku Cloud (opcional - para análisis avanzado)
            logger.info("☁️ Calculando Ichimoku...")
            try:
                ichimoku = ta.trend.IchimokuIndicator(
                    high=df_indicators['High'],
                    low=df_indicators['Low']
                )
                df_indicators['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
                df_indicators['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
                df_indicators['Ichimoku_A'] = ichimoku.ichimoku_a()
                df_indicators['Ichimoku_B'] = ichimoku.ichimoku_b()
                logger.info(f"✅ Ichimoku calculado")
            except Exception as e:
                logger.error(f"Error calculando Ichimoku: {e}")
                # No es crítico, continuar sin Ichimoku

            # 10. Pivot Points (para soporte/resistencia)
            logger.info("🎯 Calculando Pivot Points...")
            try:
                # Pivot Point simple
                df_indicators['Pivot'] = (df_indicators['High'] + df_indicators['Low'] + df_indicators['Close']) / 3
                df_indicators['Support_1'] = (2 * df_indicators['Pivot']) - df_indicators['High']
                df_indicators['Resistance_1'] = (2 * df_indicators['Pivot']) - df_indicators['Low']
                logger.info(f"✅ Pivot Points calculados")
            except Exception as e:
                logger.error(f"Error calculando Pivot Points: {e}")
                df_indicators['Pivot'] = df_indicators['Close']
                df_indicators['Support_1'] = df_indicators['Close']
                df_indicators['Resistance_1'] = df_indicators['Close']

            # Limpiar NaN values al inicio (debido a ventanas de cálculo)
            initial_nans = df_indicators.isna().sum().sum()
            df_indicators = df_indicators.bfill().ffill()
            final_nans = df_indicators.isna().sum().sum()

            logger.info(f"🧹 NaN values limpiados: {initial_nans} -> {final_nans}")
            logger.info(f"🎯 Indicadores técnicos calculados exitosamente para {len(df_indicators)} registros")

            # DEBUG: Mostrar algunas estadísticas
            if len(df_indicators) > 0:
                latest = df_indicators.iloc[-1]
                logger.info(f"📊 Últimos valores - RSI: {latest.get('RSI', 'N/A'):.1f}, "
                           f"MACD: {latest.get('MACD', 'N/A'):.3f}, "
                           f"SMA_50: {latest.get('SMA_50', 'N/A'):.2f}")

            return df_indicators

        except Exception as e:
            logger.error(f"❌ Error crítico calculando indicadores técnicos: {e}")
            import traceback
            logger.error(f"🔍 DEBUG - Traceback: {traceback.format_exc()}")

            # Retornar DataFrame original si hay error crítico
            return df.copy()

# ==========================================
# AGENTE 5: SOCIAL MEDIA ANALYZER
# ==========================================

class SocialMediaAnalyzer:
    """Agente para análisis de redes sociales"""
    
    def __init__(self):
        self.platforms = ['reddit', 'twitter', 'stocktwits']
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    async def analyze_social_sentiment(self, date: datetime, symbol: str = "SPY") -> List[Dict]:
        """Analizar sentimiento en redes sociales"""
        results = []
        
        # Reddit (simulado - en producción usar PRAW)
        reddit_data = await self._analyze_reddit(date, symbol)
        if reddit_data:
            results.append(reddit_data)
        
        # Twitter/X (simulado - en producción usar Tweepy)
        twitter_data = await self._analyze_twitter(date, symbol)
        if twitter_data:
            results.append(twitter_data)
        
        # StockTwits (simulado)
        stocktwits_data = await self._analyze_stocktwits(date, symbol)
        if stocktwits_data:
            results.append(stocktwits_data)
        
        return results
    
    async def _analyze_reddit(self, date: datetime, symbol: str) -> Dict:
        """Analizar sentimiento en Reddit (simulado)"""
        try:
            # En producción, usar PRAW para obtener posts reales
            # Aquí simulamos datos
            
            sample_posts = [
                "SPY to the moon! 🚀",
                "Market crash incoming, sell everything!",
                "Steady growth expected, holding my positions",
                "Fed news is bullish AF",
                "Uncertain times, staying cash"
            ]
            
            sentiments = []
            for post in sample_posts:
                score = self.sentiment_analyzer.polarity_scores(post)
                sentiments.append(score['compound'])
            
            return {
                'platform': 'reddit',
                'mentions_count': len(sample_posts) * 10,  # Simulado
                'sentiment_average': round(np.mean(sentiments), 3),
                'trending_topics': ['Fed', 'earnings', 'inflation'],
                'influencer_posts': [
                    {
                        'author': 'u/DeepFuckingValue',
                        'text': sample_posts[0],
                        'upvotes': 5000,
                        'sentiment': sentiments[0]
                    }
                ],
                'volume_change_pct': 25.5  # Simulado
            }
        except Exception as e:
            logger.error(f"Error analizando Reddit: {e}")
            return None
    
    async def _analyze_twitter(self, date: datetime, symbol: str) -> Dict:
        """Analizar sentimiento en Twitter (simulado)"""
        try:
            # En producción, usar Tweepy API
            # Aquí simulamos datos
            
            sample_tweets = [
                "$SPY breakout confirmed! #bullish",
                "Market overvalued, correction coming #bearish",
                "Holding $SPY for long term gains",
            ]
            
            sentiments = []
            for tweet in sample_tweets:
                score = self.sentiment_analyzer.polarity_scores(tweet)
                sentiments.append(score['compound'])
            
            return {
                'platform': 'twitter',
                'mentions_count': len(sample_tweets) * 100,
                'sentiment_average': round(np.mean(sentiments), 3),
                'trending_topics': ['#SPY', '#StockMarket', '#Fed'],
                'influencer_posts': [
                    {
                        'author': '@jimcramer',
                        'text': sample_tweets[0],
                        'retweets': 1000,
                        'sentiment': sentiments[0]
                    }
                ],
                'volume_change_pct': 15.0
            }
        except Exception as e:
            logger.error(f"Error analizando Twitter: {e}")
            return None
    
    async def _analyze_stocktwits(self, date: datetime, symbol: str) -> Dict:
        """Analizar sentimiento en StockTwits (simulado)"""
        try:
            return {
                'platform': 'stocktwits',
                'mentions_count': 500,
                'sentiment_average': 0.45,
                'trending_topics': ['$SPY', 'bull', 'moon'],
                'influencer_posts': [],
                'volume_change_pct': 10.0
            }
        except:
            return None

# ==========================================
# AGENTE 6: LLM SYNTHESIZER
# ==========================================
# ==========================================
# AGENTE 6: LLM SYNTHESIZER
# ==========================================
class LLMSynthesizer:
    """Agente sintetizador usando LLM con Groq vía OpenAI client"""
    
    def __init__(self):
        # Configuración del cliente OpenAI para Groq
        self.client = OpenAI(
            api_key=config.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = config.LLM_MODEL
    
    def generate_causality_explanation(
        self,
        date: datetime,
        market_data: Dict,
        news: List[Dict],
        technical: Dict,
        social: List[Dict],
        user_level: UserLevel = UserLevel.INTERMEDIATE
    ) -> str:
        """Generar explicación de causalidad usando LLM"""
        
        try:
            # Preparar contexto para el LLM
            news_summary = self._summarize_news(news[:5])  # Top 5 noticias
            
            prompt = self._build_causality_prompt(
                date, market_data, news_summary, technical, social, user_level
            )
            
            # Llamar al LLM usando el nuevo cliente
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "Eres un analista financiero experto que explica movimientos del mercado de forma clara, precisa y objetiva."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=500,
                top_p=0.9
            )
            
            explanation = response.choices[0].message.content
            logger.info(f"✅ Explicación de causalidad generada para {date.strftime('%Y-%m-%d')}")
            return explanation
            
        except Exception as e:
            logger.error(f"❌ Error general generando explicación con LLM: {e}")
            return self._generate_fallback_explanation(market_data, news, technical)
    
    def generate_prediction_explanation(
        self,
        prediction: Dict,
        factors: Dict,
        patterns: List[Dict],
        user_level: UserLevel = UserLevel.INTERMEDIATE
    ) -> str:
        """Generar explicación para predicción"""
        
        try:
            prompt = self._build_prediction_prompt(prediction, factors, patterns, user_level)
            
            logger.info("🔍 DEBUG - Enviando prompt de predicción a LLM")
            
            # Llamar al LLM usando el nuevo cliente
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "Eres un analista financiero que explica predicciones de mercado de forma clara, honesta y basada en datos."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=400,
                top_p=0.9
            )
            
            explanation = response.choices[0].message.content
            logger.info(f"✅ Explicación de predicción generada para {prediction.get('horizon_days', 7)} días")
            return explanation
            
        except Exception as e:
            logger.error(f"❌ Error general generando explicación de predicción: {e}")
            return self._generate_fallback_prediction_explanation(prediction, factors)
    
    def generate_news_summary(
        self,
        news_list: List[Dict],
        user_level: UserLevel = UserLevel.INTERMEDIATE
    ) -> str:
        """Generar resumen ejecutivo de noticias"""
        
        try:
            news_text = self._summarize_news_detailed(news_list[:10])
            
            level_instruction = {
                UserLevel.BEGINNER: "Explica en términos simples, evitando jerga financiera compleja.",
                UserLevel.INTERMEDIATE: "Incluye algunos términos técnicos con explicaciones breves.",
                UserLevel.EXPERT: "Proporciona análisis detallado con terminología profesional."
            }
            
            prompt = f"""
            Resume las siguientes noticias financieras y su impacto potencial en el mercado:
            
            {news_text}
            
            INSTRUCCIONES:
            - {level_instruction[user_level]}
            - Identifica los temas principales
            - Destaca las noticias con mayor impacto potencial
            - Menciona tendencias de sentimiento
            - Sé conciso (máximo 200 palabras)
            - Formato: párrafos claros y estructurados
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un editor financiero que resume noticias de forma clara y objetiva."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.5,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error general generando resumen de noticias: {e}")
            return "No se pudo generar el resumen de noticias en este momento."
    
    def _summarize_news(self, news_list: List[Dict]) -> str:
        """Resumir noticias principales para prompts"""
        if not news_list:
            return "No hay noticias significativas para esta fecha."
        
        summary = []
        for i, news in enumerate(news_list, 1):
            sentiment_emoji = "📈" if news.get('sentiment_score', 0) > 0.1 else "📉" if news.get('sentiment_score', 0) < -0.1 else "➡️"
            impact_stars = '⭐' * news.get('impact_score', 1)
            
            summary.append(
                f"{i}. {sentiment_emoji} {news['title']} "
                f"(Fuente: {news['source']}, "
                f"Sentimiento: {news.get('sentiment_label', 'NEUTRAL')}, "
                f"Impacto: {impact_stars})"
            )
        return "\n".join(summary)
    
    def _summarize_news_detailed(self, news_list: List[Dict]) -> str:
        """Resumen más detallado para análisis de noticias"""
        if not news_list:
            return "No hay noticias disponibles."
        
        detailed_summary = []
        for i, news in enumerate(news_list, 1):
            sentiment_score = news.get('sentiment_score', 0)
            impact_score = news.get('impact_score', 1)
            
            detailed_summary.append(
                f"NOTICIA {i}:\n"
                f"Título: {news['title']}\n"
                f"Fuente: {news['source']}\n"
                f"Score Sentimiento: {sentiment_score:.3f}\n"
                f"Impacto: {impact_score}/5\n"
                f"---"
            )
        return "\n".join(detailed_summary)
    
    def _build_causality_prompt(
        self, 
        date: datetime, 
        market_data: Dict,
        news_summary: str, 
        technical: Dict, 
        social: List[Dict],
        user_level: UserLevel
    ) -> str:
        """Construir prompt para explicación de causalidad"""
        
        level_instruction = {
            UserLevel.BEGINNER: """
            - Explica como si fuera para alguien nuevo en inversiones
            - Evita términos técnicos complejos
            - Usa analogías simples
            - Enfócate en conceptos fundamentales
            """,
            UserLevel.INTERMEDIATE: """
            - Usa algunos términos técnicos pero explícalos brevemente
            - Proporciona análisis balanceado
            - Incluye datos clave pero sin sobrecargar
            """,
            UserLevel.EXPERT: """
            - Proporciona análisis técnico detallado
            - Usa terminología profesional
            - Incluye métricas específicas
            - Analiza múltiples perspectivas
            """
        }
        
        # Calcular sentimiento social promedio
        social_sentiment = np.mean([s.get('sentiment_average', 0) for s in social]) if social else 0
        
        # Información de tendencia técnica
        trend_analysis = self._analyze_trend_strength(technical)
        
        return f"""
        FECHA DE ANÁLISIS: {date.strftime('%Y-%m-%d')}
        
        MOVIMIENTO DEL MERCADO:
        - S&P 500: {market_data.get('change_pct', 0):+.2f}%
        - Precio de cierre: ${market_data.get('close', 0):.2f}
        - Volumen: {market_data.get('volume', 0):,.0f}
        
        ANÁLISIS TÉCNICO:
        - RSI: {technical.get('rsi', 50):.1f} ({self._get_rsi_interpretation(technical.get('rsi', 50))})
        - Tendencia: {technical.get('trend', 'neutral')} ({trend_analysis})
        - Media 50 días: ${technical.get('sma_50', 0):.2f}
        
        NOTICIAS DESTACADAS:
        {news_summary}
        
        SENTIMIENTO SOCIAL: {social_sentiment:.3f} ({self._get_sentiment_interpretation(social_sentiment)})
        
        TAREA:
        Explica por qué el S&P 500 se movió como lo hizo en esta fecha específica.
        
        INSTRUCCIONES ESPECÍFICAS:
        {level_instruction[user_level]}
        
        ESTRUCTURA DE RESPUESTA:
        1. Identifica las 2-3 causas principales del movimiento
        2. Explica cómo interactúan estos factores
        3. Menciona cualquier factor contradictorio
        4. Proporciona contexto sobre la importancia del movimiento
        5. Sé objetivo y basado en datos
        
        LONGITUD: Máximo 150 palabras, claro y conciso.
        """
    
    def _build_prediction_prompt(
        self, 
        prediction: Dict, 
        factors: Dict, 
        patterns: List[Dict], 
        user_level: UserLevel
    ) -> str:
        """Construir prompt para explicación de predicción"""
        
        level_instruction = {
            UserLevel.BEGINNER: """
            - Explica en lenguaje simple y accesible
            - Evita jerga financiera compleja
            - Usa ejemplos concretos cuando sea posible
            - Destaca los puntos clave sin tecnicismos
            """,
            UserLevel.INTERMEDIATE: """
            - Balancea claridad con profundidad técnica
            - Explica conceptos técnicos brevemente
            - Proporciona razones fundamentales
            - Incluye consideraciones de riesgo
            """,
            UserLevel.EXPERT: """
            - Proporciona análisis técnico completo
            - Incluye métricas y ratios relevantes
            - Analiza supuestos y limitaciones
            - Considera múltiples escenarios
            """
        }
        
        # Formatear patrones históricos con manejo de errores
        patterns_text = "\n".join([
            f"- {p.get('date', 'N/A')}: {p.get('change', 0):+.1f}% (similitud: {p.get('similarity', 0):.0%})"
            for p in patterns[:3] if p
        ]) if patterns else "No se encontraron patrones significativamente similares."
        
        return f"""
        PREDICCIÓN ACTUAL:
        - Dirección: {prediction.get('direction', 'neutral')}
        - Cambio esperado: {prediction.get('change_pct', 0):+.1f}%
        - Horizonte: {prediction.get('horizon_days', 7)} días
        - Confianza: {prediction.get('confidence', 0):.0f}%
        
        FACTORES DE INFLUENCIA:
        - Análisis Técnico: {factors.get('technical', 0):.0%}
        - Fundamentos: {factors.get('fundamental', 0):.0%}
        - Patrones Históricos: {factors.get('historical', 0):.0%}
        - Sentimiento Social: {factors.get('social', 0):.0%}
        
        PATRONES HISTÓRICOS SIMILARES:
        {patterns_text}
        
        TAREA:
        Explica esta predicción de manera que sea útil para tomar decisiones informadas.
        
        INSTRUCCIONES ESPECÍFICAS:
        {level_instruction[user_level]}
        
        ESTRUCTURA DE RESPUESTA:
        1. Explica los factores principales detrás de la predicción
        2. Menciona el nivel de confianza y por qué
        3. Destaca 1-2 riesgos principales
        4. Proporciona contexto sobre los patrones históricos
        5. Sé honesto sobre limitaciones e incertidumbre
        
        LONGITUD: Máximo 120 palabras, directo al punto.
        """
    
    def _analyze_trend_strength(self, technical: Dict) -> str:
        """Analizar fuerza de la tendencia"""
        trend = technical.get('trend', 'neutral')
        
        if trend == 'fuerte_alcista':
            return "Tendencia alcista fuerte, momentum positivo"
        elif trend == 'alcista':
            return "Tendencia alcista moderada"
        elif trend == 'fuerte_bajista':
            return "Tendencia bajista fuerte, momentum negativo"
        elif trend == 'bajista':
            return "Tendencia bajista moderada"
        else:
            return "Mercado lateral, sin tendencia clara"
    
    def _get_rsi_interpretation(self, rsi: float) -> str:
        """Interpretar niveles de RSI"""
        if rsi > 70:
            return "sobrecomprado"
        elif rsi < 30:
            return "sobrevendido"
        else:
            return "neutral"
    
    def _get_sentiment_interpretation(self, sentiment: float) -> str:
        """Interpretar niveles de sentimiento"""
        if sentiment > 0.3:
            return "muy positivo"
        elif sentiment > 0.1:
            return "positivo"
        elif sentiment < -0.3:
            return "muy negativo"
        elif sentiment < -0.1:
            return "negativo"
        else:
            return "neutral"
    
    def _generate_fallback_explanation(self, market_data: Dict, news: List[Dict], technical: Dict) -> str:
        """Explicación de respaldo si falla el LLM"""
        change = market_data.get('change_pct', 0)
        
        if abs(change) < 0.1:
            return "El mercado se mantuvo estable con cambios mínimos durante esta sesión. No hubo catalizadores significativos que impulsaran movimientos importantes."
        
        direction = "subió" if change > 0 else "cayó"
        magnitude = "ligeramente" if abs(change) < 0.5 else "moderadamente" if abs(change) < 1.0 else "significativamente"
        
        # Análisis técnico de respaldo
        rsi = technical.get('rsi', 50)
        if rsi > 70:
            rsi_analysis = "en zona de sobrecompra"
        elif rsi < 30:
            rsi_analysis = "en zona de sobreventa"
        else:
            rsi_analysis = "en territorio neutral"
        
        # Noticia principal si existe
        news_impact = ""
        if news and news[0].get('impact_score', 0) >= 3:
            top_news = news[0]
            sentiment = "positiva" if top_news.get('sentiment_score', 0) > 0 else "negativa"
            news_impact = f" La noticia '{top_news['title'][:60]}...' tuvo un impacto {sentiment}. "
        
        return f"El S&P 500 {direction} {magnitude} un {abs(change):.1f}%, influenciado por condiciones técnicas {rsi_analysis}.{news_impact} El volumen fue de {market_data.get('volume', 0):,.0f} acciones."
    
    def _generate_fallback_prediction_explanation(self, prediction: Dict, factors: Dict) -> str:
        """Explicación de respaldo para predicciones"""
        direction = prediction.get('direction', 'neutral')
        change = prediction.get('change_pct', 0)
        confidence = prediction.get('confidence', 0)
        
        base_explanation = f"Se espera que el S&P 500 tenga un movimiento {direction} del {abs(change):.1f}% con {confidence:.0f}% de confianza. "
        
        # Añadir factores principales
        main_factors = []
        if factors.get('technical', 0) > 0.3:
            main_factors.append("indicadores técnicos")
        if factors.get('fundamental', 0) > 0.3:
            main_factors.append("análisis fundamental")
        if factors.get('historical', 0) > 0.3:
            main_factors.append("patrones históricos")
        
        if main_factors:
            factors_text = " y ".join(main_factors)
            base_explanation += f"Esta predicción se basa principalmente en {factors_text}. "
        
        # Añadir nota de precaución
        if confidence < 70:
            base_explanation += "La confianza moderada sugiere considerar esta predicción con cautela y diversificar riesgos."
        else:
            base_explanation += "La alta confianza respalda esta perspectiva, pero siempre existe incertidumbre en los mercados."
        
        return base_explanation
    
    def test_connection(self) -> bool:
        """Probar conexión con Groq API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Responde solo con 'OK'"},
                    {"role": "user", "content": "Test de conexión"}
                ],
                max_tokens=5
            )
            return response.choices[0].message.content.strip() == "OK"
        except Exception as e:
            logger.error(f"Error en test de conexión Groq: {e}")
            return False
            
    def generate_contextual_forecast(
        self,
        historical_context: str,
        last_price: float,
        horizon_days: int = 30
    ) -> List[Dict]:
        """
        Generar un forecast de precios basado en contexto histórico usando LLM.
        (Esta versión usa el historial de análisis de IA, como solicitaste).
        """
        
        # --- PROMPT ACTUALIZADO (EL QUE USA EL ANÁLISIS DE IA) ---
        prompt = f"""
        Eres un analista financiero cuantitativo (Quant) experto. Tu tarea es generar un pronóstico de precios realista, día por día, para el S&P 500 (SPY) durante los próximos {horizon_days} días.

        DATOS DE ENTRIDA:
        1. Último precio de cierre real: ${last_price:.2f}
        2. Análisis histórico (Movimiento, Datos Técnicos y Explicación de IA) de los últimos días de mercado:
        ---
        {historical_context}
        ---

        INSTRUCCIONES:
        1. Analiza profundamente el contexto histórico: fíjate en el momentum (Movimiento), la tendencia (Datos Técnicos) y las *causas* (Explicación de IA).
        2. Tu pronóstico debe ser una *continuación lógica* de esta narrativa. Si la IA de ayer dijo "El mercado cayó por miedo a la inflación", y hoy no hay noticias, esa tendencia debería continuar. Si dijo "El RSI indica sobreventa", quizás deba rebotar.
        3. Genera un pronóstico para {horizon_days} días, comenzando desde el día siguiente.
        4. El pronóstico DEBE mostrar variabilidad realista (días de subida y días de bajada). No dibujes una línea recta.
        5. Simula la volatilidad. Los cambios diarios (change_pct) deben ser realistas (generalmente entre -2.0% y +2.0%).
        6. La fecha (date) debe continuar secuencialmente desde mañana.
        7. El 'predicted_price' debe ser un float.
        8. El 'change_pct' debe ser el cambio % solo para ese día.

        FORMATO DE SALIDA (OBLIGATORIO):
        Responde ÚNICAMENTE con un array JSON válido. No incluyas explicaciones, texto introductorio, ni la palabra 'json'.
        Tu respuesta debe empezar con '[' y terminar con ']'.

        Ejemplo de formato de salida:
        [
          {{"day": 1, "date": "{ (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d') }", "predicted_price": {last_price * 1.001:.2f}, "change_pct": 0.001}},
          {{"day": 2, "date": "{ (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d') }", "predicted_price": {last_price * 0.999:.2f}, "change_pct": -0.002}},
          ...
        ]
        """
        # --- FIN DEL PROMPT ---
        
        try:
            logger.info(f"🔮 Generando forecast contextual con LLM para {horizon_days} días...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Eres un analista financiero Quant. Respondes solo con un array JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7, # Permitir variabilidad
                max_tokens=3000, # Suficiente para 30-60 días de JSON
            )

            raw_json_string = response.choices[0].message.content
            logger.info(f"Raw LLM response (forecast): {raw_json_string[:150]}...") # Debug
            
            # Limpiar el string por si el LLM añade markdown
            if "```json" in raw_json_string:
                raw_json_string = raw_json_string.split("```json")[1].split("```")[0]
            
            raw_json_string = raw_json_string.strip()

            import json
            predictions_list = json.loads(raw_json_string)
            
            if not isinstance(predictions_list, list):
                 raise ValueError("LLM output was not a JSON list.")

            logger.info(f"✅ LLM Forecast generado y parseado. {len(predictions_list)} días.")
            return predictions_list

        except Exception as e:
            logger.error(f"❌ Error generando/parseando forecast de LLM: {e}")
            import traceback
            logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            return [] # Devolver lista vacía en caso de error
# ==========================================
# DATA MANAGER CON ARCHIVOS CSV LOCALES
# ==========================================

class DataManager:
    """Gestor de datos locales con archivos CSV"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Archivos de datos
        self.sp500_file = self.data_dir / "sp500_data.csv"
        self.spy_file = self.data_dir / "spy_data.csv"
        self.vix_file = self.data_dir / "vix_data.csv"
        
        # DataFrames en memoria
        self.sp500_df = None
        self.spy_df = None
        self.vix_df = None
        
        self._load_data()
    def _safe_float(self, value, default=0.0):
        """Convertir valor a float seguro para JSON"""
        try:
            val = float(value)
            # Verificar si es NaN o infinito
            if np.isnan(val) or np.isinf(val):
                return default
            return val
        except (ValueError, TypeError):
            return default

    def _load_data(self):
        """Cargar datos desde archivos CSV - CORREGIDO"""
        try:
            # Cargar S&P 500
            if self.sp500_file.exists():
                self.sp500_df = pd.read_csv(self.sp500_file, index_col='Date', parse_dates=True)
                # CONVERTIR a DatetimeIndex con UTC
                self.sp500_df.index = pd.to_datetime(self.sp500_df.index, utc=True)
                logger.info(f"✅ S&P 500 data loaded: {len(self.sp500_df)} records")
            else:
                logger.warning("⚠️ S&P 500 CSV no encontrado")
                self.sp500_df = pd.DataFrame()

            # Cargar SPY
            if self.spy_file.exists():
                self.spy_df = pd.read_csv(self.spy_file, index_col='Date', parse_dates=True)
                # CONVERTIR a DatetimeIndex con UTC
                self.spy_df.index = pd.to_datetime(self.spy_df.index, utc=True)
                logger.info(f"✅ SPY data loaded: {len(self.spy_df)} records")
            else:
                logger.warning("⚠️ SPY CSV no encontrado")
                self.spy_df = pd.DataFrame()

            # Cargar VIX
            if self.vix_file.exists():
                self.vix_df = pd.read_csv(self.vix_file, index_col='Date', parse_dates=True)
                # CONVERTIR a DatetimeIndex con UTC
                self.vix_df.index = pd.to_datetime(self.vix_df.index, utc=True)
                logger.info(f"✅ VIX data loaded: {len(self.vix_df)} records")
            else:
                logger.warning("⚠️ VIX CSV no encontrado")
                self.vix_df = pd.DataFrame()

        except Exception as e:
            logger.error(f"Error cargando datos locales: {e}")
    
    def get_market_data(self, date: datetime, symbol: str = "SPY") -> Dict:
        """Obtener datos de mercado para una fecha específica - CORREGIDO"""
        try:
            # DEBUG: Tipo de fecha
            logger.info(f"🔍 DEBUG - Fecha recibida: {date}, tipo: {type(date)}, tzinfo: {date.tzinfo}")
            
            # Asegurar que la fecha sea naive (sin timezone)
            if date.tzinfo is not None:
                date = date.replace(tzinfo=None)
                logger.info(f"🔍 DEBUG - Fecha convertida a naive: {date}")
            
            date_str = date.strftime('%Y-%m-%d')
            logger.info(f"🔍 DEBUG - Buscando fecha: {date_str}")
            
            # Intentar con el símbolo solicitado primero
            df = self._get_dataframe(symbol)
            
            # DEBUG: Estado del DataFrame
            logger.info(f"🔍 DEBUG - DataFrame {symbol}: {'VACÍO' if df is None or df.empty else f'{len(df)} registros'}")
            if df is not None and not df.empty:
                logger.info(f"🔍 DEBUG - Índice tipo: {type(df.index)}, primer elemento: {df.index[0]}, último: {df.index[-1]}")
                logger.info(f"🔍 DEBUG - Rango de fechas: {df.index.min()} a {df.index.max()}")
            
            if df is None or df.empty:
                logger.error(f"No hay datos para {symbol}")
                return self._get_empty_market_data()
            
            # CORRECCIÓN: Ya no necesitamos convertir aquí porque se hizo en _load_data()
            # El índice ya debería ser DatetimeIndex con UTC
            
            # FORMATAR el índice para comparar solo fechas (sin horas)
            df_index_dates = df.index.strftime('%Y-%m-%d')
            logger.info(f"🔍 DEBUG - Primeras 5 fechas en índice: {df_index_dates[:5].tolist()}")
            
            # Buscar la fecha exacta (solo parte de fecha)
            matching_indices = np.where(df_index_dates == date_str)[0]
            
            if len(matching_indices) > 0:
                idx = matching_indices[0]
                row = df.iloc[idx]
                actual_date_in_df = df.index[idx]
                logger.info(f"✅ DEBUG - Fecha encontrada: {actual_date_in_df} (buscada: {date_str})")
                return self._extract_market_data(row, date_str, symbol, df)
                
            # Si no encuentra la fecha exacta, buscar la más cercana anterior
            logger.warning(f"🔍 DEBUG - Fecha {date_str} no encontrada, buscando más cercana...")
            
            # Convertir a datetime para comparación (con timezone para consistencia)
            target_date = pd.to_datetime(date_str).tz_localize('UTC')
            previous_dates = df[df.index <= target_date]
            
            logger.info(f"🔍 DEBUG - Fechas anteriores encontradas: {len(previous_dates)}")
            
            if len(previous_dates) > 0:
                closest_date = previous_dates.index[-1]
                row = df.loc[closest_date]
                logger.info(f"🔍 DEBUG - Usando fecha más cercana: {closest_date} (buscada: {date_str})")
                return self._extract_market_data(row, closest_date.strftime('%Y-%m-%d'), symbol, df)
            
            logger.warning(f"🔍 DEBUG - No se encontraron datos para {date_str} ni fechas cercanas")
            return self._get_empty_market_data()
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de mercado: {e}")
            import traceback
            logger.error(f"🔍 DEBUG - Traceback: {traceback.format_exc()}")
            return self._get_empty_market_data()
    
    def _extract_market_data(self, row: pd.Series, date_str: str, symbol: str, df: pd.DataFrame) -> Dict:
        """Extraer datos de mercado de una fila - CORREGIDO"""
        try:
            # DEBUG: Información de la fila
            logger.info(f"🔍 DEBUG - Extrayendo datos para {date_str}, close: {row['Close']}")

            # Calcular cambio porcentual
            change_pct = 0

            # Encontrar posición en el DataFrame - CORREGIDO
            date_in_index = pd.to_datetime(date_str).tz_localize('UTC')
            try:
                date_index = df.index.get_indexer([date_in_index], method='pad')[0]
                logger.info(f"🔍 DEBUG - Índice encontrado: {date_index}")
            except Exception as e:
                logger.error(f"Error encontrando índice: {e}")
                date_str_normalized = date_str
                index_as_str = df.index.strftime('%Y-%m-%d')
                matches = np.where(index_as_str == date_str_normalized)[0]
                if len(matches) > 0:
                    date_index = matches[0]
                else:
                    date_index = 0

            if date_index > 0:
                try:
                    prev_row = df.iloc[date_index - 1]
                    change_pct = ((row['Close'] - prev_row['Close']) / prev_row['Close']) * 100
                    logger.info(f"🔍 DEBUG - Cambio % calculado: {change_pct:.2f}%")
                except Exception as e:
                    logger.error(f"Error calculando cambio %: {e}")

            # USAR _safe_float para todos los valores
            return {
                'close': self._safe_float(row['Close']),
                'open': self._safe_float(row['Open']),
                'high': self._safe_float(row['High']),
                'low': self._safe_float(row['Low']),
                'volume': self._safe_float(row['Volume']),
                'change_pct': self._safe_float(change_pct),
                'date': date_str,
                'symbol': symbol,
                'data_source': 'local_csv'
            }
        except Exception as e:
            logger.error(f"Error extrayendo datos de mercado: {e}")
            return self._get_empty_market_data()
    def get_technical_data_range(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Obtener rango de datos para cálculo de indicadores técnicos - CORREGIDO"""
        df = self._get_dataframe(symbol)
    
        if df is None or df.empty:
            return pd.DataFrame()
    
        # El índice ya debería ser DatetimeIndex con UTC desde _load_data()
        df_working = df.copy()
        
        # Convertir fechas de entrada a UTC para consistencia
        if start_date.tzinfo is not None:
            start_date_utc = start_date.astimezone(timezone.utc)
        else:
            start_date_utc = start_date.replace(tzinfo=timezone.utc)
            
        if end_date.tzinfo is not None:
            end_date_utc = end_date.astimezone(timezone.utc)
        else:
            end_date_utc = end_date.replace(tzinfo=timezone.utc)
    
        # EVITAR el error de "truth value of a Series is ambiguous"
        try:
            mask = (df_working.index >= start_date_utc) & (df_working.index <= end_date_utc)
            result = df_working.loc[mask].copy()
        except Exception as e:
            logger.error(f"Error aplicando máscara de fechas: {e}")
            # Método alternativo usando slicing
            try:
                result = df_working.loc[start_date_utc:end_date_utc].copy()
            except Exception as e2:
                logger.error(f"Error en slicing de fechas: {e2}")
                # Último recurso: hacer todo naive
                df_working.index = df_working.index.tz_localize(None)
                start_naive = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date
                end_naive = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date
                result = df_working.loc[start_naive:end_naive].copy()
        
        return result
    
    def get_vix_data(self, date: datetime) -> Optional[float]:
        """Obtener datos VIX - CORREGIDO"""
        try:
            # Normalizar fecha a UTC
            if date.tzinfo is not None:
                date_utc = date.astimezone(timezone.utc)
            else:
                date_utc = date.replace(tzinfo=timezone.utc)

            date_str = date_utc.strftime('%Y-%m-%d')
            target_date = pd.to_datetime(date_str).tz_localize('UTC')

            if self.vix_df is None or self.vix_df.empty:
                return None

            # El índice VIX ya debería ser DatetimeIndex con UTC desde _load_data()
            vix_df_working = self.vix_df.copy()

            # Buscar fecha exacta
            exact_matches = vix_df_working.index[vix_df_working.index.normalize() == target_date]
            if len(exact_matches) > 0:
                return float(vix_df_working.loc[exact_matches[0], 'Close'])

            # Buscar fecha anterior más cercana
            previous_dates = vix_df_working.index[vix_df_working.index.normalize() <= target_date]
            if len(previous_dates) > 0:
                closest_date = previous_dates[-1]
                return float(vix_df_working.loc[closest_date, 'Close'])

            return None

        except Exception as e:
            logger.error(f"Error obteniendo VIX: {e}")
            logger.error(f"🔍 DEBUG - Tipo de índice VIX: {type(self.vix_df.index) if self.vix_df is not None else 'None'}")
            return None
    
    def download_initial_data(self, start_date: str = "2000-01-01", end_date: str = None):
        """Descargar datos iniciales si no existen"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"📥 Descargando datos desde {start_date} hasta {end_date}")
        time.sleep(1)  # Pausa para evitar límites de tasa
        # Descargar S&P 500
        if not self.sp500_file.exists():
            self._download_symbol("^GSPC", self.sp500_file, start_date, end_date)
        time.sleep(1)  # Pausa para evitar límites de tasa
        # Descargar SPY
        if not self.spy_file.exists():
            self._download_symbol("SPY", self.spy_file, start_date, end_date)
        time.sleep(1)  # Pausa para evitar límites de tasa
        # Descargar VIX
        if not self.vix_file.exists():
            self._download_symbol("^VIX", self.vix_file, start_date, end_date)
        
        # Recargar datos
        self._load_data()
    
    def _download_symbol(self, symbol: str, file_path: Path, start_date: str, end_date: str):
        """Descargar datos de un símbolo específico"""
        try:
            logger.info(f"📥 Descargando {symbol}...")
            time.sleep(1)  # Pausa para evitar límites de tasa
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            
            if not df.empty:
                # Guardar con todas las columnas
                df.to_csv(file_path)
                logger.info(f"✅ {symbol} descargado: {len(df)} registros")
            else:
                logger.error(f"❌ No se pudieron descargar datos para {symbol}")
                
        except Exception as e:
            logger.error(f"Error descargando {symbol}: {e}")
    
    def _get_dataframe(self, symbol: str) -> pd.DataFrame:
        """Obtener DataFrame según símbolo"""
        symbol_map = {
            "^GSPC": self.sp500_df,
            "SPY": self.spy_df,
            "^VIX": self.vix_df
        }
        return symbol_map.get(symbol, self.spy_df)  # Default a SPY
    
    def _get_empty_market_data(self) -> Dict:
        """Datos vacíos cuando no hay información"""
        return {
            'close': 0,
            'open': 0,
            'high': 0,
            'low': 0,
            'volume': 0,
            'change_pct': 0,
            'data_source': 'none'
        }

# Instancia global
data_manager = DataManager()



# ==========================================
# SERVICIO PRINCIPAL: ANÁLISIS DE MERCADO
# ==========================================

class MarketAnalysisService:
    """Servicio principal que coordina todos los agentes"""
    
    def __init__(self):
        self.news_scraper = NewsScraperAgent()
        self.sentiment_analyzer = SentimentAnalyzerAgent()
        self.impact_scorer = ImpactScorerAgent()
        self.technical_fetcher = TechnicalIndicatorsFetcher()
        self.social_analyzer = SocialMediaAnalyzer()
        self.llm_synthesizer = LLMSynthesizer()
        self.context_json_file = Path("data") / "historical_context_cache.json"
        self.news_service = NewsService() 
        
        # Verificar que tenemos datos
        self._ensure_data_loaded()

    def _calculate_sentiment_from_relevance(self, relevance_score: float) -> float:
        """Convierte puntuación de relevancia a sentimiento (-1 a 1)"""
        # Mapear relevancia 0-1 a sentimiento -1 a 1
        # Asumimos que mayor relevancia puede indicar mayor impacto (positivo o negativo)
        # Esto es una aproximación - podrías refinar basado en tu data real
        return (relevance_score - 0.5) * 2  # Convierte 0-1 a -1 a 1
    
    
    async def analyze_date(
        self,
        date: datetime,
        user_level: UserLevel = UserLevel.INTERMEDIATE
    ) -> DateAnalysis:
        """Análisis completo de una fecha específica"""
        
        logger.info(f"📊 Analizando fecha: {date.strftime('%Y-%m-%d')}")
        
        # Cache check
        cache_key = f"analysis_{date.strftime('%Y%m%d')}_{user_level}"
        cached = cache_manager.get(cache_key)
        if cached:
            logger.info("✅ Retornando desde caché")
            return DateAnalysis(**cached)
        
        try:
            # 1. Obtener datos del mercado
            market_data = self._get_market_data(date)
            
            # 2. Obtener y analizar noticias - NUEVO: Usar el servicio mejorado
            news_articles = await self.news_service.get_top_sp500_news_combined(date.strftime('%Y-%m-%d'))
            
            # Convertir al formato esperado por el sistema existente
            news = []
            for article in news_articles:
                news_item = {
                    'title': article['title'],
                    'source': article['source'],
                    'url': article['url'],
                    'published_at': article.get('published_at', date.isoformat()),
                    'sentiment_score': self._calculate_sentiment_from_relevance(article['relevance_score']),
                    'sentiment_label': self._get_sentiment_label_from_score(article['relevance_score']),
                    'impact_score': min(10, article['relevance_score']),  # Escalar a 1-10
                    'summary': article['summary']
                }
                news.append(news_item)
            
            
            
            # 3. Obtener indicadores técnicos
            technical = self.technical_fetcher.fetch_technical_indicators("^GSPC", date)
            
            # 4. Análisis de redes sociales
            social = await self.social_analyzer.analyze_social_sentiment(date)
            
            # 5. Generar explicación de causalidad (mejorada con noticias reales)
            explanation = self.llm_synthesizer.generate_causality_explanation(
                date, market_data, news, technical, social, user_level
            )
            
            # 6. Determinar dirección del mercado
            change = market_data.get('change_pct', 0)
            if change > 0.1:
                direction = MarketDirection.BULLISH
            elif change < -0.1:
                direction = MarketDirection.BEARISH
            else:
                direction = MarketDirection.NEUTRAL
            
            # 7. Identificar factores clave (mejorado con noticias reales)
            key_factors = self._identify_key_factors_from_news(news, technical, social)
            
            # 8. Calcular score de confianza
            confidence = self._calculate_confidence_score(news, technical, social)
            
            # Crear objeto de respuesta
            analysis = DateAnalysis(
                date=date,
                sp500_close=market_data.get('close', 0),
                sp500_change_pct=change,
                sp500_volume=market_data.get('volume', 0),
                market_direction=direction,
                news=[NewsItem(
                    title=n['title'],
                    source=n['source'],
                    url=n['url'],
                    published_at = (
    datetime.fromisoformat(n['published_at'])
    if isinstance(n['published_at'], str) and n['published_at'].strip()
    else None
),
                    sentiment_score=n['sentiment_score'],
                    sentiment_label=n['sentiment_label'],
                    impact_score=n['impact_score']
                ) for n in news[:10]],  # Máximo 10 noticias
                technical=TechnicalIndicators(
                    rsi=technical['rsi'],
                    macd=technical['macd'],
                    sma_50=technical['sma_50'],
                    volume=technical['volume'],
                    vix=technical.get('vix'),
                    support_levels=technical['support_levels'],
                    resistance_levels=technical['resistance_levels'],
                    trend=technical['trend']
                ),
                social=[SocialSentiment(**s) for s in social if s],
                causality_explanation=explanation,
                confidence_score=confidence,
                key_factors=key_factors
            )
            
            # Guardar en caché
            cache_manager.set(cache_key, analysis.dict(), config.CACHE_TTL_LONG)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error en análisis de fecha: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    
    def _save_context_to_json(self, analysis_list: List[Dict]):
        """Guarda el contexto histórico en un JSON persistente."""
        try:
            # Asegurarse de que el directorio 'data' exista
            Path("data").mkdir(exist_ok=True)
            with open(self.context_json_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_list, f, indent=2, default=str)
            logger.info(f"💾 Contexto persistente guardado en {self.context_json_file}")
        except Exception as e:
            logger.error(f"Error guardando JSON persistente: {e}")

    def _load_context_from_json(self) -> List[Dict]:
        """Carga el contexto histórico desde el JSON persistente."""
        if not self.context_json_file.exists():
            return []
        try:
            with open(self.context_json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convertir strings de fecha a datetime
            parsed_data = []
            for item in data:
                try:
                    # Si 'date' es string, convertir a datetime
                    if isinstance(item.get('date'), str):
                        item['date'] = datetime.fromisoformat(item['date'].replace('Z', '+00:00'))

                    # Solo añadir si la fecha es válida
                    if isinstance(item.get('date'), datetime):
                        parsed_data.append(item)
                    else:
                        logger.warning(f"Omitiendo registro con fecha inválida: {item.get('date')}")

                except Exception as e:
                    logger.warning(f"Omitiendo registro con fecha inválida: {item.get('date')}, error: {e}")

            logger.info(f"✅ Contexto persistente cargado desde {self.context_json_file} ({len(parsed_data)} registros)")
            return parsed_data

        except Exception as e:
            logger.error(f"Error cargando JSON persistente: {e}")
            return []
    def _ensure_data_loaded(self):
        """Verificar que tenemos datos locales"""
        if (data_manager.sp500_df is None or data_manager.sp500_df.empty) and \
           (data_manager.spy_df is None or data_manager.spy_df.empty):
            logger.warning("No hay datos locales disponibles, descargando...")
            data_manager.download_initial_data()
    
    async def build_historical_prompt_from_endpoint(self, days_back: int = 30) -> (str, float, datetime):
        """
        [NUEVA LÓGICA] Construye el prompt llamando al endpoint /api/analysis/date
        para cada día, usando caché persistente JSON.
        """
        logger.info(f"🔥 Construyendo prompt desde endpoint, {days_back} días (Lógica de usuario)...")
        today = datetime.now()

        # 1. Cargar contexto JSON persistente
        persistent_context = self._load_context_from_json()

        # 2. Determinar fechas faltantes - CORRECCIÓN AQUÍ
        # item['date'] ya es datetime, usar strftime directamente
        existing_dates_str = {item['date'].strftime('%Y-%m-%d') for item in persistent_context}
        logger.info(f"Fechas existentes en JSON: {len(existing_dates_str)}")

        tasks = []

        # 3. Iterar hacia atrás (ej. 60 días) para encontrar 30 días de mercado que nos faltan
        days_to_check = days_back * 2 

        for i in range(1, days_to_check):
            target_date = today - timedelta(days=i)
            target_date_str = target_date.strftime('%Y-%m-%d')

            if target_date_str not in existing_dates_str:
                # ¡Esta es una fecha que no tenemos!
                tasks.append(self.analyze_date(target_date, UserLevel.INTERMEDIATE))

        if tasks:
            logger.warning(f"🚀 ¡¡¡PRIMERA EJECUCIÓN O DÍA NUEVO!!!")
            logger.warning(f"Se llamará al endpoint 'analyze_date' {len(tasks)} veces (esto incluye {len(tasks)} llamadas al LLM de análisis).")
            logger.warning(f"Esto puede tardar varios minutos...")

            results = await asyncio.gather(*tasks, return_exceptions=True)

            new_context_items = []
            for res in results:
                if isinstance(res, DateAnalysis):
                    # Convertir a dict para guardarlo en JSON
                    new_context_items.append(res.dict())
                elif isinstance(res, HTTPException) and res.status_code == 404:
                    # Esperado (fines de semana, festivos)
                    logger.info(f"Día no bursátil (HTTP 404): {res.detail}")
                elif isinstance(res, Exception):
                    # Error real
                    logger.error(f"Error en gather llamando a analyze_date: {res}")

            # Añadir los nuevos items a la lista
            persistent_context.extend(new_context_items)
            logger.info(f"Añadidos {len(new_context_items)} nuevos análisis al contexto.")

        # 4. Ordenar, truncar y guardar
        if not persistent_context:
            logger.error("No se pudo construir ningún contexto histórico.")
            return "", prediction_service._get_current_price(), datetime.now()

        # Ordenar (más reciente primero)
        persistent_context.sort(key=lambda x: x['date'], reverse=True)
        # Truncar (quedarse solo con los 'days_back' más recientes)
        final_context_list = persistent_context[:days_back]
        # Re-ordenar (más antiguo primero, para el prompt)
        final_context_list.reverse()

        # 5. Guardar en JSON para la próxima vez
        # (Guardamos la lista truncada y ordenada más reciente primero)
        self._save_context_to_json(persistent_context[:days_back])

        # 6. Formatear el prompt
        prompt_lines = []
        for day_data in final_context_list:
            # Extraer la data del dict (que vino de DateAnalysis.dict())
            # CORRECCIÓN: day_data['date'] ya es datetime, usar strftime
            date_str = day_data['date'].strftime('%Y-%m-%d')
            change_pct = day_data['sp500_change_pct']

            trend = day_data['technical']['trend']
            rsi = day_data['technical']['rsi']
            explanation = day_data['causality_explanation']

            prompt_lines.append(
                f"--- DÍA: {date_str} ---\n"
                f"Movimiento: {change_pct:+.2f}%\n"
                f"Datos Técnicos: Tendencia={trend}, RSI={rsi:.1f}\n"
                f"Explicación (IA): {explanation}\n"
            )

        # CORRECCIÓN: day_data['date'] ya es datetime, no necesita fromisoformat
        last_real_price = final_context_list[-1]['sp500_close']
        last_real_date_obj = final_context_list[-1]['date']  # Ya es datetime

        return "\n".join(prompt_lines), last_real_price, last_real_date_obj
    
    def _get_market_data(self, date: datetime) -> Dict:
        """Obtener datos de mercado desde archivos locales"""
        # Intentar con S&P 500 primero
        market_data = data_manager.get_market_data(date, "^GSPC")
        
        # Si no hay datos, usar SPY
        if market_data['close'] == 0:
            market_data = data_manager.get_market_data(date, "SPY")
            logger.info(f"Usando datos de SPY para {date.strftime('%Y-%m-%d')}")
        
        if market_data['close'] == 0:
            logger.error(f"No se encontraron datos locales para {date.strftime('%Y-%m-%d')}")
        
        return market_data
    
    def _get_recent_market_data(self) -> Dict:
        """Obtener datos del día más reciente disponible"""
        try:
            ticker = yf.Ticker("SPY")
            df = ticker.history(period="1mo", auto_adjust=True)
            
            if df.empty:
                raise ValueError("No hay datos disponibles")
            
            # Tomar el día más reciente
            latest_date = df.index[-1]
            current = df.loc[latest_date]
            
            # Calcular cambio con día anterior
            if len(df) > 1:
                prev_data = df.iloc[-2]
                change_pct = ((current['Close'] - prev_data['Close']) / prev_data['Close']) * 100
            else:
                change_pct = 0
            
            logger.info(f"✅ Usando datos más recientes de {latest_date.strftime('%Y-%m-%d')}")
            
            return {
                'close': float(current['Close']),
                'open': float(current['Open']),
                'high': float(current['High']),
                'low': float(current['Low']),
                'volume': float(current['Volume']),
                'change_pct': float(change_pct),
                'symbol_used': 'SPY_recent'
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo datos recientes: {e}")
            raise HTTPException(status_code=500, detail="No se pudieron obtener datos reales del mercado")
    def _calculate_confidence_score(self, news: List[Dict], technical: Dict, social: List[Dict]) -> float:
        """Calcular score de confianza del análisis"""
        score = 50.0  # Base
        
        # Más noticias = más confianza
        if len(news) > 5:
            score += 10
        elif len(news) > 2:
            score += 5
        
        # Sentimiento consistente = más confianza
        if news:
            sentiments = [n['sentiment_score'] for n in news[:5]]
            if len(sentiments) > 1:
                std_dev = np.std(sentiments)
                if std_dev < 0.2:  # Sentimiento consistente
                    score += 15
        
        # Indicadores técnicos claros
        rsi = technical.get('rsi', 50)
        if rsi > 70 or rsi < 30:
            score += 10  # Señal clara
        
        # Datos de redes sociales
        if social:
            score += 10
        
        # VIX disponible
        if technical.get('vix') is not None:
            score += 5
        
        return min(95.0, score)  # Máximo 95%
    def calculate_sentiment_from_relevance(self, relevance_score: float) -> float:
        """Convierte puntuación de relevancia a sentimiento (-1 a 1)"""
        # Mapear relevancia 0-1 a sentimiento -1 a 1
        # Asumimos que mayor relevancia puede indicar mayor impacto (positivo o negativo)
        # Esto es una aproximación - podrías refinar basado en tu data real
        return (relevance_score - 0.5) * 2  # Convierte 0-1 a -1 a 1

    def _get_sentiment_label_from_score(self, relevance_score: int) -> str:
        """Convertir score de relevancia a etiqueta de sentimiento"""
        if relevance_score >= 7:
            return "muy positivo"
        elif relevance_score >= 5:
            return "positivo"
        elif relevance_score >= 3:
            return "neutral"
        else:
            return "negativo"

    def _identify_key_factors_from_news(self, news: List[Dict], technical: Dict, social: List[Dict]) -> List[str]:
        """Identificar factores clave basados en noticias reales"""
        factors = []

        try:
            # Factor de noticias - usar las noticias reales
            if news and len(news) > 0:
                top_news = news[0]  # La noticia más relevante
                factors.append(f" {top_news['title'][:60]}...")

            # Factor técnico - RSI
            rsi = technical.get('rsi', 50)
            if rsi > 70:
                factors.append("📊 RSI en zona de sobrecompra")
            elif rsi < 30:
                factors.append("📊 RSI en zona de sobreventa")

            # Factor de tendencia
            trend = technical.get('trend', 'neutral')
            if trend == 'fuerte_alcista':
                factors.append(" Tendencia alcista fuerte")
            elif trend == 'alcista':
                factors.append(" Tendencia alcista moderada")
            elif trend == 'fuerte_bajista':
                factors.append(" Tendencia bajista fuerte")
            elif trend == 'bajista':
                factors.append(" Tendencia bajista moderada")

            # Factor de volumen
            volume = technical.get('volume', 0)
            if volume > 4000000000:
                factors.append(" Volumen de trading elevado")
            elif volume < 2000000000:
                factors.append(" Volumen de trading bajo")

            # Si no hay factores identificados, agregar uno genérico
            if not factors:
                factors.append(" Condiciones de mercado neutrales")

            return factors[:3]  # Máximo 3 factores principales

        except Exception as e:
            logger.error(f"Error identificando factores clave: {e}")
            return [" Análisis de factores no disponible"]

# ==========================================
# SERVICIO DE PREDICCIÓN
# ==========================================

class PredictionService:
    """Servicio para generar predicciones del S&P 500"""
    
    def __init__(self):
        self.market_service = MarketAnalysisService()
        self.llm_synthesizer = LLMSynthesizer()
        self.model = None
        self.scaler = StandardScaler()
    
    async def predict(
        self,
        horizon_days: int = 7,
        user_level: UserLevel = UserLevel.INTERMEDIATE
    ) -> MarketPrediction:
        """Generar predicción del S&P 500"""
        
        logger.info(f"🔮 Generando predicción a {horizon_days} días")
        
        # Cache check
        cache_key = f"prediction_{horizon_days}_{user_level}"
        cached = cache_manager.get(cache_key)
        if cached:
            return MarketPrediction(**cached)
        
        try:
            # 1. Obtener datos históricos para entrenamiento
            features, target = await self._prepare_training_data()
            
            # 2. Entrenar modelo (o usar modelo pre-entrenado)
            self._train_model(features, target)
            
            # 3. Preparar features actuales
            current_features = await self._get_current_features()
            
            # 4. Hacer predicción
            prediction_value = self.model.predict(current_features.reshape(1, -1))[0]
            
            # 5. Calcular rangos y confianza
            confidence = self._calculate_prediction_confidence(horizon_days)
            volatility = self._estimate_volatility(horizon_days)
            
            # 6. Obtener precio actual
            current_price = self._get_current_price()
            
            # 7. Calcular precios predichos
            change_pct = prediction_value * (horizon_days ** 0.5)  # Ajuste por horizonte
            price_predicted = current_price * (1 + change_pct / 100)
            range_upper = price_predicted * (1 + volatility)
            range_lower = price_predicted * (1 - volatility)
            
            # 8. Determinar dirección
            if change_pct > 0.5:
                direction = MarketDirection.BULLISH
            elif change_pct < -0.5:
                direction = MarketDirection.BEARISH
            else:
                direction = MarketDirection.NEUTRAL
            
            # 9. Calcular factores
            factors = {
                'technical': 0.30,
                'fundamental': 0.35,
                'social': 0.10,
                'historical': 0.25
            }
            
            # 10. Buscar patrones históricos similares
            patterns = await self._find_similar_patterns(current_features)
            
            # 11. Identificar riesgos
            risks = self._identify_risks(horizon_days)
            
            # 12. Generar explicación
            explanation = self.llm_synthesizer.generate_prediction_explanation(
                {
                    'direction': direction.value,
                    'change_pct': change_pct,
                    'horizon_days': horizon_days,
                    'confidence': confidence
                },
                factors,
                patterns,
                user_level
            )
            
            prediction = MarketPrediction(
                direction=direction,
                change_pct=change_pct,
                confidence=confidence,
                price_current=current_price,
                price_predicted=price_predicted,
                range_upper=range_upper,
                range_lower=range_lower,
                horizon_days=horizon_days,
                factors=factors,
                explanation=explanation,
                risks=risks,
                similar_patterns=patterns
            )
            
            # Cache por tiempo proporcional al horizonte
            cache_ttl = min(config.CACHE_TTL_SHORT * horizon_days, config.CACHE_TTL_LONG)
            cache_manager.set(cache_key, prediction.dict(), cache_ttl)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generando predicción: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Preparar datos para entrenamiento"""
        try:
            # Obtener datos históricos
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            ticker = yf.Ticker("^GSPC")
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                raise ValueError("No hay datos históricos disponibles")
            
            # Calcular features técnicos
            df['Returns'] = df['Close'].pct_change()
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            df['MACD'] = ta.trend.MACD(df['Close']).macd()
            df['BB_width'] = ta.volatility.BollingerBands(df['Close']).bollinger_wband()
            df['Volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            
            # Limpiar NaNs
            df = df.dropna()
            
            # Features y target
            feature_cols = ['RSI', 'MACD', 'BB_width', 'Volume_ratio']
            X = df[feature_cols].values
            y = df['Returns'].shift(-1).dropna().values  # Predecir retorno siguiente
            
            # Alinear tamaños
            X = X[:-1]
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparando datos de entrenamiento: {e}")
            # Datos sintéticos de respaldo
            X = np.random.randn(100, 4)
            y = np.random.randn(100) * 0.01
            return X, y
    
    def _train_model(self, X: np.ndarray, y: np.ndarray):
        """Entrenar modelo de predicción"""
        try:
            # Escalar features
            X_scaled = self.scaler.fit_transform(X)
            
            # Entrenar Random Forest
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.model.fit(X_scaled, y)
            
            logger.info("✅ Modelo entrenado correctamente")
            
        except Exception as e:
            logger.error(f"Error entrenando modelo: {e}")
            # Modelo dummy
            self.model = lambda x: np.array([0.001])  # Predicción neutral
    
    async def _get_current_features(self) -> np.ndarray:
        """Obtener features actuales para predicción"""
        try:
            # Obtener indicadores técnicos actuales
            technical = TechnicalIndicatorsFetcher().fetch_technical_indicators(
                "^GSPC",
                datetime.now()
            )
            
            # Construir feature vector
            features = np.array([
                technical.get('rsi', 50),
                technical.get('macd', {}).get('value', 0),
                technical.get('bb_width', 0.0),  # Corregido
                technical.get('volume_ratio', 1.0)  # Corregido
            ])
            
            return self.scaler.transform(features.reshape(1, -1))[0]
            
        except Exception as e:
            logger.error(f"Error obteniendo features actuales: {e}")
            return np.random.randn(4)
    
    def _get_current_price(self) -> float:
        """Obtener precio actual del S&P 500 desde datos locales"""
        try:
            # Intentar desde data_manager (datos locales más recientes)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)  # Últimos 5 días

            df = data_manager.get_technical_data_range("^GSPC", start_date, end_date)

            if not df.empty:
                last_price = float(df['Close'].iloc[-1])
                logger.info(f"💰 Precio actual desde datos locales: ${last_price:.2f}")
                return last_price
            
            df = data_manager.get_technical_data_range("SPY", start_date, end_date)
            
            if not df.empty:
                last_price = float(df['Close'].iloc[-1])
                logger.info(f"💰 Precio actual desde datos locales: ${last_price:.2f}")
                return last_price
            
            # Si no hay datos locales, intentar yfinance
            logger.warning("⚠️ No hay datos locales recientes, intentando yfinance...")
            ticker = yf.Ticker("^GSPC")

            # Intentar info primero
            try:
                current_price = float(ticker.info.get('regularMarketPrice', 0))
                if current_price > 0:
                    logger.info(f"💰 Precio actual desde yfinance (info): ${current_price:.2f}")
                    return current_price
            except:
                pass
            
            # Intentar history
            hist = ticker.history(period="1d")
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                logger.info(f"💰 Precio actual desde yfinance (history): ${current_price:.2f}")
                return current_price

            # Fallback: usar último precio conocido del data_manager
            logger.warning("⚠️ No se pudo obtener precio actual, usando último conocido")
            all_data = data_manager.spy_df if data_manager.spy_df is not None and not data_manager.spy_df.empty else data_manager.sp500_df

            if all_data is not None and not all_data.empty:
                last_price = float(all_data['Close'].iloc[-1])
                logger.info(f"💰 Precio actual (último conocido): ${last_price:.2f}")
                return last_price

            # Último recurso
            logger.error("❌ No se pudo obtener ningún precio")
            return 6800.0  # Valor aproximado por defecto

        except Exception as e:
            logger.error(f"❌ Error obteniendo precio actual: {e}")
            return 6800.0  # Valor aproximado por defecto
    
    def _calculate_prediction_confidence(self, horizon_days: int) -> float:
        """Calcular confianza según horizonte temporal"""
        base_confidence = {
            1: 85,
            7: 75,
            30: 60,
            90: 45,
            180: 35,
            365: 25
        }
        
        # Interpolar para horizontes intermedios
        for days, conf in sorted(base_confidence.items()):
            if horizon_days <= days:
                return float(conf)
        
        return 20.0  # Muy largo plazo
    
    def _estimate_volatility(self, horizon_days: int) -> float:
        """Estimar volatilidad esperada"""
        # Volatilidad aumenta con horizonte (simplificado)
        daily_vol = 0.01  # 1% diario típico
        return daily_vol * (horizon_days ** 0.5)
    
    async def _find_similar_patterns(self, current_features: np.ndarray) -> List[Dict]:
        """Buscar patrones históricos similares"""
        # Simulación de patrones encontrados
        patterns = [
            {
                'date': '2024-03-15',
                'similarity': 0.87,
                'change': 1.8,
                'context': 'Fed pausó tasas, tech earnings positivos'
            },
            {
                'date': '2023-07-20',
                'similarity': 0.82,
                'change': 1.2,
                'context': 'Inflación bajando, dólar débil'
            },
            {
                'date': '2023-11-10',
                'similarity': 0.79,
                'change': 0.6,
                'context': 'Golden cross, sentimiento moderado'
            }
        ]
        
        return patterns
    
    def _identify_risks(self, horizon_days: int) -> List[str]:
        """Identificar riesgos principales para la predicción"""
        risks = []
        
        if horizon_days <= 7:
            risks.append("📊 Datos económicos inesperados (CPI, empleos)")
            risks.append("📰 Noticias geopolíticas sorpresa")
        elif horizon_days <= 30:
            risks.append("🏦 Cambios en política monetaria de la Fed")
            risks.append("📈 Temporada de earnings con sorpresas")
            risks.append("🌍 Eventos geopolíticos mayores")
        else:
            risks.append("🔄 Cambio de ciclo económico")
            risks.append("⚠️ Recesión no anticipada")
            risks.append("🏛️ Cambios regulatorios significativos")
        
        return risks[:3]

# ==========================================
# SERVICIO DE BÚSQUEDA HISTÓRICA
# ==========================================

class HistoricalSearchService:
    """Servicio para búsquedas y comparaciones históricas"""
    
    def __init__(self):
        self.market_service = MarketAnalysisService()
    
    async def find_similar_dates(
        self,
        reference_date: datetime,
        similarity_threshold: float = 0.7,
        max_results: int = 5
    ) -> List[Dict]:
        """Encontrar fechas históricas similares"""
        
        logger.info(f"🔍 Buscando fechas similares a {reference_date.strftime('%Y-%m-%d')}")
        
        try:
            # Obtener características de la fecha de referencia
            ref_analysis = await self.market_service.analyze_date(reference_date)
            
            # Buscar en histórico
            similar_dates = []
            
            # Simplificado: buscar en últimos 2 años
            end_date = reference_date - timedelta(days=1)
            start_date = end_date - timedelta(days=730)
            
            ticker = yf.Ticker("^GSPC")
            df = ticker.history(start=start_date, end=end_date)
            
            for date in df.index[-100:]:  # Últimos 100 días de trading
                try:
                    # Calcular similitud basada en cambio porcentual
                    date_analysis = await self.market_service.analyze_date(date.to_pydatetime())
                    
                    similarity = self._calculate_similarity(ref_analysis, date_analysis)
                    
                    if similarity >= similarity_threshold:
                        similar_dates.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'similarity': similarity,
                            'change_pct': date_analysis.sp500_change_pct,
                            'key_factors': date_analysis.key_factors[:2]
                        })
                except:
                    continue
            
            # Ordenar por similitud
            similar_dates.sort(key=lambda x: x['similarity'], reverse=True)
            
            return similar_dates[:max_results]
            
        except Exception as e:
            logger.error(f"Error buscando fechas similares: {e}")
            return []
    
    def _calculate_similarity(self, ref: DateAnalysis, comp: DateAnalysis) -> float:
        """Calcular similitud entre dos análisis de fechas"""
        similarity = 0.0
        
        # Similitud en cambio porcentual
        change_diff = abs(ref.sp500_change_pct - comp.sp500_change_pct)
        if change_diff < 0.1:
            similarity += 0.3
        elif change_diff < 0.5:
            similarity += 0.2
        elif change_diff < 1.0:
            similarity += 0.1
        
        # Similitud en dirección
        if ref.market_direction == comp.market_direction:
            similarity += 0.2
        
        # Similitud en indicadores técnicos
        if abs(ref.technical.rsi - comp.technical.rsi) < 10:
            similarity += 0.2
        
        # Similitud en tendencia
        if ref.technical.trend == comp.technical.trend:
            similarity += 0.2
        
        # Similitud en sentimiento de noticias
        if ref.news and comp.news:
            ref_sentiment = np.mean([n.sentiment_score for n in ref.news[:3]])
            comp_sentiment = np.mean([n.sentiment_score for n in comp.news[:3]])
            if abs(ref_sentiment - comp_sentiment) < 0.2:
                similarity += 0.1
        
        return min(1.0, similarity)
    
    async def compare_periods(
        self,
        period1_start: datetime,
        period1_end: datetime,
        period2_start: datetime,
        period2_end: datetime
    ) -> Dict:
        """Comparar dos períodos de tiempo"""
        
        logger.info(f"📊 Comparando períodos")
        
        try:
            # Obtener datos para ambos períodos
            period1_data = await self._get_period_data(period1_start, period1_end)
            period2_data = await self._get_period_data(period2_start, period2_end)
            
            # Calcular estadísticas
            comparison = {
                'period1': {
                    'start': period1_start.isoformat(),
                    'end': period1_end.isoformat(),
                    'total_return': period1_data['total_return'],
                    'volatility': period1_data['volatility'],
                    'avg_volume': period1_data['avg_volume'],
                    'best_day': period1_data['best_day'],
                    'worst_day': period1_data['worst_day']
                },
                'period2': {
                    'start': period2_start.isoformat(),
                    'end': period2_end.isoformat(),
                    'total_return': period2_data['total_return'],
                    'volatility': period2_data['volatility'],
                    'avg_volume': period2_data['avg_volume'],
                    'best_day': period2_data['best_day'],
                    'worst_day': period2_data['worst_day']
                },
                'comparison': {
                    'better_performer': 'period1' if period1_data['total_return'] > period2_data['total_return'] else 'period2',
                    'more_volatile': 'period1' if period1_data['volatility'] > period2_data['volatility'] else 'period2',
                    'return_difference': period1_data['total_return'] - period2_data['total_return'],
                    'volatility_ratio': period1_data['volatility'] / period2_data['volatility'] if period2_data['volatility'] > 0 else 0
                }
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparando períodos: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _get_period_data(self, start: datetime, end: datetime) -> Dict:
        """Obtener datos estadísticos de un período"""
        try:
            ticker = yf.Ticker("^GSPC")
            df = ticker.history(start=start, end=end)
            
            if df.empty:
                return {
                    'total_return': 0,
                    'volatility': 0,
                    'avg_volume': 0,
                    'best_day': {},
                    'worst_day': {}
                }
            
            # Calcular métricas
            returns = df['Close'].pct_change()
            
            total_return = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
            volatility = returns.std() * (252 ** 0.5)  # Anualizada
            avg_volume = df['Volume'].mean()
            
            # Mejor y peor día
            best_idx = returns.idxmax()
            worst_idx = returns.idxmin()
            
            return {
                'total_return': round(total_return, 2),
                'volatility': round(volatility * 100, 2),
                'avg_volume': int(avg_volume),
                'best_day': {
                    'date': best_idx.strftime('%Y-%m-%d'),
                    'return': round(returns[best_idx] * 100, 2)
                },
                'worst_day': {
                    'date': worst_idx.strftime('%Y-%m-%d'),
                    'return': round(returns[worst_idx] * 100, 2)
                }
            }
        except Exception as e:
            logger.error(f"Error obteniendo datos del período: {e}")
            return {
                'total_return': 0,
                'volatility': 0,
                'avg_volume': 0,
                'best_day': {},
                'worst_day': {}
            }
            
# Agregar al inicio del archivo, antes de crear las instancias
def initialize_system():
    """Inicializar el sistema con datos locales"""
    logger.info("🚀 Inicializando sistema con datos locales...")
    
    # Crear data manager y cargar/descargar datos
    global data_manager
    data_manager = DataManager()
    
    # Verificar si necesitamos descargar datos
    if not data_manager.sp500_file.exists() or not data_manager.spy_file.exists():
        logger.info("📥 Descargando datos históricos iniciales...")
        data_manager.download_initial_data(start_date="2000-01-01")
    else:
        logger.info("✅ Datos locales ya existen")
    
    return data_manager

# Llamar la inicialización al inicio
data_manager = initialize_system()

# ==========================================
# API FASTAPI
# ==========================================

app = FastAPI(
    title="S&P 500 Intelligence Platform API",
    version="1.0.0",
    description="Sistema de análisis multi-agente con predicción y explicación contextual del S&P 500"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancias de servicios
market_service = MarketAnalysisService()
prediction_service = PredictionService()
historical_service = HistoricalSearchService()

# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/")
async def root():
    """Endpoint raíz con información del API"""
    return {
        "name": "S&P 500 Intelligence Platform API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "analysis": "/api/analysis/date",
            "prediction": "/api/prediction",
            "historical": "/api/historical/similar",
            "comparison": "/api/historical/compare",
            "market_status": "/api/market/status"
        }
    }

@app.get("/api/analysis/date", response_model=DateAnalysis)
async def analyze_date(
    date: str = Query(..., description="Fecha a analizar (YYYY-MM-DD)"),
    user_level: UserLevel = Query(UserLevel.INTERMEDIATE, description="Nivel de conocimiento del usuario")
):
    """
    Análisis completo de una fecha específica del S&P 500.
    Incluye noticias, indicadores técnicos, sentimiento social y explicación de causalidad.
    """
    try:
        analysis_date = datetime.strptime(date, "%Y-%m-%d")
        return await market_service.analyze_date(analysis_date, user_level)
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de fecha inválido. Use YYYY-MM-DD")

@app.get("/api/prediction", response_model=MarketPrediction)
async def get_prediction(
    horizon_days: int = Query(7, ge=1, le=365, description="Horizonte de predicción en días"),
    user_level: UserLevel = Query(UserLevel.INTERMEDIATE, description="Nivel de conocimiento del usuario")
):
    """
    Generar predicción del S&P 500 para el horizonte especificado.
    Incluye explicación contextualizada según el nivel del usuario.
    """
    return await prediction_service.predict(horizon_days, user_level)

@app.get("/api/historical/similar")
async def find_similar_dates(
    reference_date: str = Query(..., description="Fecha de referencia (YYYY-MM-DD)"),
    threshold: float = Query(0.7, ge=0, le=1, description="Umbral de similitud"),
    max_results: int = Query(5, ge=1, le=20, description="Máximo número de resultados")
):
    """
    Buscar fechas históricas similares a la fecha de referencia.
    Útil para análisis de patrones y comportamientos recurrentes.
    """
    try:
        ref_date = datetime.strptime(reference_date, "%Y-%m-%d")
        return await historical_service.find_similar_dates(ref_date, threshold, max_results)
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de fecha inválido. Use YYYY-MM-DD")

@app.post("/api/historical/compare")
async def compare_periods(
    period1_start: str = Query(..., description="Inicio período 1 (YYYY-MM-DD)"),
    period1_end: str = Query(..., description="Fin período 1 (YYYY-MM-DD)"),
    period2_start: str = Query(..., description="Inicio período 2 (YYYY-MM-DD)"),
    period2_end: str = Query(..., description="Fin período 2 (YYYY-MM-DD)")
):
    """
    Comparar dos períodos de tiempo del S&P 500.
    Retorna estadísticas comparativas de rendimiento, volatilidad y eventos clave.
    """
    try:
        p1_start = datetime.strptime(period1_start, "%Y-%m-%d")
        p1_end = datetime.strptime(period1_end, "%Y-%m-%d")
        p2_start = datetime.strptime(period2_start, "%Y-%m-%d")
        p2_end = datetime.strptime(period2_end, "%Y-%m-%d")
        
        return await historical_service.compare_periods(p1_start, p1_end, p2_start, p2_end)
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de fecha inválido. Use YYYY-MM-DD")

@app.get("/api/market/status")
async def get_market_status():
    """
    Estado actual del mercado S&P 500.
    Información en tiempo real del índice.
    """
    try:
        ticker = yf.Ticker("^GSPC")
        info = ticker.info
        
        # Obtener último precio
        hist = ticker.history(period="1d")
        
        if not hist.empty:
            last_close = float(hist['Close'].iloc[-1])
            last_volume = float(hist['Volume'].iloc[-1])
            
            # Cambio del día
            hist_2d = ticker.history(period="2d")
            if len(hist_2d) >= 2:
                prev_close = float(hist_2d['Close'].iloc[-2])
                change = last_close - prev_close
                change_pct = (change / prev_close) * 100
            else:
                change = 0
                change_pct = 0
        else:
            last_close = info.get('regularMarketPrice', 0)
            last_volume = info.get('regularMarketVolume', 0)
            change = info.get('regularMarketChange', 0)
            change_pct = info.get('regularMarketChangePercent', 0)
        
        return {
            "symbol": "^GSPC",
            "name": "S&P 500",
            "last_price": last_close,
            "change": change,
            "change_pct": change_pct,
            "volume": last_volume,
            "timestamp": datetime.now().isoformat(),
            "market_state": "open" if datetime.now().hour in range(9, 16) else "closed"
        }
    except Exception as e:
        logger.error(f"Error obteniendo estado del mercado: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cache/clear")
async def clear_cache(pattern: str = "*"):
    """
    Limpiar caché del sistema.
    Solo para administradores.
    """
    cache_manager.invalidate(pattern)
    return {"message": f"Cache cleared for pattern: {pattern}"}

@app.get("/health")
async def health_check():
    """Health check del sistema"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "cache": "operational" if cache_manager.redis_client else "degraded",
            "llm": "operational",
            "market_data": "operational"
        }
    }



    
@app.get("/api/cache/stats")
async def get_cache_stats():
    """Obtener estadísticas del caché"""
    try:
        if cache_manager.redis_client:
            info = cache_manager.redis_client.info()
            return {
                'type': 'redis',
                'connected': True,
                'keys': cache_manager.redis_client.dbsize(),
                'memory_used': info.get('used_memory_human', 'N/A'),
                'hits': info.get('keyspace_hits', 0),
                'misses': info.get('keyspace_misses', 0),
                'hit_rate': round(info.get('keyspace_hits', 0) / max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0), 1) * 100, 2)
            }
        else:
            return {
                'type': 'memory',
                'connected': True,
                'keys': len(cache_manager.memory_cache),
                'memory_used': 'N/A'
            }
    except Exception as e:
        return {
            'type': 'unknown',
            'connected': False,
            'error': str(e)
        }


@app.post("/api/cache/clear")
async def clear_cache(
    pattern: str = Query("*", description="Patrón para limpiar (* para todo)"),
    confirm: bool = Query(False, description="Confirmar limpieza")
):
    """Limpiar caché"""
    if not confirm:
        return {
            'message': 'Agrega ?confirm=true para confirmar la limpieza',
            'pattern': pattern
        }
    
    cache_manager.invalidate(pattern)
    return {
        'message': f'Caché limpiado para patrón: {pattern}',
        'timestamp': datetime.now().isoformat()
    }

@app.post("/api/cache/warmup")
async def warmup_cache():
    """Pre-cargar caché con datos comunes"""
    try:
        logger.info("🔥 Iniciando warmup de caché...")
        
        # Pre-cargar predicciones comunes
        await get_advanced_prediction(horizon_days=7, include_similar_patterns=True)
        await get_advanced_prediction(horizon_days=15, include_similar_patterns=True)
        await get_advanced_prediction(horizon_days=30, include_similar_patterns=True)
        
        # Pre-cargar forecast data
        await get_forecast_data(horizon_days=15, method='ensemble')
        await get_forecast_data(horizon_days=30, method='ensemble')
        
        # Pre-cargar sentimiento
        await analyze_recent_sentiment_trend()
        
        logger.info("✅ Warmup completado")
        
        return {
            'message': 'Caché pre-cargado exitosamente',
            'items': [
                'Predicciones 7, 15, 30 días',
                'Forecast 15, 30 días',
                'Sentimiento últimas 3 semanas'
            ]
        }
        
    except Exception as e:
        logger.error(f"Error en warmup: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
# ==========================================
# NUEVO: ENDPOINT PARA PREDICCIONES AVANZADAS
# ==========================================
@app.get("/api/prediction/advanced")
async def get_advanced_prediction(
    horizon_days: int = Query(30, ge=1, le=365, description="Días a predecir"),
    include_similar_patterns: bool = Query(True, description="Incluir patrones similares"),
    user_level: UserLevel = Query(UserLevel.INTERMEDIATE, description="Nivel del usuario")
):
    """Predicción avanzada con caché - ahora incluye análisis de noticias"""
    try:
        cache_key = f"prediction_advanced_{horizon_days}_{include_similar_patterns}_{user_level.value}"
        
        ttl = CacheTTL.PREDICTIONS_SHORT if horizon_days <= 7 else CacheTTL.PREDICTIONS_LONG
        
        cached = cache_manager.get(cache_key)
        if cached:
            return cached
        
        # Computar predicción
        sentiment_analysis = await analyze_recent_sentiment_trend()
        similar_patterns = []
        
        if include_similar_patterns:
            similar_patterns = await find_similar_market_patterns(
                lookback_days=30,
                max_results=5
            )
        
        base_prediction = await prediction_service.predict(horizon_days, user_level)
        
        enriched_prediction = {
            **base_prediction.dict(),
            'sentiment_trend': sentiment_analysis,
            'similar_patterns': similar_patterns,
            'pattern_weights': calculate_pattern_weights(similar_patterns),
            'news_analysis': {
                'status': 'integrated',
                'message': 'Análisis de noticias integrado en patrones similares',
                'metrics': {
                    'patterns_with_news': len([p for p in similar_patterns if p.get('news_similarity', 0) > 0]),
                    'avg_news_similarity': round(
                        sum([p.get('news_similarity', 0) for p in similar_patterns]) / len(similar_patterns) 
                        if similar_patterns else 0, 3
                    )
                }
            },
            'cached_at': datetime.now().isoformat()
        }
        
        cache_manager.set(cache_key, enriched_prediction, ttl)
        
        return enriched_prediction
        
    except Exception as e:
        logger.error(f"Error en predicción avanzada: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/api/prediction/daily-forecast")
async def get_daily_forecast(
    days: int = Query(30, ge=1, le=90, description="Número de días a predecir"),
    method: str = Query("contextual_llm", description="Method: contextual_llm (Recomendado), robust_rf (Antiguo)")
):
    """
    Predicciones individuales día por día.
    - contextual_llm: (NUEVO) Usa LLM basado en 30 días de análisis de noticias/técnicos.
    - robust_rf: (ANTIGUO) Usa un modelo autoregresivo robusto.
    """
    
    # --- NUEVA LÓGICA: PREDICCIÓN CONTEXTUAL (LLM) ---
    if method == "contextual_llm":
        cache_key = f"daily_forecast_contextual_llm_{days}"
        cached = cache_manager.get(cache_key)
        if cached:
            logger.info(f"⚡ Retornando forecast CONTEXTUAL (LLM) desde caché")
            return cached
        
        logger.info(f"🔄 Generando forecast CONTEXTUAL (LLM): {days} días")
        
        try:
            # 1. Recopilar contexto histórico (¡Llama a la nueva función!)
            # Esta función AHORA llama a /api/analysis/date X veces (usando caché JSON)
            historical_context_str, last_real_price, last_real_date_obj = await market_service.build_historical_prompt_from_endpoint(days_back=days)
            
            if not historical_context_str:
                raise ValueError("No se pudo recopilar el contexto histórico.")

            # 2. Llamar al LLM para generar el forecast
            predictions_list = market_service.llm_synthesizer.generate_contextual_forecast(
                historical_context_str,
                last_real_price,
                days
            )
            
            if not predictions_list:
                raise ValueError("El LLM no devolvió predicciones válidas.")

            # 3. Procesar salida para el frontend (Añadir bandas de confianza)
            
            # Cargar datos históricos solo para calcular volatilidad
            hist_data_for_vol = data_manager.get_technical_data_range("SPY", datetime.now() - timedelta(days=90), datetime.now())
            hist_volatility = calculate_historical_volatility(hist_data_for_vol['Close'].values)
            
            all_predicted_prices = [p.get('predicted_price', last_real_price) for p in predictions_list]
            confidence_bands = calculate_confidence_bands(all_predicted_prices, hist_volatility, days)

            # Añadir bandas y formatear
            final_predictions = []
            for i, pred in enumerate(predictions_list):
                # Asegurarse de que los índices no se salgan de rango
                if i >= days or i >= len(confidence_bands['lower']): 
                    break 
                
                pred_price = pred.get('predicted_price', last_real_price) # Fallback al último precio
                final_predictions.append({
                    'day': pred.get('day', i + 1),
                    'date': pred.get('date', (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')),
                    'predicted_price': float(pred_price) ,
                    'predicted_return': float(pred.get('change_pct', 0)), # El LLM ya lo da
                    'change_from_current': float(((pred_price - last_real_price) / last_real_price) * 100) if last_real_price != 0 else 0,
                    'confidence_lower': float(confidence_bands['lower'][i]),
                    'confidence_upper': float(confidence_bands['upper'][i])
                })
            
            if not final_predictions:
                 raise ValueError("La lista final de predicciones está vacía.")

            result = {
                'method': method,
                'total_days': days,
                'last_real_date': last_real_date_obj.strftime('%Y-%m-%d'),
                'last_real_price': last_real_price,
                'predictions': final_predictions,
                'summary': {
                    'final_predicted_price': final_predictions[-1]['predicted_price'],
                    'total_change_pct': final_predictions[-1]['change_from_current'],
                    'avg_daily_return': float(np.mean([p['predicted_return'] for p in final_predictions]))
                },
                'cached_at': datetime.now().isoformat()
            }
            
            cache_manager.set(cache_key, result, CacheTTL.FORECAST_DATA)
            return result

        except Exception as e:
            logger.error(f"Error generando forecast CONTEXTUAL (LLM): {e}")
            import traceback
            logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            # Fallback al método antiguo (robust_rf) si el LLM falla
            logger.warning("⚠️ Fallback al método 'robust_rf'...")
            method = "robust_rf" # Esto hará que continúe al siguiente bloque

    # --- LÓGICA ANTIGUA: ROBUST RF (AHORA ES EL FALLBACK) ---
    if method == "robust_rf":
        try:
            cache_key = f"daily_forecast_robust_{days}"
            
            cached = cache_manager.get(cache_key)
            if cached:
                logger.info(f"⚡ Retornando daily forecast ROBUSTO desde caché")
                return cached
            
            logger.info(f"🔄 Generando daily forecast ROBUSTO: {days} días")

            # --- 1. ENTRENAR EL MODELO UNA SOLA VEZ ---
            logger.info("🔧 Entrenando modelo RandomForest...")
            try:
                features, target = await prediction_service._prepare_training_data()
                prediction_service._train_model(features, target)
                logger.info("✅ Modelo entrenado y listo")
            except Exception as e:
                logger.error(f"Error fatal entrenando modelo: {e}")
                raise HTTPException(status_code=500, detail="Error al entrenar el modelo de predicción")

            # --- 2. OBTENER DATOS HISTÓRICOS Y ÚLTIMAS FEATURES ---
            end_date = datetime.now()
            start_date = end_date - timedelta(days=200) # Necesitamos 200 días para los indicadores
            
            historical_data_full = data_manager.get_technical_data_range("^GSPC", start_date, end_date)
            if historical_data_full.empty:
                raise ValueError("No hay datos históricos disponibles")
            
            # Calcular indicadores para TODOS los datos históricos
            logger.info("📊 Calculando indicadores para datos históricos...")
            # (AQUÍ ESTÁ LA CORRECCIÓN DE TU ERROR ANTERIOR)
            historical_data_with_ta = market_service.technical_fetcher._calculate_technical_indicators(historical_data_full)
            
            # Listas para simulación
            simulated_data = historical_data_with_ta.copy()
            daily_predictions = []
            
            last_real_date = simulated_data.index[-1]
            last_real_price = float(simulated_data['Close'].iloc[-1])

            logger.info(f"📈 Último precio real (RF): ${last_real_price:.2f} en {last_real_date.strftime('%Y-%m-%d')}")

            # --- 3. BUCLE AUTOREGRESIVO ---
            for day in range(1, days + 1):
                
                # Obtener las features más recientes de nuestros datos simulados
                last_row = simulated_data.iloc[-1]
                
                current_features_array = np.array([
                    last_row.get('RSI', 50),
                    last_row.get('MACD', 0.0),
                    last_row.get('BB_width', 0.0),
                    last_row.get('Volume_Ratio', 1.0)
                ])
                
                # Escalar y predecir el *retorno* del siguiente día
                scaled_features = prediction_service.scaler.transform(current_features_array.reshape(1, -1))
                predicted_return = prediction_service.model.predict(scaled_features)[0]
                
                # Calcular nuevo precio
                current_price = last_row['Close']
                new_price = current_price * (1 + predicted_return)
                
                # --- 4. SIMULAR NUEVA FILA DE DATOS ---
                new_date = last_real_date + timedelta(days=day)
                
                new_row = pd.Series({
                    'Open': current_price,
                    'High': new_price if new_price > current_price else current_price,
                    'Low': new_price if new_price < current_price else current_price,
                    'Close': new_price,
                    'Volume': last_row['Volume'] # Asumir volumen constante
                }, name=new_date)
                
                # Añadir la nueva fila a los datos simulados
                simulated_data = pd.concat([simulated_data, new_row.to_frame().T])
                
                # --- 5. RECALCULAR INDICADORES ---
                simulated_data = market_service.technical_fetcher._calculate_technical_indicators(simulated_data)
                
                # Guardar la predicción
                change_from_current = ((new_price - last_real_price) / last_real_price) * 100
                
                daily_predictions.append({
                    'day': day,
                    'date': new_date.strftime('%Y-%m-%d'),
                    'predicted_price': float(new_price),
                    'predicted_return': float(predicted_return),
                    'change_from_current': float(change_from_current)
                })

            logger.info(f"✅ Secuencia autoregresiva (RF) generada para {days} días")

            # --- 6. GENERAR BANDAS DE CONFIANZA ---
            all_predictions = [p['predicted_price'] for p in daily_predictions]
            hist_volatility = calculate_historical_volatility(historical_data_full['Close'].values)
            confidence_bands = calculate_confidence_bands(all_predictions, hist_volatility, days)
            
            # Añadir bandas a las predicciones
            for i, pred in enumerate(daily_predictions):
                pred['confidence_lower'] = float(confidence_bands['lower'][i])
                pred['confidence_upper'] = float(confidence_bands['upper'][i])
                
            result = {
                'method': method,
                'total_days': days,
                'last_real_date': last_real_date.strftime('%Y-%m-%d'),
                'last_real_price': last_real_price,
                'predictions': daily_predictions, 
                'summary': {
                    'final_predicted_price': daily_predictions[-1]['predicted_price'],
                    'total_change_pct': daily_predictions[-1]['change_from_current'],
                    'avg_daily_return': float(np.mean([p['predicted_return'] for p in daily_predictions]))
                },
                'cached_at': datetime.now().isoformat()
            }
            
            cache_manager.set(cache_key, result, CacheTTL.FORECAST_DATA)
            
            logger.info(f"✅ Forecast robusto (RF) generado y guardado en caché")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generando daily forecast ROBUSTO: {e}")
            import traceback
            logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))
    
    else:
        raise HTTPException(status_code=400, detail=f"Método '{method}' no reconocido. Use 'contextual_llm' o 'robust_rf'.")
    
@app.get("/api/prediction/forecast-data")
async def get_forecast_data(
    horizon_days: int = Query(15, ge=1, le=90, description="Días a predecir"),
    method: str = Query("ensemble", description="Método: linear, fourier, ensemble")
):
    """Forecast data con caché agresivo"""
    try:
        cache_key = f"forecast_{horizon_days}_{method}"
        
        cached = cache_manager.get(cache_key)
        if cached:
            logger.info(f"⚡ Retornando forecast desde caché")
            return cached
        
        logger.info(f"🔄 Generando forecast: {horizon_days} días, método: {method}")
        
        # Obtener datos históricos recientes
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        historical_data = data_manager.get_technical_data_range("SPY", start_date, end_date)
        
        if historical_data.empty:
            raise ValueError("No hay datos históricos disponibles")
        
        prices = historical_data['Close'].values
        dates = historical_data.index
        
        # Generar predicciones
        if method == "linear":
            predictions = predict_linear_trend(prices, horizon_days)
        elif method == "fourier":
            predictions = predict_fourier_series(prices, horizon_days)
        else:  # ensemble
            pred_linear = predict_linear_trend(prices, horizon_days)
            pred_fourier = predict_fourier_series(prices, horizon_days)
            predictions = (pred_linear + pred_fourier) / 2
        
        # Generar fechas futuras
        future_dates = [end_date + timedelta(days=i+1) for i in range(horizon_days)]
        
        # Calcular intervalos de confianza
        volatility = calculate_historical_volatility(prices)
        confidence_intervals = calculate_confidence_bands(predictions, volatility, horizon_days)
        
        # Formatear respuesta
        forecast_data = []
        for i, (date, price) in enumerate(zip(future_dates, predictions)):
            forecast_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'predicted_price': float(price),
                'confidence_lower': float(confidence_intervals['lower'][i]),
                'confidence_upper': float(confidence_intervals['upper'][i])
            })
        
        result = {
            'method': method,
            'horizon_days': horizon_days,
            'last_real_date': dates[-1].strftime('%Y-%m-%d'),
            'last_real_price': float(prices[-1]),
            'forecast': forecast_data,
            'metadata': {
                'volatility': float(volatility),
                'trend': detect_trend_direction(prices),
                'confidence_level': 0.95
            },
            'cached_at': datetime.now().isoformat()
        }
        
        # Guardar en caché
        cache_manager.set(cache_key, result, CacheTTL.FORECAST_DATA)
        
        logger.info(f"✅ Forecast generado y guardado en caché")
        
        return result
        
    except Exception as e:
        logger.error(f"Error generando forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/api/prediction/daily-forecast")
async def get_daily_forecast(
    days: int = Query(7, ge=1, le=365, description="Días a predecir"),
    method: str = Query("contextual_llm", description="Método de predicción")
):
    """
    Predicción diaria con análisis de mercado e hipótesis
    """
    try:
        # Obtener la predicción existente
        forecast_data = await prediction_service.generate_daily_forecast(days, method)
        
        # Generar análisis del mercado anterior e hipótesis
        market_analysis = await generate_market_analysis_and_hypothesis()
        
        # Enriquecer la respuesta
        enriched_response = {
            **forecast_data.dict(),
            "market_analysis": market_analysis,
            "analysis_generated_at": datetime.now().isoformat()
        }
        
        return enriched_response
        
    except Exception as e:
        logger.error(f"Error en predicción diaria: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_market_analysis_and_hypothesis():
    """
    Genera un análisis del mercado reciente e hipótesis de lo que podría pasar
    """
    try:
        # Obtener datos de los últimos 30 días
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Obtener datos del mercado
        market_data = data_manager.get_technical_data_range("SPY", start_date, end_date)
        
        if market_data.empty:
            return {
                "recent_performance": "Datos no disponibles",
                "key_observations": [],
                "market_hypothesis": "No hay suficiente datos históricos para generar una hipótesis",
                "confidence_level": "baja"
            }
        
        # Calcular métricas clave
        recent_performance = calculate_recent_performance(market_data)
        key_observations = identify_key_observations(market_data)
        market_hypothesis = generate_market_hypothesis(market_data, key_observations)
        
        return {
            "recent_performance": recent_performance,
            "key_observations": key_observations,
            "market_hypothesis": market_hypothesis,
            "confidence_level": "media",  # baja, media, alta
            "analysis_period": f"Últimos {len(market_data)} días hábiles"
        }
        
    except Exception as e:
        logger.error(f"Error generando análisis de mercado: {e}")
        return {
            "recent_performance": "Error en análisis",
            "key_observations": [],
            "market_hypothesis": "No se pudo generar análisis en este momento",
            "confidence_level": "baja"
        }

def calculate_recent_performance(market_data):
    """Calcula el desempeño reciente del mercado"""
    if len(market_data) < 2:
        return "Datos insuficientes"
    
    first_close = market_data['Close'].iloc[0]
    last_close = market_data['Close'].iloc[-1]
    total_change = ((last_close - first_close) / first_close) * 100
    
    # Calcular volatilidad
    daily_returns = market_data['Close'].pct_change().dropna()
    volatility = daily_returns.std() * 100
    
    performance_text = ""
    
    if total_change > 5:
        performance_text = f"fuerte alza de {total_change:.1f}%"
    elif total_change > 2:
        performance_text = f"moderada alza de {total_change:.1f}%"
    elif total_change > -2:
        performance_text = f"estabilidad con {total_change:.1f}% de cambio"
    elif total_change > -5:
        performance_text = f"moderada baja de {abs(total_change):.1f}%"
    else:
        performance_text = f"fuerte baja de {abs(total_change):.1f}%"
    
    return f"El mercado ha tenido una {performance_text} en las últimas semanas con volatilidad del {volatility:.1f}%"

# Servir archivos estáticos del frontend
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")

@app.get("/{full_path:path}")
async def catch_all(frontend_path: str):
    """Manejar todas las rutas del frontend"""
    if frontend_path.startswith("api/"):
        raise HTTPException(status_code=404)
    
    # Servir archivos estáticos o el index.html
    file_path = f"frontend/{frontend_path}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return FileResponse("frontend/index.html")
        

def identify_key_observations(market_data):
    """Identifica observaciones clave del mercado"""
    observations = []
    
    try:
        # 1. Tendencia general
        price_change = ((market_data['Close'].iloc[-1] - market_data['Close'].iloc[0]) / market_data['Close'].iloc[0]) * 100
        if price_change > 3:
            observations.append("📈 Tendencia alcista en el período analizado")
        elif price_change < -3:
            observations.append("📉 Tendencia bajista en el período analizado")
        else:
            observations.append("➡️ Mercado lateral sin dirección clara")
        
        # 2. Volumen de trading
        avg_volume = market_data['Volume'].mean()
        if avg_volume > 4000000000:  # 4B
            observations.append("💰 Alto volumen de operaciones, mucho interés")
        elif avg_volume < 2000000000:  # 2B
            observations.append("💤 Bajo volumen, mercado tranquilo")
        
        # 3. Velocidad de movimientos
        daily_changes = market_data['Close'].pct_change().abs()
        avg_daily_move = daily_changes.mean() * 100
        if avg_daily_move > 1.5:
            observations.append("⚡ Alta volatilidad diaria")
        elif avg_daily_move < 0.5:
            observations.append("🛌 Baja volatilidad, movimientos suaves")
        
        # 4. Patrón reciente (últimos 5 días)
        if len(market_data) >= 5:
            recent_trend = market_data['Close'].iloc[-5:].pct_change().sum() * 100
            if recent_trend > 2:
                observations.append("🚀 Momentum positivo reciente")
            elif recent_trend < -2:
                observations.append("🎯 Momentum negativo reciente")
        
        return observations[:4]  # Máximo 4 observaciones
        
    except Exception as e:
        logger.error(f"Error identificando observaciones: {e}")
        return ["📊 Análisis técnico no disponible temporalmente"]

def generate_market_hypothesis(market_data, observations):
    """Genera una hipótesis comprensible sobre el mercado"""
    try:
        # Análisis simplificado para hipótesis
        total_change = ((market_data['Close'].iloc[-1] - market_data['Close'].iloc[0]) / market_data['Close'].iloc[0]) * 100
        recent_5d_change = ((market_data['Close'].iloc[-1] - market_data['Close'].iloc[-5]) / market_data['Close'].iloc[-5]) * 100 if len(market_data) >= 5 else 0
        
        # Base de la hipótesis
        if total_change > 5 and recent_5d_change > 1:
            hypothesis = "El mercado muestra fortaleza con tendencia alcista sostenida. Podría continuar subiendo si se mantiene el optimismo."
        elif total_change > 5 and recent_5d_change < -1:
            hypothesis = "Tras una buena racha, el mercado muestra signos de pausa. Podría consolidarse antes de decidir siguiente dirección."
        elif total_change < -5 and recent_5d_change < -1:
            hypothesis = "El mercado está en corrección. Podría buscar un piso de soporte antes de intentar recuperarse."
        elif total_change < -5 and recent_5d_change > 1:
            hypothesis = "Tras una corrección, el mercado muestra signos de recuperación. Podría estabilizarse en los próximos días."
        elif abs(total_change) < 3 and abs(recent_5d_change) < 1:
            hypothesis = "Mercado en fase de consolidación sin dirección clara. Podría mantenerse lateral hasta nuevo catalizador."
        else:
            hypothesis = "El mercado muestra comportamiento mixto. Se recomienda observar próximos movimientos para identificar tendencia."
        
        # Añadir contexto basado en observaciones
        if "Alto volumen" in " ".join(observations):
            hypothesis += " El alto volumen sugiere convicción en la dirección actual."
        if "Baja volatilidad" in " ".join(observations):
            hypothesis += " La baja volatilidad indica ambiente de calma en el mercado."
        
        return hypothesis
        
    except Exception as e:
        logger.error(f"Error generando hipótesis: {e}")
        return "Basado en el análisis reciente, el mercado muestra comportamiento típico. Se recomienda monitorear factores económicos clave para identificar próximas tendencias."

def predict_linear_trend(prices: np.ndarray, horizon: int) -> np.ndarray:
    """Predicción usando regresión lineal simple"""
    try:
        x = np.arange(len(prices))
        
        # Regresión lineal
        coeffs = np.polyfit(x, prices, 1)
        slope, intercept = coeffs
        
        # Proyectar hacia el futuro
        future_x = np.arange(len(prices), len(prices) + horizon)
        predictions = slope * future_x + intercept
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error en predicción lineal: {e}")
        # Fallback: mantener último precio
        return np.full(horizon, prices[-1])


def predict_fourier_series(prices: np.ndarray, horizon: int, n_harmonics: int = 3) -> np.ndarray:
    """Predicción usando series de Fourier"""
    try:
        from scipy.fft import fft, ifft
        
        # Normalizar precios
        prices_normalized = (prices - prices.mean()) / prices.std()
        
        # FFT
        fft_result = fft(prices_normalized)
        frequencies = np.fft.fftfreq(len(prices))
        
        # Mantener solo las frecuencias principales
        fft_filtered = fft_result.copy()
        threshold = np.sort(np.abs(fft_result))[-n_harmonics]
        fft_filtered[np.abs(fft_filtered) < threshold] = 0
        
        # Reconstruir señal
        reconstructed = np.real(ifft(fft_filtered))
        
        # Extrapolar usando las frecuencias dominantes
        period = 1 / np.abs(frequencies[np.argsort(np.abs(fft_result))[-1]])
        if np.isnan(period) or np.isinf(period):
            period = len(prices)
        
        # Generar predicciones
        future_indices = np.arange(len(prices), len(prices) + horizon)
        predictions_normalized = np.array([
            reconstructed[int(i % len(reconstructed))] for i in future_indices
        ])
        
        # Desnormalizar
        predictions = predictions_normalized * prices.std() + prices.mean()
        
        # Aplicar tendencia de últimos días
        recent_trend = (prices[-1] - prices[-10]) / 10 if len(prices) >= 10 else 0
        for i in range(len(predictions)):
            predictions[i] += recent_trend * (i + 1)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error en predicción Fourier: {e}")
        # Fallback: tendencia lineal simple
        return predict_linear_trend(prices, horizon)


def calculate_historical_volatility(prices: np.ndarray) -> float:
    """Calcular volatilidad histórica"""
    try:
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        return volatility
    except:
        return 0.01  # 1% por defecto


def calculate_confidence_bands(predictions: np.ndarray, volatility: float, horizon: int) -> Dict:
    """Calcular bandas de confianza"""
    try:
        # Confianza del 95% (1.96 desviaciones estándar)
        confidence_multiplier = 1.96
        
        # La incertidumbre aumenta con el horizonte
        lower_band = []
        upper_band = []
        
        for i, pred in enumerate(predictions):
            # Error aumenta con la raíz cuadrada del tiempo
            time_adjusted_volatility = volatility * np.sqrt(i + 1)
            error_margin = pred * time_adjusted_volatility * confidence_multiplier
            
            lower_band.append(pred - error_margin)
            upper_band.append(pred + error_margin)
        
        return {
            'lower': np.array(lower_band),
            'upper': np.array(upper_band)
        }
        
    except Exception as e:
        logger.error(f"Error calculando bandas: {e}")
        return {
            'lower': predictions * 0.95,
            'upper': predictions * 1.05
        }


def detect_trend_direction(prices: np.ndarray) -> str:
    """Detectar dirección de tendencia"""
    try:
        if len(prices) < 10:
            return 'neutral'
        
        # Comparar promedio de últimos 5 días vs anteriores
        recent_avg = np.mean(prices[-5:])
        previous_avg = np.mean(prices[-10:-5])
        
        change = (recent_avg - previous_avg) / previous_avg
        
        if change > 0.01:
            return 'uptrend'
        elif change < -0.01:
            return 'downtrend'
        else:
            return 'neutral'
            
    except:
        return 'neutral'

async def analyze_recent_sentiment_trend():
    """Análisis de sentimiento con caché"""
    cache_key = "sentiment_trend_3weeks"
    
    cached = cache_manager.get(cache_key)
    if cached:
        return cached
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=21)
        
        weekly_sentiments = []
        
        for week in range(3):
            week_end = end_date - timedelta(days=week * 7)
            week_start = week_end - timedelta(days=7)
            
            week_data = await market_service.analyze_date(week_start, UserLevel.INTERMEDIATE)
            
            if week_data.news:
                avg_sentiment = np.mean([n.sentiment_score for n in week_data.news])
            else:
                avg_sentiment = 0
            
            if week_data.social:
                social_sentiment = np.mean([s.sentiment_average for s in week_data.social])
            else:
                social_sentiment = 0
            
            weekly_sentiments.append({
                'week': f'Semana {3 - week}',
                'start_date': week_start.strftime('%Y-%m-%d'),
                'end_date': week_end.strftime('%Y-%m-%d'),
                'news_sentiment': round(avg_sentiment, 3),
                'social_sentiment': round(social_sentiment, 3),
                'combined_sentiment': round((avg_sentiment + social_sentiment) / 2, 3)
            })
        
        sentiments = [w['combined_sentiment'] for w in weekly_sentiments]
        trend = 'improving' if sentiments[0] < sentiments[-1] else 'declining' if sentiments[0] > sentiments[-1] else 'stable'
        
        result = {
            'weekly_data': weekly_sentiments,
            'trend': trend,
            'current_sentiment': sentiments[-1],
            'sentiment_change': round(sentiments[-1] - sentiments[0], 3)
        }
        
        cache_manager.set(cache_key, result, CacheTTL.SENTIMENT_TREND)
        
        return result
        
    except Exception as e:
        logger.error(f"Error analizando sentimiento reciente: {e}")
        return {
            'weekly_data': [],
            'trend': 'unknown',
            'current_sentiment': 0,
            'sentiment_change': 0
        }
async def find_similar_market_patterns(lookback_days: int = 30, max_results: int = 5):
    """Búsqueda de patrones con caché - versión corregida"""
    cache_key = f"similar_patterns_{lookback_days}_{max_results}"
    
    cached = cache_manager.get(cache_key)
    if cached:
        return cached
    
    try:
        logger.info(f"🔍 Buscando patrones similares...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Obtener datos actuales
        current_data = data_manager.get_technical_data_range("^GSPC", start_date, end_date)
        
        if current_data.empty:
            return []
        
        current_pattern = calculate_pattern_features(current_data)
        
        # Obtener noticias SOLO si están disponibles (opcional)
        current_news = None
        try:
            peak_change_date_current = find_peak_change_date(current_data)
            current_news = await get_news_similarity_data(peak_change_date_current, peak_change_date_current)
        except Exception as e:
            logger.warning(f"No se pudieron obtener noticias actuales: {e}")
        
        search_start = end_date - timedelta(days=1825)
        historical_data = data_manager.get_technical_data_range("^GSPC", search_start, end_date - timedelta(days=lookback_days + 1))
        
        if historical_data.empty:
            return []
        
        similar_patterns = []
        
        for i in range(len(historical_data) - lookback_days):
            window_data = historical_data.iloc[i:i+lookback_days]
            
            if len(window_data) < lookback_days:
                continue
            
            pattern_start = window_data.index[0]
            pattern_end = window_data.index[-1]
            
            pattern_features = calculate_pattern_features(window_data)
            technical_similarity = calculate_pattern_similarity(current_pattern, pattern_features)
            
            # PRIMERO: Filtro técnico (como antes)
            if technical_similarity > 0.7:
                historical_news = None
                news_similarity = 0.0
                
                # Intentar obtener noticias históricas (opcional)
                try:
                    peak_change_date_historical = find_peak_change_date(window_data)
                    historical_news = await get_news_similarity_data(peak_change_date_historical, peak_change_date_historical)
                    
                    # Calcular similitud de noticias solo si tenemos datos de ambas partes
                    if current_news and historical_news:
                        news_similarity = calculate_news_similarity(current_news, historical_news)
                    else:
                        news_similarity = 0.5  # Valor neutral si faltan datos de noticias
                except Exception as e:
                    logger.warning(f"No se pudieron obtener noticias históricas para patrón {pattern_start}: {e}")
                    news_similarity = 0.5  # Valor neutral si hay error
                
                # CALCULAR SCORE COMBINADO CON FALBACK
                if current_news and historical_news:
                    # Si tenemos datos de noticias completos, usar combinación
                    combined_similarity = (technical_similarity * 0.7) + (news_similarity * 0.3)
                    similarity_threshold = 0.65
                else:
                    # Si faltan datos de noticias, usar solo similitud técnica
                    combined_similarity = technical_similarity
                    similarity_threshold = 0.7  # Umbral más alto para compensar
                
                if combined_similarity > similarity_threshold:
                    post_pattern_start = historical_data.index[i + lookback_days]
                    post_pattern_end = post_pattern_start + timedelta(days=30)
                    post_pattern_data = data_manager.get_technical_data_range("^GSPC", post_pattern_start, post_pattern_end)
                    
                    if not post_pattern_data.empty:
                        post_behavior = analyze_post_pattern_behavior(post_pattern_data)
                        
                        similar_patterns.append({
                            'pattern_start': pattern_start.strftime('%Y-%m-%d'),
                            'pattern_end': pattern_end.strftime('%Y-%m-%d'),
                            'similarity_score': round(combined_similarity, 3),
                            'technical_similarity': round(technical_similarity, 3),
                            'news_similarity': round(news_similarity, 3) if current_news and historical_news else None,
                            'pattern_features': pattern_features,
                            'post_pattern_behavior': post_behavior,
                            'key_news_topics': historical_news.get('top_topics', []) if historical_news else [],
                            'has_news_analysis': current_news is not None and historical_news is not None,
                            'sentiment_comparison': {
                                'current_sentiment': current_news.get('avg_sentiment', 0) if current_news else 0,
                                'historical_sentiment': historical_news.get('avg_sentiment', 0) if historical_news else 0,
                                'sentiment_difference': round(
                                    (current_news.get('avg_sentiment', 0) if current_news else 0) - 
                                    (historical_news.get('avg_sentiment', 0) if historical_news else 0), 3
                                ) if current_news and historical_news else None
                            }
                        })
        
        similar_patterns.sort(key=lambda x: x['similarity_score'], reverse=True)
        result = similar_patterns[:max_results]
        
        cache_manager.set(cache_key, result, CacheTTL.SIMILAR_PATTERNS)
        
        logger.info(f"✅ Encontrados {len(result)} patrones similares")
        
        return result
        
    except Exception as e:
        logger.error(f"Error buscando patrones similares: {e}")
        return []
def find_peak_change_date(data):
    """Encuentra la fecha con el mayor cambio porcentual absoluto en el dataset"""
    if 'change_pct' in data.columns:
        peak_index = data['change_pct'].abs().idxmax()
    else:
        # Calcular cambios si no existe la columna
        data = data.copy()
        data['change_pct'] = data['close'].pct_change().abs()
        peak_index = data['change_pct'].idxmax()
    
    return peak_index

def calculate_sentiment_from_relevance(relevance_score: float) -> float:
        """Convierte puntuación de relevancia a sentimiento (-1 a 1)"""
        # Mapear relevancia 0-1 a sentimiento -1 a 1
        # Asumimos que mayor relevancia puede indicar mayor impacto (positivo o negativo)
        # Esto es una aproximación - podrías refinar basado en tu data real
        return (relevance_score - 0.5) * 2  # Convierte 0-1 a -1 a 1

@app.delete("/api/cache/patterns")
async def clear_patterns_cache():
    """Endpoint para limpiar cache de patrones similares"""
    try:
        deleted_count = cache_manager.delete_pattern("similar_patterns_*")
        return {"message": f"✅ Cache limpiado: {deleted_count} entradas de patrones"}
    except Exception as e:
        logger.error(f"Error limpiando cache: {e}")
        raise HTTPException(status_code=500, detail="Error limpiando cache")


async def get_news_similarity_data(start_date: datetime, end_date: datetime) -> dict:
    """Obtiene y procesa datos de noticias con caché"""
    cache_key = f"news_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    
    cached = cache_manager.get(cache_key)
    if cached:
        return cached
    
    try:
        news_service = NewsService()
        news_articles = await news_service.get_top_sp500_news_combined(
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
        
        if not news_articles:
            result = {
                'avg_sentiment': 0,
                'avg_impact': 0,
                'top_topics': [],
                'total_articles': 0
            }
            cache_manager.set(cache_key, result, CacheTTL.NEWS_DATA)
            return result
        
        # Procesar métricas (mismo código anterior)
        sentiment_scores = []
        impact_scores = []
        topics = []
        
        for article in news_articles:
            sentiment = calculate_sentiment_from_relevance(article['relevance_score'])
            sentiment_scores.append(sentiment)
            impact_scores.append(min(10, article['relevance_score']))
            topics.extend(extract_topics_from_title(article['title']))
        
        topic_freq = {}
        for topic in topics:
            topic_freq[topic] = topic_freq.get(topic, 0) + 1
        
        top_topics = sorted(topic_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        result = {
            'avg_sentiment': sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
            'avg_impact': sum(impact_scores) / len(impact_scores) if impact_scores else 0,
            'top_topics': [topic[0] for topic in top_topics],
            'total_articles': len(news_articles)
        }
        
        cache_manager.set(cache_key, result, CacheTTL.NEWS_DATA)
        return result
        
    except Exception as e:
        logger.warning(f"Error obteniendo datos de noticias: {e}")
        result = {
            'avg_sentiment': 0,
            'avg_impact': 0,
            'top_topics': [],
            'total_articles': 0
        }
        cache_manager.set(cache_key, result, CacheTTL.NEWS_DATA)
        return result
    
def extract_topics_from_title(title: str) -> list:
    """Extrae temas clave de un título de noticia"""
    # Palabras comunes a excluir
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    words = title.lower().split()
    topics = [word.strip('.,!?;:()[]{}"\'') for word in words 
              if word.strip('.,!?;:()[]{}"\'').isalpha() 
              and word not in stop_words 
              and len(word) > 2]
    
    return topics


def calculate_news_similarity(current_news: dict, historical_news: dict) -> float:
    """Calcula similitud entre conjuntos de noticias"""
    if not current_news or not historical_news:
        return 0.0
    
    similarity_score = 0.0
    factors_considered = 0
    
    # 1. Similitud de sentimiento (40%)
    if current_news.get('avg_sentiment') is not None and historical_news.get('avg_sentiment') is not None:
        sentiment_diff = abs(current_news['avg_sentiment'] - historical_news['avg_sentiment'])
        sentiment_similarity = max(0, 1 - sentiment_diff)  # 1 si son iguales, 0 si diferencia máxima
        similarity_score += sentiment_similarity * 0.4
        factors_considered += 1
    
    # 2. Similitud de temas (30%)
    current_topics = set(current_news.get('top_topics', []))
    historical_topics = set(historical_news.get('top_topics', []))
    
    if current_topics and historical_topics:
        topic_overlap = len(current_topics.intersection(historical_topics))
        topic_similarity = topic_overlap / max(len(current_topics), len(historical_topics))
        similarity_score += topic_similarity * 0.3
        factors_considered += 1
    
    # 3. Similitud de volumen/intensidad (30%)
    current_impact = current_news.get('avg_impact', 0)
    historical_impact = historical_news.get('avg_impact', 0)
    
    if current_impact > 0 and historical_impact > 0:
        impact_similarity = 1 - abs(current_impact - historical_impact) / 10  # Normalizar a 0-10
        similarity_score += max(0, impact_similarity) * 0.3
        factors_considered += 1
    
    # Si no hay suficientes factores, retornar 0
    if factors_considered < 2:
        return 0.0
    
    return similarity_score



def calculate_pattern_features(data: pd.DataFrame) -> Dict:
    """Calcular características de un patrón de mercado"""
    try:
        # Calcular retornos
        returns = data['Close'].pct_change().dropna()
        
        # Calcular volatilidad
        volatility = returns.std()
        
        # Calcular tendencia (regresión lineal simple)
        x = np.arange(len(data))
        y = data['Close'].values
        slope = np.polyfit(x, y, 1)[0]
        
        # Calcular indicadores técnicos promedio
        tech_fetcher = TechnicalIndicatorsFetcher()
        data_with_indicators = tech_fetcher._calculate_technical_indicators(data.copy())
        
        return {
            'avg_return': float(returns.mean()),
            'volatility': float(volatility),
            'trend_slope': float(slope),
            'price_change_pct': float((data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100),
            'avg_volume': float(data['Volume'].mean()),
            'avg_rsi': float(data_with_indicators['RSI'].mean()),
            'max_drawdown': float(calculate_max_drawdown(data['Close']))
        }
    except Exception as e:
        logger.error(f"Error calculando features del patrón: {e}")
        return {}

@app.get("/api/news/date/{date_str}")
async def get_news_for_date(date_str: str, top_n: int = Query(5, ge=1, le=10)):
    """
    Obtiene noticias relevantes para el S&P 500 de una fecha específica
    """
    try:
        # Validar fecha
        target_date = datetime.strptime(date_str, '%Y-%m-%d')

        # Obtener noticias
        news_articles = await market_service.news_service.get_top_sp500_news_combined(date_str, top_n)
        
        return {
            "date": date_str,
            "total_articles": len(news_articles),
            "articles": news_articles
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de fecha inválido. Use YYYY-MM-DD")
    except Exception as e:
        logger.error(f"Error obteniendo noticias: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calcular máxima caída desde el pico"""
    try:
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        return float(drawdown.min())
    except:
        return 0.0


def calculate_pattern_similarity(pattern1: Dict, pattern2: Dict) -> float:
    """Calcular similitud entre dos patrones (0-1)"""
    try:
        if not pattern1 or not pattern2:
            return 0.0
        
        # Normalizar y comparar features clave
        features = ['avg_return', 'volatility', 'trend_slope', 'price_change_pct', 'avg_rsi']
        
        similarities = []
        for feature in features:
            if feature in pattern1 and feature in pattern2:
                val1 = pattern1[feature]
                val2 = pattern2[feature]
                
                # Evitar división por cero
                max_val = max(abs(val1), abs(val2), 1e-10)
                diff = abs(val1 - val2) / max_val
                similarity = 1 - min(diff, 1)
                
                similarities.append(similarity)
        
        return float(np.mean(similarities)) if similarities else 0.0
        
    except Exception as e:
        logger.error(f"Error calculando similitud: {e}")
        return 0.0


def analyze_post_pattern_behavior(data: pd.DataFrame) -> Dict:
    """Analizar comportamiento del mercado después de un patrón"""
    try:
        if data.empty:
            return {}
        
        # Calcular retorno total
        total_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
        
        # Calcular volatilidad
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5) * 100  # Anualizada
        
        # Mejor y peor día
        best_day = returns.max() * 100
        worst_day = returns.min() * 100
        
        # Dirección predominante
        positive_days = (returns > 0).sum()
        total_days = len(returns)
        direction = 'bullish' if positive_days / total_days > 0.55 else 'bearish' if positive_days / total_days < 0.45 else 'neutral'
        
        return {
            'total_return_pct': round(float(total_return), 2),
            'volatility_pct': round(float(volatility), 2),
            'best_day_pct': round(float(best_day), 2),
            'worst_day_pct': round(float(worst_day), 2),
            'positive_days_ratio': round(float(positive_days / total_days), 2),
            'direction': direction,
            'duration_days': len(data)
        }
        
    except Exception as e:
        logger.error(f"Error analizando comportamiento post-patrón: {e}")
        return {}


# En tu archivo donde está calculate_pattern_weights
def calculate_pattern_weights(similar_patterns):
    """Calcula pesos para patrones similares con manejo de None"""
    if not similar_patterns:
        return []
    
    weights = []
    for pattern in similar_patterns:
        # Manejar valores None en similarity_score
        similarity = pattern.get('similarity_score', 0) or 0
        
        # Asegurar que similarity es un número válido
        if similarity is None or not isinstance(similarity, (int, float)):
            similarity = 0
            
        # Normalizar a rango 0-1 y asegurar que es > 0
        weight = max(0.1, min(1.0, similarity))
        weights.append(weight)
    
    return weights
    

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║     S&P 500 INTELLIGENCE PLATFORM - BACKEND         ║
    ║                                                      ║
    ║  Sistema Multi-Agente de Análisis Financiero        ║
    ║  Version: 1.0.0                                     ║
    ║                                                      ║
    ║  Iniciando servidor...                              ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "sp500_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
