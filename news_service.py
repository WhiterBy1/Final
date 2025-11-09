import os
import json
from datetime import datetime, timedelta
from gnews import GNews
from openai import OpenAI as Groq
from dotenv import load_dotenv
import requests
import re
from typing import List, Dict, Optional

# --- CONFIGURACIÓN ---
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")

class NewsService:
    def __init__(self):
        if not GROQ_API_KEY or not NEWSAPI_KEY:
            raise ValueError("Asegúrate de tener GROQ_API_KEY y NEWSAPI_KEY en tu .env")
        
        self.groq_client = Groq(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        self.news_cache = {}  # Cache simple en memoria

    def is_financial_news(self, title: str, description: str) -> bool:
        """Filtra noticias realmente relacionadas con mercados financieros"""
        text = f"{title} {description}".lower()
        
        # Términos específicos del S&P 500 y mercados
        financial_terms = [
            's&p', 's&p 500', 'sp500', 'standard & poor', 
            'stock market', 'equities', 'trading session', 'market close',
            'dow jones', 'nasdaq', 'index', 'points', 'percent',
            'federal reserve', 'fed', 'interest rates', 'inflation',
            'earnings', 'quarterly results', 'wall street',
            'bull market', 'bear market', 'rally', 'selloff',
            'financial sector', 'banking', 'investment'
        ]
        
        return any(term in text for term in financial_terms)

    def analyze_article_with_groq(self, title: str, description: str) -> Dict:
        """Análisis mejorado con filtro más estricto"""
        if not description or not self.is_financial_news(title, description):
            return {"summary": "No relevante para S&P500", "score": 0}

        content_to_analyze = f"""
        Título: {title}
        Descripción: {description}
        
        Analiza esta noticia y determina si está DIRECTAMENTE relacionada con:
        - S&P 500, índices bursátiles US, mercados financieros
        - Empresas del S&P 500, resultados trimestrales
        - Factores que afectan al S&P 500 (Fed, economía, etc.)
        
        Responde SOLO si es relevante. Puntúa del 1-10.
        """
        
        prompt_messages = [
            {
                "role": "system", 
                "content": "Eres un analista financiero especializado en S&P 500. Sé muy estricto con la relevancia. Responde en JSON: {\"summary\":\"resumen\",\"relevance_score\":número,\"is_relevant\":true/false}"
            },
            {"role": "user", "content": content_to_analyze}
        ]
        
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=prompt_messages,
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            response_data = json.loads(chat_completion.choices[0].message.content)
            
            # Solo considerar si es relevante
            if not response_data.get("is_relevant", True):
                return {"summary": "No relevante según IA", "score": 0}
                
            return {
                "summary": response_data.get("summary", "N/A"), 
                "score": int(response_data.get("relevance_score", 0))
            }
        except Exception as e:
            print(f"  -> Error Groq: {e}")
            return {"summary": "Error en análisis IA.", "score": 0}

    def get_google_news_sp500(self, date_str: str, top_n: int = 10) -> List[Dict]:
        """Búsqueda más específica para Google News"""
        queries = [
            '"S&P 500" market close',
            'S&P500 earnings',
            '"Wall Street" trading',
            'stock market today',
            'Federal Reserve market',
            'Dow Jones Nasdaq'
        ]
        
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        all_articles = []
        
        for query in queries:
            try:
                google_news = GNews(
                    language='en', 
                    country='US', 
                    start_date=target_date - timedelta(days=1),
                    end_date=target_date,
                    max_results=5
                )
                articles = google_news.get_news(query)
                if articles:
                    all_articles.extend(articles)
            except Exception as e:
                print(f"Error en Google News para query '{query}': {e}")
        
        # Eliminar duplicados por URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.get('url') not in seen_urls:
                seen_urls.add(article.get('url'))
                unique_articles.append(article)
        
        return unique_articles[:top_n]

    def get_newsapi_sp500(self, date_str: str, top_n: int = 10) -> List[Dict]:
        """Búsqueda más específica para NewsAPI"""
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        from_date = target_date.strftime('%Y-%m-%d')
        to_date = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')

        specific_queries = [
            'S&P 500 AND (close OR performance OR rally OR decline)',
            '"Wall Street" AND (trading OR session OR stocks)',
            'Federal Reserve AND (stocks OR markets)',
            'earnings AND (beat OR miss OR results) AND S&P',
            'stock market AND (today OR update)'
        ]
        
        all_articles = []
        
        for query in specific_queries:
            url = (
                f"https://newsapi.org/v2/everything?"
                f"q={query}&"
                f"from={from_date}&to={to_date}&"
                f"language=en&sortBy=relevancy&pageSize=5&"
                f"apiKey={NEWSAPI_KEY}"
            )
            
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    all_articles.extend(articles)
            except Exception as e:
                print(f"Error NewsAPI para query '{query}': {e}")
        
        # Eliminar duplicados
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.get('url') not in seen_urls:
                seen_urls.add(article.get('url'))
                unique_articles.append(article)
        
        return unique_articles[:top_n]

    async def get_top_sp500_news_combined(self, date_str: str, top_n: int = 5) -> List[Dict]:
        """Obtiene las mejores noticias para una fecha específica"""
        # Verificar cache primero
        cache_key = f"news_{date_str}"
        if cache_key in self.news_cache:
            return self.news_cache[cache_key]
        
        print(f"Buscando noticias financieras para {date_str}...")
        
        google_articles = self.get_google_news_sp500(date_str, top_n=8 * 2)
        newsapi_articles = self.get_newsapi_sp500(date_str, top_n=8 * 2)
        
        print(f"Google News encontrados: {len(google_articles)}")
        print(f"NewsAPI encontrados: {len(newsapi_articles)}")
        
        all_articles = google_articles + newsapi_articles
        results = []
        
        for i, article in enumerate(all_articles):
            title = article.get('title', '')
            description = article.get('description', '') or article.get('content', '')
            
            print(f"Analizando artículo {i+1}: {title[:50]}...")
            
            # Filtro inicial rápido
            if not self.is_financial_news(title, description):
                print("  -> Descartado por filtro inicial")
                continue
            
            analysis = self.analyze_article_with_groq(title, description)
            
            # Solo incluir si tiene puntuación decente
            if analysis['score'] >= 3:
                results.append({
                    "title": title,
                    "summary": analysis['summary'],
                    "author": article.get('author', 'No disponible'),
                    "source": article.get('publisher', {}).get('title', 
                             article.get('source', {}).get('name', 'No disponible')),
                    "relevance_score": analysis['score'],
                    "url": article.get('url'),
                    "published_at": article.get('publishedAt', '')
                })
                print(f"  -> Añadido con score: {analysis['score']}")
            else:
                print(f"  -> Descartado por score bajo: {analysis['score']}")
        
        # Ordenar por relevancia y devolver top
        results_sorted = sorted(results, key=lambda x: x['relevance_score'], reverse=True)
        final_results = results_sorted[:top_n]
        
        # Guardar en cache (1 hora)
        self.news_cache[cache_key] = final_results
        
        return final_results

# Instancia global del servicio de noticias
news_service = NewsService()