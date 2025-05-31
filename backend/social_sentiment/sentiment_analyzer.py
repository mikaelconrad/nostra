"""
Enhanced sentiment analysis engine for cryptocurrency social data
"""
import re
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import statistics

# Try to import advanced NLP libraries, fall back to basic analysis if not available
TEXTBLOB_AVAILABLE = False
NLTK_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available, using basic sentiment analysis")

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available, using basic sentiment analysis")

from .schemas import RedditPost, RedditComment, AggregatedSentiment
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logger import setup_logger

logger = setup_logger(__name__)

class CryptoSentimentAnalyzer:
    """Advanced sentiment analysis specifically for cryptocurrency content"""
    
    def __init__(self):
        """Initialize the sentiment analyzer"""
        # Initialize NLTK analyzer if available
        self.sia = None
        self.nltk_available = False
        try:
            import nltk
            from nltk.sentiment import SentimentIntensityAnalyzer
            self.sia = SentimentIntensityAnalyzer()
            self.nltk_available = True
        except (ImportError, LookupError) as e:
            logger.warning(f"NLTK not available: {e}")
            self.nltk_available = False
        
        # Crypto-specific sentiment lexicon
        self.crypto_sentiment_words = {
            # Extremely bullish
            'moon': 2.0, 'mooning': 2.0, 'lambo': 1.8, 'diamond_hands': 1.5,
            'hodl': 1.2, 'bullish': 1.5, 'pump': 1.3, 'rocket': 1.8, '🚀': 1.8,
            'ath': 1.5, 'breakout': 1.4, 'rally': 1.3, 'surge': 1.4,
            
            # Moderately bullish
            'buy': 0.8, 'accumulate': 0.9, 'long': 0.7, 'uptrend': 1.0,
            'support': 0.6, 'resistance': 0.3, 'institutional': 0.8,
            'adoption': 1.0, 'mainstream': 0.9, 'partnership': 0.7,
            
            # Bearish
            'dump': -1.3, 'crash': -1.8, 'bear': -1.2, 'bearish': -1.5,
            'fud': -1.4, 'panic': -1.6, 'sell': -0.8, 'short': -0.9,
            'drop': -1.0, 'fall': -0.8, 'decline': -0.9, 'correction': -0.7,
            
            # Extremely bearish
            'dead': -2.0, 'scam': -2.5, 'ponzi': -2.3, 'bubble': -1.6,
            'manipulation': -1.8, 'whale_dump': -1.9, 'rug_pull': -2.4,
            'paper_hands': -1.2, 'capitulation': -1.8,
            
            # Neutral/Analysis
            'analysis': 0.0, 'technical': 0.0, 'chart': 0.0, 'pattern': 0.0,
            'volume': 0.0, 'market_cap': 0.0, 'liquidity': 0.0
        }
        
        # Price prediction patterns
        self.price_patterns = [
            r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',  # $50,000 or $50000.00
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollars?|usd|\$)',  # 50000 dollars
            r'(\d+k)\s*(?:by|to|target)',  # 50k by
            r'(?:target|price|reach|hit)\s*(\d+(?:,\d{3})*)',  # target 50000
        ]
        
        # Confidence indicators
        self.confidence_words = {
            'definitely': 0.9, 'certainly': 0.9, 'absolutely': 0.9,
            'probably': 0.7, 'likely': 0.7, 'maybe': 0.4, 'possibly': 0.4,
            'might': 0.3, 'could': 0.3, 'perhaps': 0.3, 'uncertain': 0.2
        }
        
        # Meme and emoji patterns
        self.meme_patterns = {
            'diamond_hands': r'💎\s*🙌|diamond\s*hands',
            'paper_hands': r'🧻\s*🙌|paper\s*hands',
            'to_the_moon': r'🚀+|to\s*the\s*moon',
            'when_lambo': r'when\s*lambo|🏎️',
            'this_is_the_way': r'this\s*is\s*the\s*way',
            'number_go_up': r'number\s*go\s*up|📈'
        }
        
        logger.info("Initialized crypto sentiment analyzer")
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using multiple methods
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores and confidence
        """
        if not text or not text.strip():
            return {'sentiment_score': 0.0, 'confidence': 0.0, 'method': 'empty_text'}
        
        text = text.lower().strip()
        results = {}
        
        # Method 1: Crypto-specific lexicon analysis
        crypto_score = self._analyze_crypto_lexicon(text)
        results['crypto_lexicon'] = crypto_score
        
        # Method 2: TextBlob analysis (if available)
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                results['textblob'] = blob.sentiment.polarity
            except Exception as e:
                logger.debug(f"TextBlob analysis failed: {e}")
        
        # Method 3: NLTK VADER analysis (if available)
        if self.nltk_available and self.sia:
            try:
                scores = self.sia.polarity_scores(text)
                results['vader'] = scores['compound']
            except Exception as e:
                logger.debug(f"VADER analysis failed: {e}")
        
        # Combine results
        combined_score, confidence, method = self._combine_sentiment_scores(results, text)
        
        return {
            'sentiment_score': combined_score,
            'confidence': confidence,
            'method': method,
            'individual_scores': results
        }
    
    def _analyze_crypto_lexicon(self, text: str) -> float:
        """Analyze sentiment using crypto-specific word lexicon"""
        words = re.findall(r'\b\w+\b', text.lower())
        scores = []
        
        for word in words:
            if word in self.crypto_sentiment_words:
                scores.append(self.crypto_sentiment_words[word])
        
        if not scores:
            return 0.0
        
        # Weight by frequency and apply diminishing returns
        score = statistics.mean(scores)
        
        # Apply diminishing returns for extreme scores
        if score > 1.0:
            score = 1.0 + 0.5 * (score - 1.0)
        elif score < -1.0:
            score = -1.0 + 0.5 * (score + 1.0)
        
        # Normalize to [-1, 1] range
        return max(-1.0, min(1.0, score))
    
    def _combine_sentiment_scores(self, scores: Dict[str, float], text: str) -> Tuple[float, float, str]:
        """
        Combine multiple sentiment analysis results
        
        Returns:
            (combined_score, confidence, primary_method)
        """
        if not scores:
            return 0.0, 0.0, 'no_analysis'
        
        # Weight different methods
        weights = {
            'crypto_lexicon': 0.5,  # Higher weight for crypto-specific analysis
            'vader': 0.3,
            'textblob': 0.2
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        primary_method = 'crypto_lexicon'
        
        for method, score in scores.items():
            if method in weights:
                weight = weights[method]
                weighted_sum += score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0, 0.0, 'no_valid_analysis'
        
        combined_score = weighted_sum / total_weight
        
        # Calculate confidence based on agreement between methods
        confidence = self._calculate_confidence(scores, text)
        
        return combined_score, confidence, primary_method
    
    def _calculate_confidence(self, scores: Dict[str, float], text: str) -> float:
        """Calculate confidence based on method agreement and text features"""
        if len(scores) <= 1:
            return 0.5  # Low confidence with only one method
        
        # Calculate variance in scores (lower variance = higher confidence)
        score_values = list(scores.values())
        if len(score_values) > 1:
            variance = statistics.variance(score_values)
            agreement_confidence = max(0.0, 1.0 - variance)
        else:
            agreement_confidence = 0.5
        
        # Check for confidence indicators in text
        confidence_modifiers = 0.0
        for word, modifier in self.confidence_words.items():
            if word in text.lower():
                confidence_modifiers += modifier * 0.1  # Small boost/penalty
        
        # Check text length (longer text usually more reliable)
        length_confidence = min(1.0, len(text.split()) / 20.0)  # Max at 20 words
        
        # Combine confidence factors
        final_confidence = (
            agreement_confidence * 0.6 + 
            length_confidence * 0.3 + 
            confidence_modifiers * 0.1
        )
        
        return max(0.1, min(0.95, final_confidence))
    
    def extract_price_predictions(self, text: str) -> List[Dict]:
        """Extract price predictions from text"""
        predictions = []
        
        for pattern in self.price_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                price_str = match.group(1)
                
                # Clean and convert price
                try:
                    if 'k' in price_str.lower():
                        price = float(price_str.lower().replace('k', '')) * 1000
                    else:
                        price = float(price_str.replace(',', ''))
                    
                    # Extract context around the prediction
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    
                    predictions.append({
                        'price': price,
                        'original_text': price_str,
                        'context': context,
                        'position': match.start()
                    })
                except ValueError:
                    continue
        
        return predictions
    
    def detect_memes_and_patterns(self, text: str) -> Dict[str, bool]:
        """Detect crypto memes and patterns in text"""
        detected = {}
        
        for meme_name, pattern in self.meme_patterns.items():
            detected[meme_name] = bool(re.search(pattern, text, re.IGNORECASE))
        
        return detected
    
    def analyze_reddit_post(self, post: RedditPost) -> RedditPost:
        """Analyze sentiment for a Reddit post"""
        # Combine title and content for analysis
        full_text = f"{post.title} {post.content}"
        
        # Analyze sentiment
        sentiment_result = self.analyze_text_sentiment(full_text)
        
        # Extract price predictions
        price_predictions = self.extract_price_predictions(full_text)
        
        # Detect memes
        memes = self.detect_memes_and_patterns(full_text)
        
        # Update post with analysis results
        post.sentiment_score = sentiment_result['sentiment_score']
        post.confidence = sentiment_result['confidence']
        post.price_predictions = price_predictions
        
        # Determine sentiment label
        if post.sentiment_score > 0.1:
            post.sentiment_label = 'bullish'
        elif post.sentiment_score < -0.1:
            post.sentiment_label = 'bearish'
        else:
            post.sentiment_label = 'neutral'
        
        # Store additional analysis data
        if not hasattr(post, 'analysis_metadata'):
            post.analysis_metadata = {}
        
        post.analysis_metadata.update({
            'memes_detected': memes,
            'sentiment_method': sentiment_result['method'],
            'individual_scores': sentiment_result['individual_scores']
        })
        
        return post
    
    def analyze_reddit_comment(self, comment: RedditComment) -> RedditComment:
        """Analyze sentiment for a Reddit comment"""
        sentiment_result = self.analyze_text_sentiment(comment.body)
        
        comment.sentiment_score = sentiment_result['sentiment_score']
        comment.confidence = sentiment_result['confidence']
        
        # Determine sentiment label
        if comment.sentiment_score > 0.1:
            comment.sentiment_label = 'bullish'
        elif comment.sentiment_score < -0.1:
            comment.sentiment_label = 'bearish'
        else:
            comment.sentiment_label = 'neutral'
        
        return comment
    
    def aggregate_daily_sentiment(self, posts: List[RedditPost], 
                                 comments: List[RedditComment], 
                                 coin: str, date: str) -> AggregatedSentiment:
        """
        Aggregate sentiment data for a day
        
        Args:
            posts: List of analyzed Reddit posts
            comments: List of analyzed Reddit comments
            coin: Cryptocurrency symbol
            date: Date string (YYYY-MM-DD)
            
        Returns:
            AggregatedSentiment object
        """
        # Analyze posts if not already analyzed
        analyzed_posts = []
        for post in posts:
            if post.sentiment_score is None:
                post = self.analyze_reddit_post(post)
            analyzed_posts.append(post)
        
        # Analyze comments if not already analyzed
        analyzed_comments = []
        for comment in comments:
            if comment.sentiment_score is None:
                comment = self.analyze_reddit_comment(comment)
            analyzed_comments.append(comment)
        
        # Calculate aggregated metrics
        post_scores = [p.sentiment_score for p in analyzed_posts if p.sentiment_score is not None]
        comment_scores = [c.sentiment_score for c in analyzed_comments if c.sentiment_score is not None]
        
        # Weighted average (posts have higher weight than comments)
        all_scores = []
        for score in post_scores:
            all_scores.extend([score] * 3)  # Posts weighted 3x
        all_scores.extend(comment_scores)  # Comments weighted 1x
        
        reddit_sentiment_avg = statistics.mean(all_scores) if all_scores else 0.0
        
        # Count sentiment categories
        bullish_posts = sum(1 for p in analyzed_posts if p.sentiment_label == 'bullish')
        bearish_posts = sum(1 for p in analyzed_posts if p.sentiment_label == 'bearish')
        neutral_posts = sum(1 for p in analyzed_posts if p.sentiment_label == 'neutral')
        
        # Calculate total score (weighted by karma)
        total_score = sum(p.score for p in analyzed_posts) + sum(c.score for c in analyzed_comments)
        
        # Find notable posts (high score and strong sentiment)
        notable_posts = []
        for post in analyzed_posts:
            if (post.score > 100 and abs(post.sentiment_score or 0) > 0.5) or post.score > 1000:
                notable_posts.append(post.id)
        
        # Calculate confidence level
        confidences = [p.confidence for p in analyzed_posts if p.confidence is not None]
        confidences.extend([c.confidence for c in analyzed_comments if c.confidence is not None])
        
        confidence_level = statistics.mean(confidences) if confidences else 0.0
        
        # Determine signal strength
        abs_sentiment = abs(reddit_sentiment_avg)
        if abs_sentiment > 0.5 and confidence_level > 0.7:
            signal_strength = 'strong'
        elif abs_sentiment > 0.2 and confidence_level > 0.5:
            signal_strength = 'moderate'
        else:
            signal_strength = 'weak'
        
        aggregated = AggregatedSentiment(
            date=date,
            coin=coin,
            reddit_sentiment_avg=reddit_sentiment_avg,
            reddit_post_count=len(analyzed_posts),
            reddit_comment_count=len(analyzed_comments),
            reddit_total_score=total_score,
            reddit_bullish_posts=bullish_posts,
            reddit_bearish_posts=bearish_posts,
            reddit_neutral_posts=neutral_posts,
            combined_sentiment=reddit_sentiment_avg,  # Will be updated when market data is added
            confidence_level=confidence_level,
            signal_strength=signal_strength,
            notable_posts=notable_posts[:10],  # Top 10 notable posts
            last_updated=datetime.now(),
            data_sources=['reddit']
        )
        
        return aggregated

    def get_sentiment_summary(self, aggregated: AggregatedSentiment) -> str:
        """Generate human-readable sentiment summary"""
        sentiment = aggregated.combined_sentiment
        strength = aggregated.signal_strength
        
        if sentiment > 0.3:
            sentiment_desc = "strongly bullish"
        elif sentiment > 0.1:
            sentiment_desc = "moderately bullish"
        elif sentiment > -0.1:
            sentiment_desc = "neutral"
        elif sentiment > -0.3:
            sentiment_desc = "moderately bearish"
        else:
            sentiment_desc = "strongly bearish"
        
        post_ratio = ""
        total_posts = aggregated.reddit_post_count
        if total_posts > 0:
            bullish_pct = (aggregated.reddit_bullish_posts / total_posts) * 100
            bearish_pct = (aggregated.reddit_bearish_posts / total_posts) * 100
            post_ratio = f" ({bullish_pct:.0f}% bullish, {bearish_pct:.0f}% bearish)"
        
        return f"Reddit sentiment for {aggregated.coin} is {sentiment_desc} with {strength} signal strength{post_ratio}"