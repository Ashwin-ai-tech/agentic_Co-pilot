# analytics_manager.py - FIXED WITH CENTRAL DB MANAGEMENT & STREAMING
import logging
import threading  # <-- ADDED
import time       # <-- ADDED
from queue import Queue   # <-- ADDED
from concurrent.futures import ThreadPoolExecutor # <-- ADDED
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import numpy as np
from dataclasses import dataclass
from collections import defaultdict, Counter
import re
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
from database import get_db  # Use the central DB manager

logger = logging.getLogger(__name__)

@dataclass
class AnalyticsConfig:
    """Configuration for analytics settings"""
    retention_days: int = 90
    top_queries_limit: int = 10
    session_timeout_minutes: int = 30
    confidence_threshold: float = 0.7
    feedback_weight_recent: float = 1.2
    feedback_weight_old: float = 0.8
    enable_visualizations: bool = True
    export_path: str = "./analytics_exports"
    real_time_update_interval: int = 300   # 5 minutes

# --- ADDED CLASS ---
class RealTimeAnalyticsStream:
    """Manages real-time data streaming to dashboard subscribers"""
    def __init__(self):
        self.subscribers = []
    
    def add_subscriber(self, queue):
        self.subscribers.append(queue)
    
    def remove_subscriber(self, queue):
        if queue in self.subscribers:
            self.subscribers.remove(queue)
    
    def broadcast(self, data):
        """Sends data to all active subscribers."""
        for queue in self.subscribers[:]:
            try:
                queue.put(data)
            except Exception:
                # Remove subscriber if queue is broken
                self.remove_subscriber(queue)
# -----------------

class AnalyticsManager:
    """
    Enhanced analytics manager for comprehensive system monitoring and insights
    with user and admin-focused features
    """
    
    def __init__(self, db_path: str, config: Optional[AnalyticsConfig] = None):
        self.db_path = db_path
        self.config = config or AnalyticsConfig()
        self._setup_export_directory()
        
        # --- ADDED STREAMING LOGIC ---
        self.realtime_stream = RealTimeAnalyticsStream()
        self.analytics_executor = ThreadPoolExecutor(max_workers=8)
        self.realtime_metrics = {}  # Holds the latest metrics
        # -----------------------------
    
    def init_tables(self):
        """Creates and migrates the necessary analytics tables within the app context."""
        self._ensure_analytics_tables()
        self._migrate_analytics_tables()
        # --- REMOVED REDUNDANT CALL ---
    
    def _setup_export_directory(self):
        """Create export directory if it doesn't exist"""
        Path(self.config.export_path).mkdir(parents=True, exist_ok=True)
    
    def _ensure_analytics_tables(self):
        """Ensure all analytics tables exist with proper schema"""
        conn = get_db()
        cursor = conn.cursor()
        
        # (All table creation SQL... unchanged)
        # Enhanced analytics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enhanced_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE DEFAULT CURRENT_DATE,
                total_queries INTEGER DEFAULT 0,
                successful_queries INTEGER DEFAULT 0,
                failed_queries INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0,
                avg_response_time REAL DEFAULT 0,
                kb_usage_count INTEGER DEFAULT 0,
                user_satisfaction_score REAL DEFAULT 0,
                top_queries TEXT,
                common_issues TEXT,
                system_health_score REAL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Query patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT UNIQUE,
                frequency INTEGER DEFAULT 1,
                avg_confidence REAL DEFAULT 0,
                success_rate REAL DEFAULT 0,
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                category TEXT
            )
        """)
        
        # User behavior table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_behavior (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT,
                query_count INTEGER DEFAULT 0,
                avg_session_duration REAL DEFAULT 0,
                preferred_categories TEXT,
                feedback_frequency REAL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_activity DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                retrieval_time REAL DEFAULT 0,
                generation_time REAL DEFAULT 0,
                total_time REAL DEFAULT 0,
                memory_usage REAL DEFAULT 0,
                cpu_usage REAL DEFAULT 0,
                active_sessions INTEGER DEFAULT 0
            )
        """)
        
        # User feedback insights table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                improvement_suggestions TEXT,
                sentiment_score REAL DEFAULT 0,
                urgency_level TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Knowledge base effectiveness table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kb_effectiveness (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kb_article_id TEXT,
                usage_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0,
                avg_confidence REAL DEFAULT 0,
                user_rating REAL DEFAULT 0,
                last_used DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()

    def _migrate_analytics_tables(self):
        """Migrate analytics tables to add missing columns"""
        try:
            conn = get_db()
            cursor = conn.cursor()
            
            # Check enhanced_analytics
            cursor.execute("PRAGMA table_info(enhanced_analytics)")
            columns = [column[1] for column in cursor.fetchall()]
            
            missing_columns = [
                ('user_satisfaction_score', 'REAL DEFAULT 0'),
                ('top_queries', 'TEXT'),
                ('common_issues', 'TEXT'),
                ('system_health_score', 'REAL DEFAULT 0')
            ]
            
            for column, col_type in missing_columns:
                if column not in columns:
                    logger.info(f"Adding {column} column to enhanced_analytics table...")
                    cursor.execute(f"ALTER TABLE enhanced_analytics ADD COLUMN {column} {col_type}")
            
            # Check performance_metrics
            cursor.execute("PRAGMA table_info(performance_metrics)")
            perf_columns = [column[1] for column in cursor.fetchall()]
            
            required_perf_columns = [
                ('retrieval_time', 'REAL DEFAULT 0'),
                ('generation_time', 'REAL DEFAULT 0'),
                ('total_time', 'REAL DEFAULT 0'),
                ('memory_usage', 'REAL DEFAULT 0'),
                ('cpu_usage', 'REAL DEFAULT 0'),
                ('active_sessions', 'INTEGER DEFAULT 0') # Corrected type
            ]
            
            for column, col_type in required_perf_columns:
                if column not in perf_columns:
                    logger.info(f"Adding {column} column to performance_metrics table...")
                    cursor.execute(f"ALTER TABLE performance_metrics ADD COLUMN {column} {col_type}")
            
            conn.commit()
            logger.info("Analytics table migration completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during analytics table migration: {e}")

    # --- REMOVED REDUNDANT FUNCTION ---
    # The _migrate_performance_metrics_table function was deleted
    # as its logic is already included in _migrate_analytics_tables.
    # ---------------------------------
    
    # ==================== DATA COLLECTION METHODS ====================
    
    # (All data collection methods... unchanged)
    
    def record_query_metrics(self, metrics: Dict[str, Any]):
        """Record detailed query metrics - FIXED VERSION"""
        try:
            conn = get_db()
            cursor = conn.cursor()
            
            query = metrics.get('query', '')
            response_time = metrics.get('response_time', 0)
            confidence = metrics.get('confidence', 0)
            success = metrics.get('success', False)
            kb_used = metrics.get('kb_used', False)
            session_id = metrics.get('session_id', '')
            user_id = metrics.get('user_id', 'default')
            clarification_asked = metrics.get('clarification_asked', False)
            exact_match = metrics.get('exact_match', False)
            
            today = datetime.now().date().isoformat()
            
            cursor.execute("SELECT id, total_queries, successful_queries, avg_confidence, avg_response_time, kb_usage_count FROM enhanced_analytics WHERE date = ?", (today,))
            existing_record = cursor.fetchone()
            
            if existing_record:
                record_id, current_total, current_successful, current_avg_conf, current_avg_time, current_kb_count = existing_record
                
                new_total = current_total + 1
                new_successful = current_successful + (1 if success else 0)
                new_avg_conf = ((current_avg_conf * current_total) + confidence) / new_total
                new_avg_time = ((current_avg_time * current_total) + response_time) / new_total
                new_kb_count = current_kb_count + (1 if kb_used else 0)
                
                cursor.execute("""
                    UPDATE enhanced_analytics 
                    SET total_queries = ?, successful_queries = ?, avg_confidence = ?, 
                        avg_response_time = ?, kb_usage_count = ?
                    WHERE id = ?
                """, (new_total, new_successful, new_avg_conf, new_avg_time, new_kb_count, record_id))
                
            else:
                cursor.execute("""
                    INSERT INTO enhanced_analytics 
                    (date, total_queries, successful_queries, failed_queries, 
                     avg_confidence, avg_response_time, kb_usage_count, user_satisfaction_score)
                    VALUES (?, 1, ?, ?, ?, ?, ?, ?)
                """, (
                    today,
                    1 if success else 0,
                    1 if not success else 0,
                    confidence,
                    response_time,
                    1 if kb_used else 0,
                    0.0
                ))
            
            if query and not clarification_asked:
                self._update_query_patterns(query, confidence, success, exact_match)
            
            if session_id:
                self._update_user_behavior(session_id, user_id, query_count=1)
            
            conn.commit()
            
            logger.debug(f"Recorded query metrics: {query[:50]}... (confidence: {confidence}, success: {success})")
            
        except Exception as e:
            logger.error(f"Error recording query metrics: {e}")
    
    def record_feedback(self, session_id: str, query: str, answer: str, rating: int, comment: str = ""):
        """Record user feedback with enhanced analytics - FIXED VERSION"""
        try:
            conn = get_db()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO feedback (session_id, query, answer, rating, comment)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, query, answer, rating, comment))
            
            self._update_user_behavior(session_id, "default", rating=rating)
            
            today = datetime.now().date().isoformat()
            
            cursor.execute("""
                SELECT AVG(rating) FROM feedback 
                WHERE DATE(timestamp) = ? AND rating > 0
            """, (today,))
            avg_rating_today = cursor.fetchone()[0] or 0
            
            satisfaction_score = (avg_rating_today / 5.0) * 100 if avg_rating_today > 0 else 0
            
            cursor.execute("""
                UPDATE enhanced_analytics 
                SET user_satisfaction_score = ?
                WHERE date = ?
            """, (satisfaction_score, today))
            
            self._generate_feedback_insights(session_id, comment, rating)
            
            conn.commit()
            
            logger.info(f"Recorded feedback: rating {rating} for session {session_id}, satisfaction: {satisfaction_score}")
            
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")

    # ==================== ANALYTICS RETRIEVAL METHODS ====================
    
    # (All analytics retrieval methods... unchanged)
    # e.g., get_comprehensive_analytics, _get_fallback_analytics,
    # get_performance_stats, get_user_dashboard, _generate_user_insights,
    # get_admin_dashboard, _get_system_overview, _get_real_time_metrics,
    # _get_kb_effectiveness, _get_user_engagement_metrics,
    # _generate_system_recommendations, _get_system_alerts
    
    def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics data - FIXED VERSION"""
        try:
            conn = get_db()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT total_queries, successful_queries, avg_confidence, avg_response_time, 
                       kb_usage_count, user_satisfaction_score
                FROM enhanced_analytics 
                WHERE date = DATE('now')
                ORDER BY date DESC LIMIT 1
            """)
            today_data = cursor.fetchone()
            
            if today_data:
                total_queries, successful_queries, avg_confidence, avg_response_time, kb_usage_count, user_satisfaction = today_data
                success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
                kb_usage_rate = (kb_usage_count / total_queries * 100) if total_queries > 0 else 0
            else:
                total_queries, successful_queries, avg_confidence, avg_response_time, kb_usage_count, user_satisfaction = 0, 0, 0, 0, 0, 0
                success_rate = 0
                kb_usage_rate = 0
            
            cursor.execute("SELECT COUNT(*), AVG(rating) FROM feedback WHERE rating > 0")
            total_feedback, avg_rating = cursor.fetchone()
            total_feedback = total_feedback or 0
            avg_rating = avg_rating or 0
            
            cursor.execute("SELECT rating, COUNT(*) FROM feedback WHERE rating > 0 GROUP BY rating ORDER BY rating")
            rating_distribution = {str(rating): count for rating, count in cursor.fetchall()}
            
            cursor.execute("""
                SELECT rating, comment, timestamp 
                FROM feedback 
                ORDER BY timestamp DESC 
                LIMIT 10
            """)
            recent_feedback = [
                {"rating": r[0], "comment": r[1], "timestamp": r[2]}
                for r in cursor.fetchall()
            ]
            
            cursor.execute("""
                SELECT pattern, frequency, avg_confidence, success_rate, category
                FROM query_patterns 
                ORDER BY frequency DESC 
                LIMIT 10
            """)
            top_queries = [
                {
                    "pattern": row[0],
                    "frequency": row[1],
                    "avg_confidence": round(row[2], 3),
                    "success_rate": round(row[3], 3),
                    "category": row[4]
                }
                for row in cursor.fetchall()
            ]
            
            cursor.execute("""
                SELECT date, total_queries, successful_queries, avg_confidence, user_satisfaction_score
                FROM enhanced_analytics 
                ORDER BY date DESC 
                LIMIT 7
            """)
            daily_stats = [
                {
                    "date": row[0],
                    "total_queries": row[1],
                    "successful_queries": row[2],
                    "avg_confidence": round(row[3], 3),
                    "satisfaction_score": round(row[4], 1)
                }
                for row in cursor.fetchall()
            ]
            
            cursor.execute("""
                SELECT AVG(query_count), AVG(avg_session_duration), COUNT(DISTINCT session_id)
                FROM user_behavior 
                WHERE last_activity >= datetime('now', '-7 days')
            """)
            engagement_stats = cursor.fetchone()
            
            system_health = self._get_system_health_metrics(cursor)
            
            return {
                "basic_stats": {
                    "total_queries": total_queries,
                    "successful_queries": successful_queries,
                    "success_rate": round(success_rate, 1),
                    "kb_usage_rate": round(kb_usage_rate, 1),
                    "avg_confidence": round(avg_confidence, 3),
                    "avg_response_time": round(avg_response_time, 3),
                    "total_feedback": total_feedback,
                    "average_rating": round(avg_rating, 2),
                    "satisfaction_rate": round(user_satisfaction, 1)
                },
                "rating_distribution": rating_distribution,
                "recent_feedback": recent_feedback,
                "query_trends": {
                    "top_queries": top_queries,
                    "categories": self._get_query_categories(cursor)
                },
                "daily_metrics": daily_stats,
                "user_engagement": {
                    "avg_queries_per_session": round(engagement_stats[0] or 0, 1),
                    "avg_session_duration_minutes": round((engagement_stats[1] or 0), 1),
                    "total_sessions": engagement_stats[2] or 0,
                    "total_users": engagement_stats[2] or 0
                },
                "system_health": system_health
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive analytics: {e}")
            return self._get_fallback_analytics()

    def _get_fallback_analytics(self) -> Dict[str, Any]:
        """Provide fallback analytics when database is unavailable"""
        # (This function... unchanged)
        return {
            "basic_stats": {
                "total_queries": 0, "successful_queries": 0, "success_rate": 0,
                "kb_usage_rate": 0, "avg_confidence": 0, "avg_response_time": 0,
                "total_feedback": 0, "average_rating": 0, "satisfaction_rate": 0
            },
            "rating_distribution": {}, "recent_feedback": [],
            "query_trends": {"top_queries": [], "categories": {}},
            "daily_metrics": [],
            "user_engagement": {
                "avg_queries_per_session": 0, "avg_session_duration_minutes": 0,
                "total_sessions": 0, "total_users": 0
            },
            "system_health": {
                "avg_retrieval_time_ms": 0, "avg_generation_time_ms": 0,
                "avg_total_time_ms": 0, "avg_active_sessions": 0,
                "error_rate": 0, "health_score": 0
            }
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get system performance statistics - FIXED VERSION"""
        try:
            conn = get_db()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT total_queries, successful_queries, avg_confidence, kb_usage_count
                FROM enhanced_analytics 
                WHERE date = DATE('now')
            """)
            today_data = cursor.fetchone()
            
            if today_data:
                total_queries, successful_queries, avg_confidence, kb_usage_count = today_data
                success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
                kb_usage_rate = (kb_usage_count / total_queries * 100) if total_queries > 0 else 0
            else:
                total_queries, successful_queries, avg_confidence, kb_usage_count = 0, 0, 0, 0
                success_rate = 0
                kb_usage_rate = 0
            
            active_sessions = 0
            try:
                cursor.execute("SELECT COUNT(DISTINCT session_id) FROM sessions WHERE is_active = 1")
                active_sessions = cursor.fetchone()[0] or 0
            except Exception:
                logger.debug("sessions table not available for active sessions count")
            
            recent_perf = [0, 0, 0]
            try:
                cursor.execute("""
                    SELECT AVG(total_time), AVG(retrieval_time), COUNT(*)
                    FROM performance_metrics 
                    WHERE timestamp >= datetime('now', '-1 hour')
                """)
                recent_perf = cursor.fetchone() or [0, 0, 0]
            except Exception as e:
                logger.debug(f"Performance metrics query failed: {e}")
            
            kb_entries = 0
            try:
                cursor.execute("SELECT COUNT(*) FROM history WHERE used_kb = 1")
                kb_entries = cursor.fetchone()[0] or 0
            except Exception:
                logger.debug("history table not available for KB entries count")
            
            return {
                "total_queries": total_queries,
                "successful_queries": successful_queries,
                "kb_usage_rate": round(kb_usage_rate, 1),
                "success_rate": round(success_rate, 1),
                "avg_confidence": round(avg_confidence, 3),
                "active_sessions": active_sessions,
                "kb_entries": kb_entries,
                "recent_performance": {
                    "avg_response_time_ms": round((recent_perf[0] or 0) * 1000, 1),
                    "avg_retrieval_time_ms": round((recent_perf[1] or 0) * 1000, 1),
                    "queries_last_hour": recent_perf[2] or 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {
                "total_queries": 0, "successful_queries": 0, "kb_usage_rate": 0,
                "success_rate": 0, "avg_confidence": 0, "active_sessions": 0,
                "kb_entries": 0,
                "recent_performance": {
                    "avg_response_time_ms": 0,
                    "avg_retrieval_time_ms": 0,
                    "queries_last_hour": 0
                }
            }

    # ==================== ENHANCED USER ANALYTICS ====================
    
    def get_user_dashboard(self, user_id: str = "default") -> Dict[str, Any]:
        """Get personalized user dashboard with insights"""
        try:
            conn = get_db()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*), AVG(confidence), AVG(rating)
                FROM history h 
                LEFT JOIN feedback f ON h.session_id = f.session_id 
                WHERE h.session_id IN (SELECT session_id FROM sessions WHERE user_id = ?)
            """, (user_id,))
            user_stats = cursor.fetchone()
            
            cursor.execute("""
                SELECT strftime('%H', timestamp) as hour, COUNT(*) as query_count
                FROM history 
                WHERE session_id IN (SELECT session_id FROM sessions WHERE user_id = ?)
                GROUP BY hour 
                ORDER BY hour
            """, (user_id,))
            activity_patterns = {f"{row[0]}:00": row[1] for row in cursor.fetchall()}
            
            cursor.execute("""
                SELECT qp.category, COUNT(*) as count
                FROM query_patterns qp
                JOIN history h ON h.query LIKE '%' || qp.pattern || '%'
                WHERE h.session_id IN (SELECT session_id FROM sessions WHERE user_id = ?)
                GROUP BY qp.category 
                ORDER BY count DESC 
                LIMIT 5
            """, (user_id,))
            user_categories = {row[0]: row[1] for row in cursor.fetchall()}
            
            cursor.execute("""
                SELECT date(timestamp) as day, AVG(confidence) as avg_conf
                FROM history 
                WHERE session_id IN (SELECT session_id FROM sessions WHERE user_id = ?)
                GROUP BY day 
                ORDER BY day DESC 
                LIMIT 7
            """, (user_id,))
            learning_progress = [{"date": row[0], "confidence": round(row[1], 3)} for row in cursor.fetchall()]
            
            return {
                "user_stats": {
                    "total_queries": user_stats[0] or 0,
                    "avg_confidence": round(user_stats[1] or 0, 3),
                    "avg_rating": round(user_stats[2] or 0, 2)
                },
                "activity_patterns": activity_patterns,
                "preferred_categories": user_categories,
                "learning_progress": learning_progress,
                "personalized_insights": self._generate_user_insights(user_id, cursor)
            }
            
        except Exception as e:
            logger.error(f"Error getting user dashboard: {e}")
            return {}

    def _generate_user_insights(self, user_id: str, cursor) -> List[str]:
        """Generate personalized insights for the user"""
        # (This function... unchanged)
        insights = []
        try:
            cursor.execute("""
                SELECT qp.category, COUNT(*), AVG(h.confidence)
                FROM query_patterns qp
                JOIN history h ON h.query LIKE '%' || qp.pattern || '%'
                WHERE h.session_id IN (SELECT session_id FROM sessions WHERE user_id = ?)
                GROUP BY qp.category
            """, (user_id,))
            categories_data = cursor.fetchall()
            
            for category, count, avg_conf in categories_data:
                if count >= 3:
                    if avg_conf < 0.5:
                        insights.append(f"Consider providing more specific details when asking about {category} to get better answers")
                    else:
                        insights.append(f"You're getting good results with {category} queries - keep it up!")
            
            cursor.execute("""
                SELECT AVG(rating) FROM feedback 
                WHERE session_id IN (SELECT session_id FROM sessions WHERE user_id = ?)
            """, (user_id,))
            avg_rating = cursor.fetchone()[0]
            
            if avg_rating and avg_rating < 3:
                insights.append("Your feedback suggests we can improve. Please share specific suggestions!")
            
        except Exception as e:
            logger.error(f"Error generating user insights: {e}")
        
        return insights[:5]
    
    # ==================== ENHANCED ADMIN ANALYTICS ====================
    
    def get_admin_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive admin dashboard with system insights"""
        # (This function... unchanged)
        try:
            conn = get_db()
            cursor = conn.cursor()
            
            return {
                "system_overview": self._get_system_overview(cursor),
                "real_time_metrics": self._get_real_time_metrics(cursor),
                "kb_effectiveness": self._get_kb_effectiveness(cursor),
                "user_engagement": self._get_user_engagement_metrics(cursor),
                "recommendations": self._generate_system_recommendations(cursor),
                "alerts": self._get_system_alerts(cursor)
            }
            
        except Exception as e:
            logger.error(f"Error getting admin dashboard: {e}")
            return {}
    
    def _get_system_overview(self, cursor) -> Dict[str, Any]:
        """Get system overview statistics"""
        # (This function... unchanged)
        cursor.execute("""
            SELECT total_queries, successful_queries, avg_confidence, user_satisfaction_score
            FROM enhanced_analytics 
            WHERE date = DATE('now')
        """)
        today_data = cursor.fetchone()
        
        if today_data:
            total_queries, successful_queries, avg_confidence, user_satisfaction = today_data
            success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
        else:
            total_queries, successful_queries, avg_confidence, user_satisfaction = 0, 0, 0, 0
            success_rate = 0
        
        active_sessions = 0
        try:
            cursor.execute("SELECT COUNT(DISTINCT session_id) FROM sessions WHERE is_active = 1")
            active_sessions = cursor.fetchone()[0]
        except Exception:
            pass # sessions table might not exist
        
        avg_rating = 0
        positive_feedback = 0
        total_feedback = 0
        try:
            cursor.execute("SELECT AVG(rating) FROM feedback WHERE rating > 0")
            avg_rating = cursor.fetchone()[0] or 0
            cursor.execute("SELECT COUNT(*) FROM feedback WHERE rating >= 4")
            positive_feedback = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM feedback")
            total_feedback = cursor.fetchone()[0]
        except Exception:
            pass # feedback table might not exist

        satisfaction_rate = (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0
        
        return {
            "total_queries": total_queries,
            "active_sessions": active_sessions,
            "avg_confidence": round(avg_confidence, 3),
            "success_rate": round(success_rate, 1),
            "avg_rating": round(avg_rating, 2),
            "satisfaction_rate": round(satisfaction_rate, 1),
            "user_satisfaction": round(user_satisfaction, 1),
            "system_uptime": self._calculate_system_uptime(cursor)
        }

    def _get_real_time_metrics(self, cursor) -> Dict[str, Any]:
        """Get real-time system metrics"""
        # (This function... unchanged)
        history_stats = (0, 0, 0)
        try:
            cursor.execute("""
                SELECT 
                    COUNT(*) as queries_last_hour,
                    AVG(confidence) as avg_confidence,
                    COUNT(DISTINCT session_id) as unique_users
                FROM history 
                WHERE timestamp >= datetime('now', '-1 hour')
            """)
            history_stats = cursor.fetchone() or (0, 0, 0)
        except Exception:
            pass
        
        perf_stats = (0,)
        try:
            cursor.execute("""
                SELECT AVG(total_time) as avg_response_time
                FROM performance_metrics
                WHERE timestamp >= datetime('now', '-1 hour')
            """)
            perf_stats = cursor.fetchone() or (0,)
        except Exception:
            pass
        
        system_activity = 0
        try:
            cursor.execute("""
                SELECT COUNT(*) FROM performance_metrics 
                WHERE timestamp >= datetime('now', '-5 minutes')
            """)
            system_activity = cursor.fetchone()[0] or 0
        except Exception:
            pass
        
        return {
            "queries_last_hour": history_stats[0],
            "avg_response_time_ms": round((perf_stats[0] or 0) * 1000, 1),
            "avg_confidence_recent": round(history_stats[1] or 0, 3),
            "unique_users_last_hour": history_stats[2],
            "system_activity_level": "High" if system_activity > 10 else "Medium" if system_activity > 5 else "Low"
        }
    
    def _get_kb_effectiveness(self, cursor) -> Dict[str, Any]:
        """Get knowledge base effectiveness metrics"""
        # (This function... unchanged)
        kb_stats = (0, 0, 0)
        try:
            cursor.execute("""
                SELECT 
                    COUNT(*) as kb_queries,
                    AVG(confidence) as avg_kb_confidence,
                    AVG(CASE WHEN used_kb = 1 THEN confidence ELSE 0 END) as kb_success_rate
                FROM history 
                WHERE used_kb = 1
            """)
            kb_stats = cursor.fetchone()
        except Exception:
            pass

        strong_categories = {}
        weak_categories = {}
        try:
            cursor.execute("""
                SELECT category, COUNT(*) FROM query_patterns 
                WHERE success_rate > 0.7 GROUP BY category 
                ORDER BY COUNT(*) DESC LIMIT 5
            """)
            strong_categories = {row[0]: row[1] for row in cursor.fetchall()}
            
            cursor.execute("""
                SELECT category, COUNT(*) FROM query_patterns 
                WHERE success_rate < 0.3 GROUP BY category 
                ORDER BY COUNT(*) DESC LIMIT 5
            """)
            weak_categories = {row[0]: row[1] for row in cursor.fetchall()}
        except Exception:
            pass
        
        return {
            "kb_usage_count": kb_stats[0] or 0,
            "avg_kb_confidence": round(kb_stats[1] or 0, 3),
            "kb_success_rate": round((kb_stats[2] or 0) * 100, 1),
            "strong_categories": strong_categories,
            "weak_categories": weak_categories,
            "kb_coverage_gap": self._calculate_kb_coverage_gap(cursor)
        }
    
    def _get_user_engagement_metrics(self, cursor) -> Dict[str, Any]:
        """Get user engagement metrics"""
        # (This function... unchanged)
        engagement_stats = (0, 0, 0, 0)
        engagement_levels = {}
        try:
            cursor.execute("""
                SELECT 
                    AVG(query_count) as avg_queries_per_session,
                    AVG(avg_session_duration) as avg_session_minutes,
                    COUNT(DISTINCT user_id) as total_users,
                    COUNT(DISTINCT session_id) as total_sessions
                FROM user_behavior 
                WHERE last_activity >= datetime('now', '-30 days')
            """)
            engagement_stats = cursor.fetchone()
            
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN query_count BETWEEN 1 AND 3 THEN 'Low'
                        WHEN query_count BETWEEN 4 AND 10 THEN 'Medium'
                        ELSE 'High'
                    END as engagement_level,
                    COUNT(*) as user_count
                FROM user_behavior 
                GROUP BY engagement_level
            """)
            engagement_levels = {row[0]: row[1] for row in cursor.fetchall()}
        except Exception:
            pass

        return {
            "avg_queries_per_session": round(engagement_stats[0] or 0, 1),
            "avg_session_minutes": round(engagement_stats[1] or 0, 1),
            "total_users": engagement_stats[2] or 0,
            "total_sessions": engagement_stats[3] or 0,
            "engagement_distribution": engagement_levels,
            "retention_rate": self._calculate_retention_rate(cursor)
        }
    
    def _generate_system_recommendations(self, cursor) -> List[Dict[str, Any]]:
        """Generate system improvement recommendations"""
        # (This function... unchanged)
        recommendations = []
        try:
            cursor.execute("""
                SELECT category, AVG(success_rate) as avg_success
                FROM query_patterns 
                GROUP BY category 
                HAVING avg_success < 0.4
            """)
            weak_categories = cursor.fetchall()
            for category, success_rate in weak_categories:
                recommendations.append({
                    "type": "KB_IMPROVEMENT", "priority": "HIGH",
                    "message": f"Improve knowledge base coverage for '{category}' category (success rate: {success_rate:.1%})",
                    "action": "Add more articles and examples"
                })
            
            cursor.execute("""
                SELECT pattern, frequency, success_rate
                FROM query_patterns 
                WHERE success_rate < 0.3 AND frequency > 5
                ORDER BY frequency DESC 
                LIMIT 5
            """)
            failing_patterns = cursor.fetchall()
            for pattern, frequency, success_rate in failing_patterns:
                recommendations.append({
                    "type": "QUERY_PATTERN", "priority": "MEDIUM",
                    "message": f"Address failing query pattern: '{pattern}' (frequency: {frequency}, success: {success_rate:.1%})",
                    "action": "Analyze pattern and improve response"
                })
            
            cursor.execute("""
                SELECT AVG(total_time) FROM performance_metrics 
                WHERE timestamp >= datetime('now', '-1 day')
            """)
            avg_response_time = cursor.fetchone()[0] or 0
            
            if avg_response_time > 5.0:
                recommendations.append({
                    "type": "PERFORMANCE", "priority": "HIGH",
                    "message": f"High average response time: {avg_response_time:.1f}s",
                    "action": "Optimize retrieval and generation processes"
                })
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations[:10]
    
    def _get_system_alerts(self, cursor) -> List[Dict[str, Any]]:
        """Get system alerts and warnings"""
        # (This function... unchanged)
        alerts = []
        try:
            cursor.execute("""
                SELECT COUNT(*) FROM history 
                WHERE confidence < 0.2 AND timestamp >= datetime('now', '-1 hour')
            """)
            recent_errors = cursor.fetchone()[0]
            if recent_errors > 10:
                alerts.append({
                    "level": "ERROR",
                    "message": f"High error rate: {recent_errors} low-confidence queries in last hour",
                    "timestamp": datetime.now().isoformat()
                })
            
            cursor.execute("""
                SELECT AVG(total_time) FROM performance_metrics 
                WHERE timestamp >= datetime('now', '-1 hour')
            """)
            recent_performance = cursor.fetchone()[0] or 0
            
            cursor.execute("""
                SELECT AVG(total_time) FROM performance_metrics 
                WHERE timestamp >= datetime('now', '-24 hours')
            """)
            daily_performance = cursor.fetchone()[0] or 0
            
            if daily_performance > 0 and recent_performance > daily_performance * 1.5:
                alerts.append({
                    "level": "WARNING",
                    "message": f"Performance degradation detected: {recent_performance:.1f}s vs {daily_performance:.1f}s average",
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            logger.error(f"Error generating system alerts: {e}")

        return alerts
    
    # ==================== VISUALIZATION & REPORTING ====================
    
    def generate_visualization(self, viz_type: str, days: int = 30) -> Optional[str]:
        """Generate base64 encoded visualization for web display"""
        # (This function... unchanged)
        if not self.config.enable_visualizations:
            return None
        try:
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if viz_type == "confidence_trend":
                data = self._get_confidence_trend_data(days)
                ax.plot([d['date'] for d in data], [d['confidence'] for d in data], marker='o')
                ax.set_title('Confidence Trend Over Time')
                ax.set_ylabel('Average Confidence')
                ax.grid(True, alpha=0.3)
            
            elif viz_type == "query_categories":
                data = self._get_category_distribution_data(days)
                ax.bar(list(data.keys()), list(data.values()))
                ax.set_title('Query Category Distribution')
                ax.set_ylabel('Number of Queries')
                plt.xticks(rotation=45)
            
            elif viz_type == "user_activity":
                data = self._get_user_activity_pattern(days)
                ax.plot(list(data.keys()), list(data.values()), marker='o')
                ax.set_title('User Activity Pattern (24h)')
                ax.set_xlabel('Hour of Day')
                ax.set_ylabel('Query Count')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Error generating visualization {viz_type}: {e}")
            return None
    
    def export_analytics_report(self, report_type: str = "comprehensive") -> str:
        """Export analytics report to file"""
        # (This function... unchanged)
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.export_path}/analytics_report_{report_type}_{timestamp}.json"
            
            if report_type == "comprehensive":
                report_data = self.get_comprehensive_analytics()
            elif report_type == "admin":
                report_data = self.get_admin_dashboard()
            elif report_type == "performance":
                report_data = self.get_performance_stats()
            else:
                report_data = self.get_comprehensive_analytics()
            
            report_data["export_metadata"] = {
                "exported_at": datetime.now().isoformat(),
                "report_type": report_type,
                "data_range": "all_time"
            }
            
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Analytics report exported: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting analytics report: {e}")
            return ""
    
    # ==================== HELPER METHODS ====================
    
    # (All helper methods... unchanged)
    
    def _calculate_system_uptime(self, cursor) -> float:
        # (This function... unchanged)
        try:
            cursor.execute("SELECT COUNT(*) FROM history WHERE timestamp >= datetime('now', '-7 days')")
            recent_queries = cursor.fetchone()[0]
            return 99.9 if recent_queries > 0 else 0.0
        except Exception:
            return 0.0 # Return 0 if history table doesn't exist
    
    def _calculate_kb_coverage_gap(self, cursor) -> float:
        # (This function... unchanged)
        try:
            cursor.execute("SELECT COUNT(*) FROM history WHERE used_kb = 1")
            kb_queries = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM history")
            total_queries = cursor.fetchone()[0]
            if total_queries == 0: return 100.0
            coverage = (kb_queries / total_queries) * 100
            return round(100 - coverage, 1)
        except Exception:
            return 100.0

    def _calculate_retention_rate(self, cursor) -> float:
        # (This function... unchanged)
        try:
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM user_behavior WHERE last_activity >= datetime('now', '-30 days')")
            active_users = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM user_behavior")
            total_users = cursor.fetchone()[0]
            if total_users == 0: return 0.0
            return round((active_users / total_users) * 100, 1)
        except Exception:
            return 0.0

    def _get_confidence_trend_data(self, days: int) -> List[Dict[str, Any]]:
        # (This function... unchanged)
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT date, avg_confidence FROM enhanced_analytics WHERE date >= date('now', ?) ORDER BY date", (f'-{days} days',))
        return [{"date": row[0], "confidence": row[1]} for row in cursor.fetchall()]
    
    def _get_category_distribution_data(self, days: int) -> Dict[str, int]:
        # (This function... unchanged)
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT category, COUNT(*) FROM query_patterns WHERE last_seen >= datetime('now', ?) GROUP BY category ORDER BY COUNT(*) DESC LIMIT 10", (f'-{days} days',))
        return {row[0]: row[1] for row in cursor.fetchall()}
    
    def _get_user_activity_pattern(self, days: int) -> Dict[str, int]:
        # (This function... unchanged)
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT strftime('%H', timestamp) as hour, COUNT(*) as count FROM history WHERE timestamp >= datetime('now', ?) GROUP BY hour ORDER BY hour", (f'-{days} days',))
        return {f"{row[0]}:00": row[1] for row in cursor.fetchall()}
    
    def _update_query_patterns(self, query: str, confidence: float, success: bool, exact_match: bool = False):
        # (This function... unchanged)
        try:
            category = self._categorize_query(query)
            pattern = self._extract_query_pattern(query)
            conn = get_db()
            cursor = conn.cursor()
            success_value = 1.0 if (success or exact_match) else 0.0
            confidence_boost = confidence * 1.2 if exact_match else confidence
            
            cursor.execute("""
                INSERT INTO query_patterns (pattern, frequency, avg_confidence, success_rate, last_seen, category)
                VALUES (?, 1, ?, ?, CURRENT_TIMESTAMP, ?)
                ON CONFLICT(pattern) DO UPDATE SET
                    frequency = frequency + 1,
                    avg_confidence = (avg_confidence * frequency + ?) / (frequency + 1),
                    success_rate = (success_rate * frequency + ?) / (frequency + 1),
                    last_seen = CURRENT_TIMESTAMP
            """, (pattern, confidence_boost, success_value, category, confidence_boost, success_value))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error updating query patterns: {e}")
    
    def _categorize_query(self, query: str) -> str:
        # (This function... unchanged)
        query_lower = query.lower()
        categories = {
            'authentication': ['login', 'password', 'authenticate', 'credential', 'access denied'],
            'network': ['connection', 'network', 'connect', 'ping', 'latency', 'bandwidth'],
            'performance': ['slow', 'performance', 'lag', 'bottleneck', 'optimize'],
            'error': ['error', 'crash', 'fail', 'broken', 'not working'],
            'configuration': ['configure', 'setup', 'install', 'settings', 'configuration'],
            'data': ['database', 'query', 'data', 'storage', 'backup'],
            'general': ['how', 'what', 'why', 'when', 'where']
        }
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        return 'other'
    
    def _extract_query_pattern(self, query: str) -> str:
        # (This function... unchanged)
        pattern = re.sub(r'\b\d+\b', '<NUMBER>', query)
        pattern = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', pattern)
        pattern = re.sub(r'\bhttps?://[^\s]+\b', '<URL>', pattern)
        pattern = re.sub(r'\b[\w\.-]+\.(com|org|net|io)\b', '<DOMAIN>', pattern)
        return pattern.strip()
    
    def _update_user_behavior(self, session_id: str, user_id: str, query_count: int = 0, rating: int = None):
        # (This function... unchanged)
        try:
            conn = get_db()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*), (JULIANDAY(MAX(timestamp)) - JULIANDAY(MIN(timestamp))) * 24 * 60
                FROM history 
                WHERE session_id = ?
            """, (session_id,))
            result = cursor.fetchone()
            total_query_count = result[0] if result else 0
            session_duration = result[1] if result and result[1] else 0
            
            cursor.execute("SELECT COUNT(*) FROM feedback WHERE session_id = ?", (session_id,))
            feedback_count = cursor.fetchone()[0] or 0
            feedback_frequency = feedback_count / max(total_query_count, 1)
            
            cursor.execute("""
                SELECT qp.category, COUNT(*) FROM query_patterns qp
                JOIN history h ON h.query LIKE '%' || qp.pattern || '%'
                WHERE h.session_id = ?
                GROUP BY qp.category ORDER BY COUNT(*) DESC LIMIT 3
            """, (session_id,))
            preferred_categories = [row[0] for row in cursor.fetchall()]
            
            cursor.execute("""
                INSERT OR REPLACE INTO user_behavior 
                (session_id, user_id, query_count, avg_session_duration, preferred_categories, feedback_frequency, last_activity)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (session_id, user_id, total_query_count, session_duration, 
                  json.dumps(preferred_categories), feedback_frequency))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error updating user behavior: {e}")
    
    def _generate_feedback_insights(self, session_id: str, comment: str, rating: int):
        # (This function... unchanged)
        try:
            if not comment:
                return
            
            positive_words = ['good', 'great', 'excellent', 'helpful', 'thanks', 'thank', 'awesome', 'perfect']
            negative_words = ['bad', 'poor', 'terrible', 'useless', 'wrong', 'incorrect', 'waste']
            comment_lower = comment.lower()
            positive_count = sum(1 for word in positive_words if word in comment_lower)
            negative_count = sum(1 for word in negative_words if word in comment_lower)
            sentiment_score = (positive_count - negative_count) / max(len(comment.split()), 1)
            
            urgency_keywords = ['urgent', 'asap', 'immediately', 'critical', 'emergency']
            urgency_level = "HIGH" if any(word in comment_lower for word in urgency_keywords) else "LOW"
            
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_feedback_insights 
                (session_id, improvement_suggestions, sentiment_score, urgency_level)
                VALUES (?, ?, ?, ?)
            """, (session_id, comment, sentiment_score, urgency_level))
            conn.commit()
        except Exception as e:
            logger.error(f"Error generating feedback insights: {e}")
    
    def record_performance_metrics(self, metrics: Dict[str, Any]):
        """Record system performance metrics from new_backend.py"""
        # (This function... unchanged)
        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance_metrics 
                (retrieval_time, generation_time, total_time, active_sessions)
                VALUES (?, ?, ?, ?)
            """, (
                metrics.get('retrieval_time', 0),
                metrics.get('generation_time', 0),
                metrics.get('total_time', 0),
                metrics.get('active_sessions', 0)
            ))
            conn.commit()
            logger.debug(f"Recorded performance metrics: retrieval={metrics.get('retrieval_time', 0):.3f}s, generation={metrics.get('generation_time', 0):.3f}s")
        except Exception as e:
            logger.error(f"Error recording performance metrics: {e}")
    
    def _get_system_health_metrics(self, cursor) -> Dict[str, Any]:
        """Get system health metrics"""
        # (This function... unchanged)
        try:
            perf_stats = [0, 0, 0, 0]
            try:
                cursor.execute("""
                    SELECT AVG(retrieval_time), AVG(generation_time), AVG(total_time), AVG(active_sessions)
                    FROM performance_metrics 
                    WHERE timestamp >= datetime('now', '-1 hour')
                """)
                perf_stats = cursor.fetchone() or [0, 0, 0, 0]
            except Exception as e:
                logger.debug(f"System health metrics query failed: {e}")
            
            low_confidence_count = 0
            total_queries_24h = 1
            try:
                cursor.execute("SELECT COUNT(*) FROM history WHERE confidence < ? AND timestamp >= datetime('now', '-24 hours')", (self.config.confidence_threshold,))
                low_confidence_count = cursor.fetchone()[0] or 0
                cursor.execute("SELECT COUNT(*) FROM history WHERE timestamp >= datetime('now', '-24 hours')")
                total_queries_24h = cursor.fetchone()[0] or 1
            except Exception:
                logger.debug("history table not available for error rate calculation")
            
            error_rate = (low_confidence_count / total_queries_24h) * 100
            
            return {
                "avg_retrieval_time_ms": round((perf_stats[0] or 0) * 1000, 1),
                "avg_generation_time_ms": round((perf_stats[1] or 0) * 1000, 1),
                "avg_total_time_ms": round((perf_stats[2] or 0) * 1000, 1),
                "avg_active_sessions": round(perf_stats[3] or 0, 1),
                "error_rate": round(error_rate, 1),
                "health_score": max(0, 100 - error_rate * 2)
            }
        except Exception as e:
            logger.error(f"Error getting system health metrics: {e}")
            return {}
    
    def _get_query_categories(self, cursor) -> Dict[str, int]:
        """Get query category distribution"""
        # (This function... unchanged)
        try:
            cursor.execute("SELECT category, COUNT(*) FROM query_patterns GROUP BY category ORDER BY COUNT(*) DESC")
            return {row[0]: row[1] for row in cursor.fetchall()}
        except:
            return {}
    
    def get_trend_analysis(self, days: int = 30) -> Dict[str, Any]:
        """Get trend analysis for specified period"""
        # (This function... unchanged)
        try:
            conn = get_db()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT date, total_queries, successful_queries, avg_confidence, user_satisfaction_score
                FROM enhanced_analytics 
                WHERE date >= date('now', ?) 
                ORDER BY date
            """, (f'-{days} days',))
            
            trends = [
                {
                    "date": row[0],
                    "total_queries": row[1],
                    "success_rate": round((row[2] / row[1]) * 100, 1) if row[1] > 0 else 0,
                    "avg_confidence": round(row[3], 3),
                    "satisfaction_score": round(row[4], 1)
                }
                for row in cursor.fetchall()
            ]
            
            trends_analysis = {}
            if len(trends) >= 2:
                first_week = trends[:7]
                last_week = trends[-7:]
                
                avg_first_week = {
                    "queries": sum(d["total_queries"] for d in first_week) / len(first_week),
                    "success": sum(d["success_rate"] for d in first_week) / len(first_week),
                    "satisfaction": sum(d["satisfaction_score"] for d in first_week) / len(first_week)
                }
                
                avg_last_week = {
                    "queries": sum(d["total_queries"] for d in last_week) / len(last_week),
                    "success": sum(d["success_rate"] for d in last_week) / len(last_week),
                    "satisfaction": sum(d["satisfaction_score"] for d in last_week) / len(last_week)
                }
                
                trends_analysis = {
                    "query_growth": round(((avg_last_week["queries"] - avg_first_week["queries"]) / avg_first_week["queries"]) * 100, 1) if avg_first_week["queries"] > 0 else 0,
                    "success_growth": round(avg_last_week["success"] - avg_first_week["success"], 1),
                    "satisfaction_growth": round(avg_last_week["satisfaction"] - avg_first_week["satisfaction"], 1)
                }
            
            return {
                "daily_trends": trends,
                "trends_analysis": trends_analysis,
                "period": f"last_{days}_days"
            }
            
        except Exception as e:
            logger.error(f"Error getting trend analysis: {e}")
            return {}
    
    def cleanup_old_data(self):
        """Clean up old analytics data based on retention policy"""
        # (This function... unchanged)
        try:
            conn = get_db()
            cursor = conn.cursor()
            cutoff_date = f"-{self.config.retention_days} days"
            
            cursor.execute("DELETE FROM enhanced_analytics WHERE date < date('now', ?)", (cutoff_date,))
            cursor.execute("DELETE FROM performance_metrics WHERE timestamp < datetime('now', ?)", (cutoff_date,))
            cursor.execute("DELETE FROM user_behavior WHERE last_activity < datetime('now', ?)", (cutoff_date,))
            cursor.execute("DELETE FROM query_patterns WHERE last_seen < datetime('now', ?) AND frequency < 3", (cutoff_date,))
            
            conn.commit()
            logger.info("Cleanup of old analytics data completed")
        except Exception as e:
            logger.error(f"Error during analytics data cleanup: {e}")

    # ==================== NEW REAL-TIME STREAMING METHODS ====================
    
    def init_streaming(self, app):
        """
        Starts the background analytics thread with the app context.
        This should be called from app.py.
        """
        logger.info("Initializing real-time analytics streaming thread...")
        metrics_thread = threading.Thread(
            target=self._background_metrics_updater, 
            args=(app,),  # Pass the Flask app object
            daemon=True
        )
        metrics_thread.start()

    def _background_metrics_updater(self, app):
        """
        Background thread to update real-time metrics for animations.
        (This logic was moved from app.py)
        """
        with app.app_context():  # <-- Create app context for DB access
            while True:
                try:
                    # Use class methods to get data
                    performance_data = self.get_performance_stats()
                    comprehensive_data = self.get_comprehensive_analytics()
                    
                    basic_stats = comprehensive_data.get('basic_stats', {})
                    system_health = comprehensive_data.get('system_health', {})
                    recent_perf = performance_data.get('recent_performance', {})
                    
                    # Update the class's metrics dictionary
                    self.realtime_metrics['total_queries'] = basic_stats.get('total_queries', 0)
                    self.realtime_metrics['success_rate'] = basic_stats.get('success_rate', 0)
                    self.realtime_metrics['user_satisfaction'] = basic_stats.get('satisfaction_rate', 0)
                    self.realtime_metrics['queries_per_minute'] = round(recent_perf.get('queries_last_hour', 0) / 60, 2)
                    self.realtime_metrics['avg_response_time'] = recent_perf.get('avg_response_time_ms', 0)
                    self.realtime_metrics['system_health'] = system_health.get('health_score', 100)
                    self.realtime_metrics['active_users'] = performance_data.get('active_sessions', 0)
                    self.realtime_metrics['last_updated'] = datetime.now().isoformat()

                    # Broadcast the new data
                    data_to_send = {'type': 'metrics_update', 'metrics': self.realtime_metrics}
                    
                    # Use the class's stream object
                    self.realtime_stream.broadcast(json.dumps(data_to_send))
                    
                    time.sleep(3) # Update every 3 seconds
                except Exception as e:
                    logger.error(f"Background metrics updater error: {e}")
                    time.sleep(10) # Sleep longer on error

# ==================== SINGLETON INSTANCE ====================

# (Singleton instance logic... unchanged)
_analytics_manager = None

def get_analytics_manager(db_path: str) -> AnalyticsManager:
    """Get or create analytics manager instance"""
    global _analytics_manager
    if _analytics_manager is None:
        _analytics_manager = AnalyticsManager(db_path)
    return _analytics_manager
