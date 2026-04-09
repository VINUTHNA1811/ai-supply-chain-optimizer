# app.py - AI-Powered Inventory Management System
# Enhanced with Forecast Confidence Intervals, Model Performance Tracking, and Error Handling

from flask import Flask, jsonify, request
from datetime import datetime, timedelta
import sqlite3
import random
import threading
import time
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

def get_db():
    conn = sqlite3.connect('inventory_system.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    
    # Products table
    c.execute('''CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT,
        current_stock INTEGER DEFAULT 0,
        reorder_point INTEGER DEFAULT 50,
        unit_cost REAL DEFAULT 0,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Suppliers table
    c.execute('''CREATE TABLE IF NOT EXISTS suppliers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        reliability_score REAL DEFAULT 0.8,
        avg_delivery_time INTEGER DEFAULT 5,
        quality_rating REAL DEFAULT 4.0,
        price_competitiveness REAL DEFAULT 0.7)''')
    
    # Orders table
    c.execute('''CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER,
        supplier_id INTEGER,
        quantity INTEGER,
        order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expected_delivery TIMESTAMP,
        status TEXT DEFAULT 'pending',
        total_cost REAL DEFAULT 0)''')
    
    # Alerts table
    c.execute('''CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_type TEXT NOT NULL,
        severity TEXT DEFAULT 'medium',
        message TEXT NOT NULL,
        product_id INTEGER,
        supplier_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        resolved BOOLEAN DEFAULT 0)''')
    
    # Inventory history table
    c.execute('''CREATE TABLE IF NOT EXISTS inventory_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER,
        stock_level INTEGER,
        change_amount INTEGER,
        change_type TEXT,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Demand history table
    c.execute('''CREATE TABLE IF NOT EXISTS demand_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER,
        demand_quantity INTEGER,
        demand_date DATE,
        source TEXT DEFAULT 'sales',
        FOREIGN KEY (product_id) REFERENCES products(id))''')
    
    # NEW: Forecast accuracy tracking table
    c.execute('''CREATE TABLE IF NOT EXISTS forecast_accuracy (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER,
        forecast_date DATE,
        predicted_demand REAL,
        actual_demand INTEGER,
        error_pct REAL,
        model_type TEXT,
        confidence REAL,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (product_id) REFERENCES products(id))''')
    
    # Insert sample data if empty
    c.execute('SELECT COUNT(*) FROM products')
    if c.fetchone()[0] == 0:
        products = [
            ('Laptop', 'Electronics', 45, 50, 800),
            ('Office Chair', 'Furniture', 30, 40, 150),
            ('Printer Paper', 'Supplies', 200, 100, 5),
            ('Desk Lamp', 'Electronics', 60, 50, 35),
            ('Filing Cabinet', 'Furniture', 15, 20, 200),
            ('USB Cable', 'Electronics', 150, 100, 8),
            ('Whiteboard', 'Supplies', 25, 30, 45),
            ('Monitor', 'Electronics', 40, 45, 300)
        ]
        c.executemany('INSERT INTO products (name, category, current_stock, reorder_point, unit_cost) VALUES (?, ?, ?, ?, ?)', products)
        
        suppliers = [
            ('TechSupply Co', 0.92, 3, 4.5, 0.85),
            ('Global Electronics', 0.88, 4, 4.2, 0.80),
            ('Office Depot Plus', 0.85, 5, 4.0, 0.75),
            ('FastShip Logistics', 0.78, 7, 3.8, 0.90),
            ('Premium Supplies', 0.95, 2, 4.8, 0.70)
        ]
        c.executemany('INSERT INTO suppliers (name, reliability_score, avg_delivery_time, quality_rating, price_competitiveness) VALUES (?, ?, ?, ?, ?)', suppliers)
        
        # Initialize inventory history
        c.execute('SELECT id, current_stock FROM products')
        for pid, stock in c.fetchall():
            c.execute('INSERT INTO inventory_history (product_id, stock_level, change_amount, change_type) VALUES (?, ?, ?, ?)',
                     (pid, stock, stock, 'initial'))
        
        # Generate 90 days of historical demand data
        c.execute('SELECT id FROM products')
        product_ids = [row[0] for row in c.fetchall()]
        
        for product_id in product_ids:
            for days_ago in range(90, 0, -1):
                demand_date = (datetime.now() - timedelta(days=days_ago)).date()
                base_demand = random.randint(5, 20)
                
                # Weekday boost
                if demand_date.weekday() < 5:
                    base_demand = int(base_demand * 1.3)
                
                # Add randomness
                demand = max(0, base_demand + random.randint(-5, 5))
                
                c.execute('INSERT INTO demand_history (product_id, demand_quantity, demand_date) VALUES (?, ?, ?)',
                         (product_id, demand, demand_date))
    
    conn.commit()
    conn.close()

class DemandForecaster:
    @staticmethod
    def calculate_forecast(product_id, days_ahead=30):
        """
        Enhanced forecast with confidence intervals and error tracking
        Returns forecast with upper/lower bounds for uncertainty visualization
        """
        conn = get_db()
        c = conn.cursor()
        
        c.execute('''SELECT demand_quantity, demand_date 
                    FROM demand_history 
                    WHERE product_id = ? 
                    ORDER BY demand_date ASC''', (product_id,))
        
        history = c.fetchall()
        conn.close()
        
        if len(history) < 14:
            return {
                "forecast": [],
                "avg_daily_demand": 0,
                "trend": "insufficient_data",
                "confidence": 0,
                "model_type": "none",
                "reliability_score": 0
            }
        
        quantities = np.array([float(row[0]) for row in history])
        dates = []
        for row in history:
            date_val = row[1]
            if isinstance(date_val, str):
                dates.append(datetime.strptime(date_val, '%Y-%m-%d').date())
            else:
                dates.append(date_val)
        
        X = np.array([(date - dates[0]).days for date in dates]).reshape(-1, 1)
        y = quantities
        
        # Try polynomial regression first
        try:
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_poly, y)
            
            # Calculate residuals for prediction intervals
            predictions = model.predict(X_poly)
            residuals = y - predictions
            std_error = np.std(residuals)
            
            r2_score = model.score(X_poly, y)
            confidence = max(0, min(100, r2_score * 100))
            model_type = "polynomial"
        except:
            # Fallback to linear regression
            model = LinearRegression()
            model.fit(X, y)
            
            predictions = model.predict(X)
            residuals = y - predictions
            std_error = np.std(residuals)
            
            r2_score = model.score(X, y)
            confidence = max(0, min(100, r2_score * 100))
            model_type = "linear"
            poly = None
        
        # Trend analysis
        recent_avg = np.mean(quantities[-7:])
        older_avg = np.mean(quantities[:7])
        
        if recent_avg > older_avg * 1.1:
            trend_direction = "increasing"
        elif recent_avg < older_avg * 0.9:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        trend_strength = abs(recent_avg - older_avg) / older_avg if older_avg > 0 else 0
        
        # Calculate reliability score
        data_quality = min(1.0, len(quantities) / 90)  # Full score at 90 days
        cv = np.std(quantities) / np.mean(quantities) if np.mean(quantities) > 0 else 1
        stability = max(0, 1 - cv)
        
        reliability_score = (
            (confidence / 100) * 0.4 +  # Model accuracy
            data_quality * 0.3 +         # Data sufficiency
            stability * 0.3              # Demand stability
        ) * 100
        
        # Generate forecasts with confidence intervals
        forecast_data = []
        current_date = datetime.now().date()
        last_day_index = (dates[-1] - dates[0]).days
        
        for day in range(1, days_ahead + 1):
            forecast_date = current_date + timedelta(days=day)
            future_day_index = last_day_index + day
            
            X_future = np.array([[future_day_index]])
            
            if poly is not None:
                X_future_poly = poly.transform(X_future)
                base_prediction = model.predict(X_future_poly)[0]
            else:
                base_prediction = model.predict(X_future)[0]
            
            # Apply weekday adjustment
            if forecast_date.weekday() < 5:
                daily_forecast = base_prediction * 1.15
            else:
                daily_forecast = base_prediction * 0.85
            
            daily_forecast = max(0, daily_forecast)
            
            # Calculate 80% confidence interval (±1.28 standard errors)
            margin = 1.28 * std_error
            lower_bound = max(0, daily_forecast - margin)
            upper_bound = daily_forecast + margin
            
            forecast_data.append({
                "date": forecast_date.strftime("%Y-%m-%d"),
                "forecasted_demand": round(daily_forecast, 1),
                "lower_bound": round(lower_bound, 1),
                "upper_bound": round(upper_bound, 1),
                "confidence": round(float(confidence), 1)
            })
        
        avg_daily_demand = np.mean(quantities[-14:])
        ma_7 = np.mean(quantities[-7:]) if len(quantities) >= 7 else avg_daily_demand
        ma_30 = np.mean(quantities[-30:]) if len(quantities) >= 30 else avg_daily_demand
        
        # Calculate MAPE for recent predictions (if we have actual data)
        mape = DemandForecaster.calculate_recent_mape(product_id)
        
        return {
            "forecast": forecast_data,
            "avg_daily_demand": round(float(avg_daily_demand), 2),
            "trend": trend_direction,
            "trend_strength": round(float(trend_strength), 3),
            "confidence": round(float(confidence), 1),
            "ma_7": round(float(ma_7), 2),
            "ma_30": round(float(ma_30), 2),
            "model_type": model_type,
            "r2_score": round(float(r2_score), 3),
            "std_error": round(float(std_error), 2),
            "reliability_score": round(float(reliability_score), 1),
            "mape": mape,
            "prediction_interval": "80%"
        }
    
    @staticmethod
    def calculate_recent_mape(product_id):
        """Calculate Mean Absolute Percentage Error for recent forecasts"""
        conn = get_db()
        c = conn.cursor()
        
        c.execute('''SELECT error_pct FROM forecast_accuracy 
                    WHERE product_id = ? 
                    ORDER BY forecast_date DESC 
                    LIMIT 30''', (product_id,))
        
        errors = [row[0] for row in c.fetchall() if row[0] is not None]
        conn.close()
        
        if not errors:
            return None
        
        return round(sum(errors) / len(errors), 1)
    
    @staticmethod
    def get_reorder_recommendation(product_id):
        """Enhanced reorder recommendation with safety stock based on confidence"""
        conn = get_db()
        c = conn.cursor()
        
        c.execute('SELECT current_stock, reorder_point, unit_cost, name FROM products WHERE id = ?', (product_id,))
        product = c.fetchone()
        conn.close()
        
        if not product:
            return {"error": "Product not found"}
        
        current_stock, reorder_point, unit_cost, name = product
        
        forecast = DemandForecaster.calculate_forecast(product_id, 30)
        
        if not forecast["forecast"]:
            return {"error": "Insufficient data for forecast"}
        
        avg_daily_demand = forecast["avg_daily_demand"]
        
        # Days until stockout
        if avg_daily_demand > 0:
            days_until_stockout = current_stock / avg_daily_demand
        else:
            days_until_stockout = 999
        
        # Total 30-day demand (use upper bound for conservative ordering)
        total_30day_demand = sum(f["upper_bound"] for f in forecast["forecast"])
        
        # Enhanced safety stock calculation based on forecast confidence
        # Lower confidence = higher safety stock
        confidence_factor = (100 - forecast["confidence"]) / 100
        # Decision strategy based on forecast confidence
        confidence = forecast["confidence"]
        if confidence >= 75:
            decision_mode = "aggressive"
        elif confidence >= 50:
            decision_mode = "balanced"
        else:
            decision_mode = "conservative"

        # Inventory risk score calculation
        stockout_pressure = min(100, (30 - days_until_stockout) / 30 * 100) if days_until_stockout < 30 else 0
        
        if decision_mode == "conservative":
            decision_penalty = 30
        elif decision_mode == "balanced":
            decision_penalty = 15
        else:
            decision_penalty = 5
        
        inventory_risk_score = (
            (100 - forecast["confidence"]) * 0.4 +
            stockout_pressure * 0.4 +
            decision_penalty * 0.2
        )
        
        inventory_risk_score = round(min(100, max(0, inventory_risk_score)), 1)


        lead_time_days = 7  # Assume 7-day lead time
        
        safety_stock = avg_daily_demand * lead_time_days * (0.5 + confidence_factor)
        # Adjust safety stock based on decision strategy
        if decision_mode == "conservative":
            safety_stock *= 1.3
        elif decision_mode == "aggressive":
            safety_stock *= 0.9

        
        # Recommended order quantity
        recommended_qty = max(0, int(total_30day_demand + safety_stock - current_stock))
        
        # Urgency classification
        if days_until_stockout < 7:
            urgency = "critical"
        elif days_until_stockout < 14:
            urgency = "high"
        elif days_until_stockout < 30:
            urgency = "medium"
        else:
            urgency = "low"
        
        return {
            "product_name": name,
            "current_stock": current_stock,
            "days_until_stockout": round(days_until_stockout, 1),
            "avg_daily_demand": avg_daily_demand,
            "recommended_order_qty": recommended_qty,
            "safety_stock": round(safety_stock, 1),
            "urgency": urgency,
            "estimated_cost": round(recommended_qty * unit_cost, 2),
            "forecast_confidence": forecast["confidence"],
            "reliability_score": forecast["reliability_score"],
            "trend": forecast["trend"],
            "confidence_explanation": f"Safety stock increased by {int(confidence_factor * 100)}% due to forecast uncertainty",
            "decision_mode": decision_mode,
            "inventory_risk_score": inventory_risk_score,

        }
    
    @staticmethod
    def track_forecast_accuracy(product_id, forecast_date, predicted_demand, actual_demand, model_type, confidence):
        """Store forecast vs actual for accuracy tracking"""
        conn = get_db()
        c = conn.cursor()
        
        if actual_demand > 0:
            error_pct = abs(predicted_demand - actual_demand) / actual_demand * 100
        else:
            error_pct = 0
        
        try:
            c.execute('''INSERT INTO forecast_accuracy 
                        (product_id, forecast_date, predicted_demand, actual_demand, error_pct, model_type, confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?)''',
                     (product_id, forecast_date, predicted_demand, actual_demand, error_pct, model_type, confidence))
            conn.commit()
        except Exception as e:
            print(f"Error tracking accuracy: {e}")
        finally:
            conn.close()

class AlertsEngine:
    @staticmethod
    def check_stockouts():
        """Check for low stock and create/resolve alerts automatically"""
        conn = get_db()
        c = conn.cursor()
        
        # Auto-resolve alerts for products that are now above reorder point
        c.execute('''SELECT a.id, p.name, p.current_stock, p.reorder_point 
                    FROM alerts a 
                    JOIN products p ON a.product_id = p.id 
                    WHERE a.alert_type = "stockout" 
                    AND a.resolved = 0 
                    AND p.current_stock > p.reorder_point''')
        
        resolved_count = 0
        for alert_id, name, stock, reorder in c.fetchall():
            c.execute('UPDATE alerts SET resolved = 1 WHERE id = ?', (alert_id,))
            resolved_count += 1
            print(f"✅ Auto-resolved stockout alert for {name} (stock: {stock} > reorder: {reorder})")
        
        # Create alerts for products below reorder point
        c.execute('SELECT id, name, current_stock, reorder_point FROM products WHERE current_stock <= reorder_point')
        created_count = 0
        for pid, name, stock, reorder in c.fetchall():
            # Check if alert already exists
            c.execute('SELECT id FROM alerts WHERE product_id = ? AND alert_type = "stockout" AND resolved = 0', (pid,))
            if not c.fetchone():
                severity = 'critical' if stock < reorder * 0.3 else 'high' if stock < reorder * 0.5 else 'medium'
                c.execute('INSERT INTO alerts (alert_type, severity, message, product_id) VALUES (?, ?, ?, ?)',
                         ('stockout', severity, f"Low stock: {name} has {stock} units (reorder: {reorder})", pid))
                created_count += 1
                print(f"⚠️ Created stockout alert for {name} (stock: {stock} ≤ reorder: {reorder})")
        
        if resolved_count > 0 or created_count > 0:
            print(f"📊 Alert Summary: {created_count} created, {resolved_count} auto-resolved")
        
        conn.commit()
        conn.close()
    
    @staticmethod
    def check_forecast_accuracy():
        """Check for degrading forecast accuracy and alert"""
        conn = get_db()
        c = conn.cursor()
        
        c.execute('''SELECT product_id, AVG(error_pct) as avg_error
                    FROM forecast_accuracy
                    WHERE forecast_date >= date('now', '-7 days')
                    GROUP BY product_id
                    HAVING avg_error > 25''')
        
        for pid, avg_error in c.fetchall():
            c.execute('SELECT name FROM products WHERE id = ?', (pid,))
            product = c.fetchone()
            if product:
                c.execute('SELECT id FROM alerts WHERE product_id = ? AND alert_type = "forecast_degradation" AND resolved = 0', (pid,))
                if not c.fetchone():
                    c.execute('INSERT INTO alerts (alert_type, severity, message, product_id) VALUES (?, ?, ?, ?)',
                             ('forecast_degradation', 'medium', 
                              f"Forecast accuracy declining for {product[0]}: {avg_error:.1f}% error", pid))
        
        conn.commit()
        conn.close()
    
    @staticmethod
    def simulate_events():
        """Simulate random supply chain disruptions"""
        conn = get_db()
        c = conn.cursor()
        
        # Weather events (10% chance)
        if random.random() < 0.1:
            events = [
                ("Hurricane warning affecting supplier routes", "high"),
                ("Heavy snow delays in supplier region", "medium"),
                ("Flooding near warehouse district", "high"),
                ("Port congestion causing shipment delays", "medium"),
                ("Labor strike at major distribution center", "high")
            ]
            event, severity = random.choice(events)
            c.execute('INSERT INTO alerts (alert_type, severity, message) VALUES (?, ?, ?)',
                     ('weather', severity, f"Weather Alert: {event}"))
        
        # Supplier delays (15% chance)
        if random.random() < 0.15:
            c.execute('SELECT id, name FROM suppliers ORDER BY RANDOM() LIMIT 1')
            supplier = c.fetchone()
            if supplier:
                sid, name = supplier
                delay = random.randint(2, 7)
                c.execute('INSERT INTO alerts (alert_type, severity, message, supplier_id) VALUES (?, ?, ?, ?)',
                         ('supplier_delay', 'medium', f"Supplier delay: {name} reporting {delay}-day delay", sid))
        
        conn.commit()
        conn.close()

def background_monitor():
    """Background thread for continuous monitoring"""
    while True:
        try:
            AlertsEngine.check_stockouts()
            AlertsEngine.check_forecast_accuracy()
            AlertsEngine.simulate_events()
            simulate_daily_demand()
            time.sleep(30)  # Check every 30 seconds
        except Exception as e:
            print(f"Monitor error: {e}")
            time.sleep(30)

def simulate_daily_demand():
    """Simulate demand to generate ongoing data"""
    conn = get_db()
    c = conn.cursor()
    
    c.execute('SELECT id FROM products')
    product_ids = [row[0] for row in c.fetchall()]
    
    today = datetime.now().date()
    
    for product_id in product_ids:
        # Check if today's demand already exists
        c.execute('SELECT id FROM demand_history WHERE product_id = ? AND demand_date = ?', 
                 (product_id, today))
        if not c.fetchone():
            base_demand = random.randint(5, 20)
            
            # Weekday boost
            if datetime.now().weekday() < 5:
                base_demand = int(base_demand * 1.3)
            
            demand = max(0, base_demand + random.randint(-5, 5))
            
            c.execute('INSERT INTO demand_history (product_id, demand_quantity, demand_date) VALUES (?, ?, ?)',
                     (product_id, demand, today))
    
    conn.commit()
    conn.close()

def rank_suppliers():
    """Multi-factor supplier ranking algorithm"""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT id, name, reliability_score, avg_delivery_time, quality_rating, price_competitiveness FROM suppliers')
    ranked = []
    for sid, name, rel, delivery, quality, price in c.fetchall():
        # Risk score calculation (lower is better)
        risk = (
            (1 - rel) * 0.35 +                    # Reliability (35%)
            (delivery / 10) * 0.25 +               # Delivery time (25%)
            ((5 - quality) / 5) * 0.20 +           # Quality (20%)
            (1 - price) * 0.20                     # Price (20%)
        )
        score = (1 - risk) * 100
        ranked.append({
            "id": sid, 
            "name": name, 
            "reliability": rel, 
            "delivery_time": delivery,
            "quality": quality, 
            "price_competitiveness": price, 
            "risk_score": round(risk, 3), 
            "overall_score": round(score, 2)
        })
    conn.close()
    return sorted(ranked, key=lambda x: x['overall_score'], reverse=True)

def calculate_eoq(product_id):
    """Economic Order Quantity calculation"""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT unit_cost FROM products WHERE id = ?', (product_id,))
    result = c.fetchone()
    conn.close()
    
    if not result:
        return {"eoq": 0}
    
    unit_cost = result[0]
    forecast = DemandForecaster.calculate_forecast(product_id)

    if forecast and forecast.get("avg_daily_demand", 0) > 0:
        annual_demand = forecast["avg_daily_demand"] * 365
    else:
        annual_demand = 1000  # Simulated - should use actual forecast
    ordering_cost = 50
    holding_cost = unit_cost * 0.25
    
    eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost) if holding_cost > 0 else 100
    
    return {
        "eoq": round(eoq, 2), 
        "annual_demand": round(annual_demand),
        "ordering_cost": ordering_cost,
        "holding_cost": round(holding_cost, 2)
    }

# ========================================
# API ROUTES
# ========================================

@app.route('/')
def index():
    """Serve the main dashboard HTML"""
    return '''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Inventory Management System - E-Summit 2025</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:linear-gradient(135deg,#1e3c72 0%,#2a5298 100%);min-height:100vh;color:#2c3e50}
.header{background:rgba(255,255,255,0.98);padding:1.5rem 2rem;box-shadow:0 4px 20px rgba(0,0,0,0.1);border-bottom:3px solid #3498db}
.header h1{color:#1e3c72;font-size:1.8rem;font-weight:700}
.header p{color:#7f8c8d;margin-top:0.3rem;font-size:0.95rem}
.version-badge{display:inline-block;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:0.3rem 0.8rem;border-radius:15px;font-size:0.75rem;margin-left:1rem;font-weight:600}
.container{max-width:1400px;margin:2rem auto;padding:0 2rem}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1.5rem;margin-bottom:2rem}
.stat-card{background:rgba(255,255,255,0.95);padding:1.8rem;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,0.12);border-left:4px solid #3498db;transition:transform 0.3s;cursor:pointer}
.stat-card:hover{transform:translateY(-5px);box-shadow:0 12px 32px rgba(0,0,0,0.18)}
.stat-card.warning{border-left-color:#f39c12}
.stat-card.danger{border-left-color:#e74c3c}
.stat-card.success{border-left-color:#27ae60}
.stat-value{font-size:2.5rem;font-weight:700;color:#2c3e50}
.stat-label{color:#7f8c8d;font-size:0.9rem;text-transform:uppercase;letter-spacing:0.5px;margin-top:0.3rem}
.stat-sublabel{color:#95a5a6;font-size:0.75rem;margin-top:0.2rem}
.dashboard-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:2rem;margin-bottom:2rem}
.card{background:rgba(255,255,255,0.95);border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,0.12);overflow:hidden}
.card-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:1.2rem 1.5rem;font-weight:600;font-size:1.1rem;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem}
.card-body{padding:1.5rem;max-height:500px;overflow-y:auto}
.alert-item{padding:1rem;margin-bottom:1rem;border-radius:8px;border-left:4px solid #3498db;background:#f8f9fa;cursor:pointer;transition:all 0.3s}
.alert-item:hover{background:#e9ecef;transform:translateX(5px)}
.alert-item.high{border-left-color:#e74c3c;background:#fff5f5}
.alert-item.medium{border-left-color:#f39c12;background:#fffbf0}
.alert-item.critical{border-left-color:#c0392b;background:#ffe6e6;animation:pulse 2s infinite}
.alert-header{display:flex;justify-content:space-between;margin-bottom:0.5rem;align-items:center}
.alert-type{font-weight:600;color:#2c3e50;text-transform:uppercase;font-size:0.85rem}
.alert-time{font-size:0.8rem;color:#95a5a6}
.alert-message{color:#34495e;line-height:1.5;margin-bottom:0.5rem}
.alert-actions{display:flex;gap:0.5rem;margin-top:0.5rem}
.product-item{display:flex;justify-content:space-between;align-items:center;padding:1rem;border-bottom:1px solid #ecf0f1;transition:background 0.3s}
.product-item:hover{background:#f8f9fa}
.product-info{flex:1}
.product-name{font-weight:600;color:#2c3e50;margin-bottom:0.3rem}
.product-category{font-size:0.85rem;color:#7f8c8d;margin-bottom:0.3rem}
.product-details{font-size:0.8rem;color:#95a5a6}
.product-stock{text-align:right;min-width:120px;margin-right:1rem}
.stock-value{font-size:1.3rem;font-weight:700;color:#27ae60}
.stock-value.low{color:#e74c3c}
.stock-label{font-size:0.8rem;color:#7f8c8d}
.product-actions{display:flex;flex-direction:column;gap:0.5rem}
.btn{padding:0.6rem 1.2rem;border:none;border-radius:6px;font-weight:600;cursor:pointer;transition:all 0.3s;font-size:0.9rem}
.btn-primary{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white}
.btn-primary:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(102,126,234,0.4)}
.btn-success{background:linear-gradient(135deg,#56ab2f 0%,#a8e063 100%);color:white}
.btn-success:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(86,171,47,0.4)}
.btn-danger{background:linear-gradient(135deg,#eb3349 0%,#f45c43 100%);color:white}
.btn-danger:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(235,51,73,0.4)}
.btn-warning{background:linear-gradient(135deg,#f79d00 0%,#ffa726 100%);color:white}
.btn-warning:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(247,157,0,0.4)}
.btn-small{padding:0.4rem 0.8rem;font-size:0.85rem}
.btn-mini{padding:0.3rem 0.6rem;font-size:0.75rem}
.badge{display:inline-block;padding:0.3rem 0.7rem;border-radius:12px;font-size:0.75rem;font-weight:600;text-transform:uppercase}
.badge-high{background:#e74c3c;color:white}
.badge-medium{background:#f39c12;color:white}
.badge-low{background:#3498db;color:white}
.badge-success{background:#27ae60;color:white}
.badge-critical{background:#c0392b;color:white;animation:pulse 2s infinite}
.loading,.empty-state{text-align:center;padding:2rem;color:#7f8c8d}
.chart-container{height:400px;width:100%}
.modal{display:none;position:fixed;z-index:1000;left:0;top:0;width:100%;height:100%;background:rgba(0,0,0,0.7);animation:fadeIn 0.3s}
.modal-content{background:white;margin:5% auto;padding:2rem;border-radius:12px;max-width:900px;box-shadow:0 20px 60px rgba(0,0,0,0.3);animation:slideIn 0.3s;max-height:80vh;overflow-y:auto}
.modal-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:2px solid #ecf0f1}
.modal-title{font-size:1.5rem;font-weight:700;color:#2c3e50}
.close{font-size:2rem;cursor:pointer;color:#95a5a6;transition:color 0.3s}
.close:hover{color:#e74c3c}
.modal-body{margin-bottom:1.5rem}
.form-group{margin-bottom:1.2rem}
.form-label{display:block;margin-bottom:0.5rem;font-weight:600;color:#2c3e50}
.form-control{width:100%;padding:0.8rem;border:2px solid #ecf0f1;border-radius:6px;font-size:1rem;transition:border-color 0.3s}
.form-control:focus{outline:none;border-color:#667eea}
.form-row{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
.info-grid{display:grid;gap:1rem;margin:1rem 0}
.info-item{padding:1rem;background:#f8f9fa;border-radius:6px;border-left:3px solid #667eea}
.info-label{font-size:0.85rem;color:#7f8c8d;text-transform:uppercase;letter-spacing:0.5px}
.info-value{font-size:1.2rem;font-weight:600;color:#2c3e50;margin-top:0.3rem}
.forecast-section{margin-top:2rem;padding-top:2rem;border-top:2px solid #ecf0f1}
.forecast-header{font-size:1.2rem;font-weight:700;color:#2c3e50;margin-bottom:1rem;display:flex;justify-content:space-between;align-items:center}
.confidence-bar{width:100%;height:8px;background:#ecf0f1;border-radius:4px;overflow:hidden;margin-top:0.5rem}
.confidence-fill{height:100%;background:linear-gradient(90deg,#e74c3c 0%,#f39c12 50%,#27ae60 100%);transition:width 0.5s}
.toast{position:fixed;top:20px;right:20px;background:white;padding:1rem 1.5rem;border-radius:8px;box-shadow:0 8px 24px rgba(0,0,0,0.2);z-index:2000;animation:slideInRight 0.3s;display:none}
.toast.success{border-left:4px solid #27ae60}
.toast.error{border-left:4px solid #e74c3c}
.toast.show{display:block}
.performance-indicator{display:inline-flex;align-items:center;gap:0.5rem;padding:0.5rem 1rem;background:#f8f9fa;border-radius:6px;margin-top:0.5rem}
.perf-dot{width:8px;height:8px;border-radius:50%}
.perf-dot.good{background:#27ae60}
.perf-dot.medium{background:#f39c12}
.perf-dot.poor{background:#e74c3c}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.7}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes slideIn{from{transform:translateY(-50px);opacity:0}to{transform:translateY(0);opacity:1}}
@keyframes slideInRight{from{transform:translateX(400px);opacity:0}to{transform:translateX(0);opacity:1}}
@media(max-width:768px){.dashboard-grid{grid-template-columns:1fr}.stat-value{font-size:2rem}.form-row{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="header">
<div style="display:flex;justify-content:space-between;align-items:center">
<div>
<h1>🌐 AI-Powered Supply Chain Optimizer<span class="version-badge">V2.0</span></h1>
<p>ML-Powered Demand Forecasting | Auto-Reorder System | Real-Time Integration</p>
</div>
</div>
</div>

<div class="container">
<div class="stats-grid">
<div class="stat-card" onclick="showProducts()">
<div class="stat-value" id="totalProducts">--</div>
<div class="stat-label">Total Products</div>
<div class="stat-sublabel">Active Inventory Items</div>
</div>
<div class="stat-card warning" onclick="showLowStock()">
<div class="stat-value" id="lowStock">--</div>
<div class="stat-label">Low Stock Items</div>
<div class="stat-sublabel">Needs Attention</div>
</div>
<div class="stat-card danger" onclick="showAlerts()">
<div class="stat-value" id="activeAlerts">--</div>
<div class="stat-label">Active Alerts</div>
<div class="stat-sublabel">Unresolved Issues</div>
</div>
<div class="stat-card success" onclick="showOrders()">
<div class="stat-value" id="pendingOrders">--</div>
<div class="stat-label">Pending Orders</div>
<div class="stat-sublabel">Awaiting Delivery</div>
</div>
</div>

<div class="dashboard-grid">
<div class="card">
<div class="card-header">
<span>🚨 Active Alerts & Monitoring</span>
<div style="display:flex;gap:0.5rem">
<button class="btn btn-warning btn-small" onclick="simulateWeather()"> Simulate Event</button>
<button class="btn btn-primary btn-small" onclick="loadAlerts()">🔄 Refresh</button>
</div>
</div>
<div class="card-body" id="alertsContainer"><div class="loading">Loading alerts...</div></div>
</div>

<div class="card">
<div class="card-header"><span>📦 Inventory Status</span>
<div>
<button class="btn btn-success btn-small" onclick="showAddProduct()" style="margin-right:0.5rem">➕ Add Product</button>
<button class="btn btn-primary btn-small" onclick="loadProducts()">🔄 Refresh</button>
</div>
</div>
<div class="card-body" id="productsContainer"><div class="loading">Loading products...</div></div>
</div>
</div>

<div class="dashboard-grid">
<div class="card">
<div class="card-header"><span>📊 Supplier Performance Ranking</span>
<button class="btn btn-primary btn-small" onclick="loadSuppliers()">🔄 Refresh</button>
</div>
<div class="card-body"><div id="supplierChart" class="chart-container"></div></div>
</div>

<div class="card">
<div class="card-header"><span>📋 Recent Purchase Orders</span>
<button class="btn btn-primary btn-small" onclick="loadOrders()">🔄 Refresh</button>
</div>
<div class="card-body" id="ordersContainer"><div class="loading">Loading orders...</div></div>
</div>
</div>
</div>

<div id="productModal" class="modal">
<div class="modal-content">
<div class="modal-header">
<span class="modal-title" id="modalTitle">Product Details</span>
<span class="close" onclick="closeModal('productModal')">&times;</span>
</div>
<div class="modal-body" id="modalBody"></div>
</div>
</div>

<div id="addProductModal" class="modal">
<div class="modal-content">
<div class="modal-header">
<span class="modal-title">Add New Product</span>
<span class="close" onclick="closeModal('addProductModal')">&times;</span>
</div>
<div class="modal-body">
<form id="addProductForm">
<div class="form-group">
<label class="form-label">Product Name</label>
<input type="text" id="newProductName" class="form-control" required>
</div>
<div class="form-row">
<div class="form-group">
<label class="form-label">Category</label>
<select id="newProductCategory" class="form-control">
<option>Electronics</option>
<option>Furniture</option>
<option>Supplies</option>
<option>Hardware</option>
</select>
</div>
<div class="form-group">
<label class="form-label">Unit Cost (₹)</label>
<input type="number" id="newProductCost" class="form-control" min="0" step="0.01" required>
</div>
</div>
<div class="form-row">
<div class="form-group">
<label class="form-label">Initial Stock</label>
<input type="number" id="newProductStock" class="form-control" min="0" required>
</div>
<div class="form-group">
<label class="form-label">Reorder Point</label>
<input type="number" id="newProductReorder" class="form-control" min="0" required>
</div>
</div>
<div style="display:flex;gap:1rem;margin-top:1.5rem">
<button type="submit" class="btn btn-success" style="flex:1">✓ Add Product</button>
<button type="button" class="btn btn-danger" onclick="closeModal('addProductModal')" style="flex:1">✗ Cancel</button>
</div>
</form>
</div>
</div>
</div>

<div id="toast" class="toast"></div>

<script>
let allProducts=[];
let allAlerts=[];
let allOrders=[];

function showToast(message,type='success'){
const toast=document.getElementById('toast');
toast.textContent=message;
toast.className='toast '+type+' show';
setTimeout(()=>toast.className='toast',3000);
}

function showModal(id){document.getElementById(id).style.display='block'}
function closeModal(id){document.getElementById(id).style.display='none'}

async function loadStats(){
try{
const r=await fetch('/api/dashboard/stats');
const d=await r.json();
document.getElementById('totalProducts').textContent=d.total_products;
document.getElementById('lowStock').textContent=d.low_stock_items;
document.getElementById('activeAlerts').textContent=d.active_alerts;
document.getElementById('pendingOrders').textContent=d.pending_orders;
}catch(e){console.error(e);showToast('Error loading stats','error')}
}

async function loadAlerts(){
try{
const r=await fetch('/api/alerts');
allAlerts=await r.json();

const c=document.getElementById('alertsContainer');
if(allAlerts.length===0){c.innerHTML='<div class="empty-state">✅ No active alerts - All systems operational</div>';return}
c.innerHTML=allAlerts.map(a=>`<div class="alert-item ${a.severity}">
<div class="alert-header">
<span class="alert-type">${a.type}</span>
<span class="badge badge-${a.severity}">${a.severity}</span>
</div>
<div class="alert-message">${a.message}</div>
<div class="alert-time">🕐 ${new Date(a.created_at).toLocaleString()}</div>
<div class="alert-actions">
<button class="btn btn-success btn-mini" onclick="resolveAlert(${a.id})">✓ Resolve</button>
${a.product_id?`<button class="btn btn-primary btn-mini" onclick="viewProduct(${a.product_id})">View Product</button>`:''}</div>
</div>`).join('');
}catch(e){console.error(e);showToast('Error loading alerts','error')}
}

async function loadProducts(){
try{
const r=await fetch('/api/products');
allProducts=await r.json();
const c=document.getElementById('productsContainer');
if(allProducts.length===0){c.innerHTML='<div class="empty-state">No products found</div>';return}
c.innerHTML=allProducts.map(p=>`<div class="product-item">
<div class="product-info">
<div class="product-name">${p.name}</div>
<div class="product-category">📁 ${p.category} | 💵 ₹${p.unit_cost}</div>
<div class="product-details">Reorder Point: ${p.reorder_point} units</div>
</div>
<div class="product-stock">
<div class="stock-value ${p.status==='low'?'low':''}">${p.current_stock}</div>
<div class="stock-label">units in stock</div>
</div>
<div class="product-actions">
<button class="btn btn-primary btn-small" onclick="viewProduct(${p.id})">📊 AI Forecast</button>
${p.status==='low'?`<button class="btn btn-warning btn-small" onclick="autoReorder(${p.id},'${p.name}')">🔄 Auto Order</button>`:''}
<button class="btn btn-success btn-small" onclick="adjustStock(${p.id},${p.current_stock},'${p.name}')">📦 Adjust</button>
</div>
</div>`).join('');
}catch(e){console.error(e);showToast('Error loading products','error')}
}

async function loadSuppliers(){
try{
const r=await fetch('/api/suppliers/ranking');
const suppliers=await r.json();
const names=suppliers.map(s=>s.name);
const scores=suppliers.map(s=>s.overall_score);
const colors=suppliers.map((s,i)=>i===0?'#27ae60':i===1?'#3498db':i===2?'#9b59b6':'#95a5a6');
const trace={x:scores,y:names,type:'bar',orientation:'h',marker:{color:colors,line:{color:'white',width:2}},
text:scores.map(s=>s.toFixed(1)+'%'),textposition:'outside',
hovertemplate:'<b>%{y}</b><br>Score: %{x:.1f}%<br><extra></extra>'};
const layout={title:{text:'Multi-Factor Performance Scores',font:{size:16}},xaxis:{title:'Overall Score (%)',range:[0,100],gridcolor:'#ecf0f1'},
yaxis:{automargin:true},plot_bgcolor:'#f8f9fa',paper_bgcolor:'white',margin:{l:150,r:40,t:60,b:60},height:400};
Plotly.newPlot('supplierChart',[trace],layout,{responsive:true,displayModeBar:false});
}catch(e){console.error(e);showToast('Error loading suppliers','error')}
}

async function loadOrders(){
try{
const r=await fetch('/api/orders');
allOrders=await r.json();
const c=document.getElementById('ordersContainer');
if(allOrders.length===0){c.innerHTML='<div class="empty-state">No recent orders</div>';return}
c.innerHTML=allOrders.map(o=>`<div class="product-item">
<div class="product-info">
<div class="product-name">${o.product_name}</div>
<div class="product-category">Supplier: ${o.supplier_name}</div>
<div class="product-details">Qty: ${o.quantity} | Cost: ₹${o.total_cost.toFixed(2)} | ETA: ${o.expected_delivery}</div>
</div>
<div style="min-width:100px;text-align:right">
<span class="badge badge-${o.status==='pending'?'medium':o.status==='delivered'?'success':'low'}">${o.status}</span>
</div>
<div class="product-actions">
${o.status==='pending'?`<button class="btn btn-success btn-small" onclick="updateOrderStatus(${o.id},'delivered')">✓ Delivered</button>`:''}
${o.status==='pending'?`<button class="btn btn-danger btn-small" onclick="updateOrderStatus(${o.id},'cancelled')">✗ Cancel</button>`:''}
</div>
</div>`).join('');
}catch(e){console.error(e);showToast('Error loading orders','error')}
}

async function viewProduct(id){
try{
const r=await fetch(`/api/products/${id}`);
const p=await r.json();
const eoqR=await fetch(`/api/eoq/${id}`);
const eoq=await eoqR.json();
const forecastR=await fetch(`/api/forecast/${id}`);
const forecast=await forecastR.json();
const reorderR=await fetch(`/api/reorder-recommendation/${id}`);
const reorder=await reorderR.json();
const perfR=await fetch(`/api/model-performance/${id}`);
const perf=await perfR.json();

document.getElementById('modalTitle').textContent=`📦 ${p.name}`;

let perfIndicator='';
if(perf && !perf.error){
let perfClass='good';
let perfText='Excellent';
if(perf.accuracy_pct<70){perfClass='medium';perfText='Good'}
if(perf.accuracy_pct<50){perfClass='poor';perfText='Needs Review'}
perfIndicator=`
<div class="performance-indicator">
<span class="perf-dot ${perfClass}"></span>
<span>Model Performance: ${perfText} (${perf.accuracy_pct}% accurate over ${perf.predictions_tracked} days)</span>
</div>`;
}

let forecastChart='';
if(!forecast.forecast || forecast.forecast.length===0){
forecastChart=`<div class="forecast-section"><div class="forecast-header">🤖 ML-Powered 30-Day Demand Forecast</div><div style="background:#fff3cd;padding:1rem;border-radius:6px;border-left:3px solid #f39c12;margin:1rem 0"><strong>⚠️ Insufficient Data:</strong> This product needs at least 14 days of demand history for ML forecasting. The background monitor adds demand data every 30 seconds — check back soon.</div></div>`;
}
if(forecast.forecast && forecast.forecast.length>0){
const dates=forecast.forecast.map(f=>f.date);
const demands=forecast.forecast.map(f=>f.forecasted_demand);
const lowerBounds=forecast.forecast.map(f=>f.lower_bound);
const upperBounds=forecast.forecast.map(f=>f.upper_bound);

forecastChart=`
<div class="forecast-section">
<div class="forecast-header">
<span>🤖 ML-Powered 30-Day Demand Forecast</span>
<span class="badge badge-${forecast.confidence>70?'success':forecast.confidence>50?'medium':'high'}">${forecast.confidence}% Confidence</span>
</div>
${perfIndicator}
<div class="info-grid" style="grid-template-columns:repeat(4,1fr);margin:1rem 0">
<div class="info-item">
<div class="info-label">Model Type</div>
<div class="info-value" style="font-size:1rem;text-transform:uppercase">${forecast.model_type}</div>
</div>
<div class="info-item">
<div class="info-label">Trend Direction</div>
<div class="info-value" style="font-size:1rem;color:${forecast.trend==='increasing'?'#27ae60':forecast.trend==='decreasing'?'#e74c3c':'#3498db'}">${forecast.trend.toUpperCase()}</div>
</div>
<div class="info-item">
<div class="info-label">Avg Daily Demand</div>
<div class="info-value">${forecast.avg_daily_demand}</div>
</div>
<div class="info-item">
<div class="info-label">Reliability Score</div>
<div class="info-value">${forecast.reliability_score}%</div>
</div>
<div class="info-item">
<div class="info-label">R² Score</div>
<div class="info-value">${forecast.r2_score}</div>
</div>
<div class="info-item">
<div class="info-label">Std Error</div>
<div class="info-value">±${forecast.std_error}</div>
</div>
<div class="info-item">
<div class="info-label">7-Day MA</div>
<div class="info-value">${forecast.ma_7}</div>
</div>
<div class="info-item">
<div class="info-label">30-Day MA</div>
<div class="info-value">${forecast.ma_30}</div>
</div>
</div>
<div style="background:#f8f9fa;padding:1rem;border-radius:6px;margin:1rem 0">
<strong>Prediction Interval:</strong> ${forecast.prediction_interval} confidence band shown below (shaded area)
</div>
<div id="forecastChart" style="height:350px"></div>
</div>`;
}

let reorderInfo='';
if(reorder && reorder.error){
reorderInfo=`<div class="forecast-section"><div class="forecast-header">💡 AI-Powered Reorder Recommendation</div><div style="background:#fff3cd;padding:1rem;border-radius:6px;border-left:3px solid #f39c12;margin:1rem 0"><strong>⚠️ Note:</strong> ${reorder.error} — The system needs more demand history to generate ML forecasts. Auto-reorder still works using default estimates.</div></div>`;
}
if(reorder && !reorder.error){
reorderInfo=`
<div class="forecast-section">
<div class="forecast-header">💡 AI-Powered Reorder Recommendation</div>
<div class="info-grid" style="grid-template-columns:repeat(3,1fr)">
<div class="info-item" style="border-left-color:${reorder.urgency==='critical'?'#e74c3c':reorder.urgency==='high'?'#f39c12':'#27ae60'}">
<div class="info-label">Urgency Level</div>
<div class="info-value" style="text-transform:uppercase;color:${reorder.urgency==='critical'?'#e74c3c':reorder.urgency==='high'?'#f39c12':'#27ae60'}">${reorder.urgency}</div>
</div>
<div class="info-item">
<div class="info-label">Days Until Stockout</div>
<div class="info-value" style="color:${reorder.days_until_stockout<7?'#e74c3c':reorder.days_until_stockout<14?'#f39c12':'#27ae60'}">${reorder.days_until_stockout}</div>
</div>
<div class="info-item">
<div class="info-label">Recommended Qty</div>
<div class="info-value">${reorder.recommended_order_qty} units</div>
</div>
<div class="info-item">
<div class="info-label">Safety Stock Buffer</div>
<div class="info-value">${reorder.safety_stock} units</div>
</div>
<div class="info-item">
<div class="info-label">Estimated Cost</div>
<div class="info-value">₹${reorder.estimated_cost.toLocaleString()}</div>
</div>
<div class="info-item">
<div class="info-label">Forecast Confidence</div>
<div class="info-value">${reorder.forecast_confidence}%</div>
</div>
<div class="info-item" style="border-left-color:${reorder.inventory_risk_score>85?'#c0392b':reorder.inventory_risk_score>60?'#e74c3c':reorder.inventory_risk_score>30?'#f39c12':'#27ae60'}">
<div class="info-label">🎯 Inventory Risk Score</div>
<div class="info-value" style="font-size:1.5rem;color:${reorder.inventory_risk_score>85?'#c0392b':reorder.inventory_risk_score>60?'#e74c3c':reorder.inventory_risk_score>30?'#f39c12':'#27ae60'}">${reorder.inventory_risk_score}/100</div>
</div>
<div class="info-item">
<div class="info-label">Decision Mode</div>
<div class="info-value" style="text-transform:uppercase;color:${reorder.decision_mode==='conservative'?'#e74c3c':reorder.decision_mode==='aggressive'?'#27ae60':'#3498db'}">${reorder.decision_mode}</div>
</div>
</div>
<div style="background:#e8f5e9;padding:1rem;border-radius:6px;border-left:3px solid #27ae60;margin-top:1rem">
<strong>🛡️ Safety Net:</strong> ${reorder.confidence_explanation}
</div>
</div>`;
}

let performanceSection='';
if(perf && !perf.error && perf.recent_misses && perf.recent_misses.length>0){
performanceSection=`
<div class="forecast-section">
<div class="forecast-header">📈 Recent Forecast Performance</div>
<div class="info-grid" style="grid-template-columns:repeat(3,1fr);margin-bottom:1rem">
<div class="info-item">
<div class="info-label">Accuracy (30 Days)</div>
<div class="info-value">${perf.accuracy_pct}%</div>
</div>
<div class="info-item">
<div class="info-label">Correct Predictions</div>
<div class="info-value">${perf.correct_predictions}/${perf.predictions_tracked}</div>
</div>
<div class="info-item">
<div class="info-label">Error Trend</div>
<div class="info-value" style="text-transform:uppercase;color:${perf.error_trend==='improving'?'#27ae60':'#f39c12'}">${perf.error_trend}</div>
</div>
</div>
<div style="background:#fff3cd;padding:1rem;border-radius:6px;border-left:3px solid #f39c12">
<strong>⚠️ Recent Misses:</strong>
${perf.recent_misses.map(m=>`<div style="margin-top:0.5rem">• ${m.date}: Predicted ${m.predicted}, Actual ${m.actual} (${m.error_pct}% error)</div>`).join('')}
</div>
</div>`;
}

document.getElementById('modalBody').innerHTML=`
<div class="info-grid" style="grid-template-columns:repeat(2,1fr)">
<div class="info-item"><div class="info-label">Category</div><div class="info-value">${p.category}</div></div>
<div class="info-item"><div class="info-label">Current Stock</div><div class="info-value" style="color:${p.current_stock<=p.reorder_point?'#e74c3c':'#27ae60'}">${p.current_stock} units</div></div>
<div class="info-item"><div class="info-label">Reorder Point</div><div class="info-value">${p.reorder_point} units</div></div>
<div class="info-item"><div class="info-label">Unit Cost</div><div class="info-value">₹${p.unit_cost}</div></div>
<div class="info-item"><div class="info-label">EOQ (Economic Order Qty)</div><div class="info-value">${eoq.eoq} units</div></div>
<div class="info-item"><div class="info-label">Est. Annual Demand</div><div class="info-value">${eoq.annual_demand} units</div></div>
</div>
${reorderInfo}
${forecastChart}
${performanceSection}
<div style="display:flex;gap:1rem;margin-top:1.5rem">
<button class="btn btn-success" style="flex:1" onclick="closeModal('productModal');adjustStock(${p.id},${p.current_stock},'${p.name}')">📦 Adjust Stock</button>
<button class="btn btn-warning" style="flex:1" onclick="closeModal('productModal');autoReorder(${p.id},'${p.name}')">🔄 Auto Reorder</button>
<button class="btn btn-danger" style="flex:1" onclick="deleteProduct(${p.id},'${p.name}')">🗑️ Delete</button>
</div>`;

showModal('productModal');

if(forecast.forecast && forecast.forecast.length>0){
const dates=forecast.forecast.map(f=>f.date);
const demands=forecast.forecast.map(f=>f.forecasted_demand);
const lowerBounds=forecast.forecast.map(f=>f.lower_bound);
const upperBounds=forecast.forecast.map(f=>f.upper_bound);

const lowerTrace={x:dates,y:lowerBounds,name:'Lower Bound',
line:{color:'rgba(102,126,234,0.3)',width:1,dash:'dot'},
fill:'none',showlegend:false,hoverinfo:'skip'};

const upperTrace={x:dates,y:upperBounds,name:'Upper Bound',
line:{color:'rgba(102,126,234,0.3)',width:1,dash:'dot'},
fill:'tonexty',fillcolor:'rgba(102,126,234,0.15)',
showlegend:false,hoverinfo:'skip'};

const forecastTrace={x:dates,y:demands,type:'scatter',mode:'lines+markers',name:'Forecast',
line:{color:'#667eea',width:3},marker:{size:6,color:'#764ba2'},
hovertemplate:'<b>%{x}</b><br>Demand: %{y:.1f} units<extra></extra>'};

const layout={xaxis:{title:'Date',gridcolor:'#ecf0f1'},
yaxis:{title:'Forecasted Demand (units)',gridcolor:'#ecf0f1'},
plot_bgcolor:'#f8f9fa',paper_bgcolor:'white',
margin:{l:60,r:20,t:20,b:60},height:350,
showlegend:true,legend:{x:0.02,y:0.98}};

Plotly.newPlot('forecastChart',[lowerTrace,upperTrace,forecastTrace],layout,{responsive:true,displayModeBar:false});
}
}catch(e){console.error(e);showToast('Error loading product details','error')}
}

async function adjustStock(id,currentStock,name){
const action=prompt(`Adjust stock for ${name}\\n\\nCurrent: ${currentStock} units\\n\\nOptions:\\n1. Add stock\\n2. Remove stock\\n\\nEnter 1 or 2:`);
if(!action||!['1','2'].includes(action))return;
const amount=parseInt(prompt(`Enter quantity to ${action==='1'?'add':'remove'}:`));
if(isNaN(amount)||amount<=0){showToast('Invalid quantity','error');return}
try{
const r=await fetch(`/api/products/${id}/adjust`,{
method:'POST',
headers:{'Content-Type':'application/json'},
body:JSON.stringify({action:action==='1'?'add':'remove',amount})
});
const result=await r.json();
if(result.success){
showToast(`✅ Stock ${action==='1'?'added':'removed'} successfully! New stock: ${result.new_stock}`);
loadStats();loadProducts();loadAlerts();
}else{showToast(result.error||'Failed to adjust stock','error')}
}catch(e){console.error(e);showToast('Error adjusting stock','error')}
}

async function autoReorder(id,name){
if(!confirm(`🤖 Generate AI-powered purchase order for ${name}?\\n\\nThe system will:\\n✓ Select best supplier\\n✓ Calculate optimal quantity\\n✓ Estimate delivery date`))return;
try{
const r=await fetch(`/api/auto-reorder/${id}`,{method:'POST'});
const result=await r.json();
if(result.success){
showToast(`✅ Order Created!\\n${result.quantity} units from ${result.supplier}\\nETA: ${result.expected_delivery}\\nCost: ₹${result.total_cost}`);
loadStats();loadProducts();loadOrders();
}else{showToast(result.error||'Failed to create order','error')}
}catch(e){console.error(e);showToast('Error creating order','error')}
}

async function deleteProduct(id,name){
if(!confirm(`⚠️ Delete product "${name}"?\\n\\nThis will remove:\\n• Product data\\n• Demand history\\n• All related alerts\\n\\nThis action cannot be undone!`))return;
try{
const r=await fetch(`/api/products/${id}`,{method:'DELETE'});
const result=await r.json();
if(result.success){
showToast(`✅ Product "${name}" deleted successfully`);
closeModal('productModal');
loadStats();loadProducts();
}else{showToast(result.error||'Failed to delete product','error')}
}catch(e){console.error(e);showToast('Error deleting product','error')}
}

async function resolveAlert(id){
try{
const r=await fetch(`/api/alerts/${id}/resolve`,{method:'POST'});
const result=await r.json();
if(result.success){
showToast('✅ Alert resolved');
loadStats();loadAlerts();
}else{showToast('Failed to resolve alert','error')}
}catch(e){console.error(e);showToast('Error resolving alert','error')}
}

async function simulateWeather(){
try{
const r=await fetch('/api/alerts/simulate/weather',{method:'POST'});
const result=await r.json();
if(result.success){
showToast(' Supply chain disruption simulated');
loadStats();loadAlerts();
}
}catch(e){console.error(e)}
}

async function updateOrderStatus(id,status){
if(status==='delivered'&&!confirm('Mark this order as delivered?\\n\\nStock will be automatically updated and related alerts will be resolved.'))return;
try{
const r=await fetch(`/api/orders/${id}/status`,{
method:'POST',
headers:{'Content-Type':'application/json'},
body:JSON.stringify({status})
});
const result=await r.json();
if(result.success){
showToast(`✅ Order ${status==='delivered'?'delivered - stock updated & alerts auto-resolved':status}`);
loadStats();loadOrders();loadAlerts();
if(status==='delivered')loadProducts();
}else{showToast('Failed to update order','error')}
}catch(e){console.error(e);showToast('Error updating order','error')}
}

function showAddProduct(){showModal('addProductModal')}

document.getElementById('addProductForm').addEventListener('submit',async(e)=>{
e.preventDefault();
const data={
name:document.getElementById('newProductName').value,
category:document.getElementById('newProductCategory').value,
unit_cost:parseFloat(document.getElementById('newProductCost').value),
current_stock:parseInt(document.getElementById('newProductStock').value),
reorder_point:parseInt(document.getElementById('newProductReorder').value)
};
try{
const r=await fetch('/api/products',{
method:'POST',
headers:{'Content-Type':'application/json'},
body:JSON.stringify(data)
});
const result=await r.json();
if(result.success){
showToast(`✅ Product "${data.name}" added successfully!`);
closeModal('addProductModal');
document.getElementById('addProductForm').reset();
loadStats();loadProducts();
}else{showToast(result.error||'Failed to add product','error')}
}catch(e){console.error(e);showToast('Error adding product','error')}
});

function showProducts(){document.getElementById('productsContainer').scrollIntoView({behavior:'smooth'})}
function showLowStock(){loadProducts();setTimeout(()=>document.getElementById('productsContainer').scrollIntoView({behavior:'smooth'}),100)}
function showAlerts(){document.getElementById('alertsContainer').scrollIntoView({behavior:'smooth'})}
function showOrders(){document.getElementById('ordersContainer').scrollIntoView({behavior:'smooth'})}

document.addEventListener('DOMContentLoaded',()=>{
loadStats();loadAlerts();loadProducts();loadSuppliers();loadOrders();
setInterval(()=>{loadStats();loadAlerts();},30000);
});

window.onclick=function(event){
if(event.target.classList.contains('modal')){event.target.style.display='none'}
}
</script>
</body>
</html>'''

@app.route('/api/dashboard/stats')
def dashboard_stats():
    """Dashboard KPI statistics"""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM products')
    total = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM products WHERE current_stock <= reorder_point')
    low = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM alerts WHERE resolved = 0')
    alerts = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM orders WHERE status = "pending"')
    orders = c.fetchone()[0]
    conn.close()
    return jsonify({
        "total_products": total, 
        "low_stock_items": low, 
        "active_alerts": alerts, 
        "pending_orders": orders
    })

@app.route('/api/products')
def get_products():
    """Get all products with status"""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM products ORDER BY name')
    result = []
    for p in c.fetchall():
        result.append({
            "id": p[0], 
            "name": p[1], 
            "category": p[2], 
            "current_stock": p[3],
            "reorder_point": p[4], 
            "unit_cost": p[5], 
            "status": "low" if p[3] <= p[4] else "normal"
        })
    conn.close()
    return jsonify(result)

@app.route('/api/products/<int:id>')
def get_product(id):
    """Get single product details"""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM products WHERE id = ?', (id,))
    p = c.fetchone()
    conn.close()
    if not p:
        return jsonify({"error": "Product not found"}), 404
    return jsonify({
        "id": p[0], 
        "name": p[1], 
        "category": p[2], 
        "current_stock": p[3],
        "reorder_point": p[4], 
        "unit_cost": p[5]
    })

@app.route('/api/products', methods=['POST'])
def add_product():
    """Add new product"""
    data = request.json
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute('INSERT INTO products (name, category, current_stock, reorder_point, unit_cost) VALUES (?, ?, ?, ?, ?)',
                 (data['name'], data['category'], data['current_stock'], data['reorder_point'], data['unit_cost']))
        product_id = c.lastrowid
        c.execute('INSERT INTO inventory_history (product_id, stock_level, change_amount, change_type) VALUES (?, ?, ?, ?)',
                 (product_id, data['current_stock'], data['current_stock'], 'initial'))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "id": product_id})
    except Exception as e:
        conn.close()
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/products/<int:id>', methods=['DELETE'])
def delete_product(id):
    """Delete product and all related data"""
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute('DELETE FROM products WHERE id = ?', (id,))
        c.execute('DELETE FROM alerts WHERE product_id = ?', (id,))
        c.execute('DELETE FROM inventory_history WHERE product_id = ?', (id,))
        c.execute('DELETE FROM demand_history WHERE product_id = ?', (id,))
        c.execute('DELETE FROM forecast_accuracy WHERE product_id = ?', (id,))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        conn.close()
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/products/<int:id>/adjust', methods=['POST'])
def adjust_stock(id):
    """Adjust product stock levels and auto-resolve alerts"""
    data = request.json
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute('SELECT current_stock, reorder_point, name FROM products WHERE id = ?', (id,))
        result = c.fetchone()
        if not result:
            conn.close()
            return jsonify({"success": False, "error": "Product not found"}), 404
        
        current, reorder_point, name = result
        
        if data['action'] == 'add':
            new_stock = current + data['amount']
            change_type = 'addition'
        else:
            new_stock = max(0, current - data['amount'])
            change_type = 'removal'
        
        # Update stock
        c.execute('UPDATE products SET current_stock = ?, last_updated = CURRENT_TIMESTAMP WHERE id = ?', (new_stock, id))
        c.execute('INSERT INTO inventory_history (product_id, stock_level, change_amount, change_type) VALUES (?, ?, ?, ?)',
                 (id, new_stock, data['amount'], change_type))
        
        alert_action = None
        
        # Auto-resolve stockout alerts if stock is now above reorder point
        if new_stock > reorder_point:
            # Check if there are alerts to resolve
            c.execute('SELECT COUNT(*) FROM alerts WHERE product_id = ? AND alert_type = "stockout" AND resolved = 0', (id,))
            alert_count = c.fetchone()[0]
            
            if alert_count > 0:
                c.execute('UPDATE alerts SET resolved = 1 WHERE product_id = ? AND alert_type = "stockout" AND resolved = 0', (id,))
                print(f"✅ Auto-resolved {alert_count} stockout alert(s) for {name} (stock: {new_stock} > reorder: {reorder_point})")
                alert_action = f"resolved_{alert_count}"
        
        # Create new stockout alert if stock dropped below reorder point
        elif new_stock <= reorder_point:
            c.execute('SELECT id FROM alerts WHERE product_id = ? AND alert_type = "stockout" AND resolved = 0', (id,))
            if not c.fetchone():
                severity = 'critical' if new_stock < reorder_point * 0.3 else 'high' if new_stock < reorder_point * 0.5 else 'medium'
                c.execute('INSERT INTO alerts (alert_type, severity, message, product_id) VALUES (?, ?, ?, ?)',
                         ('stockout', severity, f"Low stock: {name} has {new_stock} units (reorder: {reorder_point})", id))
                print(f"⚠️ Created new stockout alert for {name} (stock: {new_stock} ≤ reorder: {reorder_point})")
                alert_action = "created"
        
        conn.commit()
        conn.close()
        return jsonify({
            "success": True, 
            "new_stock": new_stock,
            "alert_action": alert_action
        })
    except Exception as e:
        print(f"❌ Error in adjust_stock: {e}")
        conn.close()
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/alerts')
def get_alerts():
    """Get active alerts"""
    conn = get_db()
    c = conn.cursor()
    c.execute('''SELECT a.id, a.alert_type, a.severity, a.message, a.created_at, a.product_id, a.supplier_id,
        p.name, s.name FROM alerts a 
        LEFT JOIN products p ON a.product_id = p.id
        LEFT JOIN suppliers s ON a.supplier_id = s.id
        WHERE a.resolved = 0 
        ORDER BY 
            CASE a.severity 
                WHEN 'critical' THEN 1 
                WHEN 'high' THEN 2 
                WHEN 'medium' THEN 3 
                ELSE 4 
            END,
            a.created_at DESC 
        LIMIT 50''')
    result = []
    for a in c.fetchall():
        result.append({
            "id": a[0], 
            "type": a[1], 
            "severity": a[2], 
            "message": a[3],
            "created_at": a[4], 
            "product_id": a[5], 
            "supplier_id": a[6],
            "product_name": a[7], 
            "supplier_name": a[8]
        })
    conn.close()
    return jsonify(result)

@app.route('/api/alerts/<int:id>/resolve', methods=['POST'])
def resolve_alert(id):
    """Resolve an alert"""
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute('UPDATE alerts SET resolved = 1 WHERE id = ?', (id,))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        conn.close()
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/alerts/simulate/weather', methods=['POST'])
def simulate_weather():
    """Manually trigger weather event simulation"""
    AlertsEngine.simulate_events()
    return jsonify({"success": True})

@app.route('/api/suppliers/ranking')
def supplier_ranking():
    """Get ranked suppliers"""
    return jsonify(rank_suppliers())

@app.route('/api/orders')
def get_orders():
    """Get recent orders"""
    conn = get_db()
    c = conn.cursor()
    c.execute('''SELECT o.id, o.quantity, o.order_date, o.expected_delivery, o.status, o.total_cost,
        p.name, s.name FROM orders o
        JOIN products p ON o.product_id = p.id
        JOIN suppliers s ON o.supplier_id = s.id
        ORDER BY o.order_date DESC LIMIT 20''')
    result = []
    for o in c.fetchall():
        result.append({
            "id": o[0], 
            "quantity": o[1], 
            "order_date": o[2],
            "expected_delivery": o[3], 
            "status": o[4], 
            "total_cost": o[5],
            "product_name": o[6], 
            "supplier_name": o[7]
        })
    conn.close()
    return jsonify(result)

@app.route('/api/orders/<int:id>/status', methods=['POST'])
def update_order_status(id):
    """Update order status and auto-resolve related alerts"""
    data = request.json
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute('UPDATE orders SET status = ? WHERE id = ?', (data['status'], id))
        
        alert_action = None
        
        # If delivered, update stock and auto-resolve alerts
        if data['status'] == 'delivered':
            c.execute('SELECT product_id, quantity FROM orders WHERE id = ?', (id,))
            order = c.fetchone()
            if order:
                product_id, quantity = order
                c.execute('SELECT current_stock, reorder_point, name FROM products WHERE id = ?', (product_id,))
                product = c.fetchone()
                if product:
                    current, reorder_point, name = product
                    new_stock = current + quantity
                    
                    c.execute('UPDATE products SET current_stock = ?, last_updated = CURRENT_TIMESTAMP WHERE id = ?',
                             (new_stock, product_id))
                    c.execute('INSERT INTO inventory_history (product_id, stock_level, change_amount, change_type) VALUES (?, ?, ?, ?)',
                             (product_id, new_stock, quantity, 'delivery'))
                    
                    # Auto-resolve stockout alerts if stock is now above reorder point
                    if new_stock > reorder_point:
                        # Check if there are alerts to resolve
                        c.execute('SELECT COUNT(*) FROM alerts WHERE product_id = ? AND alert_type = "stockout" AND resolved = 0', 
                                 (product_id,))
                        alert_count = c.fetchone()[0]
                        
                        if alert_count > 0:
                            c.execute('UPDATE alerts SET resolved = 1 WHERE product_id = ? AND alert_type = "stockout" AND resolved = 0', 
                                     (product_id,))
                            resolved_count = c.rowcount
                            print(f"✅ Auto-resolved {resolved_count} stockout alert(s) for {name} after delivery (stock: {new_stock} > reorder: {reorder_point})")
                            alert_action = f"resolved_{resolved_count}"
                    
                    print(f"📦 Delivered: {name} +{quantity} units, new stock: {new_stock}")
        
        conn.commit()
        conn.close()
        return jsonify({
            "success": True,
            "alert_action": alert_action
        })
    except Exception as e:
        print(f"❌ Error in update_order_status: {e}")
        conn.close()
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/eoq/<int:product_id>')
def get_eoq(product_id):
    """Get Economic Order Quantity"""
    return jsonify(calculate_eoq(product_id))

@app.route('/api/forecast/<int:product_id>')
def get_forecast(product_id):
    """Get ML demand forecast with confidence intervals"""
    return jsonify(DemandForecaster.calculate_forecast(product_id))

@app.route('/api/reorder-recommendation/<int:product_id>')
def get_reorder_recommendation(product_id):
    """Get AI-powered reorder recommendation"""
    return jsonify(DemandForecaster.get_reorder_recommendation(product_id))

@app.route('/api/model-performance/<int:product_id>')
def get_model_performance(product_id):
    """Get forecast model performance metrics"""
    conn = get_db()
    c = conn.cursor()
    
    # Get last 30 predictions vs actuals
    c.execute('''SELECT forecast_date, predicted_demand, actual_demand, error_pct, model_type, confidence
                FROM forecast_accuracy 
                WHERE product_id = ? 
                ORDER BY forecast_date DESC 
                LIMIT 30''', (product_id,))
    
    results = c.fetchall()
    conn.close()
    
    if not results:
        return jsonify({"error": "No performance data available yet"})
    
    errors = [row[3] for row in results if row[3] is not None]
    
    if not errors:
        return jsonify({"error": "No error data available"})
    
    mape = sum(errors) / len(errors)
    accuracy = max(0, 100 - mape)
    
    correct_predictions = sum(1 for e in errors if e < 10)
    large_errors = sum(1 for e in errors if e > 20)
    
    # Trend analysis
    recent_errors = errors[:7] if len(errors) >= 7 else errors
    older_errors = errors[7:14] if len(errors) >= 14 else errors
    error_trend = "improving" if (sum(recent_errors) / len(recent_errors)) < (sum(older_errors) / len(older_errors) if older_errors else 999) else "degrading"
    
    # Recent misses
    recent_misses = []
    for row in results[:10]:
        if row[3] and row[3] > 15:
            recent_misses.append({
                "date": row[0],
                "predicted": round(row[1], 1),
                "actual": row[2],
                "error_pct": round(row[3], 1)
            })
    
    return jsonify({
        "accuracy_pct": round(accuracy, 1),
        "mape": round(mape, 1),
        "predictions_tracked": len(results),
        "correct_predictions": correct_predictions,
        "large_errors": large_errors,
        "error_trend": error_trend,
        "recent_misses": recent_misses[:5]
    })

@app.route('/api/debug/product/<int:product_id>/alerts')
def debug_product_alerts(product_id):
    """Debug endpoint to check alerts for a product"""
    conn = get_db()
    c = conn.cursor()
    
    # Get product info
    c.execute('SELECT name, current_stock, reorder_point FROM products WHERE id = ?', (product_id,))
    product = c.fetchone()
    
    if not product:
        conn.close()
        return jsonify({"error": "Product not found"}), 404
    
    name, stock, reorder = product
    
    # Get all alerts for this product
    c.execute('''SELECT id, alert_type, severity, message, resolved, created_at 
                FROM alerts 
                WHERE product_id = ? 
                ORDER BY created_at DESC''', (product_id,))
    
    alerts = []
    for row in c.fetchall():
        alerts.append({
            "id": row[0],
            "type": row[1],
            "severity": row[2],
            "message": row[3],
            "resolved": bool(row[4]),
            "created_at": row[5]
        })
    
    # Count active vs resolved
    active_count = sum(1 for a in alerts if not a['resolved'])
    resolved_count = sum(1 for a in alerts if a['resolved'])
    
    conn.close()
    
    return jsonify({
        "product": {
            "id": product_id,
            "name": name,
            "current_stock": stock,
            "reorder_point": reorder,
            "is_low_stock": stock <= reorder
        },
        "alerts": alerts,
        "summary": {
            "total": len(alerts),
            "active": active_count,
            "resolved": resolved_count
        }
    })

@app.route('/api/alerts/stats')
def get_alert_stats():
    """Get alert statistics including auto-resolution info"""
    conn = get_db()
    c = conn.cursor()
    
    # Total alerts created today
    c.execute("SELECT COUNT(*) FROM alerts WHERE DATE(created_at) = DATE('now')")
    today_created = c.fetchone()[0]
    
    # Total alerts resolved today
    c.execute("SELECT COUNT(*) FROM alerts WHERE DATE(created_at) = DATE('now') AND resolved = 1")
    today_resolved = c.fetchone()[0]
    
    # Active alerts by severity
    c.execute("SELECT severity, COUNT(*) FROM alerts WHERE resolved = 0 GROUP BY severity")
    by_severity = {row[0]: row[1] for row in c.fetchall()}
    
    # Auto-resolution rate (last 7 days)
    c.execute("""SELECT 
        COUNT(CASE WHEN resolved = 1 THEN 1 END) as resolved,
        COUNT(*) as total
        FROM alerts 
        WHERE created_at >= datetime('now', '-7 days')
        AND alert_type = 'stockout'""")
    
    stats = c.fetchone()
    auto_resolution_rate = (stats[0] / stats[1] * 100) if stats[1] > 0 else 0
    
    conn.close()
    
    return jsonify({
        "today_created": today_created,
        "today_resolved": today_resolved,
        "active_by_severity": by_severity,
        "total_resolved_7d": stats[0],
        "total_created_7d": stats[1]
    })

@app.route('/api/auto-reorder/<int:product_id>', methods=['POST'])
def auto_reorder(product_id):
    """AI-powered automatic purchase order generation"""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT name, unit_cost FROM products WHERE id = ?', (product_id,))
    product = c.fetchone()
    
    if not product:
        conn.close()
        return jsonify({"error": "Product not found"}), 404
    
    # Get best supplier
    suppliers = rank_suppliers()
    if not suppliers:
        conn.close()
        return jsonify({"error": "No suppliers available"}), 400
    
    best = suppliers[0]
    
    # Calculate optimal order quantity
    reorder_rec = DemandForecaster.get_reorder_recommendation(product_id)
    
    if "error" in reorder_rec:
        eoq_data = calculate_eoq(product_id)
        qty = max(int(eoq_data['eoq']), 10)
    else:
        qty = max(reorder_rec['recommended_order_qty'], 10)
    
    total_cost = qty * product[1]
    delivery = datetime.now() + timedelta(days=best['delivery_time'])
    
    c.execute('INSERT INTO orders (product_id, supplier_id, quantity, expected_delivery, status, total_cost) VALUES (?, ?, ?, ?, ?, ?)',
             (product_id, best['id'], qty, delivery, 'pending', total_cost))
    order_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return jsonify({
        "success": True, 
        "order_id": order_id, 
        "product": product[0],
        "supplier": best['name'], 
        "quantity": qty,
        "expected_delivery": delivery.strftime("%Y-%m-%d"),
        "total_cost": total_cost
    })

if __name__ == '__main__':
    init_db()
    
    # Start background monitoring thread
    monitor_thread = threading.Thread(target=background_monitor, daemon=True)
    monitor_thread.start()
    
    print("=" * 80)
    print("🌐 AI-POWERED SUPPLY CHAIN OPTIMIZER - V2.0")
    print("=" * 80)
    print("\n✅ Server starting...")
    print("📊 Dashboard: http://127.0.0.1:5000")
    print("\n🤖 AI FEATURES ACTIVE:")
    print("   ✓ ML Demand Forecasting (Polynomial + Linear Regression)")
    print("   ✓ Confidence Intervals & Prediction Bands")
    print("   ✓ Real-time Model Performance Tracking")
    print("   ✓ Dynamic Safety Stock Calculation")
    print("   ✓ Economic Order Quantity (EOQ) Optimization")
    print("   ✓ Multi-factor Supplier Risk Analysis")
    print("   ✓ Automated Alert System with Severity Levels")
    print("   ✓ Auto Purchase Order Generation")
    print("   ✓ Intelligent Reorder Recommendations")
    print("\n📈 IMPROVEMENTS FROM V1:")
    print("   ✓ Direct system integration (no CSV uploads)")
    print("   ✓ Forecast accuracy tracking")
    print("   ✓ Confidence-based safety stock")
    print("   ✓ Model performance dashboard")
    print("   ✓ Enhanced error handling")
    print("\n⚡ Background monitoring active (30s interval)")
    print("=" * 80)
    print("=" * 80)

if __name__ == "__main__":
    app.run(debug=True)