import urllib.request
import urllib.error
import json
import base64

def get_dashboard(uid):
    url = f'http://localhost:3000/api/dashboards/uid/{uid}'
    req = urllib.request.Request(url)
    base64string = base64.b64encode(b'admin:admin').decode('utf-8')
    req.add_header("Authorization", f"Basic {base64string}")
    
    response = urllib.request.urlopen(req)
    data = json.loads(response.read().decode())
    return data['dashboard']

def push_dashboard(dashboard):
    url = 'http://localhost:3000/api/dashboards/db'
    payload = {
        "dashboard": dashboard,
        "overwrite": True
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    req.add_header('Content-Type', 'application/json')
    req.add_header('Accept', 'application/json')
    base64string = base64.b64encode(b'admin:admin').decode('utf-8')
    req.add_header("Authorization", f"Basic {base64string}")
    
    response = urllib.request.urlopen(req)
    print("Dashboard pushed successfully:", response.read().decode())

def add_advanced_panels(dashboard):
    panels = dashboard.get('panels', [])
    
    # Calculate starting Y
    max_y = 0
    for p in panels:
        y_val = p.get('gridPos', {}).get('y', 0)
        h_val = p.get('gridPos', {}).get('h', 0)
        max_y = max(max_y, y_val + h_val)
        
    next_id = max([p.get('id', 0) for p in panels]) + 1 if panels else 1

    # Panel 4: System Health (CPU & RAM)
    sys_health_panel = {
        "id": next_id,
        "title": "System Health (CPU / RAM)",
        "type": "gauge",
        "gridPos": {"x": 0, "y": max_y, "w": 6, "h": 7},
        "targets": [
            {
                "expr": "system_cpu_usage_percent",
                "legendFormat": "CPU %",
                "refId": "A"
            },
            {
                "expr": "system_memory_usage_percent",
                "legendFormat": "RAM %",
                "refId": "B"
            }
        ],
        "fieldConfig": {
            "defaults": {
                "max": 100,
                "min": 0,
                "unit": "percent",
                "color": {"mode": "thresholds"},
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "orange", "value": 75},
                        {"color": "red", "value": 90}
                    ]
                }
            },
            "overrides": []
        }
    }
    
    # Panel 5: Swarm Win Rates
    win_rate_panel = {
        "id": next_id + 1,
        "title": "Swarm Win Rate %",
        "type": "bargauge",
        "gridPos": {"x": 6, "y": max_y, "w": 6, "h": 7},
        "targets": [
            {
                "expr": "swarm_win_rate_percent",
                "legendFormat": "{{agent}}",
                "refId": "A"
            }
        ],
        "options": {
            "displayMode": "gradient",
            "orientation": "horizontal",
            "showUnfilled": True
        },
        "fieldConfig": {
            "defaults": {
                "max": 100,
                "min": 0,
                "unit": "percent",
                "color": {"mode": "thresholds"},
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "red", "value": None},
                        {"color": "orange", "value": 45},
                        {"color": "green", "value": 55}
                    ]
                }
            },
            "overrides": []
        }
    }

    # Panel 6: Swarm Capital & Exposure
    capital_panel = {
        "id": next_id + 2,
        "title": "Agent Capital & Open Exposure",
        "type": "barchart",
        "gridPos": {"x": 12, "y": max_y, "w": 6, "h": 7},
        "targets": [
            {
                "expr": "swarm_capital_usdt",
                "legendFormat": "Capital: {{agent}}",
                "refId": "A"
            },
            {
                "expr": "swarm_latent_exposure_usdt",
                "legendFormat": "Exposure: {{agent}}",
                "refId": "B"
            }
        ],
        "options": {
            "barWidth": 0.8,
            "groupWidth": 0.7,
            "orientation": "auto",
            "stacking": "none",
            "legend": {"displayMode": "list", "placement": "bottom"}
        },
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "unit": "currencyUSD"
            },
            "overrides": []
        }
    }
    
    # Panel 7: Market Regime (Hurst & ADX)
    regime_panel = {
        "id": next_id + 3,
        "title": "Market Regime (Hurst vs ADX)",
        "type": "timeseries",
        "gridPos": {"x": 18, "y": max_y, "w": 6, "h": 7},
        "targets": [
            {
                "expr": "market_regime_hurst",
                "legendFormat": "Hurst Exponent",
                "refId": "A"
            },
            {
                "expr": "market_regime_adx",
                "legendFormat": "ADX Trend Strength",
                "refId": "B"
            }
        ],
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "custom": {
                    "drawStyle": "line",
                    "fillOpacity": 10,
                    "lineWidth": 2
                }
            },
            "overrides": [
                {
                    "matcher": {"id": "byName", "options": "Hurst Exponent"},
                    "properties": [
                        {"id": "custom.axisPlacement", "value": "left"},
                        {"id": "min", "value": 0},
                        {"id": "max", "value": 1}
                    ]
                },
                {
                    "matcher": {"id": "byName", "options": "ADX Trend Strength"},
                    "properties": [
                        {"id": "custom.axisPlacement", "value": "right"},
                        {"id": "min", "value": 0},
                        {"id": "max", "value": 100}
                    ]
                }
            ]
        }
    }

    panels.append(sys_health_panel)
    panels.append(win_rate_panel)
    panels.append(capital_panel)
    panels.append(regime_panel)
    
    dashboard['panels'] = panels
    return dashboard

if __name__ == '__main__':
    uid = 'llama-trading-bot-01'
    print("Fetching current dashboard...")
    dash = get_dashboard(uid)
    print("Modifying dashboard with new panels...")
    new_dash = add_advanced_panels(dash)
    print("Pushing updated dashboard...")
    push_dashboard(new_dash)
