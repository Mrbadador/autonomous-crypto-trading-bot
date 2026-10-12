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

    # Panel 8: AI Action Confidence
    confidence_panel = {
        "id": next_id,
        "title": "AI Action Confidence",
        "type": "gauge",
        "gridPos": {"x": 0, "y": max_y, "w": 6, "h": 7},
        "targets": [
            {
                "expr": "swarm_action_confidence",
                "legendFormat": "{{agent}}",
                "refId": "A"
            }
        ],
        "fieldConfig": {
            "defaults": {
                "max": 1,
                "min": 0,
                "unit": "percentunit",
                "color": {"mode": "thresholds"},
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "red", "value": None},
                        {"color": "orange", "value": 0.5},
                        {"color": "green", "value": 0.8}
                    ]
                }
            },
            "overrides": []
        }
    }
    
    # Panel 9: Drawdown Monitor
    drawdown_panel = {
        "id": next_id + 1,
        "title": "Agent Drawdown from Peak",
        "type": "bargauge",
        "gridPos": {"x": 6, "y": max_y, "w": 6, "h": 7},
        "targets": [
            {
                "expr": "swarm_drawdown_percent",
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
                        {"color": "green", "value": None},
                        {"color": "orange", "value": 10},
                        {"color": "red", "value": 25}
                    ]
                }
            },
            "overrides": []
        }
    }

    # Panel 10: Lifetime Trades & Volume
    volume_panel = {
        "id": next_id + 2,
        "title": "Lifetime Trades & Volume",
        "type": "stat",
        "gridPos": {"x": 12, "y": max_y, "w": 6, "h": 7},
        "targets": [
            {
                "expr": "swarm_trades_total",
                "legendFormat": "Trades: {{agent}}",
                "refId": "A"
            },
            {
                "expr": "swarm_trade_volume_usdt_total",
                "legendFormat": "Volume: {{agent}}",
                "refId": "B"
            }
        ],
        "options": {
            "colorMode": "value",
            "graphMode": "none",
            "justifyMode": "auto"
        },
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"}
            },
            "overrides": [
                {
                    "matcher": {"id": "byRegexp", "options": ".*Volume.*"},
                    "properties": [
                        {"id": "unit", "value": "currencyUSD"}
                    ]
                }
            ]
        }
    }
    
    # Panel 11: Emergency Brake Status
    brake_panel = {
        "id": next_id + 3,
        "title": "Emergency Brake Status",
        "type": "stat",
        "gridPos": {"x": 18, "y": max_y, "w": 6, "h": 7},
        "targets": [
            {
                "expr": "system_emergency_brake_active",
                "legendFormat": "Flash Crash Override",
                "refId": "A"
            }
        ],
        "options": {
            "colorMode": "background",
            "graphMode": "none",
            "justifyMode": "auto"
        },
        "fieldConfig": {
            "defaults": {
                "mappings": [
                    {
                        "options": {
                            "0": {"color": "green", "index": 0, "text": "SAFE (OFF)"},
                            "1": {"color": "red", "index": 1, "text": "VETO (ACTIVE)"}
                        },
                        "type": "value"
                    }
                ]
            },
            "overrides": []
        }
    }

    panels.append(confidence_panel)
    panels.append(drawdown_panel)
    panels.append(volume_panel)
    panels.append(brake_panel)
    
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
