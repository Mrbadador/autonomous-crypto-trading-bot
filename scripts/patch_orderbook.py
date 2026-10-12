import sys
import os
import textwrap

def patch():
    file_path = 'berserker_trader.py'
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Define the function text explicitly without python interpreter interpretation hazards
    func_body = r"""
    async def _check_orderbook_wall(self, symbol: str, side: str, entry_p: float) -> bool:
        try:
            async with LatencyTracer("ORDERBOOK_AUDIT", budget_ms=200):
                ob = await self.exchange.fetch_order_book(symbol, limit=50)
                target_side = 'bids' if side == 'long' else 'asks'
                data = ob.get(target_side, [])
                
                if not data or len(data) < 5:
                    log.warning(f"[ORDERBOOK] Insufficient depth on {target_side}. Blocking strike.")
                    return False
                
                vols = [float(x[1]) for x in data]
                avg_vol = sum(vols) / len(vols)
                
                boundary_mult = 0.995 if side == 'long' else 1.005
                boundary_price = entry_p * boundary_mult
                
                relevant_vols = []
                for p_raw, v_raw in data:
                    p_pt, v_pt = float(p_raw), float(v_raw)
                    if side == 'long':
                        if boundary_price <= p_pt <= entry_p:
                            relevant_vols.append(v_pt)
                    else:
                        if entry_p <= p_pt <= boundary_price:
                            relevant_vols.append(v_pt)
                
                if not relevant_vols:
                    log.warning(f"[ORDERBOOK] Zero {target_side.upper()} walls found within 0.5% range. Veto.")
                    return False
                
                max_local = max(relevant_vols)
                if max_local >= (avg_vol * 2.5):
                    log.info(f"[ORDERBOOK] Verified {target_side[:-1].upper()} Wall: {max_local:.1f} ({max_local/avg_vol:.1f}x avg). Support Confirmed.")
                    return True
                else:
                    log.warning(f"[ORDERBOOK] VETO: Max wall {max_local:.1f} < avg*2.5 ({avg_vol*2.5:.1f}).")
                    return False
        except Exception as e:
            log.error(f"[ORDERBOOK] Audit Exception: {e}. Fallback TRUE.")
            return True

"""
    # Fix logic using basic string replacement to minimize chance of misalignment
    
    # Injection Point 1: Just before `async def monitor_position(`
    target1 = "    async def monitor_position("
    if target1 not in content:
        print("Cannot find monitor_position")
        return False
    
    content = content.replace(target1, func_body + "\n" + target1)

    # Injection Point 2: Locate existing placement line and derive indent.
    # Because line may differ, use regex.
    import re
    pattern = r'(\s+)async with LatencyTracer\("ORDER_PLACEMENT"'
    match = re.search(pattern, content)
    if not match:
        print("Cannot find ORDER_PLACEMENT tracer")
        return False

    indent = match.group(1)
    call_block = (
        f"\n{indent}# V24.0 Order Book Wall Audit\n"
        f"{indent}is_supported = await self._check_orderbook_wall(symbol, signal, exec_p)\n"
        f"{indent}if not is_supported:\n"
        f"{indent}    await self._set_status(f\"Status: VETO by Liquidity Wall ({{symbol}})\")\n"
        f"{indent}    return\n\n"
        f"{indent}async with LatencyTracer(\"ORDER_PLACEMENT\""
    )
    
    # Replace original with block + original start
    content = content.replace(f"{indent}async with LatencyTracer(\"ORDER_PLACEMENT\"", call_block)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("Patch Success.")
    return True

if __name__ == '__main__':
    sys.exit(0 if patch() else 1)
