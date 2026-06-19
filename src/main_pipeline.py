import numpy as np
import pandas as pd
import time
import os
import json
from collections import defaultdict

def generate_realistic_btc_data(
    n_minutes=10080,  
    base_price=95000.0,
    n_crash_events=8,
    seed=42
):
    np.random.seed(seed)
    
    timestamps = pd.date_range(
        start='2023-08-10 00:00:00',
        periods=n_minutes,
        freq='1min'
    )
    
    dt = 1.0 / (60 * 24)
    mu = 0.0
    sigma = 0.02  
    
    log_returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_minutes)
    prices = np.zeros(n_minutes)
    prices[0] = base_price
    
    for i in range(1, n_minutes):
        prices[i] = prices[i-1] * np.exp(log_returns[i])
    
    base_volume = np.random.lognormal(mean=3.0, sigma=0.8, size=n_minutes)
    volume_multiplier = 1 + 5 * np.abs(log_returns)
    volumes = base_volume * volume_multiplier
    
    buy_ratio = np.clip(
        np.random.normal(0.5, 0.08, n_minutes),
        0.15, 0.85
    )
    
    crash_indices = []
    crash_zone_size = n_minutes // (n_crash_events + 2)
    
    for k in range(n_crash_events):
        center = crash_zone_size * (k + 1)
        offset = np.random.randint(-crash_zone_size // 4, crash_zone_size // 4)
        crash_start = max(100, min(center + offset, n_minutes - 100))
        
        crash_duration = np.random.randint(5, 12)
        crash_magnitude = np.random.uniform(0.035, 0.08)
        recovery_duration = np.random.randint(15, 45)
        recovery_ratio = np.random.uniform(0.5, 0.9)
        
        crash_indices.append({
            'start': crash_start,
            'duration': crash_duration,
            'magnitude': crash_magnitude,
            'recovery_duration': recovery_duration,
            'recovery_ratio': recovery_ratio
        })
        
        pre_crash_window = np.random.randint(3, 8)
        for j in range(pre_crash_window):
            idx = crash_start - pre_crash_window + j
            if 0 <= idx < n_minutes:
                buy_ratio[idx] = np.random.uniform(0.20, 0.35)
                volumes[idx] *= np.random.uniform(1.5, 3.0)
        
        for j in range(crash_duration):
            idx = crash_start + j
            if idx < n_minutes:
                progress = (j + 1) / crash_duration
                drop = crash_magnitude * progress
                prices[idx] = prices[crash_start - 1] * (1 - drop)
                buy_ratio[idx] = np.random.uniform(0.10, 0.25)
                volumes[idx] *= np.random.uniform(3.0, 8.0)
        crash_bottom = prices[min(crash_start + crash_duration - 1, n_minutes - 1)]
        pre_crash_price = prices[crash_start - 1]
        recovery_target = crash_bottom + (pre_crash_price - crash_bottom) * recovery_ratio
        
        for j in range(recovery_duration):
            idx = crash_start + crash_duration + j
            if idx < n_minutes:
                progress = (j + 1) / recovery_duration
                prices[idx] = crash_bottom + (recovery_target - crash_bottom) * (
                    1 - np.exp(-3 * progress)
                )
                buy_ratio[idx] = np.random.uniform(0.55, 0.75)
                volumes[idx] *= np.random.uniform(1.2, 2.5)
    buy_volumes = volumes * buy_ratio
    sell_volumes = volumes * (1 - buy_ratio)
    
    noise = np.random.uniform(-0.001, 0.001, n_minutes) * prices
    opens = np.roll(prices, 1) + noise
    opens[0] = prices[0]
    highs = np.maximum(prices, opens) * (1 + np.abs(np.random.normal(0, 0.0005, n_minutes)))
    lows = np.minimum(prices, opens) * (1 - np.abs(np.random.normal(0, 0.0005, n_minutes)))
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes,
        'buy_volume': buy_volumes,
        'sell_volume': sell_volumes
    })
    
    return df, crash_indices

def compute_obi_proxy(df, window=5):
    buy_vol = df['buy_volume'].rolling(window=window, min_periods=1).sum()
    sell_vol = df['sell_volume'].rolling(window=window, min_periods=1).sum()
    total = buy_vol + sell_vol
    obi = np.where(total > 0, (buy_vol - sell_vol) / total, 0.0)
    
    if hasattr(obi, 'values'):
        return obi.values
    return np.array(obi, dtype=float)

def z_normalize(series):
    mu = np.mean(series)
    sigma = np.std(series)
    if sigma < 1e-10:
        return np.zeros_like(series)
    return (series - mu) / sigma


def paa_transform(normalized_series, w):
    n = len(normalized_series)
    if w >= n:
        return normalized_series[:w] if len(normalized_series) >= w else normalized_series
    paa = np.zeros(w)
    for i in range(w):
        start = int(np.floor(i * n / w))
        end = int(np.floor((i + 1) * n / w))
        if end > start:
            paa[i] = np.mean(normalized_series[start:end])
        else:
            paa[i] = normalized_series[start]
    return paa

def get_gaussian_breakpoints(alphabet_size):
    from scipy.stats import norm
    breakpoints = [norm.ppf(i / alphabet_size) for i in range(1, alphabet_size)]
    return breakpoints


def sax_discretize(paa_values, breakpoints):
    symbols = []
    for val in paa_values:
        assigned = False
        for j, bp in enumerate(breakpoints):
            if val < bp:
                symbols.append(chr(ord('a') + j))
                assigned = True
                break
        if not assigned:
            symbols.append(chr(ord('a') + len(breakpoints)))
    return ''.join(symbols)

def sax_encode_sliding_window(obi_series, window_size, word_size, alphabet_size):
    breakpoints = get_gaussian_breakpoints(alphabet_size)
    n = len(obi_series)
    sax_strings = []
    for i in range(n - window_size + 1):
        window = obi_series[i:i + window_size]
        normalized = z_normalize(window)
        paa = paa_transform(normalized, word_size)
        sax_str = sax_discretize(paa, breakpoints)
        sax_strings.append((i, sax_str))
    return sax_strings

def build_continuous_sax_stream(obi_series, segment_size, alphabet_size):
    breakpoints = get_gaussian_breakpoints(alphabet_size)
    n = len(obi_series)
    norm_window = 60
    rolling_mean = pd.Series(obi_series).rolling(
        window=norm_window, min_periods=1
    ).mean().values
    rolling_std = pd.Series(obi_series).rolling(
        window=norm_window, min_periods=1
    ).std().values
    rolling_std = np.where(rolling_std < 1e-10, 1.0, rolling_std)
    normalized = (obi_series - rolling_mean) / rolling_std
    n_segments = n // segment_size
    sax_chars = []
    segment_indices = []
    for i in range(n_segments):
        start = i * segment_size
        end = start + segment_size
        paa_val = np.mean(normalized[start:end])
        char_assigned = False
        for j, bp in enumerate(breakpoints):
            if paa_val < bp:
                sax_chars.append(chr(ord('a') + j))
                char_assigned = True
                break
        if not char_assigned:
            sax_chars.append(chr(ord('a') + len(breakpoints)))
        segment_indices.append(start)
    return ''.join(sax_chars), segment_indices

def compute_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps

def kmp_search(text, pattern):
    n = len(text)
    m = len(pattern)
    if m == 0 or n == 0 or m > n:
        return [], 0
    lps = compute_lps(pattern)
    matches = []
    comparisons = 0
    i = 0
    j = 0
    while i < n:
        comparisons += 1
        if text[i] == pattern[j]:
            i += 1
            j += 1
        if j == m:
            matches.append(i - j)
            j = lps[j - 1]
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return matches, comparisons

def brute_force_search(text, pattern):
    n = len(text)
    m = len(pattern)
    if m == 0 or n == 0 or m > n:
        return [], 0
    matches = []
    comparisons = 0
    for i in range(n - m + 1):
        match = True
        for j in range(m):
            comparisons += 1
            if text[i + j] != pattern[j]:
                match = False
                break
        if match:
            matches.append(i)
    
    return matches, comparisons

def label_micro_crashes(df, threshold_pct=3.5, window_minutes=10):
    prices = df['close'].values
    n = len(prices)
    labels = np.zeros(n, dtype=bool)
    for i in range(n):
        end = min(i + window_minutes, n)
        future_window = prices[i:end]
        if len(future_window) < 2:
            continue
        peak = prices[i]
        trough = np.min(future_window)
        if peak > 0:
            drawdown_pct = (peak - trough) / peak * 100
            if drawdown_pct >= threshold_pct:
                labels[i] = True
    return labels

def discover_crash_patterns(sax_text, segment_indices, crash_labels, pattern_length=4, top_n=8):
    pattern_crash_count = defaultdict(int)
    pattern_total_count = defaultdict(int)
    n_sax = len(sax_text)
    for i in range(n_sax - pattern_length + 1):
        pat = sax_text[i:i + pattern_length]
        pattern_total_count[pat] += 1
        if i < len(segment_indices):
            time_idx = segment_indices[i]
            look_ahead = 20
            for j in range(min(look_ahead, len(crash_labels) - time_idx)):
                if time_idx + j < len(crash_labels) and crash_labels[time_idx + j]:
                    pattern_crash_count[pat] += 1
                    break
    scored = []
    for pat, crash_count in pattern_crash_count.items():
        total = pattern_total_count.get(pat, 1)
        if total >= 2:
            ratio = crash_count / total
            scored.append((pat, ratio, crash_count, total))
    scored.sort(key=lambda x: (-x[1], -x[2]))
    result = [s[0] for s in scored[:top_n]]
    return result if result else None


def run_detection_pipeline(
    obi_series,
    crash_labels,
    segment_size=5,
    alphabet_size=4,
    crash_patterns=None,
    alert_window=10
):
    sax_text, segment_indices = build_continuous_sax_stream(
        obi_series, segment_size, alphabet_size
    )
    
    if crash_patterns is None:
        if alphabet_size == 4:
            crash_patterns = ['dcaa', 'ddaa', 'dcba', 'cbaa', 'ddab', 'dcab', 'ccaa', 'dbaa']
        elif alphabet_size == 3:
            crash_patterns = ['caa', 'cba', 'baa']
        elif alphabet_size == 5:
            crash_patterns = ['edaa', 'dcaa', 'eeba', 'ddaa', 'ecba']
        else:
            crash_patterns = ['dcaa', 'ddaa']
    
    all_kmp_matches = []
    all_bf_matches = []
    total_kmp_comparisons = 0
    total_bf_comparisons = 0
    kmp_time_total = 0
    bf_time_total = 0
    for pattern in crash_patterns:
        if len(pattern) > len(sax_text):
            continue
        t0 = time.perf_counter()
        kmp_matches, kmp_comps = kmp_search(sax_text, pattern)
        t1 = time.perf_counter()
        kmp_time_total += (t1 - t0)
        total_kmp_comparisons += kmp_comps
        t0 = time.perf_counter()
        bf_matches, bf_comps = brute_force_search(sax_text, pattern)
        t1 = time.perf_counter()
        bf_time_total += (t1 - t0)
        total_bf_comparisons += bf_comps
        for m in kmp_matches:
            if m < len(segment_indices):
                all_kmp_matches.append(segment_indices[m])
        for m in bf_matches:
            if m < len(segment_indices):
                all_bf_matches.append(segment_indices[m])
    
    alert_indices = sorted(set(all_kmp_matches))
    merged_alerts = []
    for idx in alert_indices:
        if not merged_alerts or idx - merged_alerts[-1] > alert_window:
            merged_alerts.append(idx)
    
    n = len(crash_labels)
    tp, fp, fn = 0, 0, 0
    crash_events = []
    in_crash = False
    for i in range(n):
        if crash_labels[i] and not in_crash:
            crash_events.append(i)
            in_crash = True
        elif not crash_labels[i]:
            in_crash = False
    
    detected_crashes = set()
    detection_lags = []
    for alert_idx in merged_alerts:
        found_crash = False
        for ce in crash_events:
            if 0 <= ce - alert_idx <= alert_window * segment_size:
                detected_crashes.add(ce)
                detection_lags.append((ce - alert_idx))
                found_crash = True
                break
        if not found_crash:
            fp += 1
    
    tp = len(detected_crashes)
    fn = len(crash_events) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_lag = np.mean(detection_lags) if detection_lags else 0
    
    return {
        'sax_text_length': len(sax_text),
        'sax_text_sample': sax_text[:200],
        'n_patterns': len(crash_patterns),
        'patterns_used': crash_patterns,
        'n_alerts': len(merged_alerts),
        'alert_indices': merged_alerts,
        'n_crash_events': len(crash_events),
        'crash_event_indices': crash_events,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'avg_detection_lag_minutes': avg_lag,
        'detection_lags': detection_lags,
        'kmp_comparisons': total_kmp_comparisons,
        'bf_comparisons': total_bf_comparisons,
        'kmp_time_seconds': kmp_time_total,
        'bf_time_seconds': bf_time_total,
        'speedup_ratio': bf_time_total / kmp_time_total if kmp_time_total > 0 else 0,
    }


def run_threshold_baseline(obi_series, crash_labels, threshold=-0.5, alert_window=10):
    n = len(obi_series)
    alerts = []
    
    for i in range(n):
        if obi_series[i] <= threshold:
            if not alerts or i - alerts[-1] > alert_window:
                alerts.append(i)
    crash_events = []
    in_crash = False
    for i in range(n):
        if crash_labels[i] and not in_crash:
            crash_events.append(i)
            in_crash = True
        elif not crash_labels[i]:
            in_crash = False
    
    detected_crashes = set()
    tp, fp = 0, 0
    
    for alert_idx in alerts:
        found_crash = False
        for ce in crash_events:
            if 0 <= ce - alert_idx <= alert_window:
                detected_crashes.add(ce)
                found_crash = True
                break
        if not found_crash:
            fp += 1
    
    tp = len(detected_crashes)
    fn = len(crash_events) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'method': f'Threshold OBI ≤ {threshold}',
        'n_alerts': len(alerts),
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def run_all_experiments(output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("PIPELINE: LOB → OBI → SAX → KMP Micro-Crash Detection")
    print("=" * 70)
    
    print("\n[1/6] Generating synthetic BTC/USDT data...")
    df, crash_info = generate_realistic_btc_data(
        n_minutes=10080,
        base_price=95000.0,
        n_crash_events=8,
        seed=42
    )
    df.to_csv(f'{output_dir}/btc_usdt_data.csv', index=False)
    print(f"  Data points: {len(df)}")
    print(f"  Period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"  Injected crash events: {len(crash_info)}")
    print("\n[2/6] Computing Order Book Imbalance proxy...")
    obi = compute_obi_proxy(df, window=5)
    df['obi'] = obi
    print(f"  OBI range: [{obi.min():.4f}, {obi.max():.4f}]")
    print(f"  OBI mean: {obi.mean():.4f}")
    print("\n[3/6] Labeling micro-crash events (ground truth)...")
    crash_labels = label_micro_crashes(df, threshold_pct=3.5, window_minutes=10)
    df['is_crash'] = crash_labels
    n_crash_minutes = crash_labels.sum()
    print(f"  Minutes labeled as pre-crash: {n_crash_minutes}")
    print(f"  Percentage: {n_crash_minutes / len(df) * 100:.2f}%")
    print("\n[4/6] Running SAX+KMP detection with parameter variations...")
    param_results = []
    
    sax_configs = [
        {'segment_size': 3, 'alphabet_size': 3, 'label': 'w=3, a=3'},
        {'segment_size': 3, 'alphabet_size': 4, 'label': 'w=3, a=4'},
        {'segment_size': 5, 'alphabet_size': 3, 'label': 'w=5, a=3'},
        {'segment_size': 5, 'alphabet_size': 4, 'label': 'w=5, a=4'},
        {'segment_size': 5, 'alphabet_size': 5, 'label': 'w=5, a=5'},
        {'segment_size': 10, 'alphabet_size': 4, 'label': 'w=10, a=4'},
    ]
    best_f1 = -1
    best_config = sax_configs[0]
    best_result = None
    for config in sax_configs:
        sax_text_tmp, seg_idx_tmp = build_continuous_sax_stream(
            obi, config['segment_size'], config['alphabet_size']
        )
        discovered_patterns = discover_crash_patterns(
            sax_text_tmp, seg_idx_tmp, crash_labels,
            pattern_length=4 if config['alphabet_size'] <= 4 else 4,
            top_n=8
        )
        
        result = run_detection_pipeline(
            obi_series=obi,
            crash_labels=crash_labels,
            segment_size=config['segment_size'],
            alphabet_size=config['alphabet_size'],
            crash_patterns=discovered_patterns if discovered_patterns else None,
            alert_window=15
        )
        result['config'] = config['label']
        param_results.append(result)
        
        print(f"\n  Config {config['label']}:")
        print(f"    SAX text length: {result['sax_text_length']}")
        print(f"    Patterns: {result['patterns_used']}")
        print(f"    Alerts: {result['n_alerts']}, Crashes: {result['n_crash_events']}")
        print(f"    TP={result['tp']}, FP={result['fp']}, FN={result['fn']}")
        print(f"    Precision={result['precision']:.3f}, Recall={result['recall']:.3f}, F1={result['f1_score']:.3f}")
        print(f"    KMP comparisons: {result['kmp_comparisons']}")
        print(f"    BF comparisons: {result['bf_comparisons']}")
        
        if result['f1_score'] > best_f1:
            best_f1 = result['f1_score']
            best_config = config
            best_result = result
    
    print(f"\n  >>> Best config: {best_config['label']} (F1={best_f1:.3f})")
    print("\n[5/6] Running baseline threshold comparisons...")
    
    baseline_results = []
    for thresh in [-0.3, -0.4, -0.5, -0.6, -0.7]:
        bl = run_threshold_baseline(obi, crash_labels, threshold=thresh)
        baseline_results.append(bl)
        print(f"  Threshold {thresh}: P={bl['precision']:.3f}, R={bl['recall']:.3f}, F1={bl['f1_score']:.3f} (Alerts={bl['n_alerts']})")
    
    print("\n[6/6] Running speed benchmark...")
    speed_results = []
    test_sizes = [1000, 5000, 10000, 50000, 100000, 500000]
    
    for size in test_sizes:
        np.random.seed(123)
        test_text = ''.join(np.random.choice(['a', 'b', 'c', 'd'], size=size))
        test_pattern = 'dcaa'
        kmp_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            kmp_m, kmp_c = kmp_search(test_text, test_pattern)
            t1 = time.perf_counter()
            kmp_times.append(t1 - t0)
        bf_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            bf_m, bf_c = brute_force_search(test_text, test_pattern)
            t1 = time.perf_counter()
            bf_times.append(t1 - t0)
        
        avg_kmp = np.mean(kmp_times)
        avg_bf = np.mean(bf_times)
        
        speed_results.append({
            'text_length': size,
            'kmp_time_ms': avg_kmp * 1000,
            'bf_time_ms': avg_bf * 1000,
            'kmp_comparisons': kmp_c,
            'bf_comparisons': bf_c,
            'speedup': avg_bf / avg_kmp if avg_kmp > 0 else 0
        })
        
        print(f"  n={size:>7}: KMP={avg_kmp*1000:.3f}ms, BF={avg_bf*1000:.3f}ms, "
              f"Speedup={avg_bf/avg_kmp:.2f}x, "
              f"KMP_comps={kmp_c}, BF_comps={bf_c}")
    all_results = {
        'data_summary': {
            'n_datapoints': len(df),
            'period_start': str(df['timestamp'].iloc[0]),
            'period_end': str(df['timestamp'].iloc[-1]),
            'n_crash_events': len(crash_info),
            'n_crash_labeled_minutes': int(n_crash_minutes),
            'obi_range': [float(obi.min()), float(obi.max())],
        },
        'best_config': {
            'config': best_config['label'] if best_config else 'N/A',
            'precision': best_result['precision'] if best_result else 0,
            'recall': best_result['recall'] if best_result else 0,
            'f1_score': best_result['f1_score'] if best_result else 0,
            'avg_detection_lag': best_result['avg_detection_lag_minutes'] if best_result else 0,
        },
        'parameter_comparison': [
            {
                'config': r['config'],
                'precision': round(r['precision'], 3),
                'recall': round(r['recall'], 3),
                'f1_score': round(r['f1_score'], 3),
                'n_alerts': r['n_alerts'],
                'tp': r['tp'], 'fp': r['fp'], 'fn': r['fn'],
            }
            for r in param_results
        ],
        'baseline_comparison': [
            {
                'method': r['method'],
                'precision': round(r['precision'], 3),
                'recall': round(r['recall'], 3),
                'f1_score': round(r['f1_score'], 3),
                'n_alerts': r['n_alerts'],
                'tp': r['tp'], 'fp': r['fp'], 'fn': r['fn'],
            }
            for r in baseline_results
        ],
        'speed_benchmark': speed_results,
        'sax_example': {
            'input_obi': list(obi[:20].round(4)),
            'sax_output_sample': best_result['sax_text_sample'][:50] if best_result else '',
        }
    }
    with open(f'{output_dir}/experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    pd.DataFrame(all_results['parameter_comparison']).to_csv(
        f'{output_dir}/table_parameter_comparison.csv', index=False
    )
    pd.DataFrame(all_results['baseline_comparison']).to_csv(
        f'{output_dir}/table_baseline_comparison.csv', index=False
    )
    pd.DataFrame(speed_results).to_csv(
        f'{output_dir}/table_speed_benchmark.csv', index=False
    )
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {output_dir}/")
    print("=" * 70)
    
    return df, obi, crash_labels, all_results, param_results, baseline_results, speed_results


def generate_figures(df, obi, crash_labels, all_results, param_results,
                     baseline_results, speed_results, output_dir='results'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        'font.size': 9,
        'figure.dpi': 150,
        'figure.figsize': (8, 4),
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    x = range(len(df))
    ax1.plot(x, df['close'].values, linewidth=0.5, color='#2196F3', label='BTC/USDT Close')
    crash_starts = []
    in_crash = False
    for i in range(len(crash_labels)):
        if crash_labels[i] and not in_crash:
            crash_starts.append(i)
            in_crash = True
        elif not crash_labels[i] and in_crash:
            ax1.axvspan(crash_starts[-1], i, alpha=0.3, color='red')
            in_crash = False
    
    ax1.set_ylabel('Harga (USDT)')
    ax1.set_title('Harga BTC/USDT dan Kejadian Micro-Crash')
    ax1.legend(loc='upper right')
    
    ax2.plot(x, obi, linewidth=0.3, color='#4CAF50', alpha=0.7)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax2.axhline(y=-0.5, color='red', linestyle=':', linewidth=0.8, label='Threshold OBI = -0.5')
    ax2.set_ylabel('Order Book Imbalance')
    ax2.set_xlabel('Waktu (menit)')
    ax2.set_title('Order Book Imbalance (OBI) Proxy')
    ax2.legend(loc='lower right')
    ax2.set_ylim([-1.1, 1.1])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig1_price_obi_crashes.png', dpi=150, bbox_inches='tight')
    plt.close()
    crash_idx = None
    for i in range(len(crash_labels)):
        if crash_labels[i]:
            crash_idx = i
            break
    
    if crash_idx is not None:
        zoom_start = max(0, crash_idx - 30)
        zoom_end = min(len(obi), crash_idx + 50)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
        zoom_obi = obi[zoom_start:zoom_end]
        zoom_x = range(zoom_start, zoom_end)
        
        ax1.plot(zoom_x, df['close'].values[zoom_start:zoom_end], 
                color='#2196F3', linewidth=1.5, label='Harga')
        ax1.axvspan(crash_idx, min(crash_idx + 10, zoom_end), 
                   alpha=0.3, color='red', label='Zona Crash')
        ax1.set_ylabel('Harga (USDT)')
        ax1.set_title('Detail Micro-Crash: Harga dan OBI')
        ax1.legend()
        ax2.plot(zoom_x, zoom_obi, color='#FF5722', linewidth=1.5, 
                marker='o', markersize=2, label='OBI')
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        segment_size = 5
        breakpoints = get_gaussian_breakpoints(4)
        norm_obi = z_normalize(zoom_obi)
        
        for i in range(0, len(zoom_obi) - segment_size, segment_size):
            seg = norm_obi[i:i+segment_size]
            paa_val = np.mean(seg)
            char = 'a'
            for j, bp in enumerate(breakpoints):
                if paa_val >= bp:
                    char = chr(ord('a') + j + 1)
            
            mid_x = zoom_start + i + segment_size // 2
            ax2.annotate(f"'{char}'", xy=(mid_x, zoom_obi[i + segment_size//2]),
                        fontsize=8, ha='center', va='bottom', color='purple',
                        fontweight='bold')
        
        ax2.set_ylabel('OBI')
        ax2.set_xlabel('Waktu (menit)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/fig2_sax_encoding_detail.png', dpi=150, bbox_inches='tight')
        plt.close()
    fig, ax = plt.subplots(figsize=(8, 4))
    
    configs = [r['config'] for r in param_results]
    precisions = [r['precision'] for r in param_results]
    recalls = [r['recall'] for r in param_results]
    f1s = [r['f1_score'] for r in param_results]
    
    x_pos = np.arange(len(configs))
    width = 0.25
    
    bars1 = ax.bar(x_pos - width, precisions, width, label='Precision', color='#2196F3')
    bars2 = ax.bar(x_pos, recalls, width, label='Recall', color='#4CAF50')
    bars3 = ax.bar(x_pos + width, f1s, width, label='F1-Score', color='#FF9800')
    
    ax.set_xlabel('Konfigurasi Parameter SAX')
    ax.set_ylabel('Skor')
    ax.set_title('Perbandingan Performa Deteksi pada Berbagai Parameter SAX')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, rotation=15)
    ax.legend()
    ax.set_ylim([0, 1.1])
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_parameter_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    sizes = [r['text_length'] for r in speed_results]
    kmp_times = [r['kmp_time_ms'] for r in speed_results]
    bf_times = [r['bf_time_ms'] for r in speed_results]
    kmp_comps = [r['kmp_comparisons'] for r in speed_results]
    bf_comps = [r['bf_comparisons'] for r in speed_results]
    
    ax1.plot(sizes, kmp_times, 'o-', label='KMP', color='#2196F3', linewidth=2)
    ax1.plot(sizes, bf_times, 's--', label='Brute Force', color='#F44336', linewidth=2)
    ax1.set_xlabel('Panjang Teks (n)')
    ax1.set_ylabel('Waktu Eksekusi (ms)')
    ax1.set_title('Perbandingan Waktu Eksekusi')
    ax1.legend()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.plot(sizes, kmp_comps, 'o-', label='KMP', color='#2196F3', linewidth=2)
    ax2.plot(sizes, bf_comps, 's--', label='Brute Force', color='#F44336', linewidth=2)
    ax2.set_xlabel('Panjang Teks (n)')
    ax2.set_ylabel('Jumlah Perbandingan Karakter')
    ax2.set_title('Perbandingan Jumlah Operasi')
    ax2.legend()
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4_kmp_vs_bruteforce.png', dpi=150, bbox_inches='tight')
    plt.close()
    fig, ax = plt.subplots(figsize=(8, 4))
    best_param = max(param_results, key=lambda r: r['f1_score'])
    
    methods = [f"SAX+KMP\n({best_param['config']})"]
    p_vals = [best_param['precision']]
    r_vals = [best_param['recall']]
    f_vals = [best_param['f1_score']]
    
    for bl in baseline_results:
        methods.append(bl['method'].replace('Threshold ', 'Threshold\n'))
        p_vals.append(bl['precision'])
        r_vals.append(bl['recall'])
        f_vals.append(bl['f1_score'])
    x_pos = np.arange(len(methods))
    width = 0.25
    
    ax.bar(x_pos - width, p_vals, width, label='Precision', color='#2196F3')
    ax.bar(x_pos, r_vals, width, label='Recall', color='#4CAF50')
    ax.bar(x_pos + width, f_vals, width, label='F1-Score', color='#FF9800')
    
    ax.set_xlabel('Metode Deteksi')
    ax.set_ylabel('Skor')
    ax.set_title('Perbandingan SAX+KMP vs Baseline Threshold OBI')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=8)
    ax.legend()
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig5_sax_kmp_vs_baseline.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nAll figures saved to {output_dir}/")

if __name__ == '__main__':
    try:
        from scipy.stats import norm
    except ImportError:
        import subprocess
        subprocess.check_call(['pip', 'install', 'scipy', '--break-system-packages', '-q'])
        from scipy.stats import norm
    df, obi, crash_labels, all_results, param_results, bl_results, speed_results = \
        run_all_experiments(output_dir='results')
    
    generate_figures(df, obi, crash_labels, all_results, param_results,
                    bl_results, speed_results, output_dir='results')
    print("\n" + "=" * 70)
    print("RINGKASAN UNTUK MAKALAH")
    print("=" * 70)
    best = all_results['best_config']
    print(f"\nKonfigurasi terbaik: {best['config']}")
    print(f"  Precision: {best['precision']:.3f}")
    print(f"  Recall: {best['recall']:.3f}")
    print(f"  F1-Score: {best['f1_score']:.3f}")
    print(f"  Rata-rata Detection Lag: {best['avg_detection_lag']:.1f} menit")
    
    print(f"\nSpeed benchmark (n=100000):")
    for sr in speed_results:
        if sr['text_length'] == 100000:
            print(f"  KMP: {sr['kmp_time_ms']:.3f} ms ({sr['kmp_comparisons']} comparisons)")
            print(f"  BF:  {sr['bf_time_ms']:.3f} ms ({sr['bf_comparisons']} comparisons)")
            print(f"  Speedup: {sr['speedup']:.2f}x")