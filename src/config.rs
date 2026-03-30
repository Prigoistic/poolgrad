use std::sync::OnceLock;

/// Global runtime knobs (via env vars), cached on first access.
///
/// These are intentionally process-global: PoolGrad is a single-process research binary,
/// and reading env vars on hot paths would add avoidable overhead.
pub fn parallel_enabled() -> bool {
    static PAR_ENABLED: OnceLock<bool> = OnceLock::new();
    *PAR_ENABLED.get_or_init(|| {
        std::env::var("POOLGRAD_PAR")
            .ok()
            .as_deref()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true)
    })
}

/// Minimum `m * p` (or similar work proxy) before Rayon parallelism is enabled.
///
/// Default: 16k elements.
pub fn par_min_elems() -> usize {
    static PAR_MIN_ELEMS: OnceLock<usize> = OnceLock::new();
    *PAR_MIN_ELEMS.get_or_init(|| {
        std::env::var("POOLGRAD_PAR_MIN_ELEMS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(16 * 1024)
    })
}

/// Maximum size for which MP (block-level Strassen-form) transforms are attempted.
///
/// Above this, the MP kernel intentionally falls back to tiled accumulation.
pub fn mp_max_size() -> usize {
    static MP_MAX_SIZE: OnceLock<usize> = OnceLock::new();
    *MP_MAX_SIZE.get_or_init(|| {
        std::env::var("POOLGRAD_MP_MAX_SIZE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(128)
    })
}
