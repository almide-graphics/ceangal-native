//! Tween engine — Rust port of ceangal-anime easing/tween/stagger

use std::f64::consts::PI;

// ── Easing functions ──

pub fn linear(t: f64) -> f64 { t }
pub fn ease_out_cubic(t: f64) -> f64 { 1.0 - (1.0 - t).powi(3) }
pub fn ease_out_elastic(t: f64) -> f64 {
    if t == 0.0 || t == 1.0 { return t; }
    let p = 0.3;
    let s = p / 4.0;
    (2.0f64).powf(-10.0 * t) * ((t - s) * 2.0 * PI / p).sin() + 1.0
}
pub fn ease_out_bounce(t: f64) -> f64 {
    let n = 7.5625;
    let d = 2.75;
    if t < 1.0 / d { n * t * t }
    else if t < 2.0 / d { let u = t - 1.5 / d; n * u * u + 0.75 }
    else if t < 2.5 / d { let u = t - 2.25 / d; n * u * u + 0.9375 }
    else { let u = t - 2.625 / d; n * u * u + 0.984375 }
}
pub fn ease_in_out_sine(t: f64) -> f64 { -(( PI * t).cos() - 1.0) / 2.0 }
pub fn ease_in_out_back(t: f64) -> f64 {
    let c = 1.70158 * 1.525;
    if t < 0.5 {
        let u = 2.0 * t;
        u * u * ((c + 1.0) * u - c) / 2.0
    } else {
        let u = 2.0 * t - 2.0;
        (u * u * ((c + 1.0) * u + c) + 2.0) / 2.0
    }
}
pub fn ease_out_expo(t: f64) -> f64 {
    if t == 1.0 { 1.0 } else { 1.0 - (2.0f64).powf(-10.0 * t) }
}
pub fn ease_out_quart(t: f64) -> f64 { 1.0 - (1.0 - t).powi(4) }
pub fn ease_in_circ(t: f64) -> f64 { 1.0 - (1.0 - t * t).sqrt() }
pub fn ease_in_out_quad(t: f64) -> f64 {
    if t < 0.5 { 2.0 * t * t } else { 1.0 - (-2.0 * t + 2.0).powi(2) / 2.0 }
}

pub type EasingFn = fn(f64) -> f64;

pub const EASINGS: &[(&str, EasingFn)] = &[
    ("linear", linear),
    ("outCubic", ease_out_cubic),
    ("inOutQuad", ease_in_out_quad),
    ("outElastic", ease_out_elastic),
    ("outBounce", ease_out_bounce),
    ("inOutBack", ease_in_out_back),
    ("outExpo", ease_out_expo),
    ("inOutSine", ease_in_out_sine),
    ("outQuart", ease_out_quart),
    ("inCirc", ease_in_circ),
];

// ── Tween ──

#[derive(Clone)]
pub struct Tween {
    pub from: f64,
    pub to: f64,
    pub duration: f64,  // ms
    pub delay: f64,
    pub easing: EasingFn,
    pub alternate: bool,
    pub elapsed: f64,
    pub loop_count: i32, // current
}

impl Tween {
    pub fn new(from: f64, to: f64, duration: f64, delay: f64, easing: EasingFn) -> Self {
        Self { from, to, duration, delay, easing, alternate: true, elapsed: 0.0, loop_count: 0 }
    }

    pub fn tick(&mut self, dt_ms: f64) {
        self.elapsed += dt_ms;
        let active = self.elapsed - self.delay;
        if active >= self.duration {
            let overflow = active - self.duration;
            self.elapsed = self.delay + overflow;
            self.loop_count += 1;
        }
    }

    pub fn value(&self) -> f64 {
        let active = self.elapsed - self.delay;
        if active < 0.0 { return self.from; }
        let t = (active / self.duration).clamp(0.0, 1.0);
        let directed = if self.alternate && self.loop_count % 2 == 1 { 1.0 - t } else { t };
        let eased = (self.easing)(directed);
        self.from + (self.to - self.from) * eased
    }
}

// ── Stagger ──

pub fn stagger_from_center(index: usize, total: usize, step: f64) -> f64 {
    if total <= 1 { return 0.0; }
    let center = (total - 1) as f64 / 2.0;
    let dist = (index as f64 - center).abs();
    dist * step
}

// ── Particle (multi-property tween) ──

pub struct Particle {
    pub x: Tween,
    pub y: Tween,
    pub scale: Tween,
    pub opacity: Tween,
    pub hue: Tween,
}

impl Particle {
    pub fn tick(&mut self, dt: f64) {
        self.x.tick(dt);
        self.y.tick(dt);
        self.scale.tick(dt);
        self.opacity.tick(dt);
        self.hue.tick(dt);
    }
}
