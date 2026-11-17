# Table S4 — Experiment configurations

Moved from the main text (Phase 5.1 compression). Full per-run settings are
in the experiment scripts and JSON files.

| Exp. | Shape(s) | \(T\) | \(N\) | \(\tau\) | \(\nu\) | Notes / \(\min\|d\|\) |
|---|---|---|---|---|---|---|
| E0 | all; E0-g: ellipse | 0.05 | 128,512; 256 | \(10^{-3}\) | 1,2 | gate; 0.9996 (E0-g) |
| E1 | circle, ellipse, flower | 0.5 | 64–2048 | \(10^{-4}\) | 2 | flower \(N=1024,2048\) abort at \(T\) |
| E2 | ellipse | 0.5 | 64–512 | \(2.5\times10^{-4}\) | 1,2,3 | ref \(N=2048\), \(\tau=6.25\times10^{-5}\) |
| E3 | circle; ellipse | 0.5 | 512; 64–512 | \(0.1/2^k\); \(0.05h\) | 1,2,3 | startup; coupling |
| E4 | ellipse, flower | 2,5 | 256,512 | \(2.5\times10^{-4}\), \(10^{-4}\); \(1.25\times10^{-3}\) | 1,2,3 | safe \(\min\|d\|\ge0.9995\) |
| E5 | ellipse | 0.2; 0.5 | 128–1024; 64–512 | 0.01; \(2.5\times10^{-4}\) | 1,2,3 | cost; \(0.999797\)–\(0.999918\) |
| E6 | circle, ellipse, flower | 0.5; 2 | 512; 256,512 | \(0.1/2^k\); \(\tau/h=0.02\)–5 | 1,2,3 | BGN1 baseline |
| E7 | circle, ellipse, flower | 0.1–1.0 | 64–512 | \(2.5\times10^{-4}\) | 1,2,3 | ref \(2048/1024\); flower \(t\le0.5\) |
| E8 | ellipse, flower | 0.5 | 64–512 | \(6.25\times10^{-5}\)–\(5\times10^{-3}\) | 2 | \((h,\tau)\) map |
| E9 | circle, ellipse | 0.5 | 512 | \(0.1/2^k\) | 2 | work–precision |

E0 also uses a superellipse \(p=8\), \(N=128\) as a curvature-discontinuity
diagnostic; it is not used in E1–E9.
