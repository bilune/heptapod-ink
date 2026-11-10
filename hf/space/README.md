---
title: Heptapod Ink
emoji: 🦑
colorFrom: gray
colorTo: indigo
sdk: static
app_file: index.html
license: cc-by-nc-4.0
short_description: Write English as Arrival-style logograms, read them back
custom_headers:
  cross-origin-embedder-policy: credentialless
  cross-origin-opener-policy: same-origin
---

# Heptapod Ink

Type an English sentence. A small neural writer turns it into one ring logogram of ink marks from the film *Arrival*, and a neural reader reads the image back. Everything runs in your browser (onnxruntime-web, WebGPU when available). The first visit downloads about 170 MB of models.

Code and write-up: https://github.com/bilune/heptapod-ink

Ink: Arrival logograms by Martine Bertrand, released by Wolfram Research under CC BY-NC 4.0. Non-commercial use.
