"""
YouTube Short: Brain Fact Video
1080x1920 — 18 seconds total
Scene 1: 3s | Scene 2: 7s | Scene 3: 5s | Scene 4: 3s
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, math

# ── Output dir ─────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "public")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "brain_short.mp4")

W, H = 1080, 1920
FPS  = 30

# ── Colour palette ─────────────────────────────────────────────────────────
DEEP_BLUE    = (10,  15,  45)
MID_BLUE     = (20,  40,  100)
BRIGHT_BLUE  = (50,  120, 220)
CYAN         = (0,   210, 255)
PURPLE       = (120, 30,  180)
DARK_PURPLE  = (40,  10,  80)
WHITE        = (255, 255, 255)
YELLOW       = (255, 230, 0)
ORANGE       = (255, 140, 0)

# ── Font helpers ────────────────────────────────────────────────────────────
def get_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def draw_text_centered(draw, text, y, font, fill=WHITE, shadow=True):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = (W - tw) // 2
    if shadow:
        draw.text((x + 3, y + 3), text, font=font, fill=(0, 0, 0, 180))
    draw.text((x, y), text, font=font, fill=fill)

def draw_text_block(draw, lines, y_start, font, fill=WHITE, line_gap=20, shadow=True):
    for line in lines:
        draw_text_centered(draw, line, y_start, font, fill, shadow)
        bbox = draw.textbbox((0, 0), line, font=font)
        y_start += (bbox[3] - bbox[1]) + line_gap
    return y_start

# ── Background generators ───────────────────────────────────────────────────

def gradient_bg(color_top, color_bottom, t=0.0, noise=0):
    """Vertical gradient with optional subtle noise."""
    arr = np.zeros((H, W, 3), dtype=np.float32)
    ys  = np.linspace(0, 1, H)[:, None]
    for c in range(3):
        arr[:, :, c] = (color_top[c] * (1 - ys) + color_bottom[c] * ys) / 255.0
    if noise:
        arr += np.random.uniform(-noise, noise, arr.shape)
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def draw_particles(img, count, t, color, max_r=6):
    """Floating dot particles — positions driven by t (time 0‥1)."""
    draw = ImageDraw.Draw(img)
    rng  = np.random.default_rng(42)
    xs   = rng.integers(50, W - 50, count)
    ys   = rng.integers(50, H - 50, count)
    speeds = rng.uniform(0.3, 1.0, count)
    for i in range(count):
        offset = int(30 * math.sin(t * 2 * math.pi * speeds[i] + i))
        rx = xs[i]
        ry = int(ys[i] + offset) % H
        r  = rng.integers(2, max_r)
        draw.ellipse([rx - r, ry - r, rx + r, ry + r], fill=(*color, 160))
    return img


def draw_neural_net(img, t):
    """Animated neural-network-style lines & dots."""
    draw = ImageDraw.Draw(img)
    rng  = np.random.default_rng(7)
    nodes = [(rng.integers(100, W-100), rng.integers(100, H-100)) for _ in range(22)]
    # connections
    for i, (x1, y1) in enumerate(nodes):
        for j, (x2, y2) in enumerate(nodes):
            if j <= i:
                continue
            dist = math.hypot(x2-x1, y2-y1)
            if dist < 350:
                alpha = int(80 * (1 - dist / 350))
                pulse = 0.5 + 0.5 * math.sin(t * 2 * math.pi + i + j)
                a2 = int(alpha * pulse)
                draw.line([x1, y1, x2, y2], fill=(*CYAN, a2), width=1)
    # nodes
    for i, (x, y) in enumerate(nodes):
        pulse = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 0.7 + i)
        r = int(4 + 8 * pulse)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(*CYAN, 200))
    return img


def draw_brain_shape(img, t):
    """Simple stylised brain outline with glow pulse."""
    draw  = ImageDraw.Draw(img, "RGBA")
    cx, cy = W // 2, H // 2 - 80
    scale  = 1.0 + 0.03 * math.sin(t * 2 * math.pi)

    # outer glow rings
    for r in range(260, 200, -15):
        sr = int(r * scale)
        alpha = int(30 * (1 - (r - 200) / 60))
        draw.ellipse([cx-sr, cy-int(sr*0.7), cx+sr, cy+int(sr*0.7)],
                     outline=(*CYAN, alpha), width=3)

    # left hemisphere
    lx, ly = cx - 60, cy
    sr = int(200 * scale)
    sy = int(150 * scale)
    draw.ellipse([lx-sr, ly-sy, lx+sr, ly+sy], outline=(*BRIGHT_BLUE, 200), width=4)

    # right hemisphere
    rx2, ry = cx + 60, cy
    draw.ellipse([rx2-sr, ry-sy, rx2+sr, ry+sy], outline=(*CYAN, 200), width=4)

    # corpus callosum divider
    draw.line([cx, cy - int(80*scale), cx, cy + int(80*scale)],
              fill=(*WHITE, 100), width=3)
    return img


def draw_eye(img, t):
    """Stylised eye that blinks periodically."""
    draw = ImageDraw.Draw(img, "RGBA")
    cx, cy = W // 2, H // 2 - 50
    # blink: eye closes when t ≈ 0 or 1
    blink = abs(math.sin(t * math.pi))  # 0 closed → 1 open
    ew, eh = 320, int(160 * blink + 4)

    # whites
    draw.ellipse([cx-ew, cy-eh, cx+ew, cy+eh], fill=(*WHITE, 220))
    # iris
    ir = int(90 * blink)
    draw.ellipse([cx-ir, cy-ir, cx+ir, cy+ir], fill=(*BRIGHT_BLUE, 230))
    # pupil
    pr = int(40 * blink)
    draw.ellipse([cx-pr, cy-pr, cx+pr, cy+pr], fill=(5, 5, 15, 240))
    # highlight
    if blink > 0.3:
        draw.ellipse([cx+15, cy-30, cx+45, cy-5], fill=(*WHITE, 200))

    # eyelashes / outline
    draw.ellipse([cx-ew, cy-eh, cx+ew, cy+eh], outline=(50, 50, 70, 180), width=4)
    return img


def draw_speed_lines(img, t):
    """Tech-themed radial speed-lines — suggests fast processing."""
    draw = ImageDraw.Draw(img, "RGBA")
    cx, cy = W // 2, H // 2
    count  = 48
    for i in range(count):
        angle  = (i / count) * 2 * math.pi + t * math.pi * 0.4
        inner  = 220 + 30 * math.sin(t * 2 * math.pi + i)
        outer  = 480 + 60 * math.cos(t * math.pi + i)
        x1 = cx + inner * math.cos(angle)
        y1 = cy + inner * math.sin(angle)
        x2 = cx + outer * math.cos(angle)
        y2 = cy + outer * math.sin(angle)
        alpha = 60 + int(60 * math.sin(t * 2 * math.pi + i * 0.5))
        draw.line([x1, y1, x2, y2], fill=(*CYAN, alpha), width=2)
    return img


def draw_dna_helix(img, t):
    """Simple double-helix scroll on left edge — decorative."""
    draw = ImageDraw.Draw(img, "RGBA")
    for step in range(0, H, 12):
        ofs = t * 2 * math.pi
        a1  = step / 120 + ofs
        a2  = a1 + math.pi
        x1  = 60 + int(35 * math.sin(a1))
        x2  = 60 + int(35 * math.sin(a2))
        y   = step
        r   = 5
        draw.ellipse([x1-r, y-r, x1+r, y+r], fill=(*CYAN, 180))
        draw.ellipse([x2-r, y-r, x2+r, y+r], fill=(*ORANGE, 180))
        if abs(math.sin(a1)) < 0.25:
            draw.line([x1, y, x2, y], fill=(*WHITE, 80), width=2)
    return img

# ── Scene frame factories ────────────────────────────────────────────────────

def scene1_frame(t):
    """Did You Know? — blue→purple gradient, particles, question-mark pulse."""
    img  = gradient_bg(DEEP_BLUE, DARK_PURPLE, t)
    img  = img.convert("RGBA")
    img  = draw_particles(img, 60, t, CYAN)

    draw = ImageDraw.Draw(img)

    # giant pulsing "?"
    pulse = 1.0 + 0.08 * math.sin(t * 2 * math.pi)
    qsize = int(340 * pulse)
    qfont = get_font(qsize)
    bbox  = draw.textbbox((0, 0), "?", font=qfont)
    qx = (W - (bbox[2]-bbox[0])) // 2
    qy = H // 2 - (bbox[3]-bbox[1]) // 2 - 100
    draw.text((qx+6, qy+6), "?", font=qfont, fill=(0, 0, 0, 120))
    draw.text((qx, qy), "?", font=qfont, fill=(*CYAN, 60))

    # main text
    f1 = get_font(110)
    f2 = get_font(90)
    draw_text_centered(draw, "Did You Know?", H // 2 + 220, f1, fill=WHITE)
    draw_text_centered(draw, "🤯", H // 2 + 360, f2, fill=YELLOW)

    # top accent bar
    draw.rectangle([100, 80, W-100, 90], fill=(*CYAN, 200))
    draw.rectangle([100, H-90, W-100, H-80], fill=(*CYAN, 200))

    return np.array(img.convert("RGB"))


def scene2_frame(t):
    """Brain fact — neural network bg, brain shape, main fact text."""
    img  = gradient_bg(DEEP_BLUE, (5, 5, 30), t)
    img  = img.convert("RGBA")
    img  = draw_neural_net(img, t)
    img  = draw_brain_shape(img, t)

    draw = ImageDraw.Draw(img)

    # label above brain
    lf = get_font(52)
    draw_text_centered(draw, "YOUR BRAIN", 180, lf, fill=CYAN)

    # fact block — positioned below brain
    f_big = get_font(72)
    f_med = get_font(62)

    lines = [
        ("Your brain can recognize", f_med, WHITE),
        ("a familiar face in", f_med, WHITE),
        ("just", f_big, WHITE),
    ]
    y = H // 2 + 300
    for text, font, color in lines:
        draw_text_centered(draw, text, y, font, fill=color)
        bbox = draw.textbbox((0, 0), text, font=font)
        y += (bbox[3] - bbox[1]) + 18

    # highlighted number
    hf = get_font(160)
    draw_text_centered(draw, "13 ms!", y, hf, fill=YELLOW)

    # tiny sub-label
    sf = get_font(44)
    bbox = draw.textbbox((0, 0), "13 ms!", font=hf)
    draw_text_centered(draw, "milliseconds", y + (bbox[3]-bbox[1]) + 10, sf, fill=CYAN)

    return np.array(img.convert("RGB"))


def scene3_frame(t):
    """Eye blink + speed lines — faster than blink."""
    img  = gradient_bg(DEEP_BLUE, (10, 5, 40), t)
    img  = img.convert("RGBA")
    img  = draw_speed_lines(img, t)
    img  = draw_eye(img, t)

    draw = ImageDraw.Draw(img)

    f1 = get_font(80)
    f2 = get_font(80)

    y = H // 2 + 310
    draw_text_centered(draw, "That's faster than", y, f1, fill=WHITE)
    y += 100
    draw_text_centered(draw, "the blink of an eye", y, f2, fill=WHITE)

    # emoji
    ef = get_font(110)
    y += 120
    draw_text_centered(draw, "👀", y, ef, fill=WHITE)

    return np.array(img.convert("RGB"))


def scene4_frame(t):
    """CTA — abstract blue bg, dna helix, follow text, rocket."""
    img  = gradient_bg(MID_BLUE, DEEP_BLUE, t)
    img  = img.convert("RGBA")
    img  = draw_particles(img, 80, t, BRIGHT_BLUE, max_r=8)
    img  = draw_dna_helix(img, t)

    draw = ImageDraw.Draw(img)

    # glowing circle behind rocket
    cx, cy_c = W // 2, H // 2 - 80
    pulse = 1.0 + 0.07 * math.sin(t * 2 * math.pi)
    for r in range(int(200*pulse), int(120*pulse), -20):
        alpha = int(40 * (1 - (r - 120*pulse) / (80*pulse)))
        draw.ellipse([cx-r, cy_c-r, cx+r, cy_c+r], fill=(*BRIGHT_BLUE, alpha))

    # rocket
    rf = get_font(260)
    bbox = draw.textbbox((0, 0), "🚀", font=rf)
    rx = (W - (bbox[2]-bbox[0])) // 2
    ry = cy_c - (bbox[3]-bbox[1]) // 2
    draw.text((rx, ry), "🚀", font=rf)

    # CTA text
    f1 = get_font(78)
    f2 = get_font(68)
    f3 = get_font(90)

    y = H // 2 + 240
    draw_text_centered(draw, "Follow", y, f1, fill=WHITE)
    y += 100
    draw_text_centered(draw, "Daily Upgrade", y, f3, fill=YELLOW)
    y += 120
    draw_text_centered(draw, "For More Amazing Facts", y, f2, fill=CYAN)

    # accent lines
    draw.rectangle([80, y + 100, W-80, y + 106], fill=(*YELLOW, 200))

    return np.array(img.convert("RGB"))


# ── Main render ──────────────────────────────────────────────────────────────

def make_video():
    from moviepy import VideoClip, AudioFileClip, CompositeAudioClip
    from moviepy.audio.AudioClip import AudioArrayClip

    scenes = [
        (scene1_frame, 3),
        (scene2_frame, 7),
        (scene3_frame, 5),
        (scene4_frame, 3),
    ]

    def make_scene_clip(frame_fn, duration):
        def make_frame(t):
            progress = t / duration
            return frame_fn(progress)
        clip = VideoClip(make_frame, duration=duration)
        clip = clip.with_fps(FPS)
        return clip

    from moviepy import concatenate_videoclips

    clips = [make_scene_clip(fn, dur) for fn, dur in scenes]
    final = concatenate_videoclips(clips, method="compose")

    # ── background music: generate a gentle pulsing tone ──────────────────
    sr     = 44100
    total  = int(sr * final.duration)
    t_arr  = np.linspace(0, final.duration, total)

    # soft ambient chord (A minor feel): 220 Hz + 330 Hz + 440 Hz
    wave = (
        0.25 * np.sin(2 * np.pi * 220 * t_arr) +
        0.15 * np.sin(2 * np.pi * 330 * t_arr) +
        0.10 * np.sin(2 * np.pi * 440 * t_arr) +
        0.08 * np.sin(2 * np.pi * 110 * t_arr)
    )
    # slow amplitude pulse for "breathing" feel
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * t_arr)
    wave *= env * 0.15   # 15 % overall volume (≈ 10–20 % range asked for)

    # fade in / out 1 s each
    fade = int(sr * 1.0)
    wave[:fade]  *= np.linspace(0, 1, fade)
    wave[-fade:] *= np.linspace(1, 0, fade)

    stereo = np.column_stack([wave, wave]).astype(np.float32)
    audio  = AudioArrayClip(stereo, fps=sr)

    final  = final.with_audio(audio)

    print(f"Rendering {final.duration:.1f}s video to {OUTPUT_PATH} …")
    final.write_videofile(
        OUTPUT_PATH,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )
    print(f"Done! → {OUTPUT_PATH}")


if __name__ == "__main__":
    make_video()
