# NST — Neural Style Transfer

A small Python project for **Neural Style Transfer (NST)** — blending the **content** of one image with the **style** of another to generate a stylized result.

---

## Features
- Load a **content image** (photo) and a **style image** (painting/texture).
- Generate a stylized output image.
- Keep example inputs in `ressources/` and save outputs to `results/`.

---

## Project Structure

- `main.py` — main entry point
- `requirements.txt` — Python dependencies
- `commands` — example commands / run cheatsheet (if included in the repo)
- `ressources/` — input images (content/style)
- `results/` — generated outputs

---

## Requirements
- Python 3.11+ recommended
- `pip`

---

## Installation

```bash
# Clone the repository
git clone https://github.com/qualicc/NST.git
cd NST

# (Optional) Create and activate a virtual environment
python -3.11 -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
