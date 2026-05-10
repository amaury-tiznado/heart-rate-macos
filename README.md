# Heart Beat Detector

Proof of Concept

THIS IS NOT A MEDICAL DEVICE. DO NOT USE TO MAKE MEDICAL DECISIONS.

Heart rate detector using the Macbook pressure sensor on the trackpad.

## Requirements

- macOS
- MacBook trackpad
- Python 3.10+

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/trackpad-heartbeat-detector.git
cd trackpad-heartbeat-detector
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 trackpad_heart_rate.py
```

Rest 5 fingers (without the palm, only fingertips) in a comfortable position on the trackpad. The user should be able to limit movement as much as possible by resting the rest of their arm on a desk or other stationary object in order to get accurate results.
## Other Commands

Run without the GUI:

```bash
python3 trackpad_heart_rate.py --simulate
```

Print raw touch contact values:

```bash
python3 trackpad_heart_rate.py --touch-probe
```

Use pressure events only:

```bash
python3 trackpad_heart_rate.py --pressure-only
```

## License

This project is source-available, not open source. Personal non-commercial evaluation only. Commercial use, redistribution, derivative works, and competing implementations based on the project are prohibited. See [LICENSE](LICENSE).
