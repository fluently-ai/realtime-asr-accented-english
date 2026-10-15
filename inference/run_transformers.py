# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate==1.14.0",
#   "librosa==1.0.0",
#   "python-dotenv==1.1.0",
#   "scipy==1.18.1",
#   "soundfile==0.14.0",
#   "torch==2.13.0",
#   "transformers==5.14.0",
# ]
# ///
"""Run local Transformers ASR models through the standard offline probe."""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from run import main

if __name__ == "__main__":
    raise SystemExit(main())
