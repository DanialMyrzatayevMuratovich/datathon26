"""Video analysis (ML stream, zone of danial). The backend calls only `analyze`; everything else is internal.

    from vision import analyze
    result = analyze("/path/video.mov", progress=lambda frac, stage: ...)   # -> contract.schemas.VideoAnalysis

Until the real pipeline lands, `analyze` returns a realistic mock (real JPEG frame from the video, real rule from the
catalog), so the API and the UI can be built and deployed now.
Switch: env VISION_ENGINE=mock|real (default: mock until the real engine is merged).
"""

from vision.pipeline import analyze

__all__ = ["analyze"]
