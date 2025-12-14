from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import joblib


@dataclass
class _LightMLModel:
    model_dir: Path
    model_filename: str

    _vectorizer: object = None
    _svd: Optional[object] = None
    _model: object = None

    def load(self) -> "_LightMLModel":
        self.model_dir = Path(self.model_dir)

        vec_path = self.model_dir / "tfidf_vectorizer.joblib"
        svd_path = self.model_dir / "svd_model.joblib"
        model_path = self.model_dir / self.model_filename

        if not vec_path.exists():
            raise FileNotFoundError(f"Missing tfidf_vectorizer.joblib at {vec_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file at {model_path}")

        self._vectorizer = joblib.load(vec_path)
        self._svd = joblib.load(svd_path) if svd_path.exists() else None
        self._model = joblib.load(model_path)
        return self

    @staticmethod
    def _join(rule: str, body: str) -> str:
        # 必须与训练脚本一致
        return f"{(rule or '').strip()} [SEP] {(body or '').strip()}"

    def predict_proba(self, rules: List[str], bodies: List[str]) -> List[float]:
        if self._model is None:
            self.load()

        texts = [self._join(r, b) for r, b in zip(rules, bodies)]
        X = self._vectorizer.transform(texts)
        if self._svd is not None:
            X = self._svd.transform(X)

        proba = self._model.predict_proba(X)[:, 1].tolist()
        # 数值安全裁剪
        return [0.0 if p < 0 else 1.0 if p > 1 else float(p) for p in proba]


# ===== 单例缓存（避免每次请求重复 load） =====
_BASE: Optional[_LightMLModel] = None
_EXT: Optional[_LightMLModel] = None


def base_head() -> _LightMLModel:
    global _BASE
    if _BASE is None:
        _BASE = _LightMLModel(
            model_dir=Path("models/light_ml/base"),
            model_filename="rf_model.joblib",
        ).load()
    return _BASE


def ext_strict_head() -> _LightMLModel:
    global _EXT
    if _EXT is None:
        _EXT = _LightMLModel(
            model_dir=Path("models/light_ml/ext_strict"),
            model_filename="rf_model_ext_strict.joblib",
        ).load()
    return _EXT
